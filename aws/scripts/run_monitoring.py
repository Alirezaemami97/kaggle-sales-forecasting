"""Phase 7 launcher: monitor the batch archive, publish metrics, wire the retrain trigger.

Always: run the monitoring Processing job (portable drift + forecast-error code)
over the latest Phase-5 archive, read its headline JSON, and publish those as
CloudWatch custom metrics — the numbers a threshold can watch.

  --create-alarm   a CloudWatch alarm on ForecastWAPE > threshold.
  --wire-retrain   a DISABLED EventBridge rule: alarm ->ALARM -> StartPipelineExecution
                   on the Phase-6 pipeline. Trigger-based continuous training, the
                   complement to the Phase-6 schedule. Disabled so a breach can't
                   kick off runaway retraining until you deliberately enable it.

    python aws/scripts/run_monitoring.py --bucket <name> --role-arn <role> \
        --create-alarm --wire-retrain
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

NAMESPACE = "DemandForecasting"
ALARM_NAME = "demand-forecasting-forecast-wape"
RETRAIN_RULE_NAME = "demand-forecasting-ct-retrain"
ARCHIVE_MOUNT = "/opt/ml/processing/input/archive"
MONITOR_OUTPUT_MOUNT = "/opt/ml/processing/output/monitoring"


def metric_data(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """CloudWatch MetricData payload from the monitor's headline JSON. Pure —
    unit-tested, since a wrong metric name would make the alarm watch nothing."""
    return [
        {"MetricName": "ForecastWAPE", "Value": float(summary["forecast_wape"]), "Unit": "None"},
        {"MetricName": "MaxPSI", "Value": float(summary["max_psi"]), "Unit": "None"},
        {"MetricName": "DriftedFeatures", "Value": float(summary["n_drifted"]), "Unit": "Count"},
    ]


def retrain_event_pattern(alarm_name: str) -> dict[str, Any]:
    """EventBridge pattern matching the alarm flipping to ALARM. Pure — tested."""
    return {
        "source": ["aws.cloudwatch"],
        "detail-type": ["CloudWatch Alarm State Change"],
        "detail": {"alarmName": [alarm_name], "state": {"value": ["ALARM"]}},
    }


def latest_archive_uri(s3: Any, bucket: str) -> str:
    """Newest forecasts/<run>/archive.parquet (run names are timestamp-sorted)."""
    keys = [
        o["Key"]
        for o in s3.list_objects_v2(Bucket=bucket, Prefix="forecasts/").get("Contents", [])
        if o["Key"].endswith("archive.parquet")
    ]
    if not keys:
        raise FileNotFoundError(f"No forecasts/<run>/archive.parquet under s3://{bucket}/")
    return f"s3://{bucket}/{max(keys)}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--archive-uri", help="Archive to monitor; defaults to the latest")
    parser.add_argument("--wape-threshold", type=float, default=0.75, help="Alarm fires above this")
    parser.add_argument("--max-series", type=int, default=3000)
    parser.add_argument("--image-uri", help="Processing image; defaults to account's :latest")
    parser.add_argument("--create-alarm", action="store_true", help="Create the ForecastWAPE alarm")
    parser.add_argument(
        "--wire-retrain", action="store_true", help="Create the DISABLED alarm->retrain rule"
    )
    parser.add_argument(
        "--retrain-role-arn", help="Role EventBridge assumes (defaults to --role-arn)"
    )
    args = parser.parse_args()

    import boto3
    import sagemaker_compat
    from pipeline import PIPELINE_NAME
    from run_evaluation import CODE_MOUNT, FEATURES_MOUNT, stage_code
    from run_sagemaker_training import default_image_uri
    from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    boto_session = boto3.Session()
    s3 = boto_session.client("s3")
    uri = default_image_uri(boto_session, args.image_uri)
    run_name = f"monitor-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
    archive_uri = args.archive_uri or latest_archive_uri(s3, args.bucket)
    code_uri = stage_code(s3, args.bucket, run_name)
    output_uri = f"s3://{args.bucket}/monitoring/{run_name}/"

    processor = Processor(
        role=args.role_arn,
        image_uri=uri,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        entrypoint=["python3", f"{CODE_MOUNT}/monitor_entry.py"],
        env={"PYTHONPATH": CODE_MOUNT, "PYTHONUNBUFFERED": "1"},
        base_job_name="demand-forecasting-monitor",
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )
    processor.run(
        inputs=[
            ProcessingInput(source=code_uri, destination=CODE_MOUNT, input_name="code"),
            ProcessingInput(
                source=f"s3://{args.bucket}/features/", destination=FEATURES_MOUNT,
                input_name="features",
            ),
            ProcessingInput(source=archive_uri, destination=ARCHIVE_MOUNT, input_name="archive"),
        ],
        outputs=[
            ProcessingOutput(
                source=MONITOR_OUTPUT_MOUNT, destination=output_uri, output_name="monitoring"
            )
        ],
        arguments=[
            "--archive", f"{ARCHIVE_MOUNT}/archive.parquet",
            "--features-dir", FEATURES_MOUNT,
            "--output-dir", MONITOR_OUTPUT_MOUNT,
            "--config", f"{CODE_MOUNT}/config.yaml",
            "--max-series", str(args.max_series),
        ],
    )

    obj = s3.get_object(Bucket=args.bucket, Key=f"monitoring/{run_name}/monitoring.json")
    summary = json.loads(obj["Body"].read())
    logger.info("Monitoring summary: %s", summary)

    cw = boto_session.client("cloudwatch")
    data = metric_data(summary)
    cw.put_metric_data(Namespace=NAMESPACE, MetricData=data)
    logger.info("Published %d metrics to CloudWatch namespace %s", len(data), NAMESPACE)

    if args.create_alarm:
        cw.put_metric_alarm(
            AlarmName=ALARM_NAME,
            Namespace=NAMESPACE,
            MetricName="ForecastWAPE",
            Statistic="Average",
            Period=300,
            EvaluationPeriods=1,
            Threshold=args.wape_threshold,
            ComparisonOperator="GreaterThanThreshold",
            TreatMissingData="notBreaching",
            AlarmDescription="Batch forecast WAPE exceeded the retrain threshold.",
        )
        logger.info("Created alarm '%s' (ForecastWAPE > %.3f)", ALARM_NAME, args.wape_threshold)

    if args.wire_retrain:
        account = boto_session.client("sts").get_caller_identity()["Account"]
        region = boto_session.region_name
        pipeline_arn = f"arn:aws:sagemaker:{region}:{account}:pipeline/{PIPELINE_NAME}"
        events = boto_session.client("events")
        events.put_rule(
            Name=RETRAIN_RULE_NAME,
            EventPattern=json.dumps(retrain_event_pattern(ALARM_NAME)),
            State="DISABLED",
            Description="Retrain the pipeline when the forecast-WAPE alarm fires (disabled).",
            Tags=[{"Key": "project", "Value": "demand-forecasting"}],
        )
        events.put_targets(
            Rule=RETRAIN_RULE_NAME,
            Targets=[
                {
                    "Id": "ct-pipeline",
                    "Arn": pipeline_arn,
                    "RoleArn": args.retrain_role_arn or args.role_arn,
                }
            ],
        )
        logger.info("Created DISABLED retrain rule '%s' -> %s", RETRAIN_RULE_NAME, pipeline_arn)
        logger.info("Enable with: aws events enable-rule --name %s", RETRAIN_RULE_NAME)


if __name__ == "__main__":
    main()
