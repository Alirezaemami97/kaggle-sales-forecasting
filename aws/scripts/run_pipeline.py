"""Phase 6 launcher: register the CT pipeline, optionally run it, optionally schedule it.

Three actions, composable:
  --upsert (always)  stage the eval code bundle to S3, compile the DAG, and
                     create/update it in SageMaker. No execution, ~free — and the
                     way you validate a definition change: a broken DAG fails
                     HERE, not in a paid run. (The training source bundle is
                     uploaded to S3 as part of the compile — expected.)
  --execute          start one pipeline execution (train + eval + maybe register)
                     and wait for it. A few cents (2 managed jobs).
  --schedule         create a DISABLED EventBridge rule that would start the
                     pipeline on a cadence. Disabled on purpose: a live weekly
                     schedule spends forever, so you enable it to demo, then
                     disable. Teardown discipline, expressed in code.

    python aws/scripts/run_pipeline.py --bucket <name> --role-arn <role> --execute
"""

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

SCHEDULE_RULE_NAME = "demand-forecasting-ct-weekly"


def create_schedule(
    events: Any, rule_name: str, expression: str, pipeline_arn: str, role_arn: str
) -> None:
    """Create a DISABLED EventBridge rule targeting the pipeline. Disabled so it
    never fires (or bills) until deliberately enabled. The target role must trust
    events.amazonaws.com and allow sagemaker:StartPipelineExecution — documented,
    not enforced here, since the rule is inert until enabled."""
    events.put_rule(
        Name=rule_name,
        ScheduleExpression=expression,
        State="DISABLED",
        Description="Weekly continuous-training run of the demand-forecasting pipeline (disabled).",
        Tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )
    events.put_targets(
        Rule=rule_name,
        Targets=[{"Id": "ct-pipeline", "Arn": pipeline_arn, "RoleArn": role_arn}],
    )
    logger.info("Created DISABLED schedule '%s' (%s) -> %s", rule_name, expression, pipeline_arn)
    logger.info("Enable with: aws events enable-rule --name %s (disable after demoing)", rule_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--role-arn", required=True, help="Role SageMaker Pipelines assumes for steps"
    )
    parser.add_argument("--image-uri", help="Training/eval image; defaults to account's :latest")
    parser.add_argument("--execute", action="store_true", help="Start one execution after upsert")
    parser.add_argument(
        "--no-wait", action="store_true", help="With --execute, don't wait for completion"
    )
    parser.add_argument(
        "--schedule", action="store_true", help="Create the DISABLED weekly EventBridge rule"
    )
    parser.add_argument("--schedule-expression", default="rate(7 days)")
    parser.add_argument(
        "--schedule-role-arn",
        help="Role EventBridge assumes to start the pipeline (defaults to --role-arn)",
    )
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat
    from pipeline import get_pipeline
    from run_evaluation import stage_code
    from run_sagemaker_training import default_image_uri

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    boto_session = boto3.Session()
    session = sagemaker.workflow.pipeline_context.PipelineSession(boto_session=boto_session)
    uri = default_image_uri(boto_session, args.image_uri)

    run_name = f"pipeline-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
    code_uri = stage_code(boto_session.client("s3"), args.bucket, run_name)

    pipe = get_pipeline(
        role_arn=args.role_arn,
        bucket=args.bucket,
        image_uri=uri,
        code_uri=code_uri,
        session=session,
    )
    response = pipe.upsert(role_arn=args.role_arn)
    pipeline_arn = response["PipelineArn"]
    logger.info("Upserted pipeline: %s", pipeline_arn)

    if args.schedule:
        create_schedule(
            boto_session.client("events"),
            SCHEDULE_RULE_NAME,
            args.schedule_expression,
            pipeline_arn,
            args.schedule_role_arn or args.role_arn,
        )

    if args.execute:
        execution = pipe.start()
        logger.info("Started execution: %s", execution.arn)
        if not args.no_wait:
            execution.wait()
            logger.info("Execution finished. Steps:")
            for step in execution.list_steps():
                logger.info("  %s: %s", step["StepName"], step["StepStatus"])


if __name__ == "__main__":
    main()
