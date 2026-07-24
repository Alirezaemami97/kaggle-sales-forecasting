"""Launch the TFT-vs-LightGBM GPU comparison as a Spot training job.

The cost-optimised-training story (exam D2.2), wired for real:
  - `use_spot_instances=True` — SageMaker bids on spare capacity, typically
    ~70% off on-demand ($0.74/hr -> ~$0.22-0.30/hr for ml.g4dn.xlarge).
  - `max_run` — hard ceiling on training seconds: the job is KILLED past it,
    which doubles as our cost circuit-breaker.
  - `max_wait` (> max_run, Spot only) — total budget including time spent
    waiting for capacity and restarts after interruptions.
  - `checkpoint_s3_uri` — /opt/ml/checkpoints syncs here across interruptions.
    Honest scope: these runs are short enough that restart-from-scratch is the
    accepted trade; epoch-level resume is what a longer job would add on top.

Requires the GPU quota (ml.g4dn.xlarge for [spot] training job usage >= 1) and
the GPU image:

    python aws/scripts/build_and_push_image.py --dockerfile Dockerfile.gpu \
        --repository demand-forecasting-training-gpu

    python aws/scripts/run_tft_training.py --bucket <name> --role-arn <role> --pilot
    python aws/scripts/run_tft_training.py --bucket <name> --role-arn <role>
"""

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)

GPU_REPOSITORY = "demand-forecasting-training-gpu"
METRIC_DEFINITIONS = [
    {"Name": "gpu_available", "Regex": "METRIC gpu_available=([0-9]+)"},
    {"Name": "lgbm_wape_item_store", "Regex": "METRIC lgbm_wape_item_store=([0-9.]+)"},
    {"Name": "tft_wape_item_store", "Regex": "METRIC tft_wape_item_store=([0-9.]+)"},
]


def tft_settings(pilot: bool) -> dict[str, int]:
    """Series/epochs/runtime caps for pilot vs full. Pure — unit-tested.

    Pilot: 1 epoch on 100 series proves GPU use, the Spot lifecycle and
    checkpoint sync for ~$0.05 before the full run. Full: bounded to caps that
    keep the job well under an hour of GPU time (max_run is the circuit-breaker).

    max_samples_per_ts bounds darts' per-series sliding-window sample count —
    without it, dataset size is train_history_days x n_series regardless of
    epochs, and the first real GPU attempt (1000 series, unbounded -> 647,000
    samples) never finished one epoch before hitting max_run. loader_workers
    parallelises the CPU-side window slicing across ml.g4dn.xlarge's 4 vCPUs
    (one held back for the main process).
    """
    if pilot:
        return {
            "max_series": 100, "epochs": 1, "max_run": 1800, "max_wait": 3600,
            "max_samples_per_ts": 30, "loader_workers": 3,
        }
    return {
        "max_series": 1000, "epochs": 15, "max_run": 5400, "max_wait": 9000,
        "max_samples_per_ts": 50, "loader_workers": 3,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge")
    parser.add_argument("--image-uri", help="Defaults to this account's GPU repo :latest")
    parser.add_argument("--pilot", action="store_true", help="1 epoch, 100 series, ~$0.05")
    parser.add_argument(
        "--no-spot", action="store_true",
        help="On-demand fallback if Spot capacity is unavailable",
    )
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat
    from build_and_push_image import image_uri
    from sagemaker.estimator import Estimator

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    settings = tft_settings(args.pilot)
    boto_session = boto3.Session()
    session = sagemaker.Session(boto_session=boto_session)
    uri = args.image_uri or image_uri(
        boto_session.client("sts").get_caller_identity()["Account"],
        boto_session.region_name,
        GPU_REPOSITORY,
        "latest",
    )

    spot: dict[str, Any] = (
        {}
        if args.no_spot
        else {
            "use_spot_instances": True,
            "max_wait": settings["max_wait"],
            "checkpoint_s3_uri": f"s3://{args.bucket}/checkpoints/tft/",
        }
    )
    estimator = Estimator(
        image_uri=uri,
        entry_point="tft_entry.py",
        source_dir="aws/sagemaker",
        dependencies=["src/demand_forecasting", "config/config.yaml"],
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        max_run=settings["max_run"],
        base_job_name="demand-forecasting-tft",
        output_path=f"s3://{args.bucket}/models/",
        sagemaker_session=session,
        hyperparameters={
            "tft-max-series": settings["max_series"],
            "tft-epochs": settings["epochs"],
            "tft-max-samples-per-ts": settings["max_samples_per_ts"],
            "tft-loader-workers": settings["loader_workers"],
        },
        metric_definitions=METRIC_DEFINITIONS,
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
        **spot,
    )

    mode = "PILOT" if args.pilot else "FULL"
    logger.info(
        "%s TFT run: %d series, %d epochs, %s, max_run=%ds",
        mode, settings["max_series"], settings["epochs"],
        "on-demand" if args.no_spot else "Spot", settings["max_run"],
    )
    estimator.fit(inputs={"features": f"s3://{args.bucket}/features/"})

    desc = boto_session.client("sagemaker").describe_training_job(
        TrainingJobName=estimator.latest_training_job.name
    )
    billable = desc.get("BillableTimeInSeconds")
    elapsed = desc.get("TrainingTimeInSeconds")
    if billable and elapsed:
        logger.info(
            "Training %ds, billable %ds — Spot saved ~%.0f%%",
            elapsed, billable, 100 * (1 - billable / max(elapsed, 1))
            if not args.no_spot else 0.0,
        )
    for metric in desc.get("FinalMetricDataList", []):
        logger.info("%s = %s", metric["MetricName"], metric["Value"])
    logger.info("Comparison artifact: %s", estimator.model_data)


if __name__ == "__main__":
    main()
