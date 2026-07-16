"""Launch the LightGBM baseline as a SageMaker script-mode training job.

Runs LOCALLY and only submits the job: it defines an Estimator over our own ECR
image (see aws/docker/Dockerfile and build_and_push_image.py), points it at the
Glue feature output in S3, and waits. SageMaker then provisions an ml.m5.xlarge,
downloads the data, runs train_entry.py, uploads model.tar.gz to S3, and tears
the instance down. Ephemeral by construction — you pay for the minutes it ran,
nothing idles afterwards.

Three details do the heavy lifting:
  - `image_uri` — our custom image. AWS's prebuilt framework images pin their own
    Python (the newest SKLearn one is 3.9; this project is 3.12), so we own the
    image. The image still speaks script mode via the sagemaker-training toolkit,
    which is why train_entry.py is identical either way.
  - `dependencies` — SageMaker tars source_dir PLUS these paths and unpacks them
    side by side in the container's working directory. That is how the portable
    `demand_forecasting` package and config.yaml get in without being pip-
    installable, published, or baked into the image (so code changes need no
    rebuild — only dependency changes do).
  - `metric_definitions` — regexes SageMaker applies to the container's stdout;
    matches become metrics in Experiments and CloudWatch. Printing a line is how
    a number escapes the container.

    python aws/scripts/run_sagemaker_training.py --bucket <name> \
        --role-arn <sagemaker-role-arn>
"""

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

from aws_errors import already_exists

logger = logging.getLogger(__name__)

# The sagemaker SDK is imported lazily inside main() so this module stays
# importable (and its pure helpers testable) without the optional `aws` group.

EXPERIMENT_NAME = "demand-forecasting-lgbm"
BASE_JOB_NAME = "demand-forecasting-lgbm"
# The container's stdout is the only channel out; these turn printed lines into
# tracked metrics. train_entry.py logs "METRIC train_rows=12345".
METRIC_DEFINITIONS = [
    {"Name": "train_rows", "Regex": "METRIC train_rows=([0-9.]+)"},
    {"Name": "n_series", "Regex": "METRIC n_series=([0-9.]+)"},
]


def trial_name(now: datetime | None = None) -> str:
    """A unique, sortable trial name per run. Pure — unit-tested in CI."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d-%H-%M-%S")
    return f"{BASE_JOB_NAME}-{stamp}"


def ensure_experiment(sm: Any, experiment: str, trial: str) -> None:
    """Create the Experiment and Trial before submitting the job.

    CreateTrainingJob does NOT create these implicitly when `experiment_config`
    is passed directly — it fails with ResourceNotFound. (The SDK only creates
    them for you via the `Run` context manager.) Idempotent: an existing
    experiment is reused, and each run gets its own timestamped trial.
    """
    try:
        sm.create_experiment(
            ExperimentName=experiment,
            Description="LightGBM quantile baseline trained via SageMaker script mode",
            Tags=[{"Key": "project", "Value": "demand-forecasting"}],
        )
        logger.info("Created experiment %s", experiment)
    except sm.exceptions.ClientError as exc:
        if not already_exists(exc):
            raise
        logger.info("Experiment %s already exists", experiment)
    sm.create_trial(TrialName=trial, ExperimentName=experiment)
    logger.info("Created trial %s", trial)


def build_hyperparameters(max_series: int | None, n_estimators: int | None) -> dict[str, object]:
    """Only non-None overrides are sent; anything omitted falls back to
    config.yaml inside the container. Pure — unit-tested in CI."""
    hyperparameters: dict[str, object] = {}
    if max_series is not None:
        hyperparameters["max-series"] = max_series
    if n_estimators is not None:
        hyperparameters["n-estimators"] = n_estimators
    return hyperparameters


def build_estimator(
    uri: str,
    role_arn: str,
    instance_type: str,
    bucket: str,
    hyperparameters: dict[str, object],
    session: Any,
) -> Any:
    """The one Estimator definition, shared by the plain training run and the
    tuner (run_tuning.py) — so a tuned job and a baseline job cannot drift in
    image, code delivery, or metric scraping."""
    from sagemaker.estimator import Estimator

    return Estimator(
        image_uri=uri,
        entry_point="train_entry.py",
        source_dir="aws/sagemaker",
        # Ship the portable package + config into the container unchanged.
        dependencies=["src/demand_forecasting", "config/config.yaml"],
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        base_job_name=BASE_JOB_NAME,
        output_path=f"s3://{bucket}/models/",
        sagemaker_session=session,
        hyperparameters=hyperparameters,
        metric_definitions=METRIC_DEFINITIONS,
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )


def default_image_uri(boto_session: Any, override: str | None) -> str:
    """Resolve the training image: an explicit --image-uri, else this account's
    :latest in the Phase-3 repository."""
    from build_and_push_image import REPOSITORY, image_uri

    if override:
        return override
    return image_uri(
        boto_session.client("sts").get_caller_identity()["Account"],
        boto_session.region_name,
        REPOSITORY,
        "latest",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument(
        "--image-uri", help="ECR training image; defaults to this account's :latest"
    )
    parser.add_argument(
        "--max-series", type=int, default=3000,
        help="Cap series for a bounded, cheap first run; omit to use config.yaml",
    )
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument(
        "--no-wait", action="store_true", help="Submit and return without streaming logs"
    )
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    boto_session = boto3.Session()
    session = sagemaker.Session(boto_session=boto_session)
    uri = default_image_uri(boto_session, args.image_uri)
    estimator = build_estimator(
        uri,
        args.role_arn,
        args.instance_type,
        args.bucket,
        build_hyperparameters(args.max_series, args.n_estimators),
        session,
    )

    name = trial_name()
    ensure_experiment(boto_session.client("sagemaker"), EXPERIMENT_NAME, name)
    logger.info("Submitting training job (trial %s)", name)
    estimator.fit(
        inputs={"features": f"s3://{args.bucket}/features/"},
        experiment_config={
            "ExperimentName": EXPERIMENT_NAME,
            "TrialName": name,
            "TrialComponentDisplayName": "training",
        },
        wait=not args.no_wait,
    )

    if not args.no_wait:
        logger.info("Training job: %s", estimator.latest_training_job.name)
        logger.info("Model artifact: %s", estimator.model_data)
        logger.info("Register it with: python aws/scripts/register_model.py --model-data %s "
                    "--role-arn %s", estimator.model_data, args.role_arn)


if __name__ == "__main__":
    main()
