"""Launch a capped Automatic Model Tuning (AMT) run over the LightGBM knobs.

AMT wraps the SAME Estimator as run_sagemaker_training.py (imported, not
copied) in a HyperparameterTuner: it launches up to `max_jobs` training jobs
with hyperparameters drawn from the declared ranges and picks the draw that
minimises `val_wape` — the holdout metric train_entry.py prints in
--validation-mode holdout and the tuner scrapes via regex. The tuner can only
optimise what escapes the container as a printed line.

Deliberate caps (cost discipline, docs/AWS_DEPLOYMENT.md):
  - max_jobs=6 (the spec's 6-8 ceiling), max_parallel_jobs=2 — Bayesian
    strategy learns from completed jobs, so some sequentiality is the point:
    fully parallel Bayesian degenerates toward random search.
  - early_stopping_type=Auto lets SageMaker kill jobs that trend worse than
    the best so far.
  - --pilot: 2 jobs, 1 parallel, 300 series — a ~$0.05 dress rehearsal that
    proves the objective metric is scraped and the tuner completes, before
    the full run spends real money.

    python aws/scripts/run_tuning.py --bucket <name> --role-arn <role> --pilot
    python aws/scripts/run_tuning.py --bucket <name> --role-arn <role>
"""

import argparse
import logging
from typing import Any

from run_sagemaker_training import build_estimator, default_image_uri

logger = logging.getLogger(__name__)

# The tuner's objective — must match a metric_definitions regex on the
# estimator plus the METRIC line train_entry.py prints in holdout mode.
OBJECTIVE_METRIC = "val_wape"
OBJECTIVE_REGEX = {"Name": OBJECTIVE_METRIC, "Regex": "METRIC val_wape=([0-9.]+)"}


def tuning_settings(pilot: bool) -> dict[str, int]:
    """Job counts + series cap for pilot vs full runs. Pure — unit-tested."""
    if pilot:
        return {"max_jobs": 2, "max_parallel_jobs": 1, "max_series": 300}
    return {"max_jobs": 6, "max_parallel_jobs": 2, "max_series": 3000}


def hyperparameter_ranges() -> dict[str, Any]:
    """The search space. Bounds sit safely inside LGBMConfig's pydantic
    constraints, so every draw the tuner can make is a valid config."""
    from sagemaker.parameter import ContinuousParameter, IntegerParameter

    return {
        "learning-rate": ContinuousParameter(0.01, 0.1, scaling_type="Logarithmic"),
        "num-leaves": IntegerParameter(31, 255),
        "min-child-samples": IntegerParameter(20, 200),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--image-uri", help="ECR image; defaults to this account's :latest")
    parser.add_argument(
        "--pilot", action="store_true",
        help="Dress rehearsal: 2 tiny jobs to prove the plumbing before the full run",
    )
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat
    from sagemaker.tuner import HyperparameterTuner

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    settings = tuning_settings(args.pilot)
    boto_session = boto3.Session()
    session = sagemaker.Session(boto_session=boto_session)
    estimator = build_estimator(
        default_image_uri(boto_session, args.image_uri),
        args.role_arn,
        args.instance_type,
        args.bucket,
        # Fixed (non-tuned) hyperparameters, applied to every job.
        {"max-series": settings["max_series"], "validation-mode": "holdout"},
        session,
    )
    # The objective regex must be on the estimator's metric list too.
    estimator.metric_definitions = [*estimator.metric_definitions, OBJECTIVE_REGEX]

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=OBJECTIVE_METRIC,
        objective_type="Minimize",
        hyperparameter_ranges=hyperparameter_ranges(),
        metric_definitions=[OBJECTIVE_REGEX],
        strategy="Bayesian",
        max_jobs=settings["max_jobs"],
        max_parallel_jobs=settings["max_parallel_jobs"],
        early_stopping_type="Auto",
        base_tuning_job_name="demand-forecasting-amt",
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )
    mode = "PILOT" if args.pilot else "FULL"
    logger.info(
        "%s tuning: %d jobs (%d parallel), %d series",
        mode, settings["max_jobs"], settings["max_parallel_jobs"], settings["max_series"],
    )
    tuner.fit(inputs={"features": f"s3://{args.bucket}/features/"})

    best = tuner.best_training_job()
    logger.info("Best training job: %s", best)
    desc = boto_session.client("sagemaker").describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.name
    )
    best_summary = desc.get("BestTrainingJob", {})
    logger.info(
        "Best %s=%s with hyperparameters: %s",
        OBJECTIVE_METRIC,
        best_summary.get("FinalHyperParameterTuningJobObjectiveMetric", {}).get("Value"),
        best_summary.get("TunedHyperParameters"),
    )


if __name__ == "__main__":
    main()
