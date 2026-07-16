"""Train SageMaker's built-in DeepAR on the sampled M5 series.

The third point of the container spectrum (built-in / script mode / BYOC):
here there is NO entry point of ours at all — AWS's forecasting-deepar image
runs AWS's code, and our entire contribution is the JSON Lines data
(prep_deepar_data.py) and these hyperparameters. Evaluation is built in too:
the `test` channel makes DeepAR score the trailing prediction_length days per
series and emit test:mean_wQuantileLoss / test:RMSE as scraped metrics.

Comparison caveat (honesty over impressiveness): wQuantileLoss and our pinball
loss are the same quantity up to normalisation, but DeepAR scores ONE trailing
window per series while our panel averages three backtest folds — directional
comparison only, recorded as such.

    python aws/scripts/run_deepar.py --bucket <name> --role-arn <role>
"""

import argparse
import logging

logger = logging.getLogger(__name__)

# The knobs that make this an honest, bounded comparison with the other models.
HYPERPARAMETERS: dict[str, str] = {
    "time_freq": "D",
    "prediction_length": "28",  # same horizon as everything else
    "context_length": "56",     # 2x horizon lookback, same as the TFT config
    "likelihood": "negative-binomial",  # counts, 68% zeros — never Gaussian here
    "epochs": "20",
    "early_stopping_patience": "3",
    "cardinality": "auto",      # sizes the [store, dept] embeddings from data
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--epochs", type=int, help="Override the bounded default")
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat
    from sagemaker.estimator import Estimator

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    boto_session = boto3.Session()
    session = sagemaker.Session(boto_session=boto_session)
    image = sagemaker.image_uris.retrieve("forecasting-deepar", boto_session.region_name)
    logger.info("Built-in image: %s", image)

    hyperparameters = dict(HYPERPARAMETERS)
    if args.epochs is not None:
        hyperparameters["epochs"] = str(args.epochs)

    estimator = Estimator(
        image_uri=image,
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        base_job_name="demand-forecasting-deepar",
        output_path=f"s3://{args.bucket}/models/",
        sagemaker_session=session,
        hyperparameters=hyperparameters,
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )
    estimator.fit(
        inputs={
            "train": f"s3://{args.bucket}/deepar/train/",
            "test": f"s3://{args.bucket}/deepar/test/",
        }
    )

    desc = boto_session.client("sagemaker").describe_training_job(
        TrainingJobName=estimator.latest_training_job.name
    )
    for metric in desc.get("FinalMetricDataList", []):
        logger.info("%s = %s", metric["MetricName"], metric["Value"])
    logger.info("Model artifact: %s", estimator.model_data)


if __name__ == "__main__":
    main()
