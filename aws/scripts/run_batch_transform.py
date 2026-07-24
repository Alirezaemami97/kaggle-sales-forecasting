"""Phase 5: score the catalogue with SageMaker Batch Transform, archive forecasts.

Batch Transform spins up an ephemeral serving container (no idle endpoint), POSTs
the prep step's feature CSV to /invocations in line-split mini-batches, and writes
one `.out` file of quantile predictions back to S3 — cents per run. We use AWS's
prebuilt SKLearn *inference* container with our self-contained `inference.py` as
the entry point (the 3.12 package can't run in its 3.9 runtime; only LightGBM is
added via requirements.txt), so there is no image to build or push.

Row order is preserved (SplitType=Line + AssembleWith=Line), so the predictions
join back to the meta sidecar by position — that reassembly + the write-time
behavioural gate (`check_forecasts`) produce the prediction archive, the M6 batch
job's cloud twin.

    python aws/scripts/prep_batch_input.py --bucket <name>
    python aws/scripts/run_batch_transform.py --bucket <name> --role-arn <role> \
        --model-data s3://<name>/models/<job>/output/model.tar.gz
"""

import argparse
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from demand_forecasting.config import load_config
from demand_forecasting.inference.batch import check_forecasts
from demand_forecasting.training.model import quantile_column

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = REPO_ROOT / "aws" / "sagemaker"


def reassemble(
    predictions: pd.DataFrame, meta: pd.DataFrame, quantiles: list[float]
) -> pd.DataFrame:
    """Join the headerless transform output (quantile columns, in input row order)
    back to the identity/horizon sidecar by position. Pure — unit-tested in CI, so
    a row-count mismatch is caught before it silently corrupts the archive."""
    if len(predictions) != len(meta):
        raise ValueError(f"row mismatch: {len(predictions)} predictions vs {len(meta)} meta rows")
    predictions = predictions.copy()
    predictions.columns = [quantile_column(q) for q in sorted(quantiles)]
    return pd.concat([meta.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--model-data", required=True, help="s3:// URI of the model.tar.gz")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--input-prefix", default="batch-input", help="Prefix written by prep_batch_input.py"
    )
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--framework-version", default="1.2-1")
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat
    from sagemaker.sklearn.model import SKLearnModel

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    boto_session = boto3.Session()
    session = sagemaker.Session(boto_session=boto_session)
    run_name = f"batch-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
    output_path = f"s3://{args.bucket}/forecasts/{run_name}"

    model = SKLearnModel(
        model_data=args.model_data,
        role=args.role_arn,
        entry_point="inference.py",
        source_dir=str(SOURCE_DIR),
        framework_version=args.framework_version,
        py_version="py3",
        sagemaker_session=session,
    )
    transformer = model.transformer(
        instance_count=1,
        instance_type=args.instance_type,
        output_path=output_path,
        accept="text/csv",
        assemble_with="Line",  # reassemble output in input order
        strategy="MultiRecord",
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )
    transformer.transform(
        data=f"s3://{args.bucket}/{args.input_prefix}/input/",
        content_type="text/csv",
        split_type="Line",
        wait=True,
    )
    logger.info("Transform output: %s", output_path)

    # Reassemble: predictions (one .out per input file, row order preserved) +
    # the meta sidecar -> the archive, gated by the same behavioural checks as M6.
    config = load_config(args.config)
    quantiles = config.training.quantiles
    s3 = boto_session.client("s3")

    meta_obj = s3.get_object(Bucket=args.bucket, Key=f"{args.input_prefix}/meta/meta.parquet")
    meta = pd.read_parquet(io.BytesIO(meta_obj["Body"].read()))

    out_prefix = f"forecasts/{run_name}/"
    out_keys = sorted(
        o["Key"]
        for o in s3.list_objects_v2(Bucket=args.bucket, Prefix=out_prefix).get("Contents", [])
        if o["Key"].endswith(".out")
    )
    frames = [
        pd.read_csv(
            io.BytesIO(s3.get_object(Bucket=args.bucket, Key=k)["Body"].read()), header=None
        )
        for k in out_keys
    ]
    predictions = pd.concat(frames, ignore_index=True)

    archive = reassemble(predictions, meta, quantiles)
    archive = archive.assign(
        run_date=datetime.now(timezone.utc).date().isoformat(), model_data=args.model_data
    )
    check_forecasts(archive, quantiles)  # non-negative + non-crossing, before we trust it

    buffer = io.BytesIO()
    archive.to_parquet(buffer, index=False)
    archive_key = f"{out_prefix}archive.parquet"
    s3.put_object(Bucket=args.bucket, Key=archive_key, Body=buffer.getvalue())
    logger.info("Archived %d forecasts to s3://%s/%s", len(archive), args.bucket, archive_key)


if __name__ == "__main__":
    main()
