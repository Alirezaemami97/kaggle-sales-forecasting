"""Phase 5 prep: build the SageMaker Batch Transform input from the Glue features.

Batch Transform is a row-in / row-out scorer, so feature-building happens HERE —
reusing the shared, point-in-time-correct `build_direct_table`/`to_model_frame`,
not the serving container. That split is not a hack: it is exactly the
`features -> transform` decomposition Phase 6's pipeline will chain.

We anchor at the latest fully-observed origin (`latest_full_origin`), so every
forecast day falls inside the data and can be scored against actuals later. Two
artifacts, written row-for-row aligned:
  - input.csv  — the model-input rows, headerless, in FEATURE_COLS order (this is
                 what Batch Transform splits by line and POSTs to /invocations);
  - meta.parquet — the identity + horizon sidecar, so predictions can be joined
                   back to (id, target_day) after the transform (order preserved).

Warm-series only: the M6 cold-start hierarchy-prior fallback is a pure groupby
that needs no model, so it stays a documented post-merge, deferred to keep the
serving path a clean single-model scorer. The series cap keeps the demo bounded,
same as training and evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from demand_forecasting.config import Config, load_config
from demand_forecasting.inference.batch import latest_full_origin
from demand_forecasting.training.dataset import STATIC_COLS, build_direct_table, to_model_frame

logger = logging.getLogger(__name__)

# The sidecar carries series identity + where each row sits in the horizon, so
# the archive can be rebuilt and forecasts scored against actuals by target day.
META_COLS = ["id", *STATIC_COLS, "origin", "horizon", "target_day"]


def build_batch_inputs(features: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feature_rows, meta), row-aligned. `feature_rows` is exactly what
    the model consumes (FEATURE_COLS); `meta` is the reassembly sidecar. Pure —
    unit-tested in CI, so a schema drift is caught before any cloud spend."""
    origin = latest_full_origin(features, horizon)
    table = build_direct_table(features, [origin], horizon).reset_index(drop=True)
    logger.info(
        "Batch input: origin day %d, %d rows, %d series",
        origin, len(table), table["id"].nunique(),
    )
    return to_model_frame(table), table[META_COLS]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Features source; defaults to s3://<bucket>/features/ (the Glue output "
        "the model trained on). pyarrow reads s3:// directly via your AWS creds.",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--prefix", default="batch-input", help="S3 key prefix for this run")
    parser.add_argument("--staging-dir", default="data/batch_staging")
    parser.add_argument("--max-series", type=int)
    parser.add_argument(
        "--skip-upload", action="store_true", help="Build + write locally, no S3 (CI/dev)"
    )
    args = parser.parse_args()

    # apply_overrides/load_features live with the training entry point (sibling
    # dir); reuse them so the batch loader can't drift from the training loader
    # (same OOM-avoiding pyarrow pushdown). Launchers run locally, never in the
    # container, so putting the sibling dir on sys.path here is safe.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sagemaker"))
    from train_entry import apply_overrides, load_features

    config: Config = apply_overrides(load_config(args.config), vars(args))
    # Read the Glue features from S3 by default (str, not Path — a Path would
    # mangle the s3:// URI on Windows). load_features' pyarrow scan takes either.
    features_dir = args.features_dir or f"s3://{args.bucket}/features/"
    features = load_features(features_dir, config)
    feature_rows, meta = build_batch_inputs(features, config.training.horizon)

    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    csv_path = staging / "input.csv"
    meta_path = staging / "meta.parquet"
    # Headerless: Batch Transform splits by line, so a header would corrupt every
    # mini-batch after the first. input_fn assigns FEATURE_COLS by position.
    feature_rows.to_csv(csv_path, index=False, header=False)
    meta.to_parquet(meta_path, index=False)
    logger.info("Wrote %s (%d rows) and %s", csv_path, len(feature_rows), meta_path)

    if args.skip_upload:
        return

    import boto3

    s3 = boto3.client("s3")
    # The transform input prefix must hold ONLY the CSV (Batch Transform scores
    # every object under it); the sidecar lives in a sibling prefix.
    s3.upload_file(str(csv_path), args.bucket, f"{args.prefix}/input/input.csv")
    s3.upload_file(str(meta_path), args.bucket, f"{args.prefix}/meta/meta.parquet")
    logger.info("Uploaded to s3://%s/%s/{input,meta}/", args.bucket, args.prefix)


if __name__ == "__main__":
    main()
