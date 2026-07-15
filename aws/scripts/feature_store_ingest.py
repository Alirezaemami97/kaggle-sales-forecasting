"""Ingest a bounded sample of the feature table into SageMaker Feature Store
(offline only — never enable the online store, it bills hourly).

Demonstrates the point-in-time / time-travel value of a managed feature store:
each record carries an event_time (derived from the M5 day index `d`), so the
offline store (Parquet in S3, auto-cataloged in Glue) can answer "what did this
feature look like as of day N" queries via Athena. This is a governance/
reproducibility capability layered on features that are ALREADY point-in-time
correct by construction (see demand_forecasting/features/pipeline.py's
no-leakage guarantee) — not a fix for leakage, which the pipeline already
prevents upstream.

Only a bounded SAMPLE is ingested, not the full ~46.8M-row catalogue: the
record-at-a-time Feature Store API is not built for bulk backfills at that
scale, and the concept is fully demonstrated on a small, explicit sample.

Prerequisite: download a small piece of the Glue output locally first, e.g.
one part file:
    aws s3 cp s3://<bucket>/features/ data/aws_download/features/ --recursive \
        --exclude "*" --include "part-00000-*"

    python aws/scripts/feature_store_ingest.py --bucket <name> \
        --role-arn <sagemaker-role-arn> --features-path data/aws_download/features/
"""

import argparse
import logging
import time
from typing import Any

import pandas as pd

# boto3/sagemaker are imported lazily inside main() so the pure build_sample
# helper (and its CI test) work without the optional `aws` dependency group.

logger = logging.getLogger(__name__)

FEATURE_GROUP_NAME = "demand-forecasting-features-sample"
# The M5 dataset's real day-1 calendar date; `d` is a 1-indexed day offset from it.
M5_EPOCH = pd.Timestamp("2011-01-29")


def build_sample(features: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """A small, explicit, bounded sample with the record-id/event-time Feature
    Store needs. Pure — no AWS calls — so it is unit-testable without an account."""
    sample = features.sort_values(["id", "d"]).head(max_rows).reset_index(drop=True).copy()
    sample["record_id"] = sample["id"].astype(str) + "_" + sample["d"].astype(str)
    event_time = M5_EPOCH + pd.to_timedelta(sample["d"].astype(int) - 1, unit="D")
    sample["event_time"] = event_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return sample


def create_feature_group(session: Any, role_arn: str, bucket: str, sample: pd.DataFrame) -> Any:
    from sagemaker.feature_store.feature_group import FeatureGroup

    fg = FeatureGroup(name=FEATURE_GROUP_NAME, sagemaker_session=session)
    fg.load_feature_definitions(data_frame=sample)
    try:
        fg.create(
            s3_uri=f"s3://{bucket}/feature-store/",
            record_identifier_name="record_id",
            event_time_feature_name="event_time",
            role_arn=role_arn,
            enable_online_store=False,  # offline only — the online store bills hourly
        )
        logger.info("Creating feature group %s", FEATURE_GROUP_NAME)
    except Exception as exc:
        if "ResourceInUse" not in str(exc):
            raise
        logger.info("Feature group %s already exists", FEATURE_GROUP_NAME)
    return fg


def wait_for_active(fg: Any) -> None:
    while fg.describe().get("FeatureGroupStatus") == "Creating":
        time.sleep(10)
    logger.info("Feature group status: %s", fg.describe().get("FeatureGroupStatus"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument(
        "--features-path", required=True,
        help="Local Parquet path/dir with a piece of the Glue feature output",
    )
    parser.add_argument("--max-rows", type=int, default=2000)
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    features = pd.read_parquet(args.features_path)
    sample = build_sample(features, args.max_rows)

    session = sagemaker.Session(boto_session=boto3.Session())
    fg = create_feature_group(session, args.role_arn, args.bucket, sample)
    wait_for_active(fg)
    fg.ingest(data_frame=sample, wait=True)
    logger.info("Ingested %d sample rows into %s", len(sample), FEATURE_GROUP_NAME)

    logger.info(
        "Example point-in-time query — check the SageMaker console for the "
        "auto-created offline-store Glue table name, then in Athena:\n"
        "  SELECT * FROM <offline_store_database>.<offline_store_table>\n"
        "  WHERE event_time <= '2011-03-01T00:00:00Z'\n"
        "  ORDER BY event_time DESC LIMIT 10;"
    )


if __name__ == "__main__":
    main()
