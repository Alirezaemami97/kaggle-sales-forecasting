"""Upload the Athena-ready Parquet datasets to s3://<bucket>/raw/.

Each dataset goes under its own prefix so an Athena external table can point its
LOCATION at that prefix:

    raw/sales_long/sales_long.parquet
    raw/calendar/calendar.parquet
    raw/prices/prices.parquet

    python aws/scripts/upload_data.py --bucket demand-forecasting-<your-suffix>
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger(__name__)


def upload_file(s3: Any, local: Path, bucket: str, key: str) -> None:
    if not local.exists():
        raise FileNotFoundError(f"Expected {local} — run prep_athena_data / the M1 convert first")
    s3.upload_file(str(local), bucket, key)
    logger.info("Uploaded %s → s3://%s/%s", local, bucket, key)


def upload_all(
    bucket: str, processed_dir: Path, staging_dir: Path, s3: Any = None
) -> list[str]:
    """Upload sales_long + calendar + prices; return the S3 keys written."""
    s3 = s3 or boto3.client("s3")
    uploads = [
        (staging_dir / "sales_long.parquet", "raw/sales_long/sales_long.parquet"),
        (processed_dir / "calendar.parquet", "raw/calendar/calendar.parquet"),
        (processed_dir / "prices.parquet", "raw/prices/prices.parquet"),
    ]
    for local, key in uploads:
        upload_file(s3, local, bucket, key)
    return [key for _, key in uploads]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Target S3 bucket")
    parser.add_argument("--processed-dir", default="data/processed", help="calendar/prices Parquet")
    parser.add_argument("--staging-dir", default="data/aws_staging", help="sales_long Parquet")
    args = parser.parse_args()
    upload_all(args.bucket, Path(args.processed_dir), Path(args.staging_dir))


if __name__ == "__main__":
    main()
