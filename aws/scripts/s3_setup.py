"""Create and configure the project's S3 bucket for the AWS-native track.

Idempotent — safe to re-run. Creates the bucket, the raw/features/models/forecasts
prefixes (plus athena-results/ for Athena query output), turns on default
encryption and a public-access block, and applies the `project` cost-allocation
tag so spend can be traced and orphaned resources found.

Bucket name and region come from CLI args / env — never hard-coded, never committed.

    python aws/scripts/s3_setup.py --bucket demand-forecasting-<your-suffix>
"""

import argparse
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# S3 has no real folders; these zero-byte keys make the layout visible and mirror
# the local data/ tree (raw → features → models → forecasts).
PREFIXES = ["raw/", "features/", "models/", "forecasts/", "athena-results/"]
PROJECT_TAG = {"Key": "project", "Value": "demand-forecasting"}


def create_bucket(s3: Any, bucket: str, region: str = "us-east-1") -> None:
    """Create the bucket, tolerating re-runs (already-owned is not an error)."""
    try:
        if region == "us-east-1":
            # us-east-1 must NOT pass a LocationConstraint (API quirk).
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        logger.info("Created bucket %s", bucket)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info("Bucket %s already exists — continuing", bucket)
        else:
            raise


def enable_encryption(s3: Any, bucket: str) -> None:
    """Default server-side encryption (SSE-S3/AES256) on every new object."""
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )


def block_public_access(s3: Any, bucket: str) -> None:
    """Fully block public access — this is a private data lake."""
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )


def tag_bucket(s3: Any, bucket: str) -> None:
    """Apply the project cost-allocation tag."""
    s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": [PROJECT_TAG]})


def create_prefixes(s3: Any, bucket: str, prefixes: list[str] = PREFIXES) -> None:
    for prefix in prefixes:
        s3.put_object(Bucket=bucket, Key=prefix)


def setup_bucket(bucket: str, region: str = "us-east-1", s3: Any = None) -> str:
    """Full idempotent setup. `s3` is injectable so tests can pass a mocked client."""
    s3 = s3 or boto3.client("s3", region_name=region)
    create_bucket(s3, bucket, region)
    enable_encryption(s3, bucket)
    block_public_access(s3, bucket)
    tag_bucket(s3, bucket)
    create_prefixes(s3, bucket)
    logger.info("Bucket %s ready with prefixes %s", bucket, PREFIXES)
    return bucket


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Globally-unique S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region (stay in one)")
    args = parser.parse_args()
    setup_bucket(args.bucket, args.region)


if __name__ == "__main__":
    main()
