"""Tests for the AWS Phase-1 scripts.

`to_sales_long` is pure and runs in CI. The S3 setup test uses moto to mock AWS
and is skipped unless the optional `aws` group is installed (CI installs neither
boto3 nor moto)."""

import pandas as pd
import prep_athena_data
import pytest


def test_to_sales_long_shape_and_columns(sales_df: pd.DataFrame) -> None:
    long = prep_athena_data.to_sales_long(sales_df)
    assert list(long.columns) == prep_athena_data.SALES_LONG_COLS
    # 3 series x 10 days in the fixture → one long row each.
    assert len(long) == 3 * 10
    assert long["sales"].notna().all()


def test_s3_setup_creates_configured_bucket() -> None:
    pytest.importorskip("moto")
    import boto3
    import s3_setup
    from moto import mock_aws

    bucket = "demand-forecasting-test"
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        s3_setup.setup_bucket(bucket, s3=client)
        # Re-run to prove idempotency (already-owned bucket is not an error).
        s3_setup.setup_bucket(bucket, s3=client)

        client.head_bucket(Bucket=bucket)  # raises if missing
        keys = [o["Key"] for o in client.list_objects_v2(Bucket=bucket).get("Contents", [])]
        assert set(s3_setup.PREFIXES).issubset(keys)

        enc = client.get_bucket_encryption(Bucket=bucket)
        algo = enc["ServerSideEncryptionConfiguration"]["Rules"][0][
            "ApplyServerSideEncryptionByDefault"
        ]["SSEAlgorithm"]
        assert algo == "AES256"

        pab = client.get_public_access_block(Bucket=bucket)["PublicAccessBlockConfiguration"]
        assert pab["BlockPublicAcls"] is True
        assert pab["RestrictPublicBuckets"] is True

        tags = client.get_bucket_tagging(Bucket=bucket)["TagSet"]
        assert {"Key": "project", "Value": "demand-forecasting"} in tags
