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


def test_prepare_calendar_casts_date_to_string(calendar_df: pd.DataFrame) -> None:
    # pandas/pyarrow writes datetime64[ns] as Parquet TIMESTAMP(NANOS), which
    # Spark's Parquet reader (Glue) cannot read at all. Casting to a plain
    # 'YYYY-MM-DD' string sidesteps the incompatibility for every reader.
    # The fixture stores `date` as a string already (for the feature-pipeline
    # tests); the real data/processed/calendar.parquet is datetime64[ns], so
    # convert here to match production input.
    calendar = calendar_df.assign(date=pd.to_datetime(calendar_df["date"]))
    out = prep_athena_data.prepare_calendar(calendar)
    assert out["date"].dtype == object
    assert (out["date"] == calendar_df["date"]).all()  # already 'YYYY-MM-DD' strings
    assert out["date"].iloc[0] == "2011-01-29"  # the real M5 start date
    # Every other column is untouched.
    assert list(out.columns) == list(calendar.columns)


def test_build_sample_adds_record_id_and_event_time(feature_table: pd.DataFrame) -> None:
    import feature_store_ingest

    sample = feature_store_ingest.build_sample(feature_table, max_rows=5)
    assert len(sample) == 5
    assert (sample["record_id"] == sample["id"].astype(str) + "_" + sample["d"].astype(str)).all()
    # sorted by (id, d) ascending, so the first row is S1's day 1 — the M5 epoch itself.
    assert sample.iloc[0]["id"] == "S1"
    assert sample.iloc[0]["d"] == 1
    assert sample.iloc[0]["event_time"] == "2011-01-29T00:00:00Z"
    # event_time advances one calendar day per unit of `d`.
    assert sample.iloc[1]["event_time"] == "2011-01-30T00:00:00Z"


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
