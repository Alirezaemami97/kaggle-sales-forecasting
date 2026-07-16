"""Tests for the AWS operational scripts.

The pure helpers (`to_sales_long`, `prepare_calendar`, `build_sample`, the
Phase-3 hyperparameter/override translation) run in CI. The S3 setup test uses
moto to mock AWS and is skipped unless the optional `aws` group is installed (CI
installs neither boto3 nor moto); the SageMaker calls themselves are exercised
against the real account, not mocked."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import prep_athena_data
import pytest
import run_sagemaker_training
import train_entry
from pydantic import ValidationError

from demand_forecasting.config import load_config


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


def test_apply_overrides_only_replaces_supplied_values() -> None:
    """SageMaker hyperparameters override individual knobs; everything else must
    keep coming from config.yaml, which stays the single source of truth."""
    config = load_config("config/config.yaml")
    out = train_entry.apply_overrides(
        config, {"max_series": 50, "n_estimators": 10, "learning_rate": None}
    )
    assert out.training.max_series == 50
    assert out.training.lgbm.n_estimators == 10
    # Untouched knobs fall through unchanged.
    assert out.training.lgbm.learning_rate == config.training.lgbm.learning_rate
    assert out.training.horizon == config.training.horizon
    assert out.training.quantiles == config.training.quantiles
    # Overriding must not mutate the input config.
    assert config.training.max_series != 50


def test_apply_overrides_with_nothing_supplied_is_a_no_op() -> None:
    config = load_config("config/config.yaml")
    out = train_entry.apply_overrides(config, {"max_series": None, "state_filter": None})
    assert out.training.max_series == config.training.max_series
    assert out.data.state_filter == config.data.state_filter


def test_apply_overrides_rejects_an_invalid_hyperparameter() -> None:
    # Overrides are re-validated, so a hyperparameter that violates a Config
    # constraint fails immediately rather than deep inside training, after the
    # instance has spun up and pulled the data. Note this only protects fields
    # Config actually constrains — LGBMConfig's knobs are currently unbounded.
    config = load_config("config/config.yaml")
    with pytest.raises(ValidationError):
        train_entry.apply_overrides(config, {"horizon": 0})


def test_build_hyperparameters_omits_unset_values() -> None:
    # Sending an unset hyperparameter would silently shadow config.yaml, so only
    # explicit overrides may be transmitted to the container.
    assert run_sagemaker_training.build_hyperparameters(None, None) == {}
    assert run_sagemaker_training.build_hyperparameters(3000, None) == {"max-series": 3000}
    assert run_sagemaker_training.build_hyperparameters(3000, 50) == {
        "max-series": 3000,
        "n-estimators": 50,
    }


def test_load_features_pushes_the_series_cap_into_the_scan(
    feature_table: pd.DataFrame, tmp_path: Path
) -> None:
    """The cap must be applied AT SCAN TIME, not after loading: reading all 46.8M
    Spark-written rows first is what OOM-killed the first real training job."""
    # Mimic the Glue output: several part-files in one directory.
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for i, (_, chunk) in enumerate(feature_table.groupby("id")):
        chunk.to_parquet(features_dir / f"part-{i:05d}.parquet", index=False)

    config = load_config("config/config.yaml")
    capped = train_entry.load_features(features_dir, train_entry.apply_overrides(
        config, {"max_series": 2}
    ))
    assert capped["id"].nunique() == 2
    # Only the columns the training path consumes are materialised.
    assert "sales" in capped.columns and "d" in capped.columns
    # Every row of a kept series survives — capping selects series, never rows.
    kept = capped["id"].unique()
    expected = len(feature_table[feature_table["id"].isin(kept)])
    assert len(capped) == expected

    # Deterministic: the same seed picks the same series across runs.
    again = train_entry.load_features(features_dir, train_entry.apply_overrides(
        config, {"max_series": 2}
    ))
    assert sorted(again["id"].unique()) == sorted(kept)


def test_already_exists_recognises_both_signal_shapes() -> None:
    """Re-running any setup script must be safe, and SageMaker's create APIs
    signal a duplicate as an untyped ValidationException — the message is the
    only clue. Pinning both shapes here so a re-run never hard-fails again."""
    import aws_errors

    class FakeClientError(Exception):
        def __init__(self, code: str, message: str = "") -> None:
            self.response = {"Error": {"Code": code, "Message": message}}

    # SageMaker CreateExperiment / CreateModelPackageGroup: untyped, message only.
    assert aws_errors.already_exists(
        FakeClientError(
            "ValidationException",
            "Experiment names must be unique within an AWS account and region. "
            "Experiment with name (demand-forecasting-lgbm) already exists.",
        )
    )
    # Typed variants used by other services.
    assert aws_errors.already_exists(FakeClientError("ResourceInUse"))
    assert aws_errors.already_exists(FakeClientError("RepositoryAlreadyExistsException"))
    # Genuine failures must still propagate — never swallow these.
    assert not aws_errors.already_exists(FakeClientError("AccessDeniedException"))
    assert not aws_errors.already_exists(FakeClientError("ResourceLimitExceeded"))
    assert not aws_errors.already_exists(
        FakeClientError("ValidationException", "1 validation error detected")
    )


def test_image_uri_is_a_valid_ecr_reference() -> None:
    import build_and_push_image

    uri = build_and_push_image.image_uri("1234567890", "us-east-1", "demand-forecasting", "latest")
    assert uri == "1234567890.dkr.ecr.us-east-1.amazonaws.com/demand-forecasting:latest"


def test_trial_name_is_unique_and_sortable() -> None:
    now = datetime(2026, 7, 16, 9, 30, 5, tzinfo=timezone.utc)
    assert run_sagemaker_training.trial_name(now) == "demand-forecasting-lgbm-2026-07-16-09-30-05"


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
