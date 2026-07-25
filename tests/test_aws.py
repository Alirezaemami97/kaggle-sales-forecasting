"""Tests for the AWS operational scripts.

The pure helpers (`to_sales_long`, `prepare_calendar`, `build_sample`, the
Phase-3 hyperparameter/override translation) run in CI. The S3 setup test uses
moto to mock AWS and is skipped unless the optional `aws` group is installed (CI
installs neither boto3 nor moto); the SageMaker calls themselves are exercised
against the real account, not mocked."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
    # instance has spun up and pulled the data. LGBMConfig is bounded too since
    # 4b — AMT draws exactly those knobs, so a bad draw must die in validation.
    config = load_config("config/config.yaml")
    with pytest.raises(ValidationError):
        train_entry.apply_overrides(config, {"horizon": 0})
    with pytest.raises(ValidationError):
        train_entry.apply_overrides(config, {"num_leaves": 1})
    with pytest.raises(ValidationError):
        train_entry.apply_overrides(config, {"learning_rate": 1.5})


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


def test_split_holdout_holds_out_the_newest_origin() -> None:
    # The tuner must validate at the leading edge of history, never randomly.
    assert train_entry.split_holdout([10, 24, 38]) == ([10, 24], 38)
    with pytest.raises(ValueError):
        train_entry.split_holdout([38])


def test_tuning_settings_caps_pilot_and_full_runs() -> None:
    import run_tuning

    pilot = run_tuning.tuning_settings(pilot=True)
    full = run_tuning.tuning_settings(pilot=False)
    assert pilot == {"max_jobs": 2, "max_parallel_jobs": 1, "max_series": 300}
    # The full run must respect the 6-8 job cost ceiling from the AWS plan.
    assert full["max_jobs"] <= 8
    assert full["max_parallel_jobs"] <= full["max_jobs"]


def test_train_holdout_mode_emits_the_objective_metric(
    feature_table: pd.DataFrame, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The AMT objective only exists if holdout mode prints METRIC val_wape —
    a silent regression here would make every tuning job fail metric-less."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    feature_table.to_parquet(features_dir / "part-00000.parquet", index=False)

    config = load_config("config/config.yaml")
    config = train_entry.apply_overrides(config, {"max_series": 3, "n_estimators": 5})
    with caplog.at_level("INFO"):
        train_entry.train(config, features_dir, tmp_path / "model", validation_mode="holdout")

    metric_lines = [r.getMessage() for r in caplog.records if "METRIC val_wape=" in r.getMessage()]
    assert len(metric_lines) == 1
    val_wape = float(metric_lines[0].split("=")[1])
    assert val_wape >= 0.0
    assert (tmp_path / "model" / "meta.json").exists()


def test_code_files_layout_supports_the_container_imports() -> None:
    """The staged code tree must let evaluate_entry do `from train_entry import`
    and `import demand_forecasting` with PYTHONPATH at the code root: entry
    scripts + config.yaml at the root, the package as a subtree."""
    import run_evaluation

    pairs = dict(
        (key, local) for local, key in run_evaluation.code_files(run_evaluation.REPO_ROOT)
    )
    assert "evaluate_entry.py" in pairs
    assert "train_entry.py" in pairs
    assert "config.yaml" in pairs
    assert "demand_forecasting/config.py" in pairs
    assert "demand_forecasting/training/model.py" in pairs
    # Every staged file actually exists locally (a rename would break the job
    # only at container start otherwise).
    for key, local in pairs.items():
        assert local.exists(), f"staged file missing locally: {key}"


def test_evaluate_entry_writes_the_panel(feature_table: pd.DataFrame, tmp_path: Path) -> None:
    """End-to-end wiring of the Processing entry point on the synthetic fixture —
    the free local check that replaces a billed cloud round-trip."""
    import evaluate_entry

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    feature_table.to_parquet(features_dir / "part-00000.parquet", index=False)

    config = load_config("config/config.yaml")
    config = train_entry.apply_overrides(config, {"max_series": 3, "n_estimators": 5})
    out = evaluate_entry.evaluate(config, features_dir, tmp_path / "panel")

    for name in ["by_level.csv", "by_horizon.csv", "calibration.csv", "panel.md"]:
        assert (out / name).exists(), f"missing panel artifact: {name}"

    # The pipeline's ConditionStep gates on this JSON via PropertyFile + JsonGet,
    # so its presence and shape are part of the contract, not just a log line.
    import json

    headline = json.loads((tmp_path / "panel" / "evaluation.json").read_text(encoding="utf-8"))
    assert set(headline) == {"wape_item_store", "wape_total", "wrmsse"}
    assert headline["wape_item_store"] >= 0.0


def test_deepar_jsonl_trims_zeros_and_splits_train_test() -> None:
    import json

    import prep_deepar_data

    n = prep_deepar_data.PREDICTION_LENGTH
    days = list(range(1, 2 * n + 12))  # 67 days
    frames = []
    # S1: first sale on day 6 — leading zeros must be trimmed.
    frames.append(pd.DataFrame({
        "id": "S1", "store_id": "CA_1", "dept_id": "FOODS_1", "d": days,
        "sales": [0] * 5 + [2] * (len(days) - 5),
    }))
    # S2: never sells — must be skipped entirely.
    frames.append(pd.DataFrame({
        "id": "S2", "store_id": "CA_1", "dept_id": "FOODS_2", "d": days,
        "sales": [0] * len(days),
    }))
    # S3: sells, but too short after the train holdout — skipped.
    frames.append(pd.DataFrame({
        "id": "S3", "store_id": "TX_1", "dept_id": "FOODS_1", "d": days,
        "sales": [0] * (len(days) - n - 3) + [1] * (n + 3),
    }))
    sales = pd.concat(frames, ignore_index=True)

    train_lines, test_lines = prep_deepar_data.build_jsonl(sales)
    assert len(train_lines) == len(test_lines) == 1  # only S1 survives

    train, test = json.loads(train_lines[0]), json.loads(test_lines[0])
    # Trimmed start: day 6 = M5 epoch (2011-01-29, d=1) + 5 days.
    assert train["start"] == test["start"] == "2011-02-03"
    # Train drops exactly the trailing window DeepAR will score on test.
    assert len(test["target"]) == len(train["target"]) + n
    assert train["target"] == test["target"][:-n]
    assert all(isinstance(v, int) for v in test["cat"])


def test_clarify_frame_binarises_the_label_and_derives_the_facet() -> None:
    import run_clarify

    sales = pd.DataFrame(
        {
            "id": ["A", "A", "B", "B"],
            "store_id": ["CA_1", "CA_1", "TX_2", "TX_2"],
            "dept_id": ["FOODS_1", "FOODS_1", "HOBBIES_2", "HOBBIES_2"],
            "d": [1, 2, 1, 2],
            "sales": [0, 3, 5, 0],
        }
    )
    frame = run_clarify.build_clarify_frame(sales)
    # Column order must match DataConfig headers exactly — Clarify reads by position.
    assert list(frame.columns) == run_clarify.CLARIFY_COLUMNS
    assert frame["sold"].tolist() == [0, 1, 1, 0]
    assert frame["state_id"].tolist() == ["CA", "CA", "TX", "TX"]


def test_tft_settings_bound_the_gpu_spend() -> None:
    import run_tft_training

    for pilot in (True, False):
        s = run_tft_training.tft_settings(pilot)
        # max_run is the cost circuit-breaker; Spot requires max_wait >= max_run.
        assert s["max_wait"] >= s["max_run"]
        assert s["max_run"] <= 2 * 3600  # never more than 2h of GPU
        # Total windows must stay bounded regardless of history length or
        # series count — the exact blowup that starved the first GPU attempt
        # (1000 series x unbounded history = 647,000 samples, zero epochs done).
        assert s["max_samples_per_ts"] * s["max_series"] <= 100_000
        assert 0 < s["loader_workers"] <= 4  # ml.g4dn.xlarge has 4 vCPUs
    assert run_tft_training.tft_settings(True)["epochs"] == 1


def test_tft_config_overrides_are_validated() -> None:
    import tft_entry

    config = load_config("config/config.yaml")
    cfg = tft_entry.tft_config_with_overrides(config, max_series=1000, n_epochs=15)
    assert cfg.max_series == 1000
    assert cfg.n_epochs == 15
    # Untouched knobs fall through from config.yaml.
    assert cfg.input_chunk_length == config.training.tft.input_chunk_length  # type: ignore[union-attr]
    # Overrides are validated — an invalid draw dies before GPU time is billed.
    with pytest.raises(ValidationError):
        tft_entry.tft_config_with_overrides(config, max_series=None, n_epochs=0)


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


def test_inference_columns_match_dataset() -> None:
    """inference.py hardcodes the feature schema because the 3.9 serving container
    cannot import the 3.12 package; this pins those lists to the shared dataset
    module so a column change on one side can't silently skew the other."""
    import inference

    from demand_forecasting.training import dataset

    assert inference.FEATURE_COLS == dataset.FEATURE_COLS
    assert inference.CATEGORICAL_COLS == dataset.CATEGORICAL_COLS


def test_inference_roundtrip_matches_quantile_lgbm(
    feature_table: pd.DataFrame, tmp_path: Path
) -> None:
    """The self-contained handler must reproduce QuantileLGBM.predict *exactly*,
    through the headerless-CSV Batch Transform boundary — the zero-skew proof. If
    the categorical round-trip corrupted an encoding (e.g. a value read back as
    NaN, or codes reordered), these predictions would diverge."""
    import inference

    from demand_forecasting.training.dataset import (
        build_direct_table,
        select_origins,
        to_model_frame,
    )
    from demand_forecasting.training.model import QuantileLGBM

    # Real M5 stores no-event days as NaN and only genuine events as strings, so
    # mirror that (the fixture's literal "None" is atypical and a read_csv NA
    # token) to exercise both the string-category and the missing case honestly.
    # Use None, not np.nan: np.where(mask, "SuperBowl", np.nan) coerces to a
    # string array and writes the literal "nan", which is not the missing case.
    table_src = feature_table.copy()
    is_event = table_src["d"] % 10 == 0
    table_src["event_name_1"] = np.where(is_event, "SuperBowl", None)
    table_src["event_type_1"] = np.where(is_event, "Sporting", None)

    horizon = 7
    origins = select_origins(table_src, n_origins=2, stride=7, horizon=horizon)
    table = build_direct_table(table_src, origins, horizon)
    features = to_model_frame(table)

    quantiles = [0.5, 0.9]
    params: dict[str, object] = {"n_estimators": 5, "num_leaves": 7, "min_child_samples": 1}
    model = QuantileLGBM(quantiles, params).fit(features, table["sales"])
    model.save(tmp_path)

    # Serve path: reload from disk (no QuantileLGBM import) + cross the CSV boundary.
    served = inference.model_fn(str(tmp_path))
    parsed = inference.input_fn(features.to_csv(index=False, header=False))
    served_preds = inference.predict_fn(parsed, served)

    direct_preds = model.predict(features)
    assert list(served_preds.columns) == list(direct_preds.columns)
    assert np.allclose(served_preds.to_numpy(), direct_preds.to_numpy())


def test_build_batch_inputs_aligns_features_and_meta(feature_table: pd.DataFrame) -> None:
    """The feature CSV and the meta sidecar must be row-for-row aligned, since the
    predictions are joined back to identity/horizon by position after the transform."""
    import prep_batch_input

    from demand_forecasting.training.dataset import FEATURE_COLS

    feature_rows, meta = prep_batch_input.build_batch_inputs(feature_table, horizon=7)
    assert len(feature_rows) == len(meta)
    assert list(feature_rows.columns) == FEATURE_COLS
    assert list(meta.columns) == prep_batch_input.META_COLS
    # Every row is anchored at the single latest fully-observed origin.
    origin = int(feature_table["d"].max()) - 7
    assert (meta["origin"] == origin).all()
    assert set(meta["horizon"]) == set(range(1, 8))
    assert (meta["target_day"] == meta["origin"] + meta["horizon"]).all()


def test_reassemble_joins_predictions_to_meta_by_position() -> None:
    import run_batch_transform

    meta = pd.DataFrame({"id": ["A", "A", "B"], "horizon": [1, 2, 1], "target_day": [74, 75, 74]})
    predictions = pd.DataFrame([[1.0, 3.0], [1.5, 3.5], [2.0, 4.0]])  # headerless, q order
    archive = run_batch_transform.reassemble(predictions, meta, quantiles=[0.9, 0.5])

    # Quantiles sorted -> q0.5 then q0.9, joined row-for-row onto the meta.
    assert list(archive.columns) == ["id", "horizon", "target_day", "q0.5", "q0.9"]
    assert archive["q0.5"].tolist() == [1.0, 1.5, 2.0]
    assert archive.loc[0, "id"] == "A" and archive.loc[2, "id"] == "B"
    # A row-count mismatch must fail loudly, never silently misalign the archive.
    with pytest.raises(ValueError):
        run_batch_transform.reassemble(predictions.iloc[:2], meta, quantiles=[0.9, 0.5])


def test_inference_output_fn_is_headerless_csv() -> None:
    import inference

    preds = pd.DataFrame({"q0.5": [1.0, 2.0], "q0.9": [3.0, 4.0]})
    body = inference.output_fn(preds)
    # No header (Batch Transform reassembles by row position) and one line per row.
    assert body.splitlines() == ["1.0,3.0", "2.0,4.0"]


def test_pipeline_module_wires_the_shared_pieces() -> None:
    """CI-safe smoke: `pipeline.py` imports cleanly and reuses the shared
    constants from the sibling launchers (so the DAG can't drift from the
    hand-run pieces). The full definition is validated by `run_pipeline --upsert`
    against the account — building it offline hangs the SDK, and upsert is nearly
    free, so that is the right layer for it."""
    import pipeline
    import register_model
    import run_evaluation

    assert pipeline.PIPELINE_NAME == "demand-forecasting-ct"
    assert 0.0 < pipeline.DEFAULT_WAPE_THRESHOLD < 1.0
    # The gate registers under the same group the hand-run register_model uses.
    assert pipeline.MODEL_PACKAGE_GROUP == register_model.MODEL_PACKAGE_GROUP
    # The eval step mounts code at the same path run_evaluation set the PYTHONPATH to.
    assert pipeline.CODE_MOUNT == run_evaluation.CODE_MOUNT
    assert callable(pipeline.get_pipeline)


def test_monitor_summarize_reports_wape_and_psi(feature_table: pd.DataFrame) -> None:
    """The monitor's headline metrics (the numbers the CloudWatch alarm gates on)
    must compute from the archive + actuals: forecast WAPE and worst-feature PSI."""
    import monitor_entry

    from demand_forecasting.training.model import quantile_column

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    # A tiny archive: forecasts for the observed tail days, a flat median forecast.
    archive = (
        feature_table[feature_table["d"] >= 74][["id", "d"]]
        .rename(columns={"d": "target_day"})
        .assign(origin=73, horizon=1)
    )
    for q, val in zip(quantiles, [0.0, 1.0, 2.0, 3.0, 4.0]):
        archive[quantile_column(q)] = val

    summary = monitor_entry.summarize(archive, feature_table, quantiles)
    assert set(summary) == {"forecast_wape", "max_psi", "n_drifted", "n_scored", "n_series"}
    assert summary["n_scored"] == len(archive)  # every forecast day is observed here
    assert summary["forecast_wape"] >= 0.0
    assert summary["max_psi"] >= 0.0


def test_monitoring_metric_data_and_retrain_pattern() -> None:
    """The CloudWatch payload and the retrain event pattern are what connect the
    monitor to the alarm to the pipeline; pin their shapes so the wiring holds."""
    import run_monitoring

    summary = {"forecast_wape": 0.7, "max_psi": 0.12, "n_drifted": 1, "n_scored": 100}
    data = run_monitoring.metric_data(summary)
    assert {m["MetricName"] for m in data} == {"ForecastWAPE", "MaxPSI", "DriftedFeatures"}
    assert next(m for m in data if m["MetricName"] == "ForecastWAPE")["Value"] == 0.7

    pattern = run_monitoring.retrain_event_pattern("my-alarm")
    assert pattern["source"] == ["aws.cloudwatch"]
    assert pattern["detail"]["alarmName"] == ["my-alarm"]
    # Only a breach (ALARM), never a recovery (OK), triggers a retrain.
    assert pattern["detail"]["state"]["value"] == ["ALARM"]


def test_create_schedule_is_disabled_and_targets_the_pipeline() -> None:
    """The weekly rule must be created DISABLED (a live schedule bills forever)
    and wired to the pipeline ARN with the given role. Injected fake client — no
    AWS, CI-safe."""
    import run_pipeline

    calls: dict[str, Any] = {}

    class FakeEvents:
        def put_rule(self, **kw: Any) -> None:
            calls["rule"] = kw

        def put_targets(self, **kw: Any) -> None:
            calls["targets"] = kw

    run_pipeline.create_schedule(
        FakeEvents(), "demand-forecasting-ct-weekly", "rate(7 days)", "arn:pipe", "arn:role"
    )
    assert calls["rule"]["State"] == "DISABLED"
    assert calls["rule"]["ScheduleExpression"] == "rate(7 days)"
    target = calls["targets"]["Targets"][0]
    assert target["Arn"] == "arn:pipe" and target["RoleArn"] == "arn:role"


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
