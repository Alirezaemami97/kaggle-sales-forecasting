"""Tests for the monitoring report: operational summary and forecast-error scorer."""

import pandas as pd

from demand_forecasting.inference.batch import forecast_catalogue, latest_full_origin
from demand_forecasting.monitoring.report import operational_summary, score_forecasts
from demand_forecasting.training.dataset import build_direct_table, select_origins, to_model_frame
from demand_forecasting.training.model import QuantileLGBM

TINY_PARAMS: dict[str, object] = {
    "n_estimators": 15,
    "num_leaves": 7,
    "min_child_samples": 1,
    "learning_rate": 0.3,
    "seed": 0,
}
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _archive(feature_table: pd.DataFrame, config) -> pd.DataFrame:
    origins = select_origins(feature_table, n_origins=6, stride=8, horizon=5)
    table = build_direct_table(feature_table, origins, horizon=5)
    model = QuantileLGBM(QUANTILES, dict(TINY_PARAMS)).fit(to_model_frame(table), table["sales"])
    origin = latest_full_origin(feature_table, horizon=5)
    return forecast_catalogue(feature_table, model, config, origin).assign(
        run_date="2026-07-13", model_version="local"
    )


def _config():
    from demand_forecasting.config import load_config

    config = load_config("config/config.yaml")
    config.training.horizon = 5
    config.training.quantiles = QUANTILES
    return config


def test_operational_summary_counts(feature_table: pd.DataFrame) -> None:
    archive = _archive(feature_table, _config())
    summary = operational_summary(archive)
    assert summary["n_series"] == 3
    assert summary["horizons"] == 5
    assert summary["origin"] == 75
    assert summary["run_date"] == "2026-07-13"
    assert summary["n_cold_start_series"] == 0  # full-history fixture


def test_score_forecasts_by_horizon(feature_table: pd.DataFrame) -> None:
    archive = _archive(feature_table, _config())
    scored = score_forecasts(archive, feature_table, QUANTILES)
    # Target days 76..80 are all observed → every horizon scores.
    assert list(scored["horizon"]) == [1, 2, 3, 4, 5]
    assert scored["wape"].notna().all()
    assert "coverage_90" in scored.columns
    assert ((scored["coverage_90"] >= 0) & (scored["coverage_90"] <= 1)).all()


def test_score_forecasts_empty_when_no_actuals(feature_table: pd.DataFrame) -> None:
    archive = _archive(feature_table, _config())
    # Shift target days beyond the observed range → nothing to join.
    future = archive.assign(target_day=archive["target_day"] + 1000)
    assert score_forecasts(future, feature_table, QUANTILES).empty


def test_run_monitoring_writes_all_layers(tmp_path, feature_table: pd.DataFrame) -> None:
    from demand_forecasting.monitoring.run import run_monitoring

    config = _config()
    config.paths.processed_dir = tmp_path
    config.paths.forecasts_dir = tmp_path / "forecasts"
    config.paths.models_dir = tmp_path / "models"
    feature_table.to_parquet(tmp_path / "features.parquet", index=False)
    run_dir = config.paths.forecasts_dir / "run_date=2026-07-13"
    run_dir.mkdir(parents=True)
    _archive(feature_table, config).to_parquet(run_dir / "forecast.parquet", index=False)

    paths = run_monitoring(config, window=28)
    assert paths["operational"].exists()
    assert paths["forecast_error"].exists()
    assert paths["data_drift"].exists()
    drift = pd.read_csv(paths["data_drift"])
    assert {"feature", "psi", "drifted"}.issubset(drift.columns)
