"""Tests for the batch inference job and the prediction archive."""

import pandas as pd
import pytest

from demand_forecasting.config import load_config
from demand_forecasting.inference.batch import (
    check_forecasts,
    cold_start_forecasts,
    forecast_catalogue,
    generate_forecasts,
    hierarchy_priors,
    identify_cold_start,
    latest_full_origin,
    write_archive,
)
from demand_forecasting.training.dataset import build_direct_table, select_origins, to_model_frame
from demand_forecasting.training.model import QuantileLGBM

TINY_PARAMS: dict[str, object] = {
    "n_estimators": 15,
    "num_leaves": 7,
    "min_child_samples": 1,
    "learning_rate": 0.3,
    "seed": 0,
}
QUANTILES = [0.05, 0.5, 0.95]


def _trained_model(feature_table: pd.DataFrame, horizon: int) -> QuantileLGBM:
    origins = select_origins(feature_table, n_origins=6, stride=8, horizon=horizon)
    table = build_direct_table(feature_table, origins, horizon)
    return QuantileLGBM(QUANTILES, dict(TINY_PARAMS)).fit(to_model_frame(table), table["sales"])


def test_latest_full_origin(feature_table: pd.DataFrame) -> None:
    # last day 80, horizon 5 → most recent fully-observed origin is 75.
    assert latest_full_origin(feature_table, horizon=5) == 75


def test_generate_forecasts_shape_and_contract(feature_table: pd.DataFrame) -> None:
    horizon = 5
    model = _trained_model(feature_table, horizon)
    origin = latest_full_origin(feature_table, horizon)
    preds = generate_forecasts(feature_table, model, QUANTILES, horizon, origin)

    # One row per (series, horizon); identity + forecast columns present.
    assert len(preds) == 3 * horizon
    cols = {"id", "origin", "horizon", "target_day", "q0.05", "q0.5", "q0.95"}
    assert cols.issubset(preds.columns)
    assert set(preds["horizon"]) == {1, 2, 3, 4, 5}
    assert (preds["target_day"] == preds["origin"] + preds["horizon"]).all()
    # The archive gate passes on a well-formed forecast.
    check_forecasts(preds, QUANTILES)


def test_check_forecasts_rejects_bad_output(feature_table: pd.DataFrame) -> None:
    model = _trained_model(feature_table, horizon=5)
    preds = generate_forecasts(feature_table, model, QUANTILES, 5, 75)
    negative = preds.copy()
    negative.loc[negative.index[0], "q0.05"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        check_forecasts(negative, QUANTILES)

    crossing = preds.copy()
    crossing.loc[crossing.index[0], "q0.95"] = 0.0  # now below q0.05/q0.5
    with pytest.raises(ValueError, match="crossing"):
        check_forecasts(crossing, QUANTILES)


def test_write_archive_roundtrip(tmp_path, feature_table: pd.DataFrame) -> None:
    # Minimal config pointing the archive at tmp_path.
    config = load_config("config/config.yaml")
    config.paths.forecasts_dir = tmp_path
    model = _trained_model(feature_table, horizon=5)
    preds = generate_forecasts(feature_table, model, QUANTILES, 5, 75)

    path = write_archive(preds, config, run_date="2026-07-13", model_version="local")
    assert path.exists()
    back = pd.read_parquet(path)
    assert (back["run_date"] == "2026-07-13").all()
    assert (back["model_version"] == "local").all()
    assert len(back) == len(preds)


def _with_cold_series(feature_table: pd.DataFrame) -> pd.DataFrame:
    """S1/S2 keep full history; S3 is truncated to 10 days → cold-start at origin 75."""
    warm = feature_table[feature_table["id"].isin(["S1", "S2"])]
    cold = feature_table[(feature_table["id"] == "S3") & (feature_table["d"] <= 10)]
    return pd.concat([warm, cold], ignore_index=True)


def test_identify_cold_start(feature_table: pd.DataFrame) -> None:
    frame = _with_cold_series(feature_table)
    cold = identify_cold_start(frame, origin=75, min_history=28)
    assert cold == {"S3"}
    # A lenient threshold treats even the short series as warm.
    assert identify_cold_start(frame, origin=75, min_history=5) == set()


def test_hierarchy_priors_are_group_quantiles(feature_table: pd.DataFrame) -> None:
    priors = hierarchy_priors(feature_table, origin=75, quantiles=[0.05, 0.5, 0.95], window=28)
    # One row for the single (store_id, dept_id) group in the fixture.
    row = priors[(priors["store_id"] == "CA_1") & (priors["dept_id"] == "FOODS_1")].iloc[0]
    # Sales equal the day number; recent window (47, 75] → median around 61.
    assert row["q0.05"] <= row["q0.5"] <= row["q0.95"]
    assert 55 <= row["q0.5"] <= 68


def test_cold_start_forecasts_broadcast_and_flagged(feature_table: pd.DataFrame) -> None:
    frame = _with_cold_series(feature_table)
    priors = hierarchy_priors(frame, origin=75, quantiles=[0.05, 0.5, 0.95], window=28)
    out = cold_start_forecasts(
        frame, {"S3"}, [0.05, 0.5, 0.95], horizon=5, origin=75, priors=priors
    )
    assert set(out["horizon"]) == {1, 2, 3, 4, 5}
    assert (out["target_day"] == 75 + out["horizon"]).all()
    # Same prior vector broadcast across every horizon (one distinct value per quantile).
    assert out["q0.5"].nunique() == 1


def test_forecast_catalogue_splits_warm_and_cold(feature_table: pd.DataFrame) -> None:
    frame = _with_cold_series(feature_table)
    config = load_config("config/config.yaml")
    config.training.horizon = 5
    config.training.quantiles = QUANTILES
    config.inference.cold_start_min_history_days = 28
    model = _trained_model(feature_table, horizon=5)

    preds = forecast_catalogue(frame, model, config, origin=75)
    cold_rows = preds[preds["is_cold_start"]]
    warm_rows = preds[~preds["is_cold_start"]]
    assert set(cold_rows["id"]) == {"S3"}
    assert set(warm_rows["id"]) == {"S1", "S2"}
    check_forecasts(preds, QUANTILES)
