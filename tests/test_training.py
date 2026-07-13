"""Tests for the direct-multihorizon dataset builder and the quantile model."""

import pandas as pd

from demand_forecasting.training.dataset import (
    FEATURE_COLS,
    build_direct_table,
    select_origins,
    to_model_frame,
)
from demand_forecasting.training.model import QuantileLGBM, quantile_column

TINY_PARAMS: dict[str, object] = {
    "n_estimators": 15,
    "num_leaves": 7,
    "min_child_samples": 1,
    "learning_rate": 0.3,
    "seed": 0,
}


def test_select_origins_spacing(feature_table: pd.DataFrame) -> None:
    origins = select_origins(feature_table, n_origins=3, stride=14, horizon=5)
    # last_day 80, latest origin 75, then 61, 47.
    assert origins == [47, 61, 75]


def test_build_direct_table_relabel_and_target(feature_table: pd.DataFrame) -> None:
    table = build_direct_table(feature_table, origins=[40], horizon=5)
    # 3 series x 5 horizons, all target days (41..45) exist.
    assert len(table) == 3 * 5
    assert set(table["horizon"]) == {1, 2, 3, 4, 5}
    # target_day = origin + horizon, and target sales == target_day (fixture design).
    assert (table["target_day"] == table["origin"] + table["horizon"]).all()
    assert (table["sales"] == table["target_day"]).all()
    # Dynamic block came from day O+1 = 41 (lag_7 == 41 in the fixture), not O+h.
    assert (table["lag_7"] == 41).all()


def test_build_direct_table_drops_unavailable_targets(feature_table: pd.DataFrame) -> None:
    # Origin 79 with horizon 5 wants target days 80..84; only 80 exists.
    table = build_direct_table(feature_table, origins=[79], horizon=5)
    assert set(table["horizon"]) == {1}


def test_to_model_frame_types(feature_table: pd.DataFrame) -> None:
    table = build_direct_table(feature_table, origins=[40], horizon=5)
    X = to_model_frame(table)
    assert list(X.columns) == FEATURE_COLS
    assert str(X["item_id"].dtype) == "category"
    assert str(X["snap"].dtype) == "category"


def test_model_predict_nonnegative_and_monotone(feature_table: pd.DataFrame) -> None:
    origins = select_origins(feature_table, n_origins=6, stride=8, horizon=5)
    table = build_direct_table(feature_table, origins, horizon=5)
    quantiles = [0.1, 0.5, 0.9]
    model = QuantileLGBM(quantiles, dict(TINY_PARAMS)).fit(to_model_frame(table), table["sales"])
    preds = model.predict(to_model_frame(table))

    assert list(preds.columns) == [quantile_column(q) for q in quantiles]
    assert (preds.to_numpy() >= 0).all()
    # Quantiles never cross, row by row.
    lo, mid, hi = preds["q0.1"], preds["q0.5"], preds["q0.9"]
    assert (lo <= mid + 1e-9).all()
    assert (mid <= hi + 1e-9).all()


def test_model_save_load_roundtrip(tmp_path, feature_table: pd.DataFrame) -> None:
    origins = select_origins(feature_table, n_origins=6, stride=8, horizon=5)
    table = build_direct_table(feature_table, origins, horizon=5)
    X = to_model_frame(table)
    model = QuantileLGBM([0.5], dict(TINY_PARAMS)).fit(X, table["sales"])

    model.save(tmp_path / "m")
    reloaded = QuantileLGBM.load(tmp_path / "m")
    pd.testing.assert_frame_equal(model.predict(X), reloaded.predict(X))
