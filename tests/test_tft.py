"""Tests for the TFT path. The pure helpers run in CI (no darts needed — the
Darts/torch imports live inside the functions that use them). The end-to-end
contract test is skipped unless darts is installed (the optional `tft` group)."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.config import TFTConfig
from demand_forecasting.evaluation.backtest import fold_origins
from demand_forecasting.training.tft import (
    _FUTURE_COV,
    _add_event_codes,
    _contiguous,
    _select_series,
)

TINY_TFT = TFTConfig(
    max_series=10,
    train_history_days=60,
    input_chunk_length=10,
    hidden_size=4,
    lstm_layers=1,
    num_attention_heads=1,
    dropout=0.0,
    batch_size=16,
    n_epochs=1,
    learning_rate=0.01,
)


def test_add_event_codes_numeric_and_stable(feature_table: pd.DataFrame) -> None:
    out = _add_event_codes(feature_table)
    assert out["event_name_code"].dtype == np.float32
    assert out["event_type_code"].dtype == np.float32
    # A single event value ("None" in the fixture) maps to one code everywhere.
    assert out["event_name_code"].nunique() == feature_table["event_name_1"].nunique()


def test_contiguous_fills_gaps_and_zeros_sales(feature_table: pd.DataFrame) -> None:
    one = _add_event_codes(feature_table[feature_table["id"] == "S1"]).copy()
    # Punch a hole: drop day 5 so the grid has a gap to fill.
    gapped = one[one["d"] != 5]
    filled = _contiguous(gapped, _FUTURE_COV, end_day=40)
    # Contiguous integer day grid up to end_day, with the missing day restored.
    assert list(filled["d"]) == list(range(int(gapped["d"].min()), 41))
    assert (filled["d"] == 5).any()
    # The restored day has zero sales and no NaN covariates.
    assert filled.loc[filled["d"] == 5, "sales"].iloc[0] == 0.0
    assert filled[_FUTURE_COV].notna().all().all()


def test_select_series_respects_cap_and_history(feature_table: pd.DataFrame) -> None:
    # min_history 20 with earliest fold at day 40 → all 3 series (80 days) qualify.
    chosen = _select_series(feature_table, max_series=2, earliest_fold=40,
                            min_history=20, seed=0)
    assert len(chosen) == 2
    assert set(chosen).issubset({"S1", "S2", "S3"})
    # A history requirement no series can meet yields an empty selection.
    none = _select_series(feature_table, max_series=5, earliest_fold=40,
                          min_history=999, seed=0)
    assert none == []


def test_tft_backtest_frame_contract(feature_table: pd.DataFrame) -> None:
    pytest.importorskip("darts")
    from demand_forecasting.training.tft import backtest_tft_predictions

    horizon = 5
    folds = fold_origins(feature_table, n_folds=1, stride=20, horizon=horizon)
    quantiles = [0.1, 0.5, 0.9]
    preds = backtest_tft_predictions(feature_table, quantiles, TINY_TFT, horizon, folds, seed=0)

    # Same column contract the panel consumes for the LightGBM path.
    expected = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                "origin", "horizon", "target_day", "actual", "q0.1", "q0.5", "q0.9"}
    assert expected.issubset(preds.columns)
    assert set(preds["horizon"]) <= set(range(1, horizon + 1))
    # Behavioural guarantees: non-negative and non-crossing quantiles.
    q = preds[["q0.1", "q0.5", "q0.9"]].to_numpy()
    assert (q >= 0).all()
    assert (q[:, 0] <= q[:, 1] + 1e-6).all()
    assert (q[:, 1] <= q[:, 2] + 1e-6).all()
