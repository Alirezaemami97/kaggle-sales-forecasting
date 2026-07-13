"""Tests for metrics and the rolling-origin backtest."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.evaluation.backtest import (
    backtest_predictions,
    fold_origins,
    rolling_origin_backtest,
)
from demand_forecasting.evaluation.metrics import (
    coverage,
    mase,
    mean_pinball,
    naive_mse,
    pinball_loss,
    rmsse,
    seasonal_naive_mae,
    wape,
)
from demand_forecasting.evaluation.panel import build_panel

TINY_PARAMS: dict[str, object] = {
    "n_estimators": 15,
    "num_leaves": 7,
    "min_child_samples": 1,
    "learning_rate": 0.3,
    "seed": 0,
}


def test_wape_known_value() -> None:
    # |2-1| + |0-0| + |4-4| = 1 ; sum|actual| = 6.
    assert wape([2, 0, 4], [1, 0, 4]) == pytest.approx(1 / 6)


def test_wape_all_zero_actual_is_nan() -> None:
    assert np.isnan(wape([0, 0], [1, 2]))


def test_pinball_symmetry_at_median() -> None:
    # At q=0.5 the loss is half the absolute error.
    assert pinball_loss([1.0], [0.0], 0.5) == pytest.approx(0.5)


def test_pinball_penalises_underprediction_more_at_high_q() -> None:
    under = pinball_loss([10.0], [0.0], 0.9)  # missed a high actual, penalty q
    over = pinball_loss([0.0], [10.0], 0.9)  # over-forecast, penalty 1-q
    assert under == pytest.approx(9.0)
    assert over == pytest.approx(1.0)


def test_mean_pinball_averages_columns() -> None:
    actual = np.array([5.0, 5.0])
    preds = pd.DataFrame({"q0.1": [4.0, 4.0], "q0.9": [6.0, 6.0]})
    expected = (pinball_loss(actual, [4.0, 4.0], 0.1) + pinball_loss(actual, [6.0, 6.0], 0.9)) / 2
    assert mean_pinball(actual, preds) == pytest.approx(expected)


def test_seasonal_naive_mae_known_value() -> None:
    # season=7: |8-1| and |9-2| = 7, 7 -> mean 7.
    assert seasonal_naive_mae([1, 2, 3, 4, 5, 6, 7, 8, 9], season=7) == pytest.approx(7.0)


def test_seasonal_naive_mae_too_short_is_nan() -> None:
    assert np.isnan(seasonal_naive_mae([1, 2, 3], season=7))


def test_naive_mse_known_value() -> None:
    # diffs 2, 3 -> squares 4, 9 -> mean 6.5.
    assert naive_mse([1, 3, 6]) == pytest.approx(6.5)


def test_mase_scales_by_naive() -> None:
    # MAE = mean(|1|, 0) = 0.5 ; scale 2 -> 0.25.
    assert mase([2, 4], [1, 4], scale=2.0) == pytest.approx(0.25)


def test_mase_zero_scale_is_nan() -> None:
    assert np.isnan(mase([1, 2], [1, 2], scale=0.0))


def test_rmsse_known_value() -> None:
    # squared errors 4, 0 -> mean 2 ; /scale 4 = 0.5 ; sqrt = 0.7071.
    assert rmsse([2, 4], [0, 4], scale_mse=4.0) == pytest.approx(np.sqrt(0.5))


def test_coverage_fraction_inside_interval() -> None:
    # Only the value 1 lies in [0, 4]; 5 and 9 fall outside -> 1/3.
    assert coverage([1, 5, 9], [0, 0, 0], [4, 4, 4]) == pytest.approx(1 / 3)


def test_fold_origins_spacing(feature_table: pd.DataFrame) -> None:
    assert fold_origins(feature_table, n_folds=2, stride=28, horizon=5) == [47, 75]


def test_backtest_runs_and_reports(feature_table: pd.DataFrame) -> None:
    folds = fold_origins(feature_table, n_folds=2, stride=20, horizon=5)
    results = rolling_origin_backtest(
        feature_table,
        quantiles=[0.5],
        lgbm_params=dict(TINY_PARAMS),
        horizon=5,
        folds=folds,
        n_train_origins=4,
        origin_stride=6,
    )
    assert len(results) == len(folds)
    assert {"fold_origin", "wape", "mean_pinball"}.issubset(results.columns)
    assert results["wape"].notna().all()


def test_panel_structure_and_levels(feature_table: pd.DataFrame) -> None:
    folds = fold_origins(feature_table, n_folds=2, stride=20, horizon=5)
    preds = backtest_predictions(
        feature_table,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        lgbm_params=dict(TINY_PARAMS),
        horizon=5,
        folds=folds,
        n_train_origins=4,
        origin_stride=6,
    )
    panel = build_panel(preds, feature_table)

    # by_level covers every hierarchy level and reports the three point metrics.
    levels = set(panel["by_level"]["level"])
    assert {"total", "store", "item_store"}.issubset(levels)
    assert {"wape", "mase", "rmsse"}.issubset(panel["by_level"].columns)

    # by_horizon spans the whole horizon and carries the coverage curve.
    assert list(panel["by_horizon"]["horizon"]) == [1, 2, 3, 4, 5]
    assert "coverage_90" in panel["by_horizon"].columns

    # calibration reports nominal vs empirical, empirical a valid probability.
    cal = panel["calibration"]
    assert set(cal["nominal_coverage"]) == {0.5, 0.9}
    assert ((cal["empirical_coverage"] >= 0) & (cal["empirical_coverage"] <= 1)).all()
