"""Tests for metrics and the rolling-origin backtest."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.evaluation.backtest import fold_origins, rolling_origin_backtest
from demand_forecasting.evaluation.metrics import mean_pinball, pinball_loss, wape

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
