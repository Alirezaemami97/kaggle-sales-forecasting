"""Forecast metrics for the evaluation panel.

WAPE and pinball are chosen deliberately over MAPE: 68% of item-days are zero
sales, and MAPE divides by the actual — it is undefined/explosive here. MASE and
RMSSE scale error against a naive in-sample forecast, so they are comparable
across series of wildly different volume — essential when aggregating thousands.
"""

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

Array: TypeAlias = "pd.Series | npt.NDArray[Any] | Sequence[float]"


def wape(actual: Array, predicted: Array) -> float:
    """Weighted Absolute Percentage Error: sum|a-p| / sum|a|.

    Volume-weighted (high-selling series count more) and zero-safe unless the
    whole actual series is zero.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denom)


def pinball_loss(actual: Array, predicted: Array, q: float) -> float:
    """Pinball (quantile) loss at quantile q — the proper scoring rule for a
    quantile forecast. Under-prediction is penalised by q, over-prediction by 1-q,
    so the loss is minimised by the true q-th quantile."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    diff = actual - predicted
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def mean_pinball(actual: Array, quantile_preds: pd.DataFrame) -> float:
    """Average pinball loss across quantile columns named 'q<value>'."""
    losses = [
        pinball_loss(actual, quantile_preds[col], float(col[1:]))
        for col in quantile_preds.columns
    ]
    return float(np.mean(losses))


def seasonal_naive_mae(history: Array, season: int = 7) -> float:
    """In-sample MAE of a seasonal-naive forecast (y_t predicted by y_{t-season}).

    This is the MASE denominator: it captures each series' own baseline
    difficulty, so a forecast's error can be judged 'better than naive' (<1).
    """
    h = np.asarray(history, dtype=float)
    if len(h) <= season:
        return float("nan")
    return float(np.mean(np.abs(h[season:] - h[:-season])))


def naive_mse(history: Array) -> float:
    """In-sample MSE of a one-step-naive forecast — the RMSSE denominator (M5)."""
    h = np.asarray(history, dtype=float)
    if len(h) <= 1:
        return float("nan")
    return float(np.mean((h[1:] - h[:-1]) ** 2))


def mase(actual: Array, predicted: Array, scale: float) -> float:
    """Mean Absolute Scaled Error: MAE(forecast) / seasonal-naive in-sample MAE.

    < 1 means the forecast beats a seasonal-naive baseline. NaN if the series is
    constant in-sample (scale 0) — such series are dropped from aggregates.
    """
    if not scale > 0:
        return float("nan")
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)) / scale)


def rmsse(actual: Array, predicted: Array, scale_mse: float) -> float:
    """Root Mean Squared Scaled Error — the per-series building block of WRMSSE."""
    if not scale_mse > 0:
        return float("nan")
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2) / scale_mse))


def coverage(actual: Array, lower: Array, upper: Array) -> float:
    """Empirical coverage: fraction of actuals inside [lower, upper].

    Calibration check — a nominal 90% interval should cover ~90% of actuals.
    """
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((a >= lo) & (a <= hi)))
