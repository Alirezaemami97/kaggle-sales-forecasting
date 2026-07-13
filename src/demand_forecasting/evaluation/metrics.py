"""Forecast metrics. M3 ships the two headline numbers; M4 adds the full panel
(MASE, WRMSSE, coverage, hierarchy levels).

WAPE and pinball are chosen deliberately over MAPE: 68% of item-days are zero
sales, and MAPE divides by the actual — it is undefined/explosive here.
"""

import numpy as np
import pandas as pd


def wape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
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


def pinball_loss(
    actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray, q: float
) -> float:
    """Pinball (quantile) loss at quantile q — the proper scoring rule for a
    quantile forecast. Under-prediction is penalised by q, over-prediction by 1-q,
    so the loss is minimised by the true q-th quantile."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    diff = actual - predicted
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def mean_pinball(actual: pd.Series | np.ndarray, quantile_preds: pd.DataFrame) -> float:
    """Average pinball loss across quantile columns named 'q<value>'."""
    losses = [
        pinball_loss(actual, quantile_preds[col], float(col[1:]))
        for col in quantile_preds.columns
    ]
    return float(np.mean(losses))
