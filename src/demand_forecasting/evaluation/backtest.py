"""Expanding-window, rolling-origin backtesting — the honest way to estimate
production performance for a time series.

For each fold origin O_f (spaced `fold_stride` apart, newest last):
  1. train on origins strictly earlier than O_f, whose entire horizon ends at or
     before O_f (train target days <= O_f) — so no fold ever sees its own future;
  2. forecast O_f+1 … O_f+H and score against the held-out actuals.
Retraining every fold is deliberate: it measures exactly what Continuous
Training (M7) will do on a schedule, retraining cost included.
"""

import logging

import pandas as pd

from demand_forecasting.evaluation.metrics import mean_pinball, wape
from demand_forecasting.training.dataset import build_direct_table, to_model_frame
from demand_forecasting.training.model import QuantileLGBM, quantile_column

logger = logging.getLogger(__name__)


def fold_origins(features: pd.DataFrame, n_folds: int, stride: int, horizon: int) -> list[int]:
    """The most recent `n_folds` origins, `stride` apart, that have full horizons."""
    last_day = int(features["d"].max())
    latest = last_day - horizon
    origins = [latest - i * stride for i in range(n_folds)]
    return sorted(o for o in origins if o > 0)


def _train_origins_before(
    fold: int, n_train_origins: int, origin_stride: int, horizon: int
) -> list[int]:
    """Training origins whose horizon ends on or before the fold origin."""
    latest = fold - horizon
    origins = [latest - i * origin_stride for i in range(n_train_origins)]
    return sorted(o for o in origins if o > 0)


def rolling_origin_backtest(
    features: pd.DataFrame,
    quantiles: list[float],
    lgbm_params: dict[str, object],
    horizon: int,
    folds: list[int],
    n_train_origins: int,
    origin_stride: int,
) -> pd.DataFrame:
    """Run the backtest and return one metrics row per fold (retraining each fold)."""
    rows = []
    median_col = quantile_column(0.5)

    for fold in folds:
        train_origins = _train_origins_before(fold, n_train_origins, origin_stride, horizon)
        if not train_origins:
            logger.warning("Fold origin %d has no valid training origins; skipping", fold)
            continue

        train_table = build_direct_table(features, train_origins, horizon)
        model = QuantileLGBM(quantiles, dict(lgbm_params))
        model.fit(to_model_frame(train_table), train_table["sales"])

        test_table = build_direct_table(features, [fold], horizon)
        preds = model.predict(to_model_frame(test_table))
        actual = test_table["sales"]

        rows.append(
            {
                "fold_origin": fold,
                "n_train_rows": len(train_table),
                "n_test_rows": len(test_table),
                "wape": wape(actual, preds[median_col]),
                "mean_pinball": mean_pinball(actual, preds),
            }
        )
        logger.info(
            "Fold %d — WAPE %.4f | pinball %.4f | train rows %d",
            fold,
            rows[-1]["wape"],
            rows[-1]["mean_pinball"],
            len(train_table),
        )

    return pd.DataFrame(rows)
