"""Monitoring layers 1 and 3: operational health and forecast error.

Layer 1 (operational) — did the job run and produce sane volume: row counts,
series counts, cold-start share, the origin it forecast from.

Layer 3 (forecast error / concept drift) — once actuals land, how wrong were we.
`score_forecasts` joins an archived forecast to the observed actuals and reports
WAPE, pinball, and interval coverage by horizon. Only target days that are now
observed can be scored; the rest wait for their actuals (the whole reason the
archive exists). Data drift (layer 2, Evidently) lives in `drift.py`.
"""

import logging

import pandas as pd

from demand_forecasting.evaluation.metrics import coverage, mean_pinball, wape
from demand_forecasting.training.model import quantile_column

logger = logging.getLogger(__name__)

# (nominal, lower-quantile col, upper-quantile col) for coverage checks.
_INTERVALS = [(50, quantile_column(0.25), quantile_column(0.75)),
              (90, quantile_column(0.05), quantile_column(0.95))]


def _first(archive: pd.DataFrame, col: str) -> object:
    return str(archive[col].iloc[0]) if col in archive.columns else None


def operational_summary(archive: pd.DataFrame) -> dict[str, object]:
    """Layer 1: a compact health record for one batch run."""
    cold = 0
    if "is_cold_start" in archive.columns:
        cold = int(archive.loc[archive["is_cold_start"], "id"].nunique())
    return {
        "run_date": _first(archive, "run_date"),
        "model_version": _first(archive, "model_version"),
        "origin": int(archive["origin"].iloc[0]),
        "n_forecasts": int(len(archive)),
        "n_series": int(archive["id"].nunique()),
        "n_cold_start_series": cold,
        "horizons": int(archive["horizon"].nunique()),
    }


def score_forecasts(
    archive: pd.DataFrame, features: pd.DataFrame, quantiles: list[float]
) -> pd.DataFrame:
    """Layer 3: join archived forecasts to observed actuals and score by horizon."""
    actuals = features[["id", "d", "sales"]].rename(columns={"d": "target_day", "sales": "actual"})
    scored = archive.merge(actuals, on=["id", "target_day"], how="inner")
    if scored.empty:
        logger.info("No archived forecasts have observed actuals yet; nothing to score")
        return pd.DataFrame()

    median = quantile_column(0.5)
    qcols = [quantile_column(q) for q in sorted(quantiles)]
    rows = []
    for h, g in scored.groupby("horizon", observed=True):
        row: dict[str, object] = {
            "horizon": int(h),
            "n": int(len(g)),
            "wape": wape(g["actual"], g[median]),
            "pinball": mean_pinball(g["actual"], g[qcols]),
        }
        for nominal, lo, hi in _INTERVALS:
            if lo in g.columns and hi in g.columns:
                row[f"coverage_{nominal}"] = coverage(g["actual"], g[lo], g[hi])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("horizon", ignore_index=True)
