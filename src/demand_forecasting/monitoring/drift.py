"""Monitoring layer 2: data drift on the inference inputs.

Two tiers, on purpose:

- **PSI (Population Stability Index)** is the lightweight, always-on signal — a
  single number per feature comparing today's input distribution to a training
  reference. It has no heavy dependencies, runs in CI, and is what the M7
  retraining trigger will watch. Rule of thumb: <0.1 stable, 0.1-0.25 moderate
  shift, >0.25 significant.
- **Evidently** produces the rich human-facing drift report (per-feature plots,
  statistical tests). It is a heavy optional dependency (the `monitoring` group)
  and is lazy-imported, so the always-on PSI path never depends on it.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

# Classic PSI decision line: above this a feature is flagged as drifted.
DRIFT_THRESHOLD = 0.2


def population_stability_index(
    reference: npt.NDArray[Any] | pd.Series,
    current: npt.NDArray[Any] | pd.Series,
    bins: int = 10,
) -> float:
    """PSI between a reference and current sample, binned on reference quantiles.

    Returns 0 when a feature is (near-)constant in the reference (no bins to
    compare). Empty bins are floored by a small epsilon so the log is finite.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    ref_pct = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    cur_pct = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    threshold: float = DRIFT_THRESHOLD,
) -> pd.DataFrame:
    """PSI per feature with a drifted flag — the always-on drift table."""
    rows = []
    for f in features:
        if f not in reference.columns or f not in current.columns:
            continue
        psi = population_stability_index(reference[f], current[f])
        rows.append({"feature": f, "psi": psi, "drifted": psi > threshold})
    result = pd.DataFrame(rows)
    if not result.empty and result["drifted"].any():
        drifted = result.loc[result["drifted"], "feature"].tolist()
        logger.warning("Data drift detected on %d feature(s): %s", len(drifted), drifted)
    return result


def evidently_data_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, output_path: str | Path
) -> Path:
    """Rich Evidently HTML drift report (optional `monitoring` group). Lazy import
    so the always-on PSI path never requires the heavy dependency."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:  # pragma: no cover - exercised only without the group
        raise ImportError(
            "Evidently is not installed. Run: poetry install --with monitoring"
        ) from exc

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    logger.info("Wrote Evidently drift report to %s", out)
    return out
