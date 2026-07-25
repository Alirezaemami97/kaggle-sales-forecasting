"""SageMaker Processing entry point for monitoring the batch forecast archive.

Reuses the portable monitoring code unchanged — `feature_drift` (PSI) and
`score_forecasts` (forecast error by horizon) — so the cloud monitor computes the
same signals as the local M6 job, no skew. It writes:
  - monitoring.json  — the headline metrics the launcher publishes to CloudWatch;
  - forecast_error.csv / data_drift.csv / operational.json — the full artifacts.

Ground truth already exists: Phase 5 anchored forecasts at the latest fully
observed origin, so the actuals to score against are in the feature set now.

Same Processing contract as evaluate_entry: no SM_* env vars, paths arrive as CLI
args over /opt/ml/processing/..., and defaults stay overridable for a free local
run:

    python aws/sagemaker/monitor_entry.py --archive data/forecasts/<run>/archive.parquet \
        --features-dir data/processed/features.parquet --output-dir /tmp/mon
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

# Same-directory import: the launcher ships train_entry.py alongside this file.
from train_entry import apply_overrides, load_features

from demand_forecasting.config import Config, load_config
from demand_forecasting.evaluation.metrics import wape
from demand_forecasting.monitoring.drift import feature_drift
from demand_forecasting.monitoring.report import operational_summary, score_forecasts
from demand_forecasting.training.model import quantile_column

logger = logging.getLogger(__name__)

# Model-input features worth watching for distribution shift (matches monitoring/run.py).
MONITORED_FEATURES = [
    "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28", "sell_price", "price_rel_dept",
]
DRIFT_WINDOW = 28


def summarize(
    archive: pd.DataFrame, features: pd.DataFrame, quantiles: list[float]
) -> dict[str, object]:
    """Headline metrics for CloudWatch: overall forecast WAPE (median vs actuals)
    and the worst per-feature PSI. Pure — unit-tested in CI, so the numbers the
    alarm gates on can't silently break."""
    actuals = features[["id", "d", "sales"]].rename(columns={"d": "target_day", "sales": "actual"})
    scored = archive.merge(actuals, on=["id", "target_day"], how="inner")
    forecast_wape = (
        float(wape(scored["actual"], scored[quantile_column(0.5)])) if not scored.empty else -1.0
    )

    last = int(features["d"].max())
    current = features[features["d"] > last - DRIFT_WINDOW]
    reference = features[
        (features["d"] <= last - DRIFT_WINDOW) & (features["d"] > last - 2 * DRIFT_WINDOW)
    ]
    drift = feature_drift(reference, current, MONITORED_FEATURES)
    max_psi = float(drift["psi"].max()) if not drift.empty else 0.0
    n_drifted = int(drift["drifted"].sum()) if not drift.empty else 0

    return {
        "forecast_wape": forecast_wape,
        "max_psi": max_psi,
        "n_drifted": n_drifted,
        "n_scored": int(len(scored)),
        "n_series": int(archive["id"].nunique()),
    }


def monitor(config: Config, archive_path: Path, features_dir: Path, output_dir: Path) -> Path:
    archive = pd.read_parquet(archive_path)
    features = load_features(features_dir, config)
    logger.info("Monitoring %d forecasts against %d feature rows", len(archive), len(features))

    summary = summarize(archive, features, config.training.quantiles)
    operational = operational_summary(archive)
    scored = score_forecasts(archive, features, config.training.quantiles)
    last = int(features["d"].max())
    current = features[features["d"] > last - DRIFT_WINDOW]
    reference = features[
        (features["d"] <= last - DRIFT_WINDOW) & (features["d"] > last - 2 * DRIFT_WINDOW)
    ]
    drift = feature_drift(reference, current, MONITORED_FEATURES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monitoring.json").write_text(json.dumps(summary), encoding="utf-8")
    (output_dir / "operational.json").write_text(json.dumps(operational), encoding="utf-8")
    scored.to_csv(output_dir / "forecast_error.csv", index=False)
    drift.to_csv(output_dir / "data_drift.csv", index=False)
    logger.info(
        "Monitoring: forecast_wape=%(forecast_wape).4f max_psi=%(max_psi).4f "
        "n_drifted=%(n_drifted)d n_scored=%(n_scored)d", summary
    )
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="/opt/ml/processing/input/archive/archive.parquet")
    parser.add_argument("--features-dir", default="/opt/ml/processing/input/features")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output/monitoring")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), vars(args))
    monitor(config, Path(args.archive), Path(args.features_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
