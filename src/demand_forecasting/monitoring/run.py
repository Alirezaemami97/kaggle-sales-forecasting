"""Monitoring job: read the latest prediction archive and emit the three
always-on monitoring layers as artifacts.

  1. operational — a health record (rows, series, cold-start share, origin);
  2. data drift  — PSI per input feature, recent window vs an earlier reference;
  3. forecast error — archive joined to actuals, scored by horizon.

The rich Evidently report (drift.py) is a separate optional step. This job needs
only pandas + the always-on PSI signal, so it runs anywhere the batch job does.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from demand_forecasting.config import Config, load_config
from demand_forecasting.monitoring.drift import feature_drift
from demand_forecasting.monitoring.report import operational_summary, score_forecasts

logger = logging.getLogger(__name__)

# Model input features worth watching for distribution shift.
MONITORED_FEATURES = [
    "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28", "sell_price", "price_rel_dept",
]


def latest_archive(config: Config) -> pd.DataFrame:
    """Load the most recent run's prediction archive."""
    runs = sorted(config.paths.forecasts_dir.glob("run_date=*"))
    if not runs:
        raise FileNotFoundError(f"No prediction archives under {config.paths.forecasts_dir}")
    path = runs[-1] / "forecast.parquet"
    logger.info("Monitoring latest archive: %s", path)
    return pd.read_parquet(path)


def run_monitoring(config: Config, window: int = 28) -> dict[str, Path]:
    """Compute the three layers and write them under paths.models_dir/monitoring."""
    features = pd.read_parquet(config.paths.processed_dir / "features.parquet")
    archive = latest_archive(config)

    operational = operational_summary(archive)
    scored = score_forecasts(archive, features, config.training.quantiles)

    last = int(features["d"].max())
    current = features[features["d"] > last - window]
    reference = features[(features["d"] <= last - window) & (features["d"] > last - 2 * window)]
    drift = feature_drift(reference, current, MONITORED_FEATURES)

    out_dir = config.paths.models_dir / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "operational.json").write_text(json.dumps(operational, indent=2), encoding="utf-8")
    scored.to_csv(out_dir / "forecast_error.csv", index=False)
    drift.to_csv(out_dir / "data_drift.csv", index=False)
    logger.info("Monitoring: %s; drift on %d feature(s)", operational, int(drift["drifted"].sum()))
    return {
        "operational": out_dir / "operational.json",
        "forecast_error": out_dir / "forecast_error.csv",
        "data_drift": out_dir / "data_drift.csv",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    run_monitoring(load_config(args.config))


if __name__ == "__main__":
    main()
