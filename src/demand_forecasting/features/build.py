"""Build the feature table from processed Parquet:

    python -m demand_forecasting.features.build --config config/config.yaml

Reads sales/calendar/prices from processed_dir, writes features.parquet there.
Training (M3) and batch inference (M6) both consume this table — and both
import the same `pipeline.py` functions for any at-forecast-time computation.
"""

import argparse
import logging
import time

import pandas as pd

from demand_forecasting.config import Config, load_config
from demand_forecasting.features.pipeline import build_features

logger = logging.getLogger(__name__)


def build(config: Config) -> pd.DataFrame:
    processed = config.paths.processed_dir
    sales = pd.read_parquet(processed / "sales.parquet")
    calendar = pd.read_parquet(processed / "calendar.parquet")
    prices = pd.read_parquet(processed / "prices.parquet")
    logger.info("Loaded processed tables: %d series", len(sales))

    start = time.perf_counter()
    features = build_features(
        sales,
        calendar,
        prices,
        lags=config.features.lags,
        rolling_windows=config.features.rolling_windows,
        drop_pre_release=config.features.drop_pre_release,
    )
    elapsed = time.perf_counter() - start

    mem_gb = features.memory_usage(deep=True).sum() / 1e9
    logger.info(
        "Built features: %d rows x %d cols | %.2f GB in memory | %.1fs",
        len(features),
        features.shape[1],
        mem_gb,
        elapsed,
    )

    out_path = processed / "features.parquet"
    features.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
