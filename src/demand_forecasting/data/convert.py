"""One-time conversion step: raw M5 CSVs → validated, typed Parquet.

Run once after downloading the data:

    python -m demand_forecasting.data.convert --config config/config.yaml

Everything downstream (features, training, inference) reads only the Parquet
outputs — smaller, typed, and the format the future S3/Athena layer expects.
"""

import argparse
import logging

from demand_forecasting.config import Config, load_config
from demand_forecasting.data.loader import load_calendar, load_prices, load_sales

logger = logging.getLogger(__name__)


def convert_all(config: Config) -> None:
    """Load, validate, and write the three tables as Parquet into processed_dir."""
    raw = config.paths.raw_dir
    out = config.paths.processed_dir
    out.mkdir(parents=True, exist_ok=True)

    sales, _ = load_sales(raw / config.data.sales_file)
    if config.data.subsample_series > 0:
        sales = sales.sample(n=config.data.subsample_series, random_state=config.random_seed)
        logger.info("Subsampled to %d series (fast-dev mode)", len(sales))
    sales.to_parquet(out / "sales.parquet", index=False)

    calendar, _ = load_calendar(raw / config.data.calendar_file)
    calendar.to_parquet(out / "calendar.parquet", index=False)

    prices, _ = load_prices(raw / config.data.prices_file)
    prices.to_parquet(out / "prices.parquet", index=False)

    logger.info("Wrote sales/calendar/prices Parquet to %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    convert_all(load_config(args.config))


if __name__ == "__main__":
    main()
