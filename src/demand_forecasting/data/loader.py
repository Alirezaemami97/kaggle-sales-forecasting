"""Loaders for the three M5 tables with memory-safe dtypes.

The wide sales table is ~30,490 rows x ~1,941 day columns. Loaded naively as
int64 it costs ~470 MB; downcast to int16 (max M5 daily sale is 763) it is
~118 MB — the difference between comfortable and painful on a 16 GB machine.
"""

import logging
from pathlib import Path

import pandas as pd

from demand_forecasting.data.schema import (
    CalendarStats,
    PricesStats,
    SalesStats,
    day_columns,
    validate_calendar,
    validate_prices,
    validate_sales,
)

logger = logging.getLogger(__name__)


def load_sales(path: str | Path) -> tuple[pd.DataFrame, SalesStats]:
    """Load the wide sales table, validate it, and downcast day columns to int16."""
    logger.info("Reading sales from %s", path)
    sales = pd.read_csv(path)
    stats = validate_sales(sales)

    d_cols = day_columns(sales)
    sales[d_cols] = sales[d_cols].astype("int16")
    for col in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        sales[col] = sales[col].astype("category")

    logger.info(
        "Sales OK — %d series x %d days | zero rate %.1f%%",
        stats.n_series,
        stats.n_days,
        stats.zero_rate * 100,
    )
    return sales, stats


def load_calendar(path: str | Path) -> tuple[pd.DataFrame, CalendarStats]:
    """Load the calendar table, validate it, and type the columns."""
    logger.info("Reading calendar from %s", path)
    calendar = pd.read_csv(path)
    stats = validate_calendar(calendar)

    calendar["date"] = pd.to_datetime(calendar["date"])
    for col in ("wday", "month", "snap_CA", "snap_TX", "snap_WI"):
        calendar[col] = calendar[col].astype("int8")
    calendar["year"] = calendar["year"].astype("int16")
    calendar["wm_yr_wk"] = calendar["wm_yr_wk"].astype("int32")
    for col in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        calendar[col] = calendar[col].astype("category")

    logger.info("Calendar OK — %d days (%s → %s)", stats.n_days, stats.first_date, stats.last_date)
    return calendar, stats


def load_prices(path: str | Path) -> tuple[pd.DataFrame, PricesStats]:
    """Load the weekly sell_prices table, validate it, and downcast."""
    logger.info("Reading prices from %s", path)
    prices = pd.read_csv(path)
    stats = validate_prices(prices)

    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")
    prices["sell_price"] = prices["sell_price"].astype("float32")

    logger.info("Prices OK — %d rows | %d items", stats.n_rows, stats.n_items)
    return prices, stats
