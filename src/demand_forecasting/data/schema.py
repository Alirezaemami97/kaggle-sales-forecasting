"""Schema validation for the three M5 tables.

Bad data should fail loudly here, before it reaches the feature pipeline.
Checks are vectorised (never per-row pydantic) because the sales table has
30,490 rows x ~1,941 day-columns; pydantic models carry the validated summary.
"""

import pandas as pd
from pydantic import BaseModel, Field

# Identity columns the rest of the system depends on.
SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

CALENDAR_COLUMNS = [
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "d",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

PRICES_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]


class SalesStats(BaseModel):
    """Summary produced after successful validation of the sales table."""

    n_series: int
    n_days: int
    zero_rate: float = Field(ge=0.0, le=1.0)


class CalendarStats(BaseModel):
    n_days: int
    first_date: str
    last_date: str


class PricesStats(BaseModel):
    n_rows: int
    n_items: int


def _require_columns(df: pd.DataFrame, required: list[str], table: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{table}: missing required columns {sorted(missing)}")


def day_columns(sales: pd.DataFrame) -> list[str]:
    """Return the d_1..d_N day columns in day order."""
    cols = [c for c in sales.columns if c.startswith("d_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def validate_sales(sales: pd.DataFrame) -> SalesStats:
    """Validate the wide sales table. Raises ValueError on any violation."""
    _require_columns(sales, SALES_ID_COLUMNS, "sales")

    d_cols = day_columns(sales)
    if not d_cols:
        raise ValueError("sales: no d_* day columns found")

    # Day columns must be contiguous d_1..d_N — a gap means a silently dropped day.
    day_numbers = [int(c.split("_")[1]) for c in d_cols]
    if day_numbers != list(range(1, len(d_cols) + 1)):
        raise ValueError("sales: d_* columns are not contiguous from d_1")

    if sales[SALES_ID_COLUMNS].isnull().any().any():
        raise ValueError("sales: null values in identity columns")

    if sales["id"].duplicated().any():
        raise ValueError("sales: duplicated series ids")

    values = sales[d_cols]
    if values.isnull().any().any():
        raise ValueError("sales: null values in day columns")
    if (values < 0).any().any():
        raise ValueError("sales: negative unit sales")

    return SalesStats(
        n_series=len(sales),
        n_days=len(d_cols),
        zero_rate=float((values == 0).to_numpy().mean()),
    )


def validate_calendar(calendar: pd.DataFrame) -> CalendarStats:
    """Validate the calendar table. Raises ValueError on any violation."""
    _require_columns(calendar, CALENDAR_COLUMNS, "calendar")

    dates = pd.to_datetime(calendar["date"], errors="raise")
    if not dates.is_monotonic_increasing:
        raise ValueError("calendar: dates are not sorted ascending")
    gaps = dates.diff().dropna().dt.days
    if not (gaps == 1).all():
        raise ValueError("calendar: dates are not contiguous daily")

    if calendar["d"].duplicated().any():
        raise ValueError("calendar: duplicated d keys")

    for col in ("snap_CA", "snap_TX", "snap_WI"):
        if not calendar[col].isin([0, 1]).all():
            raise ValueError(f"calendar: {col} contains values outside {{0, 1}}")

    return CalendarStats(
        n_days=len(calendar),
        first_date=str(dates.iloc[0].date()),
        last_date=str(dates.iloc[-1].date()),
    )


def validate_prices(prices: pd.DataFrame) -> PricesStats:
    """Validate the sell_prices table. Raises ValueError on any violation."""
    _require_columns(prices, PRICES_COLUMNS, "sell_prices")

    if prices[PRICES_COLUMNS].isnull().any().any():
        raise ValueError("sell_prices: null values")
    if (prices["sell_price"] <= 0).any():
        raise ValueError("sell_prices: non-positive prices")
    if prices.duplicated(subset=["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("sell_prices: duplicated (store, item, week) keys")

    return PricesStats(
        n_rows=len(prices),
        n_items=int(prices["item_id"].nunique()),
    )
