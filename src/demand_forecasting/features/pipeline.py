"""Point-in-time feature pipeline, shared by training and batch inference.

The contract that makes backtests honest: a feature row for day *t* may use
  - sales strictly BEFORE t (lags, rolling stats on a shift(1) of the series),
  - calendar and price information AT t (known in advance in retail).
`tests/test_features.py::test_no_future_leakage` fails if this contract breaks.

All functions are pure (DataFrame in, DataFrame out) so the exact same code
runs in training and in the batch inference job — no training–serving skew.
"""

import numpy as np
import pandas as pd

from demand_forecasting.data.schema import SALES_ID_COLUMNS, day_columns

# Calendar columns carried into the feature table.
_CALENDAR_KEEP = ["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1", "event_type_1"]


def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Wide (one row per series) → long (one row per series-day).

    Day columns are renamed to integers *before* melting so the resulting `d`
    column is numeric from the start — melting 59M rows into an object-dtype
    column of "d_123" strings would cost gigabytes.
    """
    d_cols = day_columns(sales)
    renamed = sales.rename(columns={c: int(c.split("_")[1]) for c in d_cols})
    long = renamed.melt(id_vars=SALES_ID_COLUMNS, var_name="d", value_name="sales")
    long["d"] = long["d"].astype("int16")
    long["sales"] = long["sales"].astype("int16")
    return long


def join_calendar(long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar attributes for day t (legitimately known in advance).

    The three state-level SNAP columns collapse into one `snap` flag matching
    each row's own state — the model sees "is SNAP active for me today".
    """
    cal = calendar.copy()
    cal["d"] = cal["d"].str.split("_").str[1].astype("int16")

    snap_cols = ["snap_CA", "snap_TX", "snap_WI"]
    out = long.merge(cal[_CALENDAR_KEEP + snap_cols], on="d", how="left")

    state = out["state_id"].astype(str)
    out["snap"] = np.select(
        [state == "CA", state == "TX", state == "WI"],
        [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
        default=0,
    ).astype("int8")
    return out.drop(columns=snap_cols)


def add_price_features(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach weekly price features: level, week-over-week change, dept-relative.

    Computed on the compact weekly prices table (6.8M rows) *before* joining
    to the 59M-row daily frame — same result, ~9x less work.
    Rows before an item's first listed week get NaN prices (see
    `drop_pre_release` in `build_features`).
    """
    pr = prices.copy()
    pr = pr.sort_values(["store_id", "item_id", "wm_yr_wk"], ignore_index=True)

    grp = pr.groupby(["store_id", "item_id"], observed=True, sort=False)["sell_price"]
    pr["price_pct_change"] = grp.pct_change(fill_method=None).astype("float32")

    # Price relative to the item's department in the same store and week —
    # "is this item cheap or expensive for its shelf right now".
    dept = pr["item_id"].astype(str).str.rsplit("_", n=1).str[0]
    dept_mean = pr.groupby([pr["store_id"], dept, pr["wm_yr_wk"]], observed=True)[
        "sell_price"
    ].transform("mean")
    pr["price_rel_dept"] = (pr["sell_price"] / dept_mean).astype("float32")

    return long.merge(pr, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def add_lag_features(long: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Sales lag features: lag_k at day t is the sales at day t-k.

    Requires `long` sorted by (id, d). shift() never crosses series thanks to
    the groupby, so the first k days of each series are NaN — LightGBM handles
    NaN natively, and pre-history rows are honest unknowns, not zeros.
    """
    grp = long.groupby("id", observed=True, sort=False)["sales"]
    for lag in lags:
        long[f"lag_{lag}"] = grp.shift(lag).astype("float32")
    return long


def add_rolling_features(long: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Trailing rolling mean/std of sales over the last w days ENDING at t-1.

    The shift(1) before rolling is the point-in-time guarantee: the window for
    day t covers [t-w, t-1] and never includes t itself (the target).
    """
    shifted = long.groupby("id", observed=True, sort=False)["sales"].shift(1)
    sgrp = shifted.groupby(long["id"], observed=True, sort=False)
    for w in windows:
        roll = sgrp.rolling(window=w, min_periods=w)
        long[f"rolling_mean_{w}"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
        long[f"rolling_std_{w}"] = roll.std().reset_index(level=0, drop=True).astype("float32")
    return long


def build_features(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    drop_pre_release: bool = True,
) -> pd.DataFrame:
    """Full pipeline: melt → calendar → prices → lags → rolling → filter.

    Lags and rolling stats are computed on the complete day grid BEFORE any
    row filtering, so a shift of k rows is always a shift of k calendar days.
    """
    long = melt_sales(sales)
    long = join_calendar(long, calendar)
    long = add_price_features(long, prices)

    long = long.sort_values(["id", "d"], ignore_index=True)
    long = add_lag_features(long, lags)
    long = add_rolling_features(long, rolling_windows)

    if drop_pre_release:
        long = long[long["sell_price"].notna()].reset_index(drop=True)

    return long
