"""Synthetic M5-shaped fixtures — tiny, so CI never needs the real dataset."""

import numpy as np
import pandas as pd
import pytest

N_DAYS = 10


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """Three series (2 stores, 2 items) over 10 days, M5 wide format."""
    rows = [
        ("FOODS_1_001_CA_1_evaluation", "FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002_CA_1_evaluation", "FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_001_TX_1_evaluation", "FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX"),
    ]
    df = pd.DataFrame(rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    for day in range(1, N_DAYS + 1):
        # Deterministic small integers with plenty of zeros, like real M5 demand.
        df[f"d_{day}"] = [(day + i) % 4 for i in range(len(df))]
    return df


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """Ten contiguous days starting 2011-01-29 (the real M5 start date)."""
    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11101] * 7 + [11102] * 3,
            "weekday": dates.day_name(),
            "wday": ((dates.dayofweek + 2) % 7 + 1),
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, N_DAYS + 1)],
            "event_name_1": [None] * N_DAYS,
            "event_type_1": [None] * N_DAYS,
            "event_name_2": [None] * N_DAYS,
            "event_type_2": [None] * N_DAYS,
            "snap_CA": [1, 0] * (N_DAYS // 2),
            "snap_TX": [0, 1] * (N_DAYS // 2),
            "snap_WI": [0] * N_DAYS,
        }
    )


@pytest.fixture
def prices_df() -> pd.DataFrame:
    """Weekly prices for the item-store pairs in sales_df."""
    return pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "CA_1", "TX_1", "TX_1"],
            "item_id": ["FOODS_1_001", "FOODS_1_001", "FOODS_1_002", "FOODS_1_001", "FOODS_1_001"],
            "wm_yr_wk": [11101, 11102, 11101, 11101, 11102],
            "sell_price": [2.48, 2.48, 1.98, 2.58, 2.68],
        }
    )


@pytest.fixture
def feature_table() -> pd.DataFrame:
    """A synthetic M2-style feature table: 3 series x 80 days, with all columns
    the training/backtest code consumes. Dynamic `lag_7` is set to the row's own
    day and `sales` to the row's day, so tests can pin down the origin-relabel
    join (dynamic block must come from day O+1; target sales from day O+h)."""
    rng = np.random.default_rng(0)
    series = ["S1", "S2", "S3"]
    days = range(1, 81)
    rows = []
    for s in series:
        for d in days:
            rows.append(
                {
                    "id": s,
                    "d": d,
                    "item_id": s,
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "lag_7": float(d),
                    "lag_28": float(d),
                    "rolling_mean_7": float(rng.integers(0, 5)),
                    "rolling_std_7": float(rng.random()),
                    "rolling_mean_28": float(rng.integers(0, 5)),
                    "rolling_std_28": float(rng.random()),
                    "wday": (d % 7) + 1,
                    "month": ((d // 30) % 12) + 1,
                    "event_name_1": "None",
                    "event_type_1": "None",
                    "snap": d % 2,
                    "sell_price": 2.0 + (d % 5) * 0.1,
                    "price_rel_dept": 1.0,
                    "sales": float(d),
                }
            )
    return pd.DataFrame(rows)
