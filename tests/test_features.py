"""Tests for the point-in-time feature pipeline — above all, the no-leakage test."""

import pandas as pd
import pytest

from demand_forecasting.features.pipeline import build_features, join_calendar, melt_sales

# Small lags/windows sized for the 10-day synthetic fixtures.
LAGS = [2, 3]
WINDOWS = [3]


@pytest.fixture
def features_df(
    sales_df: pd.DataFrame, calendar_df: pd.DataFrame, prices_df: pd.DataFrame
) -> pd.DataFrame:
    return build_features(
        sales_df, calendar_df, prices_df, lags=LAGS, rolling_windows=WINDOWS,
        drop_pre_release=False,
    )


def test_melt_shape_and_values(sales_df: pd.DataFrame) -> None:
    long = melt_sales(sales_df)
    assert len(long) == 3 * 10  # 3 series x 10 days
    assert str(long["d"].dtype) == "int16"
    # Spot-check one cell against the wide table.
    row = long[(long["id"] == sales_df.loc[0, "id"]) & (long["d"] == 4)]
    assert row["sales"].iloc[0] == sales_df.loc[0, "d_4"]


def test_snap_matches_own_state(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> None:
    long = join_calendar(melt_sales(sales_df), calendar_df)
    ca = long[long["state_id"] == "CA"].set_index("d")["snap"]
    tx = long[long["state_id"] == "TX"].set_index("d")["snap"]
    cal = calendar_df.assign(d=calendar_df["d"].str.split("_").str[1].astype(int))
    for day in range(1, 11):
        assert (ca.loc[day] == cal.loc[cal["d"] == day, "snap_CA"].iloc[0]).all()
        assert (tx.loc[day] == cal.loc[cal["d"] == day, "snap_TX"].iloc[0]).all()


def test_price_features(features_df: pd.DataFrame) -> None:
    # FOODS_1_001 in TX_1: 2.58 (wk 11101) → 2.68 (wk 11102).
    tx = features_df[(features_df["store_id"] == "TX_1") & (features_df["wm_yr_wk"] == 11102)]
    assert tx["price_pct_change"].iloc[0] == pytest.approx((2.68 - 2.58) / 2.58)
    # Dept FOODS_1 in CA_1, week 11101: prices 2.48 and 1.98 → mean 2.23.
    ca = features_df[
        (features_df["store_id"] == "CA_1")
        & (features_df["item_id"] == "FOODS_1_001")
        & (features_df["wm_yr_wk"] == 11101)
    ]
    assert ca["price_rel_dept"].iloc[0] == pytest.approx(2.48 / 2.23)


def test_lag_is_sales_k_days_earlier(features_df: pd.DataFrame) -> None:
    one = features_df[features_df["id"] == features_df["id"].iloc[0]].set_index("d")
    for lag in LAGS:
        for day in range(lag + 1, 11):
            assert one.loc[day, f"lag_{lag}"] == one.loc[day - lag, "sales"]
        # Days without enough history are NaN, never fabricated zeros.
        assert one.loc[range(1, lag + 1), f"lag_{lag}"].isna().all()


def test_rolling_window_ends_at_t_minus_1(features_df: pd.DataFrame) -> None:
    one = features_df[features_df["id"] == features_df["id"].iloc[0]].set_index("d")
    w = WINDOWS[0]
    for day in range(w + 1, 11):
        window = [one.loc[day - k, "sales"] for k in range(1, w + 1)]  # [t-w, t-1]
        assert one.loc[day, f"rolling_mean_{w}"] == pytest.approx(
            sum(window) / w, abs=1e-6
        )
    assert one.loc[range(1, w + 1), f"rolling_mean_{w}"].isna().all()


def test_drop_pre_release_removes_unpriced_rows(
    sales_df: pd.DataFrame, calendar_df: pd.DataFrame, prices_df: pd.DataFrame
) -> None:
    kept = build_features(
        sales_df, calendar_df, prices_df, lags=LAGS, rolling_windows=WINDOWS,
        drop_pre_release=True,
    )
    # FOODS_1_002 in CA_1 has no price for week 11102 (days 8-10) → rows dropped.
    assert kept["sell_price"].notna().all()
    f2 = kept[(kept["item_id"] == "FOODS_1_002") & (kept["store_id"] == "CA_1")]
    assert set(f2["d"]) == set(range(1, 8))


def test_no_future_leakage(
    sales_df: pd.DataFrame, calendar_df: pd.DataFrame, prices_df: pd.DataFrame
) -> None:
    """THE contract of this project: perturbing sales on day t must not change
    any feature at day <= t (same series) nor anything in other series."""
    perturb_day, perturb_series = 5, sales_df.loc[0, "id"]

    baseline = build_features(
        sales_df, calendar_df, prices_df, lags=LAGS, rolling_windows=WINDOWS,
        drop_pre_release=False,
    )
    corrupted_sales = sales_df.copy()
    corrupted_sales.loc[0, f"d_{perturb_day}"] = 999
    corrupted = build_features(
        corrupted_sales, calendar_df, prices_df, lags=LAGS, rolling_windows=WINDOWS,
        drop_pre_release=False,
    )

    feature_cols = [c for c in baseline.columns if c != "sales"]

    # 1) For every series, features at day <= t are untouched.
    past = baseline["d"] <= perturb_day
    pd.testing.assert_frame_equal(
        baseline.loc[past, feature_cols], corrupted.loc[past, feature_cols]
    )
    # 2) Other series are untouched everywhere — no cross-series leakage.
    others = baseline["id"] != perturb_series
    pd.testing.assert_frame_equal(baseline.loc[others], corrupted.loc[others])
    # 3) Sanity: the perturbation DID flow into the future of its own series
    #    (otherwise this test would pass vacuously on a broken pipeline).
    future_own = (baseline["id"] == perturb_series) & (baseline["d"] > perturb_day)
    assert not baseline.loc[future_own, feature_cols].equals(
        corrupted.loc[future_own, feature_cols]
    )
