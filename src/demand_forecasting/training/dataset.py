"""Build direct multi-horizon training tables from the M2 feature table.

Direct multi-horizon: a training row is an (origin O, horizon h) pair whose
target is sales at day O+h. Features split into three groups, all point-in-time
correct:
  - static     — series identity (item/dept/cat/store/state);
  - dynamic    — lag/rolling values known AS OF the origin. We reuse the M2
                 feature row at day O+1, whose lags/rolling already use only
                 sales <= O — no new lag math, just a relabel (d -> origin);
  - target-day — calendar and price at day O+h, legitimately known in advance.

The model also sees `horizon` itself, so one model covers all 28 days.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Static series identity (does not depend on origin or horizon).
STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
# Origin-anchored dynamic features (taken from the M2 row at day O+1).
DYNAMIC_COLS = [
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
]
# Target-day features known in advance at the origin (calendar + weekly price).
TARGET_KNOWN_COLS = [
    "wday",
    "month",
    "event_name_1",
    "event_type_1",
    "snap",
    "sell_price",
    "price_rel_dept",
]

# What the model actually consumes, and which of those are categorical.
CATEGORICAL_COLS = STATIC_COLS + ["wday", "month", "event_name_1", "event_type_1", "snap"]
NUMERIC_COLS = DYNAMIC_COLS + ["horizon", "sell_price", "price_rel_dept"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def cap_series(features: pd.DataFrame, max_series: int, seed: int) -> pd.DataFrame:
    """Deterministically restrict to `max_series` series (0 = keep all)."""
    if max_series <= 0 or features["id"].nunique() <= max_series:
        return features
    ids = pd.Series(features["id"].unique())
    keep = ids.sample(n=max_series, random_state=seed)
    logger.info("Capped to %d series (of %d)", max_series, len(ids))
    return features[features["id"].isin(keep)].reset_index(drop=True)


def filter_state(features: pd.DataFrame, state: str) -> pd.DataFrame:
    """Restrict to one state (e.g. CA); '' keeps all states."""
    if not state:
        return features
    out = features[features["state_id"].astype(str) == state].reset_index(drop=True)
    logger.info("Filtered to state %s: %d series", state, out["id"].nunique())
    return out


def select_origins(features: pd.DataFrame, n_origins: int, stride: int, horizon: int) -> list[int]:
    """Pick the most recent `n_origins` origin days on a `stride`, newest first.

    An origin O is usable only if its whole horizon O+1..O+horizon exists in the
    data (so targets are observed) — hence we start `horizon` days before the end.
    """
    last_day = int(features["d"].max())
    latest_origin = last_day - horizon
    origins = [latest_origin - i * stride for i in range(n_origins)]
    return sorted(o for o in origins if o > 0)


def build_direct_table(
    features: pd.DataFrame, origins: list[int], horizon: int
) -> pd.DataFrame:
    """Assemble (origin, horizon) examples via two joins onto the M2 table.

    Returns one row per (series, origin, horizon) carrying FEATURE_COLS plus
    `sales` (the target), `origin`, and `target_day`.
    """
    # Dynamic block: the row at day O+1 holds features known as of O.
    dyn = features[["id", "d", *DYNAMIC_COLS, *STATIC_COLS]].copy()
    dyn["origin"] = dyn["d"].astype("int32") - 1
    dyn = dyn.drop(columns="d")
    dyn = dyn[dyn["origin"].isin(origins)]

    # Cross the chosen origins with horizons 1..H, then locate each target day.
    horizons = pd.DataFrame({"horizon": range(1, horizon + 1)}, dtype="int16")
    base = dyn.merge(horizons, how="cross")
    base["target_day"] = base["origin"] + base["horizon"]

    # Target-day block: sales (the label) + calendar/price known in advance.
    tgt = features[["id", "d", "sales", *TARGET_KNOWN_COLS]].rename(columns={"d": "target_day"})

    out = base.merge(tgt, on=["id", "target_day"], how="inner")
    return out.reset_index(drop=True)


def to_model_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Return just FEATURE_COLS with categoricals typed for LightGBM auto-detection."""
    X = table[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    return X
