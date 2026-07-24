"""SageMaker serving handler for the quantile LightGBM forecaster (Batch Transform).

Self-contained on purpose. This runs inside AWS's prebuilt SKLearn *inference*
container, whose Python is 3.9 and so cannot import the 3.12-syntax
`demand_forecasting` package. It therefore reloads the LightGBM boosters straight
from `model.tar.gz` and reproduces the two guarantees `QuantileLGBM.predict`
makes — non-negative demand and non-crossing quantiles.

The feature pipeline is NOT duplicated here: the model-input rows are built
upstream by the shared `build_direct_table`/`to_model_frame` in the prep step, so
there is no training–serving skew in the features. Only the trivial model-load +
clip/sort lives in this file. The column lists below mirror
`demand_forecasting.training.dataset`; `test_inference_columns_match_dataset`
pins them so they cannot silently drift.

Why CSV is safe for the categoricals: a saved booster stores its training
`pandas_categorical` mapping, and `Booster.predict` realigns any incoming
`category` column to that mapping by its string *value*, not by its transient
code — so the headerless-CSV round-trip cannot corrupt the encoding.

Contract — the SageMaker inference toolkit calls these four callbacks:
  model_fn   -> load the model once, at container startup
  input_fn   -> deserialize one CSV mini-batch (headerless, FEATURE_COLS order)
  predict_fn -> quantile predictions, guaranteed non-negative and monotone
  output_fn  -> serialize the quantile columns back to CSV (headerless)
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# Mirror of demand_forecasting.training.dataset (pinned by the CI test). The
# prep step writes feature rows in exactly this column order, headerless.
STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
DYNAMIC_COLS = [
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
]
CATEGORICAL_COLS = STATIC_COLS + ["wday", "month", "event_name_1", "event_type_1", "snap"]
NUMERIC_COLS = DYNAMIC_COLS + ["horizon", "sell_price", "price_rel_dept"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def _quantile_column(q: float) -> str:
    return f"q{q}"


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load the saved QuantileLGBM artifact: one booster per quantile + meta.json.

    Reproduces QuantileLGBM.load without importing it, since the 3.12 package is
    absent in the 3.9 serving container.
    """
    with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    quantiles = sorted(meta["quantiles"])
    boosters = {
        q: lgb.Booster(model_file=os.path.join(model_dir, f"booster_{q}.txt")) for q in quantiles
    }
    return {"quantiles": quantiles, "boosters": boosters}


def input_fn(request_body: str | bytes, content_type: str = "text/csv") -> pd.DataFrame:
    """Parse one headerless CSV mini-batch into a FEATURE_COLS DataFrame.

    Batch Transform splits the input file by line, so a mini-batch carries no
    header; columns are assigned by position. Categoricals are cast to `category`
    so the booster's stored pandas_categorical realigns them to training.
    """
    if "csv" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type}")
    text = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    # pandas defaults are the exact inverse of the prep step's to_csv: a missing
    # value written as "" is read back as NaN. M5 event names are never NA tokens,
    # so no real category value collides with that mapping.
    frame = pd.read_csv(io.StringIO(text), header=None, names=FEATURE_COLS)
    for col in CATEGORICAL_COLS:
        frame[col] = frame[col].astype("category")
    return frame


def predict_fn(data: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """One column per quantile, clipped non-negative and sorted so intervals never
    invert — the same two guarantees QuantileLGBM.predict enforces."""
    quantiles = model["quantiles"]
    boosters = model["boosters"]
    raw = np.column_stack([boosters[q].predict(data) for q in quantiles])
    raw = np.clip(raw, a_min=0.0, a_max=None)
    raw = np.sort(raw, axis=1)
    return pd.DataFrame(raw, columns=[_quantile_column(q) for q in quantiles])


def output_fn(prediction: pd.DataFrame, accept: str = "text/csv") -> str:
    """Serialize the quantile columns back to headerless CSV, one row per input
    row, so Batch Transform can reassemble output in input order."""
    if "csv" not in accept:
        raise ValueError(f"Unsupported accept type: {accept}")
    return prediction.to_csv(index=False, header=False)
