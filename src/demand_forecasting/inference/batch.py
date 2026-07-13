"""Batch inference: load the production model + the SAME feature pipeline used in
training, produce 28-day quantile forecasts for the catalogue, and write them to
the prediction archive.

Replay semantics: forecasts are anchored at the most recent fully-observed origin
(last_day - horizon), so the target days fall inside the data and every archived
forecast can later be scored against actuals — which is how the monitoring loop
(forecast error) is demonstrated. True open-ended future inference (extending the
calendar +28 days and rebuilding as-of-origin features from the last day) is a
documented extension for the AWS track.

No API, no latency budget: correctness and throughput are what matter, and every
forecast is archived (Parquet, keyed by run date + model version) for later scoring.
"""

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecasting.config import Config, load_config
from demand_forecasting.training.dataset import build_direct_table, to_model_frame
from demand_forecasting.training.model import QuantileLGBM, quantile_column

logger = logging.getLogger(__name__)

# Identity columns carried into the archive so forecasts can be rolled up / joined.
_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def latest_full_origin(features: pd.DataFrame, horizon: int) -> int:
    """The most recent origin whose whole horizon is observed (so features exist
    for every target day and the forecast can be scored against actuals)."""
    return int(features["d"].max()) - horizon


def load_production_model(config: Config) -> QuantileLGBM:
    """Load the model artifact registered as production. For now this is the saved
    booster set (path from config); resolving the registry *stage* is wired in M8."""
    model_dir = config.paths.models_dir / config.mlflow.model_name
    return QuantileLGBM.load(model_dir)


def generate_forecasts(
    features: pd.DataFrame,
    model: QuantileLGBM,
    quantiles: list[float],
    horizon: int,
    origin: int,
) -> pd.DataFrame:
    """Produce one 28-day quantile forecast per series from `origin`, reusing the
    exact training feature construction (no training–serving skew)."""
    table = build_direct_table(features, [origin], horizon)
    preds = model.predict(to_model_frame(table))
    meta = table[[*_ID_COLS, "origin", "horizon", "target_day"]].reset_index(drop=True)
    return pd.concat([meta, preds.reset_index(drop=True)], axis=1)


def identify_cold_start(features: pd.DataFrame, origin: int, min_history: int) -> set[str]:
    """Series with fewer than `min_history` observed days at the origin. Their lag
    and rolling features are undefined/unreliable, so the model can't be trusted."""
    hist = features[features["d"] <= origin]
    counts = hist.groupby("id", observed=True)["d"].count()
    return set(counts[counts < min_history].index)


def hierarchy_priors(
    features: pd.DataFrame, origin: int, quantiles: list[float], window: int
) -> pd.DataFrame:
    """Store+department demand quantiles over the trailing `window` days — the
    prior a new item inherits until it accumulates its own history."""
    recent = features[(features["d"] <= origin) & (features["d"] > origin - window)]
    qcols = [quantile_column(q) for q in sorted(quantiles)]
    grouped = recent.groupby(["store_id", "dept_id"], observed=True)["sales"]
    priors = grouped.quantile(sorted(quantiles)).unstack()
    priors.columns = qcols
    return priors.reset_index()


def cold_start_forecasts(
    features: pd.DataFrame,
    cold_ids: set[str],
    quantiles: list[float],
    horizon: int,
    origin: int,
    priors: pd.DataFrame,
) -> pd.DataFrame:
    """Build flat, hierarchy-prior forecasts for the cold-start series: the same
    store+dept quantile vector broadcast across all horizons."""
    qcols = [quantile_column(q) for q in sorted(quantiles)]
    ident = features[features["id"].isin(cold_ids)][_ID_COLS].drop_duplicates()
    if ident.empty:
        return pd.DataFrame(columns=[*_ID_COLS, "origin", "horizon", "target_day", *qcols])
    ident = ident.merge(priors, on=["store_id", "dept_id"], how="left")
    ident[qcols] = ident[qcols].fillna(0.0)  # unseen group → zero-demand prior
    horizons = pd.DataFrame({"horizon": range(1, horizon + 1)}, dtype="int64")
    out = ident.merge(horizons, how="cross")
    out["origin"] = origin
    out["target_day"] = origin + out["horizon"]
    return out[[*_ID_COLS, "origin", "horizon", "target_day", *qcols]]


def forecast_catalogue(
    features: pd.DataFrame,
    model: QuantileLGBM,
    config: Config,
    origin: int,
) -> pd.DataFrame:
    """Forecast every series: the model for warm series, hierarchy priors for
    cold-start ones, tagged with `is_cold_start`."""
    quantiles = config.training.quantiles
    horizon = config.training.horizon
    cold = identify_cold_start(features, origin, config.inference.cold_start_min_history_days)

    warm_features = features[~features["id"].isin(cold)]
    warm = generate_forecasts(warm_features, model, quantiles, horizon, origin)
    warm = warm.assign(is_cold_start=False)

    logger.info("Forecast %d warm + %d cold-start series", warm["id"].nunique(), len(cold))
    if not cold:
        return warm

    priors = hierarchy_priors(features, origin, quantiles, config.inference.prior_window_days)
    cold_df = cold_start_forecasts(features, cold, quantiles, horizon, origin, priors)
    cold_df = cold_df.assign(is_cold_start=True)
    return pd.concat([warm, cold_df], ignore_index=True)


def check_forecasts(preds: pd.DataFrame, quantiles: list[float]) -> None:
    """Write-time behavioural gate: forecasts non-negative and quantiles ordered."""
    qcols = [quantile_column(q) for q in sorted(quantiles)]
    q = preds[qcols].to_numpy()
    if (q < 0).any():
        raise ValueError("Forecast archive rejected: negative quantile forecast")
    if (np.diff(q, axis=1) < -1e-6).any():
        raise ValueError("Forecast archive rejected: crossing quantiles")


def write_archive(
    preds: pd.DataFrame, config: Config, run_date: str, model_version: str
) -> Path:
    """Append run metadata and write the archive, partitioned by run date."""
    out_dir = config.paths.forecasts_dir / f"run_date={run_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    archived = preds.assign(
        run_date=run_date, model_name=config.mlflow.model_name, model_version=model_version
    )
    path = out_dir / "forecast.parquet"
    archived.to_parquet(path, index=False)
    logger.info("Wrote %d forecasts to %s", len(archived), path)
    return path


def run_batch(config: Config, model_version: str = "local") -> Path:
    """Full batch job: load model + features, forecast from the replay origin,
    check, and archive. Returns the archive path."""
    features = pd.read_parquet(config.paths.processed_dir / "features.parquet")
    horizon = config.training.horizon
    quantiles = config.training.quantiles
    origin = latest_full_origin(features, horizon)
    logger.info("Batch inference: origin day %d, %d series", origin, features["id"].nunique())

    model = load_production_model(config)
    preds = forecast_catalogue(features, model, config, origin)
    check_forecasts(preds, quantiles)

    run_date = datetime.date.today().isoformat()
    return write_archive(preds, config, run_date, model_version)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--model-version", default="local", help="Model version tag for the archive"
    )
    args = parser.parse_args()
    run_batch(load_config(args.config), args.model_version)


if __name__ == "__main__":
    main()
