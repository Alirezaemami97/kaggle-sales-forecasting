"""Config-driven training entry point:

    python -m demand_forecasting.training.train --config config/config.yaml

Runs an honest rolling-origin backtest, trains the final model on the most
recent origins, logs everything to MLflow, and registers the model version.
One command, fixed seed, same result every run.
"""

import argparse
import logging
import random

import mlflow
import numpy as np
import pandas as pd

from demand_forecasting.config import Config, load_config
from demand_forecasting.evaluation.backtest import fold_origins, rolling_origin_backtest
from demand_forecasting.training.dataset import build_direct_table, select_origins, to_model_frame
from demand_forecasting.training.model import QuantileLGBM

logger = logging.getLogger(__name__)


def _cap_series(features: pd.DataFrame, max_series: int, seed: int) -> pd.DataFrame:
    """Deterministically restrict to `max_series` series (0 = keep all)."""
    if max_series <= 0 or features["id"].nunique() <= max_series:
        return features
    ids = pd.Series(features["id"].unique())
    keep = ids.sample(n=max_series, random_state=seed)
    logger.info("Capped to %d series (of %d)", max_series, len(ids))
    return features[features["id"].isin(keep)].reset_index(drop=True)


def _lgbm_params(config: Config) -> dict[str, object]:
    lg = config.training.lgbm
    return {
        "n_estimators": lg.n_estimators,
        "learning_rate": lg.learning_rate,
        "num_leaves": lg.num_leaves,
        "min_child_samples": lg.min_child_samples,
        "seed": config.random_seed,
    }


def train(config: Config) -> None:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    features = pd.read_parquet(config.paths.processed_dir / "features.parquet")
    features = _cap_series(features, config.training.max_series, config.random_seed)
    logger.info("Loaded features: %d rows, %d series", len(features), features["id"].nunique())

    horizon = config.training.horizon
    quantiles = config.training.quantiles
    params = _lgbm_params(config)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "horizon": horizon,
                "quantiles": quantiles,
                "n_train_origins": config.training.n_train_origins,
                "origin_stride": config.training.origin_stride,
                "max_series": config.training.max_series,
                **{f"lgbm_{k}": v for k, v in params.items()},
            }
        )

        # 1) Honest backtest (retrains per fold).
        folds = fold_origins(
            features, config.backtest.n_folds, config.backtest.fold_stride, horizon
        )
        logger.info("Backtesting on fold origins %s", folds)
        results = rolling_origin_backtest(
            features,
            quantiles,
            params,
            horizon,
            folds,
            config.training.n_train_origins,
            config.training.origin_stride,
        )
        for _, row in results.iterrows():
            mlflow.log_metrics(
                {"wape": row["wape"], "mean_pinball": row["mean_pinball"]},
                step=int(row["fold_origin"]),
            )
        mlflow.log_metrics(
            {
                "backtest_wape_mean": float(results["wape"].mean()),
                "backtest_pinball_mean": float(results["mean_pinball"].mean()),
            }
        )
        logger.info(
            "Backtest mean WAPE %.4f | mean pinball %.4f",
            results["wape"].mean(),
            results["mean_pinball"].mean(),
        )

        # 2) Final model on the most recent origins, then register it.
        train_origins = select_origins(
            features, config.training.n_train_origins, config.training.origin_stride, horizon
        )
        table = build_direct_table(features, train_origins, horizon)
        model = QuantileLGBM(quantiles, params).fit(to_model_frame(table), table["sales"])

        model_dir = config.paths.models_dir / config.mlflow.model_name
        model.save(model_dir)
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", config.mlflow.model_name
        )
        logger.info(
            "Registered %s version %s (run %s)",
            config.mlflow.model_name,
            version.version,
            run.info.run_id,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
