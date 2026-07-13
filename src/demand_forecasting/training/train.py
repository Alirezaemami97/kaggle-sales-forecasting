"""Config-driven training entry point:

    python -m demand_forecasting.training.train --config config/config.yaml

Runs an honest rolling-origin backtest, trains the final model on the most
recent origins, logs everything to MLflow, and registers the model version.
One command, fixed seed, same result every run.
"""

import argparse
import logging
import random
import time

import mlflow
import numpy as np
import pandas as pd

from demand_forecasting.config import Config, load_config
from demand_forecasting.evaluation.backtest import backtest_predictions, fold_origins
from demand_forecasting.evaluation.panel import build_panel, save_panel
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


def _filter_state(features: pd.DataFrame, state: str) -> pd.DataFrame:
    """Restrict to one state (e.g. CA); '' keeps all states."""
    if not state:
        return features
    out = features[features["state_id"].astype(str) == state].reset_index(drop=True)
    logger.info("Filtered to state %s: %d series", state, out["id"].nunique())
    return out


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
    features = _filter_state(features, config.data.state_filter)
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

        # 1) Honest backtest (retrains per fold) → full prediction frame → panel.
        folds = fold_origins(
            features, config.backtest.n_folds, config.backtest.fold_stride, horizon
        )
        logger.info("Backtesting on fold origins %s", folds)
        preds = backtest_predictions(
            features,
            quantiles,
            params,
            horizon,
            folds,
            config.training.n_train_origins,
            config.training.origin_stride,
        )

        panel = build_panel(preds, features)
        panel_dir = save_panel(panel, config.paths.models_dir / "evaluation")
        mlflow.log_artifacts(str(panel_dir), artifact_path="evaluation")

        # Headline metrics from the panel: bottom-level WAPE, WRMSSE, coverage.
        by_lvl = panel["by_level"].set_index("level")
        wrmsse = float(panel["by_level"]["rmsse"].mean())
        headline = {
            "wape_item_store": float(by_lvl.loc["item_store", "wape"]),
            "wape_total": float(by_lvl.loc["total", "wape"]),
            "wrmsse": wrmsse,
        }
        for _, cal in panel["calibration"].iterrows():
            headline[f"coverage_{int(cal['nominal_coverage'] * 100)}"] = float(
                cal["empirical_coverage"]
            )
        mlflow.log_metrics(headline)
        logger.info("Panel headline: %s", {k: round(v, 4) for k, v in headline.items()})

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


def _comparison_table(lgbm_level: pd.DataFrame, tft_level: pd.DataFrame) -> pd.DataFrame:
    """Merge the two by_level panels into one side-by-side WAPE/MASE/RMSSE table."""
    merged = lgbm_level.merge(tft_level, on="level", suffixes=("_lgbm", "_tft"))
    cols = ["level"]
    for m in ["wape", "mase", "rmsse"]:
        cols += [f"{m}_lgbm", f"{m}_tft"]
    return merged[cols]


def compare(config: Config) -> None:
    """M5: train LightGBM and TFT on the SAME CA slice and compare them through the
    same evaluation panel, plus wall-clock retraining cost. The honest verdict —
    which may favour the simpler model — is the deliverable, not a TFT win."""
    from demand_forecasting.training.tft import _select_series, backtest_tft_predictions

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    horizon = config.training.horizon
    quantiles = config.training.quantiles
    tft_cfg = config.training.tft
    if tft_cfg is None:
        raise ValueError("training.tft config is required when training.model == 'tft'")

    features = pd.read_parquet(config.paths.processed_dir / "features.parquet")
    features = _filter_state(features, config.data.state_filter or "CA")

    folds = fold_origins(features, config.backtest.n_folds, config.backtest.fold_stride, horizon)
    # Pick the shared series set once, so both models see EXACTLY the same series.
    min_history = tft_cfg.input_chunk_length + horizon
    series = _select_series(
        features, tft_cfg.max_series, min(folds), min_history, config.random_seed
    )
    features = features[features["id"].isin(series)].reset_index(drop=True)
    logger.info("Comparison on %d shared CA series, folds %s", len(series), folds)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="lgbm-vs-tft") as run:
        mlflow.log_params(
            {"comparison": "lgbm_vs_tft", "n_series": len(series), "horizon": horizon,
             "tft_epochs": tft_cfg.n_epochs, "tft_hidden": tft_cfg.hidden_size}
        )

        # LightGBM on the shared slice.
        t0 = time.perf_counter()
        lgbm_preds = backtest_predictions(
            features, quantiles, _lgbm_params(config), horizon, folds,
            config.training.n_train_origins, config.training.origin_stride,
        )
        lgbm_secs = time.perf_counter() - t0
        lgbm_panel = build_panel(lgbm_preds, features)

        # TFT on the same shared slice.
        t0 = time.perf_counter()
        tft_preds = backtest_tft_predictions(
            features, quantiles, tft_cfg, horizon, folds, config.random_seed
        )
        tft_secs = time.perf_counter() - t0
        tft_panel = build_panel(tft_preds, features)

        table = _comparison_table(lgbm_panel["by_level"], tft_panel["by_level"])
        out_dir = config.paths.models_dir / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "by_level.csv", index=False)
        timing = pd.DataFrame(
            [{"model": "lgbm", "backtest_seconds": round(lgbm_secs, 1),
              "serving": "5 boosters; CPU; no batch inference framework"},
             {"model": "tft", "backtest_seconds": round(tft_secs, 1),
              "serving": "1 network; torch runtime; heavier deploy"}]
        )
        timing.to_csv(out_dir / "cost.csv", index=False)
        (out_dir / "comparison.md").write_text(
            "# LightGBM vs TFT (CA slice)\n\n## Accuracy by level\n\n"
            + table.to_markdown(index=False)
            + "\n\n## Retraining cost & serving\n\n"
            + timing.to_markdown(index=False)
            + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifacts(str(out_dir), artifact_path="comparison")

        bottom = table[table["level"] == "item_store"].iloc[0]
        mlflow.log_metrics(
            {"lgbm_wape_item_store": float(bottom["wape_lgbm"]),
             "tft_wape_item_store": float(bottom["wape_tft"]),
             "lgbm_wrmsse": float(lgbm_panel["by_level"]["rmsse"].mean()),
             "tft_wrmsse": float(tft_panel["by_level"]["rmsse"].mean()),
             "lgbm_backtest_seconds": lgbm_secs, "tft_backtest_seconds": tft_secs}
        )
        logger.info("Comparison written to %s (run %s)", out_dir, run.info.run_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.training.model == "tft":
        compare(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
