"""SageMaker Processing entry point for the evaluation panel.

Runs the M4 evaluation — rolling-origin backtest -> WAPE/MASE/RMSSE x hierarchy
x horizon panel + calibration — on the Glue feature output, using the portable
`demand_forecasting` package unchanged. Same code as the local run; only the
orchestration differs.

Processing jobs have NO env-var contract (that is the training toolkit's
SM_CHANNEL_*/SM_MODEL_DIR magic): the launcher mounts inputs and collects
outputs at the /opt/ml/processing/... paths *it* chose and passes them here as
plain CLI args. Defaults below match what run_evaluation.py mounts, and stay
overridable so the same script runs locally for free:

    python aws/sagemaker/evaluate_entry.py --features-dir data/processed/features.parquet \
        --output-dir /tmp/eval --config config/config.yaml --max-series 200

Unlike training jobs, Processing has no metric_definitions scraping — results
leave as FILES in the output dir (panel CSVs), which is exactly what an
evaluation should produce anyway. Headline numbers are logged for the console.
"""

import argparse
import json
import logging
from pathlib import Path

# Same-directory import: the launcher ships train_entry.py alongside this file,
# so the pushdown loader and override logic are shared, not duplicated.
from train_entry import apply_overrides, load_features

from demand_forecasting.config import Config, load_config
from demand_forecasting.evaluation.backtest import backtest_predictions, fold_origins
from demand_forecasting.evaluation.panel import build_panel, save_panel
from demand_forecasting.training.model import lgbm_params

logger = logging.getLogger(__name__)


def evaluate(config: Config, features_dir: Path, output_dir: Path) -> Path:
    features = load_features(features_dir, config)
    logger.info("Loaded features: %d rows, %d series", len(features), features["id"].nunique())

    horizon = config.training.horizon
    folds = fold_origins(features, config.backtest.n_folds, config.backtest.fold_stride, horizon)
    logger.info("Backtesting on fold origins %s", folds)
    preds = backtest_predictions(
        features,
        config.training.quantiles,
        lgbm_params(config),
        horizon,
        folds,
        config.training.n_train_origins,
        config.training.origin_stride,
    )

    panel = build_panel(preds, features)
    panel_dir = save_panel(panel, output_dir)

    by_lvl = panel["by_level"].set_index("level")
    headline = {
        "wape_item_store": float(by_lvl.loc["item_store", "wape"]),
        "wape_total": float(by_lvl.loc["total", "wape"]),
        "wrmsse": float(panel["by_level"]["rmsse"].mean()),
    }
    # A flat JSON the pipeline's ConditionStep reads via PropertyFile + JsonGet:
    # a printed log line can't gate a DAG, only a file a downstream step queries.
    (Path(output_dir) / "evaluation.json").write_text(json.dumps(headline), encoding="utf-8")
    logger.info(
        "Panel headline: wape_item_store=%(wape_item_store).4f "
        "wape_total=%(wape_total).4f wrmsse=%(wrmsse).4f", headline
    )
    logger.info("Panel written to %s", panel_dir)
    return panel_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="/opt/ml/processing/input/features")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output/evaluation")
    # config.yaml is staged next to this script by the launcher; __file__-relative
    # because Processing makes no promise about the working directory.
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--state-filter")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), vars(args))
    evaluate(config, Path(args.features_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
