"""SageMaker entry point for the TFT-vs-LightGBM comparison on GPU.

The deferred "full-scale-ish" TFT run: the same src/training/tft.py code that
trained on 120 CA series locally (CPU, small net) now trains on a larger CA
slice on a GPU instance. Both models are backtested on the SAME series and the
SAME fold through the SAME panel, and the comparison tables are written to
SM_MODEL_DIR — the honest-comparison artifact is produced by one job.

GPU use is automatic: QuantileTFT passes accelerator="auto" to Lightning, so
this identical code uses CUDA here and CPU on the laptop. The entry logs
torch.cuda availability early — if the log says no GPU on a g4dn instance,
something is wrong with the image, and the job should be stopped rather than
billed as a slow CPU run.

Spot checkpointing: Darts persists Lightning checkpoints under --checkpoint-dir
(/opt/ml/checkpoints), which SageMaker syncs to checkpoint_s3_uri. Honest
scope note: automatic RESUME from those checkpoints is not wired — with runs
bounded to well under an hour, restart-from-scratch after a Spot interruption
is the cheaper engineering trade. The checkpoint sync demonstrates the
mechanism; epoch-level resume is what a longer job would add.
"""

import argparse
import logging
import os
from pathlib import Path

from train_entry import load_features

from demand_forecasting.config import Config, TFTConfig, load_config
from demand_forecasting.evaluation.backtest import backtest_predictions, fold_origins
from demand_forecasting.evaluation.panel import build_panel
from demand_forecasting.training.model import lgbm_params
from demand_forecasting.training.tft import backtest_tft_predictions

logger = logging.getLogger(__name__)


def tft_config_with_overrides(
    config: Config, max_series: int | None, n_epochs: int | None
) -> TFTConfig:
    """Rebuild (not model_copy) so overrides are validated — same lesson as
    apply_overrides: an invalid draw must die before compute is spent."""
    if config.training.tft is None:
        raise ValueError("training.tft must be configured for the TFT comparison")
    raw = config.training.tft.model_dump()
    if max_series is not None:
        raw["max_series"] = max_series
    if n_epochs is not None:
        raw["n_epochs"] = n_epochs
    return TFTConfig(**raw)


def log_device() -> None:
    import torch

    available = torch.cuda.is_available()
    logger.info("METRIC gpu_available=%d", int(available))
    if available:
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))


def compare(config: Config, tft_cfg: TFTConfig, features_dir: Path, out_dir: Path,
            checkpoint_dir: str | None, max_samples_per_ts: int | None,
            num_loader_workers: int) -> None:
    features = load_features(features_dir, config)
    logger.info("Loaded features: %d rows, %d series", len(features), features["id"].nunique())

    horizon = config.training.horizon
    quantiles = config.training.quantiles
    # One fold (the most recent) keeps GPU minutes bounded; both models see it.
    folds = fold_origins(features, 1, config.backtest.fold_stride, horizon)

    tft_preds = backtest_tft_predictions(
        features, quantiles, tft_cfg, horizon, folds, config.random_seed,
        work_dir=checkpoint_dir, max_samples_per_ts=max_samples_per_ts,
        num_loader_workers=num_loader_workers,
    )
    # The TFT selected the series it had enough history for; LightGBM gets
    # EXACTLY those, or the comparison is meaningless.
    shared = tft_preds["id"].unique()
    lgbm_features = features[features["id"].isin(shared)].reset_index(drop=True)
    lgbm_preds = backtest_predictions(
        lgbm_features, quantiles, lgbm_params(config), horizon, folds,
        config.training.n_train_origins, config.training.origin_stride,
    )

    tft_panel = build_panel(tft_preds, features)
    lgbm_panel = build_panel(lgbm_preds, lgbm_features)
    merged = lgbm_panel["by_level"].merge(
        tft_panel["by_level"], on="level", suffixes=("_lgbm", "_tft")
    )
    cols = ["level"] + [f"{m}_{s}" for m in ("wape", "mase", "rmsse") for s in ("lgbm", "tft")]
    table = merged[cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison_by_level.csv", index=False)
    tft_panel["calibration"].to_csv(out_dir / "tft_calibration.csv", index=False)
    lgbm_panel["calibration"].to_csv(out_dir / "lgbm_calibration.csv", index=False)
    (out_dir / "comparison.md").write_text(
        f"# LightGBM vs TFT — {len(shared)} shared CA series, fold {folds[0]} (GPU run)\n\n"
        + table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    bottom = table[table["level"] == "item_store"].iloc[0]
    logger.info("METRIC lgbm_wape_item_store=%.6f", float(bottom["wape_lgbm"]))
    logger.info("METRIC tft_wape_item_store=%.6f", float(bottom["wape_tft"]))
    logger.info("Comparison written to %s (%d shared series)", out_dir, len(shared))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=os.environ.get("SM_CHANNEL_FEATURES"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--checkpoint-dir", default="/opt/ml/checkpoints")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--state-filter", default="CA", help="The honest-comparison slice")
    parser.add_argument("--tft-max-series", type=int)
    parser.add_argument("--tft-epochs", type=int)
    parser.add_argument(
        "--tft-max-samples-per-ts", type=int,
        help="Bounds darts' sliding-window sample count per series (see tft.py QuantileTFT.fit)",
    )
    parser.add_argument("--tft-loader-workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    tft_cfg = tft_config_with_overrides(config, args.tft_max_series, args.tft_epochs)
    raw = config.model_dump()
    raw["data"]["state_filter"] = args.state_filter
    # Load 2x what the TFT's eligibility filter will select from — headroom for
    # short-history rejects without materialising all ~12k CA series on an
    # instance with the same 16GB that OOM-killed the Phase 3 first attempt.
    raw["training"]["max_series"] = 2 * tft_cfg.max_series
    config = Config(**raw)

    log_device()
    compare(
        config, tft_cfg, Path(args.features_dir), Path(args.model_dir), args.checkpoint_dir,
        args.tft_max_samples_per_ts, args.tft_loader_workers,
    )


if __name__ == "__main__":
    main()
