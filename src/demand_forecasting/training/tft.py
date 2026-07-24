"""TFT (Temporal Fusion Transformer) via Darts — the M5 deep-model upgrade.

Where the LightGBM baseline consumes a flat (origin, horizon) table of hand-built
lag/rolling features, the TFT consumes each series as a `TimeSeries` and learns the
temporal structure itself. It emits the whole 28-day quantile forecast from a single
network (one QuantileRegression head), instead of one booster per quantile.

The point of this module is an HONEST comparison, not "deep beats trees": we train
the TFT on the same CA slice the LightGBM comparison uses, run it through the SAME
evaluation panel, and produce a prediction frame with the identical column contract
(`backtest_tft_predictions`), so `evaluation.panel.build_panel` consumes both.

CPU-bound by design (no GPU on this machine): small network, few epochs, a few
hundred series. Full-scale TFT is deferred to the AWS GPU track.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from demand_forecasting.config import TFTConfig
from demand_forecasting.training.model import quantile_column

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only where the optional `tft` group is installed
    from pytorch_lightning.callbacks import Callback as _CallbackBase
except ImportError:  # keeps this module importable (pure helpers, tests) without darts
    _CallbackBase = object  # type: ignore[assignment,misc]


class _EpochProgress(_CallbackBase):
    """One log line per epoch — the only signal a long GPU run gives before
    completion, since the progress bar and Lightning's own logger are both
    disabled below (neither plays well with piped/CloudWatch stdout). A
    silent multi-hour job is a debugging dead end; this print is cheap
    insurance. Must live at module level (not nested in QuantileTFT.fit): darts
    pickles the whole model, callbacks included, and a class defined inside a
    function has an unpicklable `<locals>` qualname.
    """

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        logger.info("Epoch %d/%d complete", trainer.current_epoch + 1, trainer.max_epochs)


# Identity columns carried into the prediction frame so the panel can roll up.
_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Numeric future covariates — legitimately known ahead at forecast time (calendar
# + weekly price). The TFT gets no engineered lags/rolling: it learns those from
# the target window, which is the whole reason to compare it against the tree.
_FUTURE_COV = ["wday", "month", "snap", "event_name_code", "event_type_code",
               "sell_price", "price_rel_dept"]


def _add_event_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode the categorical calendar-event columns as stable integer codes
    (Darts covariates must be numeric). Codes are global across the frame so a
    given event maps to the same number in every fold."""
    out = frame.copy()
    for src, dst in [("event_name_1", "event_name_code"), ("event_type_1", "event_type_code")]:
        out[dst] = out[src].astype("category").cat.codes.astype("float32")
    return out


def _contiguous(series_df: pd.DataFrame, value_cols: list[str], end_day: int) -> pd.DataFrame:
    """Reindex one series onto a gap-free integer day grid [min_d, end_day].

    Darts requires a regular time index. Missing sales become 0 (a not-listed /
    absent day is no demand); covariates are forward/back-filled from known weeks.
    """
    g = series_df.sort_values("d")
    full = pd.RangeIndex(int(g["d"].min()), end_day + 1)
    g = g.set_index("d").reindex(full)
    g["sales"] = g["sales"].fillna(0.0).astype("float32")
    for c in value_cols:
        g[c] = g[c].ffill().bfill().astype("float32")
    return g.reset_index(names="d")


def _build_series(
    frame: pd.DataFrame, series_ids: list[str], cutoff: int, horizon: int, train_history: int
) -> tuple[list[Any], list[Any], list[str]]:
    """Build (target, future-covariate) TimeSeries lists for the given series.

    Target spans the most recent `train_history` days up to the fold origin
    `cutoff`; future covariates span through `cutoff + horizon` (known in advance),
    as Darts needs covariates to cover the forecast window.
    """
    from darts import TimeSeries

    targets, fcovs, used = [], [], []
    last_needed = cutoff + horizon
    start = cutoff - train_history
    for sid in series_ids:
        s = frame[frame["id"] == sid]
        hist = s[(s["d"] > start) & (s["d"] <= cutoff)]
        if len(hist) < 1:
            continue
        cov_src = s[(s["d"] > start) & (s["d"] <= last_needed)]
        tgt_df = _contiguous(hist, _FUTURE_COV, cutoff)
        cov_df = _contiguous(cov_src, _FUTURE_COV, last_needed)
        targets.append(
            TimeSeries.from_dataframe(tgt_df, time_col="d", value_cols="sales", freq=1)
        )
        fcovs.append(
            TimeSeries.from_dataframe(cov_df, time_col="d", value_cols=_FUTURE_COV, freq=1)
        )
        used.append(sid)
    return targets, fcovs, used


class QuantileTFT:
    """Thin wrapper around Darts `TFTModel` with a QuantileRegression head.

    Exposes the same fit/predict shape the backtest needs; `predict` reads the
    quantile parameters straight from the likelihood (no sampling), then applies
    the same behavioural guarantees as the LightGBM model: non-negative demand
    and non-crossing quantiles.
    """

    def __init__(
        self,
        quantiles: list[float],
        cfg: TFTConfig,
        horizon: int,
        seed: int,
        work_dir: str | None = None,
    ) -> None:
        self.quantiles = sorted(quantiles)
        self.cfg = cfg
        self.horizon = horizon
        self.seed = seed
        # When set, Darts persists Lightning checkpoints under this directory —
        # pointed at /opt/ml/checkpoints on SageMaker so Spot interruptions do
        # not lose training state (the dir is synced to S3 by the platform).
        self.work_dir = work_dir
        self.model: Any = None

    def fit(
        self,
        targets: list[Any],
        fcovs: list[Any],
        max_samples_per_ts: int | None = None,
        num_loader_workers: int = 0,
    ) -> "QuantileTFT":
        """`max_samples_per_ts`/`num_loader_workers` default to darts' own
        defaults (unlimited windows, single-process loading) — the exact
        behaviour the local M5 CPU comparison was verified against — so
        neither argument changes anything unless a caller opts in (the GPU
        entry point does; train.py's compare() does not).

        Why this knob exists: darts builds one training SAMPLE per valid
        sliding-window start per series (train_history_days - input_chunk -
        horizon + 1 windows each), all sliced in a single Python process by
        default. At 1000 CA series x 730-day history that is 647,000 samples
        x 15 epochs — ~9.7M slice operations that starved a tiny (hidden_size
        16) network of GPU work entirely; the first real GPU run hit its
        max_run cost cap without completing even one epoch. Capping samples
        per series bounds the dataset size independent of history length or
        series count; num_loader_workers parallelises the slicing itself.
        """
        import torch
        from darts.models import TFTModel
        from darts.utils.likelihood_models import QuantileRegression

        torch.manual_seed(self.seed)
        checkpointing = (
            {"save_checkpoints": True, "work_dir": self.work_dir, "model_name": "tft"}
            if self.work_dir
            else {}
        )
        self.model = TFTModel(
            input_chunk_length=self.cfg.input_chunk_length,
            output_chunk_length=self.horizon,
            hidden_size=self.cfg.hidden_size,
            lstm_layers=self.cfg.lstm_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            dropout=self.cfg.dropout,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            optimizer_kwargs={"lr": self.cfg.learning_rate},
            likelihood=QuantileRegression(quantiles=self.quantiles),
            random_state=self.seed,
            add_relative_index=True,
            **checkpointing,
            pl_trainer_kwargs={
                # "auto" = whatever the machine has: CPU on the 16GB laptop
                # (identical to the old hardcoded value there), CUDA on a GPU
                # training instance. Hardware choice is not the model's business.
                "accelerator": "auto",
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
                "callbacks": [_EpochProgress()],
            },
        )
        self.model.fit(
            series=targets,
            future_covariates=fcovs,
            max_samples_per_ts=max_samples_per_ts,
            dataloader_kwargs={"num_workers": num_loader_workers},
        )
        return self

    def predict(self, horizon: int, targets: list[Any], fcovs: list[Any]) -> list[pd.DataFrame]:
        """Return one DataFrame per series: index=target_day, columns=quantiles,
        non-negative and non-crossing."""
        assert self.model is not None, "fit() must be called before predict()"
        preds = self.model.predict(
            n=horizon,
            series=targets,
            future_covariates=fcovs,
            predict_likelihood_parameters=True,
            num_samples=1,
        )
        preds_list = preds if isinstance(preds, list) else [preds]
        cols = [quantile_column(q) for q in self.quantiles]
        out = []
        for ts in preds_list:
            values = np.asarray(ts.values(), dtype=float)  # (horizon, n_quantiles)
            values = np.sort(np.clip(values, a_min=0.0, a_max=None), axis=1)
            df = pd.DataFrame(values, columns=cols)
            df["target_day"] = np.asarray(ts.time_index, dtype=int)
            out.append(df)
        return out


def _select_series(
    frame: pd.DataFrame, max_series: int, earliest_fold: int, min_history: int, seed: int
) -> list[str]:
    """Deterministically pick up to `max_series` series with enough history before
    the earliest fold origin to form at least one training window."""
    counts = frame[frame["d"] <= earliest_fold].groupby("id", observed=True)["d"].count()
    eligible = counts[counts >= min_history].index.to_series()
    if len(eligible) > max_series:
        eligible = eligible.sample(n=max_series, random_state=seed)
    return sorted(eligible.tolist())


def backtest_tft_predictions(
    features: pd.DataFrame,
    quantiles: list[float],
    cfg: TFTConfig,
    horizon: int,
    folds: list[int],
    seed: int,
    work_dir: str | None = None,
    max_samples_per_ts: int | None = None,
    num_loader_workers: int = 0,
) -> pd.DataFrame:
    """Rolling-origin backtest for the TFT, returning the same per-row prediction
    frame the panel consumes: identity + origin/horizon/target_day + actual + one
    column per quantile. Retrains once per fold, exactly like the LightGBM path.
    """
    frame = _add_event_codes(features)

    min_history = cfg.input_chunk_length + horizon
    series_ids = _select_series(frame, cfg.max_series, min(folds), min_history, seed)
    frame = frame[frame["id"].isin(series_ids)].reset_index(drop=True)
    logger.info("TFT backtest on %d CA series, folds %s", len(series_ids), folds)

    id_meta = frame.groupby("id", observed=True)[_ID_COLS[1:]].first().reset_index()
    actuals = frame[["id", "d", "sales"]].rename(columns={"d": "target_day", "sales": "actual"})
    qcols = [quantile_column(q) for q in sorted(quantiles)]

    frames = []
    for fold in folds:
        targets, fcovs, used = _build_series(
            frame, series_ids, fold, horizon, cfg.train_history_days
        )
        if not targets:
            logger.warning("Fold origin %d has no usable series; skipping", fold)
            continue
        model = QuantileTFT(quantiles, cfg, horizon, seed, work_dir=work_dir).fit(
            targets, fcovs,
            max_samples_per_ts=max_samples_per_ts, num_loader_workers=num_loader_workers,
        )
        per_series = model.predict(horizon, targets, fcovs)

        parts = []
        for sid, df in zip(used, per_series):
            df = df.assign(id=sid, origin=fold)
            df["horizon"] = (df["target_day"] - fold).astype(int)
            parts.append(df)
        fold_df = pd.concat(parts, ignore_index=True)
        fold_df = fold_df.merge(id_meta, on="id").merge(actuals, on=["id", "target_day"])
        frames.append(fold_df[[*_ID_COLS, "origin", "horizon", "target_day", "actual", *qcols]])
        logger.info("Fold %d — %d TFT predictions", fold, len(fold_df))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
