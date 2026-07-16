"""SageMaker script-mode entry point for the LightGBM baseline.

SageMaker runs THIS file inside a prebuilt container; everything it does is
imported from the portable `demand_forecasting` package — same training code as
the local `python -m demand_forecasting.training.train`, different orchestration.
That is the entire point of Phase 3.

The only AWS-specific surface is the container contract, all of it environment
variables SageMaker sets before the script starts:
  - SM_CHANNEL_FEATURES — local dir where SageMaker downloaded the `features`
    input channel (our Glue output from s3://<bucket>/features/) before we ran.
  - SM_MODEL_DIR        — local dir whose contents SageMaker tars into
    model.tar.gz and uploads to S3 when the script exits 0.
  - SM_OUTPUT_DATA_DIR  — local dir for non-model artifacts (uploaded separately).
Hyperparameters arrive as command-line flags. Nothing else knows it is on AWS,
which is why the same code runs unchanged on a laptop.

MLflow is deliberately NOT used here: on AWS, SageMaker Experiments records the
run (hyperparameters automatically, metrics via the regex patterns declared in
run_sagemaker_training.py, which scrape the METRIC lines this script prints).
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecasting.config import Config, load_config
from demand_forecasting.training.dataset import (
    DYNAMIC_COLS,
    STATIC_COLS,
    TARGET_KNOWN_COLS,
    build_direct_table,
    cap_series,
    select_origins,
    to_model_frame,
)
from demand_forecasting.training.model import QuantileLGBM, lgbm_params

logger = logging.getLogger(__name__)


def apply_overrides(config: Config, overrides: dict[str, object]) -> Config:
    """Fold non-None hyperparameter overrides into the config from config.yaml.

    Config stays the single source of truth (the repo convention); SageMaker
    hyperparameters only override individual knobs, which is what Automatic
    Model Tuning will vary in Phase 4. Pure — no AWS, no I/O — so it is the part
    worth unit-testing in CI.

    Rebuilds through `Config(**raw)` rather than `model_copy(update=...)`: the
    latter assigns without validating, leaving nested dicts un-coerced (so
    `config.training.horizon` would blow up mid-job). Re-validating also rejects
    overrides that violate a Config constraint before the instance has pulled any
    data — though note LGBMConfig's knobs are currently unbounded, so that safety
    net only covers the fields Config actually constrains.
    """
    raw = config.model_dump()
    training = raw["training"]
    lgbm = training["lgbm"]
    for key in ("max_series", "horizon", "n_train_origins", "origin_stride"):
        if overrides.get(key) is not None:
            training[key] = overrides[key]
    for key in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples"):
        if overrides.get(key) is not None:
            lgbm[key] = overrides[key]
    if overrides.get("state_filter") is not None:
        raw["data"]["state_filter"] = overrides["state_filter"]
    return Config(**raw)


def load_features(features_dir: Path, config: Config) -> pd.DataFrame:
    """Read the Glue feature output, pushing the series cap INTO the Parquet scan.

    The local entry point can afford a plain `pd.read_parquet(...)` then cap: its
    features.parquet was written by pandas with M1's downcast dtypes (int16/
    float32/categorical). The Glue output is written by SPARK — int64/double and
    plain strings — so the same 46.8M rows are tens of GB in memory and the
    instance is OOM-killed (SIGKILL/exit -9) before the cap ever runs. Same
    features, different physical encoding.

    So we select the ids first, then let pyarrow filter at scan time and only
    materialise the rows we keep. Feature semantics are untouched — this is a
    loader concern, and the sampling itself still goes through the shared
    `cap_series` so the cloud and local runs cannot drift apart.

    Note `max_series: 0` (the full catalogue) would still not fit on an
    ml.m5.xlarge; that needs a larger instance, which is a config + instance-type
    decision, not a code change.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(features_dir), format="parquet")
    state_filter = (
        ds.field("state_id") == config.data.state_filter if config.data.state_filter else None
    )

    # Scanning one column to pick the series is far cheaper than reading all 25.
    id_table = dataset.to_table(columns=["id"], filter=state_filter)
    # Sorted so the choice depends on the seed alone, not on Spark's part-file order.
    unique_ids = pd.DataFrame({"id": sorted(id_table.column("id").unique().to_pylist())})
    keep = cap_series(unique_ids, config.training.max_series, config.random_seed)["id"].tolist()

    row_filter = ds.field("id").isin(keep)
    if state_filter is not None:
        row_filter = row_filter & state_filter
    columns = list(
        dict.fromkeys(["id", "d", "sales", *STATIC_COLS, *DYNAMIC_COLS, *TARGET_KNOWN_COLS])
    )
    return dataset.to_table(columns=columns, filter=row_filter).to_pandas()


def train(config: Config, features_dir: Path, model_dir: Path) -> None:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    features = load_features(features_dir, config)
    logger.info("Loaded features: %d rows, %d series", len(features), features["id"].nunique())

    origins = select_origins(
        features, config.training.n_train_origins, config.training.origin_stride,
        config.training.horizon,
    )
    table = build_direct_table(features, origins, config.training.horizon)
    logger.info("Training table: %d rows over origins %s", len(table), origins)

    model = QuantileLGBM(config.training.quantiles, lgbm_params(config))
    model.fit(to_model_frame(table), table["sales"])
    model.save(model_dir)

    # SageMaker Experiments scrapes stdout for the metric regexes declared on the
    # Estimator, so a printed line is how a metric leaves this container.
    logger.info("METRIC train_rows=%d", len(table))
    logger.info("METRIC n_series=%d", features["id"].nunique())
    logger.info("Saved %d boosters to %s", len(model.boosters), model_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    # Paths default to the SageMaker container contract, but stay overridable so
    # this script is runnable locally too (the AWS-readiness rule).
    parser.add_argument("--features-dir", default=os.environ.get("SM_CHANNEL_FEATURES"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--config", default="config.yaml")
    # Hyperparameter overrides; None means "use config.yaml".
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--n-train-origins", type=int)
    parser.add_argument("--origin-stride", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-leaves", type=int)
    parser.add_argument("--min-child-samples", type=int)
    parser.add_argument("--state-filter")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), vars(args))
    train(config, Path(args.features_dir), Path(args.model_dir))


if __name__ == "__main__":
    main()
