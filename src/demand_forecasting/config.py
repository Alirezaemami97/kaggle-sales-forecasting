from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    forecasts_dir: Path


class DataConfig(BaseModel):
    sales_file: str
    calendar_file: str
    prices_file: str
    # 0 means "use every series"; a positive N subsamples N series for fast dev runs.
    subsample_series: int = Field(default=0, ge=0)
    # "" = all states; a state code (e.g. "CA") restricts to that state — used to
    # keep the LightGBM-vs-TFT comparison on a CPU-affordable subset.
    state_filter: str = ""


class FeaturesConfig(BaseModel):
    lags: list[int]
    rolling_windows: list[int]
    drop_pre_release: bool


class LGBMConfig(BaseModel):
    # Bounded so a bad hyperparameter (hand-edited or drawn by SageMaker AMT)
    # fails at config validation, before any training compute is spent.
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0, lt=1.0)
    num_leaves: int = Field(gt=1)  # LightGBM needs at least one split
    min_child_samples: int = Field(gt=0)


class TFTConfig(BaseModel):
    # TFT is CPU-bound here, so it trains on a small CA slice; the LightGBM
    # comparison runs on the SAME slice. Full-scale TFT is the AWS GPU track.
    max_series: int = Field(gt=0)
    # Train on the most recent N days per series (bounds CPU cost; ~2 years still
    # covers two full yearly cycles). Full history is the AWS GPU track.
    train_history_days: int = Field(gt=0)
    input_chunk_length: int = Field(gt=0)  # lookback window fed to the network
    hidden_size: int = Field(gt=0)
    lstm_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    batch_size: int = Field(gt=0)
    n_epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)


class TrainingConfig(BaseModel):
    horizon: int = Field(gt=0)
    quantiles: list[float]
    n_train_origins: int = Field(gt=0)
    origin_stride: int = Field(gt=0)
    max_series: int = Field(ge=0)
    # Which model the entry point trains: "lgbm" (baseline) or "tft" (runs the
    # honest LightGBM-vs-TFT comparison on the CA slice).
    model: str = "lgbm"
    lgbm: LGBMConfig
    tft: TFTConfig | None = None


class BacktestConfig(BaseModel):
    n_folds: int = Field(gt=0)
    fold_stride: int = Field(gt=0)


class InferenceConfig(BaseModel):
    # A series with fewer than this many observed days at the origin is treated as
    # cold-start and forecast from a hierarchy prior instead of the model.
    cold_start_min_history_days: int = Field(default=28, gt=0)
    # Trailing window used to build the store+department demand prior.
    prior_window_days: int = Field(default=28, gt=0)


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    model_name: str


class Config(BaseModel):
    project_name: str
    random_seed: int
    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig
    training: TrainingConfig
    backtest: BacktestConfig
    inference: InferenceConfig = InferenceConfig()
    mlflow: MLflowConfig


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
