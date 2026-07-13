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


class FeaturesConfig(BaseModel):
    lags: list[int]
    rolling_windows: list[int]
    drop_pre_release: bool


class Config(BaseModel):
    project_name: str
    random_seed: int
    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
