"""A multi-quantile LightGBM forecaster: one booster per quantile.

Quantile regression: LightGBM's `quantile` objective with `alpha=q` trains a
model whose prediction targets the q-th quantile of the conditional demand
distribution. We fit one booster per quantile (0.05 … 0.95) and stack them, so
the output is a predictive distribution, not a point — the reason this project
exists (asymmetric under/over-forecast cost; the business picks the service level).

`predict` enforces two behavioural guarantees the rest of the system relies on:
non-negative demand, and quantiles that do not cross (q05 <= q50 <= q95).
"""

import json
from pathlib import Path
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_forecasting.training.dataset import CATEGORICAL_COLS


def quantile_column(q: float) -> str:
    """Stable column name for a quantile, e.g. 0.5 -> 'q0.5'."""
    return f"q{q}"


class QuantileLGBM:
    def __init__(self, quantiles: list[float], params: dict[str, object]) -> None:
        self.quantiles = sorted(quantiles)
        self.params = params
        self.boosters: dict[float, lgb.Booster] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileLGBM":
        categoricals = [c for c in CATEGORICAL_COLS if c in X.columns]
        dataset = lgb.Dataset(X, label=y, categorical_feature=categoricals, free_raw_data=False)
        n_rounds = cast(int, self.params.get("n_estimators", 300))
        base = {k: v for k, v in self.params.items() if k != "n_estimators"}
        for q in self.quantiles:
            params = {**base, "objective": "quantile", "alpha": q, "verbose": -1}
            self.boosters[q] = lgb.train(params, dataset, num_boost_round=n_rounds)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of quantile columns, non-negative and non-crossing."""
        raw = np.column_stack([self.boosters[q].predict(X) for q in self.quantiles])
        # Non-negativity, then sort across quantiles so intervals never invert.
        raw = np.clip(raw, a_min=0.0, a_max=None)
        raw = np.sort(raw, axis=1)
        cols = [quantile_column(q) for q in self.quantiles]
        return pd.DataFrame(raw, columns=cols, index=X.index)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for q, booster in self.boosters.items():
            booster.save_model(str(directory / f"booster_{q}.txt"))
        meta = {"quantiles": self.quantiles, "params": self.params}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "QuantileLGBM":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        model = cls(quantiles=meta["quantiles"], params=meta["params"])
        for q in model.quantiles:
            model.boosters[q] = lgb.Booster(model_file=str(directory / f"booster_{q}.txt"))
        return model
