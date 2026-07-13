"""Tests for data-drift monitoring. PSI runs in CI; the Evidently report is
skipped unless the optional `monitoring` group is installed."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.monitoring.drift import (
    feature_drift,
    population_stability_index,
)


def test_psi_zero_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    # Two samples from the same distribution → PSI near zero.
    assert population_stability_index(x[:2500], x[2500:]) < 0.05


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(loc=0.0, size=5000)
    cur = rng.normal(loc=3.0, size=5000)  # strong mean shift
    assert population_stability_index(ref, cur) > 0.25


def test_psi_constant_reference_is_zero() -> None:
    assert population_stability_index(np.ones(100), np.arange(100)) == 0.0


def test_feature_drift_flags_shifted_feature() -> None:
    rng = np.random.default_rng(2)
    reference = pd.DataFrame(
        {"stable": rng.normal(size=3000), "shifted": rng.normal(loc=0.0, size=3000)}
    )
    current = pd.DataFrame(
        {"stable": rng.normal(size=3000), "shifted": rng.normal(loc=4.0, size=3000)}
    )
    table = feature_drift(reference, current, ["stable", "shifted"])
    flags = dict(zip(table["feature"], table["drifted"]))
    assert flags["shifted"] is True or flags["shifted"] == True  # noqa: E712
    assert not flags["stable"]


def test_evidently_report_smoke(tmp_path) -> None:
    pytest.importorskip("evidently")
    from demand_forecasting.monitoring.drift import evidently_data_drift_report

    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    current = pd.DataFrame({"x": rng.normal(loc=1.0, size=500), "y": rng.normal(size=500)})
    out = evidently_data_drift_report(reference, current, tmp_path / "drift.html")
    assert out.exists()
