"""Tests for XarrayRBIG class."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rbig import XarrayRBIG


@pytest.fixture
def simple_da():
    rng = np.random.default_rng(42)
    return xr.DataArray(
        rng.normal(size=(50, 4, 4)),
        dims=["time", "lat", "lon"],
        coords={"time": np.arange(50), "lat": np.arange(4), "lon": np.arange(4)},
    )


def test_xarray_rbig_fit(simple_da):
    model = XarrayRBIG(n_layers=3, random_state=42)
    summary = model.fit(simple_da)
    assert "entropy" in summary
    assert "total_correlation" in summary


def test_xarray_rbig_transform(simple_da):
    model = XarrayRBIG(n_layers=3, random_state=42)
    model.fit(simple_da)
    Xt = model.transform(simple_da)
    assert Xt is not None


def test_xarray_rbig_score_samples(simple_da):
    model = XarrayRBIG(n_layers=3, random_state=42)
    model.fit(simple_da)
    log_probs = model.score_samples(simple_da)
    assert log_probs.ndim == 1
    assert len(log_probs) > 0
