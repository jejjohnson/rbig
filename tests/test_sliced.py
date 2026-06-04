"""Tests for Sliced Iterative Gaussianization (SIGLayer, GIS, SIG)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rbig import GIS, SIG, SIGLayer


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def banana(rng):
    """Banana-shaped 2D distribution."""
    X = rng.standard_normal((600, 2))
    X[:, 1] += 0.5 * X[:, 0] ** 2
    return X


@pytest.fixture
def gmm_2d(rng):
    """Two-component 2D Gaussian mixture."""
    n = 600
    comp = rng.integers(0, 2, n)
    means = np.array([[-3.0, -3.0], [3.0, 3.0]])
    X = rng.standard_normal((n, 2)) + means[comp]
    return X


@pytest.fixture
def high_d(rng):
    """Moderate-dimensional correlated data."""
    d = 20
    A = rng.standard_normal((d, d))
    return rng.standard_normal((400, d)) @ A


# ── SIGLayer ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["stiefel", "random"])
def test_layer_roundtrip(banana, method):
    layer = SIGLayer(n_directions=2, direction_method=method, random_state=0)
    layer.fit(banana)
    Z, log_det = layer.transform(banana)
    Xr = layer.inverse_transform(Z)
    np.testing.assert_allclose(Xr, banana, atol=1e-6)
    assert log_det.shape == (banana.shape[0],)
    assert np.all(np.isfinite(log_det))


def test_layer_directions_orthonormal(high_d):
    layer = SIGLayer(n_directions=5, direction_method="stiefel", random_state=0)
    layer.fit(high_d)
    assert layer.A_.shape == (20, 5)
    np.testing.assert_allclose(layer.A_.T @ layer.A_, np.eye(5), atol=1e-7)


def test_layer_makes_projection_more_gaussian(rng):
    from scipy import stats

    # Strongly non-Gaussian along the first axis.
    X = rng.standard_normal((1000, 3))
    X[:, 0] = rng.exponential(1.0, 1000)
    layer = SIGLayer(n_directions=3, direction_method="stiefel", random_state=0)
    layer.fit(X)
    Z, _ = layer.transform(X)
    # The most non-Gaussian projected direction should be more Gaussian after.
    a = layer.A_[:, 0]
    before = stats.shapiro(X @ a).statistic
    after = stats.shapiro((Z @ a)[:500]).statistic
    assert after >= before - 0.05


# ── GIS ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("data", ["banana", "gmm_2d", "high_d"])
def test_gis_transform_roundtrip(data, request):
    X = request.getfixturevalue(data)
    model = GIS(n_layers=10, random_state=0).fit(X)
    Z = model.transform(X)
    Xr = model.inverse_transform(Z)
    assert Z.shape == X.shape
    np.testing.assert_allclose(Xr, X, atol=1e-4)


def test_gis_transform_is_gaussianish(gmm_2d):
    model = GIS(n_layers=15, random_state=0).fit(gmm_2d)
    Z = model.transform(gmm_2d)
    assert np.all(np.abs(Z.mean(axis=0)) < 0.3)
    assert np.all(np.abs(Z.std(axis=0) - 1.0) < 0.3)


def test_gis_score_samples(banana):
    model = GIS(n_layers=10, random_state=0).fit(banana)
    ll = model.score_samples(banana)
    assert ll.shape == (banana.shape[0],)
    assert np.all(np.isfinite(ll))
    assert np.isfinite(model.score(banana))


def test_gis_sample_shape(banana):
    model = GIS(n_layers=8, random_state=0).fit(banana)
    samples = model.sample(150, random_state=1)
    assert samples.shape == (150, 2)
    assert np.all(np.isfinite(samples))


def test_gis_early_stopping(rng):
    # Already-Gaussian data should converge well before the layer cap.
    X = rng.standard_normal((500, 3))
    model = GIS(n_layers=200, patience=3, random_state=0).fit(X)
    assert model.n_layers_ < 200


def test_gis_random_directions(banana):
    model = GIS(n_layers=10, direction_method="random", random_state=0).fit(banana)
    Z = model.transform(banana)
    Xr = model.inverse_transform(Z)
    np.testing.assert_allclose(Xr, banana, atol=1e-4)


def test_gis_no_whiten(banana):
    model = GIS(n_layers=8, whiten=False, random_state=0).fit(banana)
    assert model.whitener_ is None
    Xr = model.inverse_transform(model.transform(banana))
    np.testing.assert_allclose(Xr, banana, atol=1e-4)


# ── SIG ──────────────────────────────────────────────────────────────────────


def test_sig_sample_resembles_data(gmm_2d):
    model = SIG(n_layers=15, random_state=0).fit(gmm_2d)
    samples = model.sample(600, random_state=1)
    assert samples.shape == (600, 2)
    # Generated mean should be near the data mean (mixture centred near 0).
    assert np.all(np.abs(samples.mean(axis=0) - gmm_2d.mean(axis=0)) < 1.5)


def test_sig_inverse_roundtrip(banana):
    model = SIG(n_layers=8, random_state=0).fit(banana)
    Z = model.transform(banana)
    Xr = model.inverse_transform(Z)
    np.testing.assert_allclose(Xr, banana, atol=1e-4)


def test_sig_score_finite(banana):
    model = SIG(n_layers=8, random_state=0).fit(banana)
    assert np.isfinite(model.score(banana))


# ── sklearn compatibility ────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [GIS, SIG])
def test_get_params_clone(cls):
    model = cls(n_layers=7, n_directions=2, random_state=3)
    params = model.get_params()
    assert params["n_layers"] == 7
    assert params["random_state"] == 3
    cloned = clone(model)
    assert cloned.get_params()["n_layers"] == 7


@pytest.mark.parametrize("cls", [GIS, SIG])
def test_pipeline(cls, banana):
    pipe = Pipeline(
        [("scale", StandardScaler()), ("flow", cls(n_layers=5, random_state=0))]
    )
    pipe.fit(banana)
    Z = pipe.transform(banana)
    assert Z.shape == banana.shape
