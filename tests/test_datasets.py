"""Tests for the synthetic distribution zoo (issue #132)."""

from __future__ import annotations

import numpy as np
import pytest

from rbig import (
    make_banana,
    make_bimodal,
    make_cross,
    make_equicorrelated_gaussian,
    make_heavy_tail,
    make_rings,
    make_signal_plus_noise_dims,
    make_variance_leak,
    make_xor_labels,
)
from tests.truth import GAUSSIAN_1D_ENTROPY, equicorr_tc, gaussian_entropy, gaussian_mi

GENERATORS = [
    make_banana,
    make_rings,
    make_bimodal,
    make_heavy_tail,
    make_cross,
    make_equicorrelated_gaussian,
    make_xor_labels,
    make_variance_leak,
    make_signal_plus_noise_dims,
]


@pytest.mark.parametrize("generator", GENERATORS)
def test_seed_deterministic(generator):
    """Two calls with identical arguments return bit-identical arrays."""
    X1, meta1 = generator(seed=123)
    X2, meta2 = generator(seed=123)
    np.testing.assert_array_equal(X1, X2)
    assert meta1["name"] == meta2["name"]


@pytest.mark.parametrize("generator", GENERATORS)
def test_shapes_and_finiteness(generator):
    X, meta = generator(n_samples=500, seed=0)
    assert X.shape[0] == 500
    assert X.ndim == 2
    assert np.isfinite(X).all()
    assert "name" in meta


@pytest.mark.parametrize("generator", GENERATORS)
def test_different_seeds_differ(generator):
    X1, _ = generator(seed=0)
    X2, _ = generator(seed=1)
    assert not np.array_equal(X1, X2)


def test_banana_meta_entropy():
    """The banana map is volume-preserving: H = 2 * H(N(0,1))."""
    _X, meta = make_banana(seed=0)
    assert meta["entropy"] == pytest.approx(2 * GAUSSIAN_1D_ENTROPY, abs=1e-12)


def test_equicorrelated_meta_matches_truth_helpers():
    _X, meta = make_equicorrelated_gaussian(d=3, rho=0.5, seed=0)
    assert meta["entropy"] == pytest.approx(gaussian_entropy(meta["cov"]), abs=1e-12)
    assert meta["tc"] == pytest.approx(equicorr_tc(3, 0.5), abs=1e-12)
    _X2, meta2 = make_equicorrelated_gaussian(d=2, rho=0.8, seed=0)
    assert meta2["mi"] == pytest.approx(gaussian_mi(0.8), abs=1e-12)
    # The classic reference value for rho = 0.5.
    assert equicorr_tc(3, 0.5) == pytest.approx(0.3466, abs=5e-4)


def test_equicorrelated_sample_covariance():
    X, meta = make_equicorrelated_gaussian(n_samples=20_000, d=3, rho=0.5, seed=0)
    np.testing.assert_allclose(np.cov(X.T), meta["cov"], atol=0.05)


def test_xor_labels_are_synergistic():
    """Univariate correlation with y is ~0 for both XOR features."""
    X, meta = make_xor_labels(n_samples=5000, seed=0)
    y = meta["labels"]
    for j in meta["informative"]:
        assert abs(np.corrcoef(X[:, j], y)[0, 1]) < 0.05
    # But the product carries the signal.
    assert np.corrcoef(X[:, 0] * X[:, 1], y)[0, 1] > 0.5


def test_variance_leak_groups():
    X, meta = make_variance_leak(n_samples=5000, scale_ratio=2.0, seed=0)
    A = meta["A"]
    var0 = X[A == 0].var(axis=0).mean()
    var1 = X[A == 1].var(axis=0).mean()
    # Equal means, unequal variances.
    np.testing.assert_allclose(X[A == 0].mean(axis=0), 0.0, atol=0.1)
    np.testing.assert_allclose(X[A == 1].mean(axis=0), 0.0, atol=0.15)
    assert var1 / var0 == pytest.approx(4.0, rel=0.15)


def test_signal_plus_noise_variances_interleave():
    """Signal dims have *lower* variance than noise dims (the PCA trap)."""
    X, meta = make_signal_plus_noise_dims(
        n_samples=5000, k_signal=2, m_noise=4, snr=0.5, seed=0
    )
    variances = X.var(axis=0)
    assert variances[meta["signal_dims"]].max() < variances[meta["noise_dims"]].min()
