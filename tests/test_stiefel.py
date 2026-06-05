"""Tests for Stiefel-manifold projection-direction optimization."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from rbig._src.stiefel import (
    max_sliced_wasserstein_directions,
    random_orthogonal_directions,
    wasserstein_1d,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_wasserstein_identical_is_zero(rng):
    x = rng.standard_normal(500)
    assert wasserstein_1d(x, x) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_shift(rng):
    x = rng.standard_normal(5000)
    assert wasserstein_1d(x, x + 3.0) == pytest.approx(3.0, abs=0.05)


def test_wasserstein_unequal_sizes(rng):
    x = rng.standard_normal(1000)
    y = rng.standard_normal(700) + 2.0
    assert wasserstein_1d(x, y) == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("d,k", [(5, 1), (5, 2), (8, 8), (10, 5)])
def test_random_directions_orthonormal(d, k):
    A = random_orthogonal_directions(d, k, random_state=0)
    assert A.shape == (d, k)
    np.testing.assert_allclose(A.T @ A, np.eye(k), atol=1e-10)


def test_random_directions_capped_to_d():
    A = random_orthogonal_directions(4, 10, random_state=0)
    assert A.shape == (4, 4)


def _non_gaussian_data(rng, n=2000, d=4):
    # First two coordinates are heavy-tailed / skewed; rest are Gaussian.
    X = rng.standard_normal((n, d))
    X[:, 0] = rng.exponential(1.0, n)
    X[:, 1] = rng.standard_t(2.5, n)
    return X


@pytest.mark.parametrize("k", [1, 2, 4])
def test_directions_orthonormal(rng, k):
    X = _non_gaussian_data(rng)
    Z = rng.standard_normal((2000, 4))
    A = max_sliced_wasserstein_directions(X, Z, k, max_iter=30, random_state=0)
    assert A.shape == (4, k)
    np.testing.assert_allclose(A.T @ A, np.eye(k), atol=1e-8)


def test_objective_strictly_improves(rng):
    X = _non_gaussian_data(rng)
    Z = rng.standard_normal((2000, 4))
    _, history = max_sliced_wasserstein_directions(
        X, Z, 2, max_iter=80, random_state=0, return_history=True
    )
    # The optimizer must actually move: a no-op (returning the initial frame)
    # would leave history at length 1 with no gain.
    assert len(history) > 1
    assert history[-1] > history[0] + 1e-3
    # Monotone ascent: every accepted iterate improves the objective.
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(history))


def test_finds_more_non_gaussian_direction_than_random(rng):
    X = _non_gaussian_data(rng)
    Z = rng.standard_normal((2000, 4))
    # Averaged over several random initializations, the optimized direction
    # is meaningfully more non-Gaussian than a random one.
    swd_opt, swd_rand = [], []
    for seed in range(5):
        A_opt = max_sliced_wasserstein_directions(
            X, Z, 1, max_iter=80, random_state=seed
        )
        A_rand = random_orthogonal_directions(4, 1, random_state=seed)
        swd_opt.append(wasserstein_1d(X @ A_opt[:, 0], Z @ A_opt[:, 0]))
        swd_rand.append(wasserstein_1d(X @ A_rand[:, 0], Z @ A_rand[:, 0]))
    assert np.mean(swd_opt) > np.mean(swd_rand) + 0.05
