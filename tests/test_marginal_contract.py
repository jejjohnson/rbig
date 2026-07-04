"""Marginal Gaussianizer contract tests (issue #121).

One explicit contract for every marginal Gaussianizer — ``fit`` /
``transform`` / ``inverse_transform`` (+ ``log_det_jacobian`` where density
estimation is supported), monotonicity, and documented boundary behavior —
plus the new tail-extension and dithering strategies.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtri

from rbig import (
    GMMGaussianizer,
    KDEGaussianizer,
    MarginalGaussianize,
    MarginalKDEGaussianize,
    QuantileGaussianizer,
    SplineGaussianizer,
)
from tests.truth import GAUSSIAN_1D_ENTROPY, assert_close_stat

# Classes implementing the Gaussianizing-marginal contract.  Values are
# interior round-trip tolerances — looser for parametric/smoothing families.
GAUSSIANIZERS: dict = {
    MarginalGaussianize: 0.05,
    MarginalKDEGaussianize: 0.1,
    QuantileGaussianizer: 0.05,
    KDEGaussianizer: 0.1,
    SplineGaussianizer: 0.05,
    GMMGaussianizer: 0.35,
}


def _make(cls):
    return cls(random_state=0) if "random_state" in cls().get_params() else cls()


@pytest.fixture(scope="module")
def skewed_train():
    rng = np.random.default_rng(0)
    return rng.gamma(shape=4.0, scale=1.0, size=(3000, 2))


@pytest.mark.parametrize("cls", list(GAUSSIANIZERS), ids=lambda c: c.__name__)
def test_contract_round_trip_interior(cls, skewed_train):
    """inverse(transform(x)) recovers x at interior quantiles."""
    tol = GAUSSIANIZERS[cls]
    model = _make(cls).fit(skewed_train)
    lo, hi = np.quantile(skewed_train, [0.05, 0.95], axis=0)
    interior = skewed_train[((skewed_train > lo) & (skewed_train < hi)).all(axis=1)][
        :500
    ]
    X_rec = model.inverse_transform(model.transform(interior))
    scale = skewed_train.std(axis=0)
    assert np.max(np.abs(X_rec - interior) / scale) < tol


@pytest.mark.parametrize("cls", list(GAUSSIANIZERS), ids=lambda c: c.__name__)
def test_contract_monotone(cls, skewed_train):
    """The forward map is monotone non-decreasing per feature."""
    model = _make(cls).fit(skewed_train)
    lo, hi = np.quantile(skewed_train[:, 0], [0.02, 0.98])
    grid = np.linspace(lo, hi, 200)
    G = np.column_stack([grid, np.full(grid.size, skewed_train[:, 1].mean())])
    z = model.transform(G)[:, 0]
    assert np.all(np.diff(z) > -1e-6)


@pytest.mark.parametrize("cls", list(GAUSSIANIZERS), ids=lambda c: c.__name__)
def test_contract_output_approximately_gaussian(cls, skewed_train):
    model = _make(cls).fit(skewed_train)
    Z = model.transform(skewed_train)
    assert np.abs(Z.mean(axis=0)).max() < 0.25
    assert np.abs(Z.std(axis=0) - 1.0).max() < 0.25


@pytest.mark.parametrize("cls", list(GAUSSIANIZERS), ids=lambda c: c.__name__)
def test_contract_picklable(cls, skewed_train):
    model = _make(cls).fit(skewed_train)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.transform(skewed_train[:50]), model.transform(skewed_train[:50])
    )


def test_log_det_matches_finite_differences():
    """log|dz/dx| vs central differences of the smooth (KDE) forward map."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 1))
    model = MarginalKDEGaussianize().fit(x)
    lo, hi = np.quantile(x, [0.05, 0.95])
    grid = np.linspace(lo, hi, 100).reshape(-1, 1)
    h = 1e-5
    z_plus = model.transform(grid + h)
    z_minus = model.transform(grid - h)
    fd = np.log((z_plus - z_minus) / (2 * h)).ravel()
    ldj = model.log_det_jacobian(grid)
    assert np.max(np.abs(ldj - fd)) < 1e-3


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"tail": "gaussian"}, {"tail": "pareto"}],
    ids=["empirical", "tail-gaussian", "tail-pareto"],
)
def test_entropy_identity_per_strategy(kwargs):
    """1-D entropy via -E[log_det] + H(N(0,1)) reproduces closed forms."""

    def estimate(seed: int) -> float:
        x = np.random.default_rng(seed).standard_normal((5000, 1))
        model = MarginalGaussianize(**kwargs).fit(x)
        return GAUSSIAN_1D_ENTROPY - float(np.mean(model.log_det_jacobian(x)))

    assert_close_stat(
        estimate, GAUSSIAN_1D_ENTROPY, seeds=(0, 1, 2), tol=0.03, label="H(N(0,1))"
    )


# ── Tail extension ───────────────────────────────────────────────────────────


def test_tail_extension_finite_monotone_accurate():
    """N(0,1) train: transform(5) is finite, monotone, within 0.5 of truth."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 1))
    model = MarginalGaussianize(tail="gaussian").fit(x)
    grid = np.linspace(-6.0, 6.0, 400).reshape(-1, 1)
    z = model.transform(grid).ravel()
    assert np.isfinite(z).all()
    assert np.all(np.diff(z) > -1e-3)  # monotone up to the seam discontinuity
    z5 = model.transform(np.array([[5.0]]))[0, 0]
    assert abs(z5 - ndtri(stats.norm.cdf(5.0))) < 0.5
    # The plain empirical version clips (documented boundary behavior).
    clipped = MarginalGaussianize().fit(x)
    z5_clip = clipped.transform(np.array([[5.0]]))[0, 0]
    z9_clip = clipped.transform(np.array([[9.0]]))[0, 0]
    assert z5_clip == pytest.approx(z9_clip)  # saturated: no ranking beyond


@pytest.mark.parametrize("tail", ["gaussian", "pareto"])
def test_tail_round_trip_out_of_support(tail):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 1))
    model = MarginalGaussianize(tail=tail).fit(x)
    grid = np.linspace(-7.0, 7.0, 100).reshape(-1, 1)
    x_rec = model.inverse_transform(model.transform(grid))
    assert np.max(np.abs(x_rec - grid)) < 0.05
    assert np.isfinite(model.log_det_jacobian(grid)).all()


def test_tail_invalid_name_raises():
    with pytest.raises(ValueError, match="tail"):
        MarginalGaussianize(tail="cauchy").fit(
            np.random.default_rng(0).standard_normal((100, 1))
        )


# ── Dithering ────────────────────────────────────────────────────────────────


def test_dither_breaks_ties_deterministically():
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(1000, 1)).astype(float)
    model = MarginalGaussianize(dither=True, random_state=0).fit(X)
    Z1, Z2 = model.transform(X), model.transform(X)
    np.testing.assert_array_equal(Z1, Z2)  # same seed => same jitter
    # Ties are broken: many more unique latents than unique inputs.
    assert len(np.unique(Z1)) > 10 * len(np.unique(X))


def test_dither_gaussianizes_discrete_column():
    """Regression: dithering removes the atoms that stall Gaussianization.

    Undithered, a Poisson column maps to a handful of repeated latents (the
    empirical CDF is a step function on ties) and the output is far from
    N(0, 1); with seeded dithering the ties break and the KS distance to a
    standard normal drops by an order of magnitude.
    """
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(2000, 1)).astype(float)
    ks = {}
    for dither in [False, True]:
        mg = MarginalGaussianize(dither=dither, random_state=0).fit(X)
        z = mg.transform(X).ravel()
        ks[dither] = stats.ks_1samp(z, stats.norm.cdf).statistic
    assert ks[True] < ks[False] / 10.0
    assert ks[True] < 0.02


# ── n_quantiles memory cap ───────────────────────────────────────────────────


def test_n_quantiles_caps_memory_with_small_accuracy_loss():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20_000, 1))
    full = MarginalGaussianize().fit(x)
    capped = MarginalGaussianize(n_quantiles=1000).fit(x)
    assert capped.support_.shape[0] == 1000
    grid = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    assert np.max(np.abs(full.transform(grid) - capped.transform(grid))) < 0.02
