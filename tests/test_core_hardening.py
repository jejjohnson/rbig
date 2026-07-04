"""Core-engine hardening tests for AnnealedRBIG (issue #120).

Covers the public ``log_det_jacobian``, versioned ``to_dict``/``from_dict``
serialization, determinism, validation edges, and the analytic entropy gate
that regression-guards the Jacobian density term.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import stats

from rbig import AnnealedRBIG, make_banana
from tests.truth import GAUSSIAN_1D_ENTROPY, assert_close_stat, gaussian_entropy


@pytest.fixture(scope="module")
def banana_model():
    X, _ = make_banana(n_samples=2000, seed=0)
    return X, AnnealedRBIG(n_layers=15, random_state=0).fit(X)


# ── log_det_jacobian ─────────────────────────────────────────────────────────


def test_log_det_jacobian_shape_and_consistency(banana_model):
    """score_samples == sum(log phi(z)) + log_det_jacobian, by definition."""
    X, model = banana_model
    ldj = model.log_det_jacobian(X[:100])
    assert ldj.shape == (100,)
    assert np.isfinite(ldj).all()
    Z = model.transform(X[:100])
    np.testing.assert_allclose(
        model.score_samples(X[:100]),
        stats.norm.logpdf(Z).sum(axis=1) + ldj,
        rtol=1e-10,
    )


def test_log_det_jacobian_requires_fit():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        AnnealedRBIG().log_det_jacobian(np.zeros((5, 2)))


# ── Analytic entropy gate (Jacobian density term regression guard) ──────────


def _rbig_entropy(X: np.ndarray) -> float:
    """H(X) = d/2 log(2*pi*e) - E[log |det J|] under a converged flow."""
    model = AnnealedRBIG(n_layers=15, random_state=0).fit(X)
    d = X.shape[1]
    return d * GAUSSIAN_1D_ENTROPY - float(np.mean(model.log_det_jacobian(X)))


def test_entropy_gate_fast():
    """Small-n per-PR gate: half the tolerance budget of the nightly run."""

    def estimate(seed: int) -> float:
        x = np.random.default_rng(seed).standard_normal((2000, 1))
        return _rbig_entropy(x)

    assert_close_stat(
        estimate, GAUSSIAN_1D_ENTROPY, seeds=(0, 1, 2), tol=0.1, label="H(N(0,1))"
    )


@pytest.mark.statistical
@pytest.mark.parametrize(
    ("sampler", "truth", "tol"),
    [
        (lambda rng, n: rng.standard_normal((n, 1)), GAUSSIAN_1D_ENTROPY, 0.05),
        # Bounded/one-sided supports carry a small positive KDE boundary
        # bias in the marginal density term (~+0.05 for U(0,1), ~+0.08 for
        # Exp(1) at n=5000) — a deterministic estimator property, not
        # flakiness, hence the documented wider tolerance (policy: widen,
        # never retry).  The spacing-based 1-D estimators arriving with
        # #124 are the unbiased path for bounded marginals.
        (lambda rng, n: rng.uniform(size=(n, 1)), 0.0, 0.1),
        (lambda rng, n: rng.exponential(size=(n, 1)), 1.0, 0.1),
    ],
    ids=["normal", "uniform", "exponential"],
)
def test_entropy_gate_statistical(sampler, truth, tol):
    def estimate(seed: int) -> float:
        x = sampler(np.random.default_rng(seed), 5000)
        return _rbig_entropy(x)

    assert_close_stat(estimate, truth, tol=tol, label="1-D entropy")


@pytest.mark.statistical
def test_joint_entropy_correlated_gaussian():
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    truth = gaussian_entropy(cov)

    def estimate(seed: int) -> float:
        X = np.random.default_rng(seed).multivariate_normal(np.zeros(2), cov, size=5000)
        return _rbig_entropy(X)

    assert_close_stat(estimate, truth, tol=0.05, label="H(2D rho=0.8)")


def _grid_integral(model, X: np.ndarray, n_grid: int = 300) -> float:
    lim = np.abs(X).max() + 2.0
    grid = np.linspace(-lim, lim, n_grid)
    GX, GY = np.meshgrid(grid, grid)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)  # P: (n_grid^2, 2)
    density = np.exp(model.score_samples(P))
    return float(density.sum() * (grid[1] - grid[0]) ** 2)


@pytest.mark.statistical
def test_density_integrates_to_one_shallow():
    """exp(score_samples) integrates to ~1 for a shallow flow."""
    X, _ = make_banana(n_samples=2000, seed=0)
    model = AnnealedRBIG(n_layers=3, random_state=0).fit(X)
    assert _grid_integral(model, X) == pytest.approx(1.0, abs=0.03)


@pytest.mark.statistical
def test_density_normalization_degrades_with_depth():
    """Pinned finding: clipped empirical marginals leak mass with depth.

    Each layer's probit clipping puts a derivative spike at the support
    boundary; stacking layers compounds it, so the grid integral of
    ``exp(score_samples)`` inflates well past 1 (measured ~1.6 at 15
    layers on the banana).  Tail-aware marginals through the flow (#123)
    are the fix; this test documents the current behavior so the fix has
    a regression target.
    """
    X, _ = make_banana(n_samples=2000, seed=0)
    shallow = AnnealedRBIG(n_layers=3, random_state=0).fit(X)
    deep = AnnealedRBIG(n_layers=15, random_state=0).fit(X)
    assert _grid_integral(deep, X) > _grid_integral(shallow, X) + 0.2


# ── Determinism ──────────────────────────────────────────────────────────────


def test_fit_and_sample_deterministic():
    X, _ = make_banana(n_samples=800, seed=3)
    m1 = AnnealedRBIG(n_layers=8, random_state=42).fit(X)
    m2 = AnnealedRBIG(n_layers=8, random_state=42).fit(X)
    np.testing.assert_array_equal(m1.transform(X), m2.transform(X))
    np.testing.assert_array_equal(
        m1.sample(50, random_state=7), m2.sample(50, random_state=7)
    )


# ── Serialization ────────────────────────────────────────────────────────────


def test_to_dict_round_trip(banana_model):
    X, model = banana_model
    state = model.to_dict()
    assert state["format_version"] == 1
    restored = AnnealedRBIG.from_dict(state)
    np.testing.assert_allclose(
        restored.transform(X[:200]), model.transform(X[:200]), atol=1e-10
    )
    np.testing.assert_allclose(
        restored.score_samples(X[:200]), model.score_samples(X[:200]), atol=1e-8
    )
    Z = model.transform(X[:200])
    np.testing.assert_allclose(
        restored.inverse_transform(Z), model.inverse_transform(Z), atol=1e-8
    )
    np.testing.assert_allclose(
        restored.sample(20, random_state=0),
        model.sample(20, random_state=0),
        atol=1e-8,
    )


def test_from_dict_rejects_unknown_version(banana_model):
    _X, model = banana_model
    state = model.to_dict()
    state["format_version"] = 999
    with pytest.raises(ValueError, match="format_version"):
        AnnealedRBIG.from_dict(state)


def test_pickle_round_trip(banana_model):
    X, model = banana_model
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.score_samples(X[:100]), model.score_samples(X[:100])
    )


# ── Validation edges ─────────────────────────────────────────────────────────


def test_nan_input_raises():
    X = np.random.default_rng(0).standard_normal((100, 2))
    X[3, 1] = np.nan
    with pytest.raises(ValueError):
        AnnealedRBIG(n_layers=3).fit(X)


def test_zero_variance_column_warns_and_fits():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.standard_normal(300), np.full(300, 2.5)])
    with pytest.warns(UserWarning, match="zero variance"):
        model = AnnealedRBIG(n_layers=3, random_state=0).fit(X)
    assert np.isfinite(model.score_samples(X[:20])).all()


# ── Golden regression ────────────────────────────────────────────────────────


def test_score_samples_golden(golden):
    """Pin fixed-seed log-densities to detect numeric drift.

    ``score_samples`` (not ``transform``) is the cross-platform regression
    target: PCA eigenvector *signs* differ between BLAS backends (Linux
    OpenBLAS vs macOS Accelerate), flipping latent columns, but the
    log-density is exactly invariant to those flips (``phi(z) == phi(-z)``
    and the log-det carries no rotation sign).
    """
    X, _ = make_banana(n_samples=500, seed=11)
    model = AnnealedRBIG(n_layers=5, random_state=0).fit(X)
    golden(
        "annealed_rbig_score_samples_banana",
        model.score_samples(X[:20]),
        rtol=1e-8,
        atol=1e-10,
    )
