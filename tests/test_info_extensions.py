"""Extended information-theory measures (issue #124).

Fast lane exercises structure (ranges, symmetry, caching, code paths) at
small n; the analytic value gates run in the statistical lane at n=5000
using reduced-but-converged RBIG settings.
"""

from __future__ import annotations

import numpy as np
import pytest

from rbig import (
    InformationMeasures,
    estimate_cmi,
    estimate_dtc,
    estimate_jsd,
    estimate_mi,
    estimate_o_information,
    estimate_tc,
    make_equicorrelated_gaussian,
)
from tests.truth import assert_close_stat, gaussian_mi

# Reduced-but-converged settings so a single estimate is seconds, not
# minutes (defaults are n_layers=100).
FAST = {"n_layers": 25, "patience": 5, "random_state": 0}


# ── Fast structural tests ────────────────────────────────────────────────────


def test_cmi_non_negative_and_conditioning_removes_dependence():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((1200, 1))
    x = z + 0.3 * rng.standard_normal((1200, 1))
    y = z + 0.3 * rng.standard_normal((1200, 1))
    cmi = estimate_cmi(x, y, z, **FAST)
    mi = estimate_mi(x, y, **FAST)
    assert cmi >= 0.0
    # X and Y are dependent only through Z: conditioning collapses the MI.
    assert cmi < mi


def test_dtc_non_negative_and_requires_2d():
    X, _ = make_equicorrelated_gaussian(n_samples=1200, d=3, rho=0.5, seed=0)
    assert estimate_dtc(X, **FAST) >= 0.0
    with pytest.raises(ValueError, match="2 dimensions"):
        estimate_dtc(X[:, :1], **FAST)


def test_jsd_bounds():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((1200, 2))
    Q = rng.standard_normal((1200, 2)) + 5.0
    jsd_same = estimate_jsd(P, P.copy(), **FAST)
    assert 0.0 <= jsd_same <= 0.05
    # Disjoint supports saturate at the ln 2 bound; the raw estimate
    # overshoots it (mixture-entropy bias), which the clamp reports.
    with pytest.warns(UserWarning, match="clamped"):
        jsd_far = estimate_jsd(P, Q, **FAST)
    assert jsd_far == pytest.approx(np.log(2.0), abs=0.1)
    assert jsd_far <= np.log(2.0)


def test_information_measures_cache_and_consistency():
    X, _ = make_equicorrelated_gaussian(n_samples=1000, d=3, rho=0.5, seed=0)
    im = InformationMeasures(**FAST).fit(X)
    tc = im.tc()
    n_fits_after_tc = len(im._cache)
    tc2 = im.tc()  # fully cached: no new fits
    assert tc == tc2
    assert len(im._cache) == n_fits_after_tc == 4  # 3 marginals + joint
    # TC(d=2) == MI on the same cached entropies, by construction.
    mi = im.mi([0], [1])
    tc_2d = im.entropy([0]) + im.entropy([1]) - im.entropy([0, 1])
    assert mi == pytest.approx(max(tc_2d, 0.0))
    # O-information is TC - DTC exactly.
    assert im.o_information() == pytest.approx(im.tc() - im.dtc())


def test_information_measures_requires_fit():
    with pytest.raises(ValueError, match="fit"):
        InformationMeasures().entropy()


def test_pairwise_mi_matrix_structure():
    X, _ = make_equicorrelated_gaussian(n_samples=1000, d=3, rho=0.5, seed=0)
    im = InformationMeasures(**FAST).fit(X)
    M = im.pairwise_mi_matrix()
    assert M.shape == (3, 3)
    np.testing.assert_allclose(M, M.T)
    assert (M[np.triu_indices(3, k=1)] >= 0.0).all()
    Mn = im.pairwise_mi_matrix(normalized=True)
    np.testing.assert_allclose(np.diag(Mn), np.diag(M))


def test_clamping_warns_on_large_violation():
    from rbig._src.rbig_measures import _clamp

    with pytest.warns(UserWarning, match="clamped"):
        assert _clamp(-0.5, 0.0, np.inf, "MI") == 0.0
    # Small violations clamp silently.
    assert _clamp(-0.001, 0.0, np.inf, "MI") == 0.0


# ── Statistical analytic gates ───────────────────────────────────────────────


@pytest.mark.statistical
def test_mi_gaussian_analytic_values():
    for rho, truth in [(0.5, gaussian_mi(0.5)), (0.8, gaussian_mi(0.8))]:

        def estimate(seed: int, rho=rho) -> float:
            X, _ = make_equicorrelated_gaussian(n_samples=5000, d=2, rho=rho, seed=seed)
            return estimate_mi(X[:, :1], X[:, 1:], **FAST)

        assert_close_stat(
            estimate, truth, seeds=(0, 1, 2), tol=0.05, label=f"MI rho={rho}"
        )


@pytest.mark.statistical
def test_mi_invariant_under_monotone_transforms():
    """The killer property Pearson fails: MI(exp(X); Y) == MI(X; Y)."""
    X, _meta = make_equicorrelated_gaussian(n_samples=5000, d=2, rho=0.8, seed=0)
    base = estimate_mi(X[:, :1], X[:, 1:], **FAST)
    for name, f in [("exp", np.exp), ("cube", lambda v: v**3)]:
        transformed = estimate_mi(f(X[:, :1]), X[:, 1:], **FAST)
        assert transformed == pytest.approx(base, abs=0.05), name


@pytest.mark.statistical
def test_tc_equicorrelated_analytic():
    def estimate(seed: int) -> float:
        X, _ = make_equicorrelated_gaussian(n_samples=5000, d=3, rho=0.5, seed=seed)
        return estimate_tc(X, **FAST)

    assert_close_stat(estimate, 0.3466, seeds=(0, 1, 2), tol=0.05, label="TC d=3")


@pytest.mark.statistical
def test_o_information_signs():
    rng = np.random.default_rng(0)
    # Redundancy: three noisy copies of one source -> positive Omega.
    s = rng.standard_normal((5000, 1))
    redundant = np.hstack([s + 0.3 * rng.standard_normal((5000, 1)) for _ in range(3)])
    assert estimate_o_information(redundant, **FAST) > 0.1
    # Synergy: XOR-style triple -> negative Omega.
    a = rng.standard_normal((5000, 1))
    b = rng.standard_normal((5000, 1))
    c = np.sign(a * b) + 0.1 * rng.standard_normal((5000, 1))
    synergy = np.hstack([a, b, c])
    assert estimate_o_information(synergy, **FAST) < -0.05
    # Correlated Gaussians: Omega matches the analytic value.  (The draft
    # design doc claimed "Omega = 0 for multivariate Gaussians" — that is
    # false: for equicorrelated rho=0.6, d=3, TC = 0.522 and DTC = 0.375,
    # so Omega = +0.147, redundancy-dominated.  The estimator agrees.)
    from tests.truth import equicorr_tc, gaussian_entropy

    rho, d = 0.6, 3
    cov3 = np.full((d, d), rho)
    np.fill_diagonal(cov3, 1.0)
    h3 = gaussian_entropy(cov3)
    h2 = gaussian_entropy(cov3[:2, :2])
    omega_true = equicorr_tc(d, rho) - (d * h2 - (d - 1) * h3)
    X, _ = make_equicorrelated_gaussian(n_samples=5000, d=d, rho=rho, seed=1)
    assert estimate_o_information(X, **FAST) == pytest.approx(omega_true, abs=0.07)
