"""Density-estimation workflow tests for AnnealedRBIG (issue #123).

Covers conditional sampling (exact grid regime and ABC regime), AIC/BIC,
sample jitter, and the generative-quality statistical gates.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rbig import AnnealedRBIG, make_banana


@pytest.fixture(scope="module")
def rho_gaussian_model():
    """A *shallow* converged flow: Gaussian data needs ~2 layers.

    Depth matters here — the clipped-marginal distortion pinned in
    test_core_hardening widens fitted conditionals as layers stack
    (measured conditional variance 0.38 / 0.50 / 0.84 at 2 / 10 / 22
    layers against an analytic 0.36), so conditional-accuracy tests must
    run at the converged shallow depth.
    """
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    X = np.random.default_rng(0).multivariate_normal(np.zeros(2), cov, size=5000)
    return AnnealedRBIG(n_layers=2, random_state=0).fit(X)


# ── sample_conditional ───────────────────────────────────────────────────────


def test_sample_conditional_matches_analytic_gaussian(rho_gaussian_model):
    """Conditional of a rho=0.8 Gaussian: x1 | x0=c ~ N(rho*c, 1-rho^2)."""
    model = rho_gaussian_model
    c, rho = 1.0, 0.8
    draws = model.sample_conditional(
        np.array([c]), cond_dims=[0], n_samples=4000, random_state=1
    )
    assert draws.shape == (4000, 2)
    np.testing.assert_allclose(draws[:, 0], c)  # conditioned dim exact
    cond = draws[:, 1]
    assert cond.mean() == pytest.approx(rho * c, abs=0.05 * max(rho * c, 1.0))
    assert cond.var() == pytest.approx(1 - rho**2, rel=0.10)


def test_sample_conditional_abc_regime():
    """Multi-free-dim regime: conditioned dims exact, shape right, finite."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 3))
    X[:, 2] = 0.7 * X[:, 0] + 0.7 * X[:, 1] + 0.3 * rng.standard_normal(2000)
    model = AnnealedRBIG(n_layers=8, random_state=0).fit(X)
    draws = model.sample_conditional(
        np.array([0.5]), cond_dims=[2], n_samples=200, random_state=1
    )
    # One free dim -> exact grid path; force ABC by conditioning on one dim
    # with two free (already the case here: free dims = {0, 1}).
    assert draws.shape == (200, 3)
    np.testing.assert_allclose(draws[:, 2], 0.5)
    assert np.isfinite(draws).all()
    # Conditional mean of x0+x1 should be positive given x2 = 0.5 > 0.
    assert (draws[:, 0] + draws[:, 1]).mean() > 0.1


def test_sample_conditional_validation(rho_gaussian_model):
    model = rho_gaussian_model
    with pytest.raises(ValueError, match="cond_dims"):
        model.sample_conditional(np.array([1.0, 2.0]), cond_dims=[0])
    with pytest.raises(ValueError, match="at most"):
        model.sample_conditional(np.array([1.0, 2.0]), cond_dims=[0, 1])


# ── AIC / BIC ────────────────────────────────────────────────────────────────


def test_aic_bic_finite_and_ordered(rho_gaussian_model):
    model = rho_gaussian_model
    X_val = np.random.default_rng(3).multivariate_normal(
        np.zeros(2), np.array([[1.0, 0.8], [0.8, 1.0]]), size=1000
    )
    aic, bic = model.aic(X_val), model.bic(X_val)
    assert np.isfinite(aic) and np.isfinite(bic)
    # ln(1000) > 2, so the BIC penalty dominates at equal fit.
    assert bic > aic


def test_score_selects_converged_model_in_gridsearch():
    """Held-out mean log-likelihood prefers the more converged model.

    Uses the rings target, where one layer cannot capture the radial
    structure (on the near-Gaussianizable banana, held-out LL is flat or
    even mildly depth-averse — see the depth-normalization finding pinned
    in test_core_hardening).
    """
    from rbig import make_rings

    X_train, _ = make_rings(n_samples=1500, seed=0)
    X_val, _ = make_rings(n_samples=1500, seed=1)
    scores = {
        nl: AnnealedRBIG(n_layers=nl, random_state=0).fit(X_train).score(X_val)
        for nl in [1, 8]
    }
    assert scores[8] > scores[1]


# ── sample jitter ────────────────────────────────────────────────────────────


def test_sample_jitter_perturbs_without_changing_distribution():
    """Jitter adds ~sigma/sqrt(n) noise and leaves the distribution intact.

    (With this implementation the inverse *interpolates* between quantile
    nodes, so exact grid duplicates do not occur even at tiny n — the
    jitter is a defensive option for duplicate-sensitive downstream code,
    not a bug fix here.)
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 2))
    model = AnnealedRBIG(n_layers=3, random_state=0).fit(X)
    plain = model.sample(2000, random_state=1)
    jittered = model.sample(2000, random_state=1, jitter=True)
    diff = jittered - plain
    assert np.any(diff != 0.0)
    # Noise scale ~ sigma_k / sqrt(n_train) with n_train = 40.
    assert 0.0 < np.abs(diff).mean() < 3.0 * X.std() / np.sqrt(40)
    # Distributions remain statistically indistinguishable.
    assert stats.ks_2samp(plain[:, 0], jittered[:, 0]).pvalue > 0.01


# ── Generative quality (statistical lane) ────────────────────────────────────


@pytest.mark.statistical
def test_two_sample_classifier_cannot_distinguish_samples():
    """GBM AUC distinguishing generated from real banana data is ~chance."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    aucs = []
    for seed in range(5):
        X_real, _ = make_banana(n_samples=2000, seed=seed)
        model = AnnealedRBIG(n_layers=20, random_state=seed).fit(X_real)
        X_hold, _ = make_banana(n_samples=2000, seed=100 + seed)
        X_gen = model.sample(2000, random_state=seed)
        X_all = np.vstack([X_hold, X_gen])
        y_all = np.concatenate([np.zeros(2000), np.ones(2000)])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.5, random_state=seed
        )
        clf = GradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
    assert np.mean(aucs) <= 0.55


@pytest.mark.statistical
def test_marginal_ks_between_samples_and_data():
    X_real, _ = make_banana(n_samples=3000, seed=0)
    model = AnnealedRBIG(n_layers=20, random_state=0).fit(X_real)
    X_hold, _ = make_banana(n_samples=3000, seed=1)
    X_gen = model.sample(3000, random_state=2)
    for j in range(X_real.shape[1]):
        assert stats.ks_2samp(X_hold[:, j], X_gen[:, j]).pvalue > 0.01
