"""Tests for new metric functions."""

import numpy as np

from rbig import (
    AnnealedRBIG,
    entropy_marginal,
    entropy_quantile_spacing,
    entropy_rbig,
    entropy_univariate,
    information_reduction,
    information_summary,
    negative_log_likelihood,
)


def test_entropy_univariate(simple_2d):
    h = entropy_univariate(simple_2d[:, 0])
    assert isinstance(h, float)
    assert np.isfinite(h)


def test_entropy_marginal_shape(simple_2d):
    h = entropy_marginal(simple_2d)
    assert h.shape == (2,)
    assert np.all(np.isfinite(h))


def test_entropy_quantile_spacing(simple_2d):
    h = entropy_quantile_spacing(simple_2d[:, 0])
    assert isinstance(h, float)
    assert np.isfinite(h)


def test_entropy_rbig(simple_2d):
    model = AnnealedRBIG(n_layers=5)
    model.fit(simple_2d)
    h = entropy_rbig(model, simple_2d)
    assert isinstance(h, float)
    assert np.isfinite(h)


def test_negative_log_likelihood(simple_2d):
    model = AnnealedRBIG(n_layers=5)
    model.fit(simple_2d)
    nll = negative_log_likelihood(model, simple_2d)
    assert isinstance(nll, float)
    assert np.isfinite(nll)


def test_information_summary_keys(simple_2d):
    model = AnnealedRBIG(n_layers=5)
    model.fit(simple_2d)
    summary = information_summary(model, simple_2d)
    assert "entropy" in summary
    assert "total_correlation" in summary
    assert "neg_log_likelihood" in summary


def test_information_reduction(simple_2d):
    reduction = information_reduction(
        simple_2d, np.random.default_rng(42).normal(size=simple_2d.shape)
    )
    assert isinstance(reduction, float)


# ── negentropy_kde ─────────────────────────────────────────────────────────


class TestNegentropyKDE:
    def test_shape(self, simple_2d):
        from rbig import negentropy_kde

        neg = negentropy_kde(simple_2d)
        assert neg.shape == (2,)

    def test_near_zero_for_gaussian(self):
        from rbig import negentropy_kde

        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 3))
        neg = negentropy_kde(X)
        np.testing.assert_allclose(neg, 0.0, atol=0.15)

    def test_non_negative(self):
        from rbig import negentropy_kde

        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 2))
        neg = negentropy_kde(X)
        assert np.all(neg >= -0.05)  # allow small numerical margin

    def test_positive_for_non_gaussian(self):
        from rbig import negentropy_kde

        rng = np.random.default_rng(42)
        # Uniform data is non-Gaussian
        X = rng.uniform(-1, 1, size=(1000, 2))
        # Standardize to zero mean, unit variance for fair comparison
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        neg = negentropy_kde(X)
        assert np.all(neg > 0.01)


# ── Coverage gap tests ────────────────────────────────────────────────────


def test_mutual_information_rbig():
    """mutual_information_rbig returns a finite float for correlated data."""
    from rbig import mutual_information_rbig

    rng = np.random.default_rng(42)
    # Simple correlated 2D data
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    XY = rng.multivariate_normal([0, 0], cov, size=200)
    X = XY[:, :1]
    Y = XY[:, 1:]

    model_X = AnnealedRBIG(n_layers=5).fit(X)
    model_Y = AnnealedRBIG(n_layers=5).fit(Y)
    model_XY = AnnealedRBIG(n_layers=5).fit(XY)

    mi = mutual_information_rbig(model_X, model_Y, model_XY)
    assert isinstance(mi, float)
    assert np.isfinite(mi)


def test_kl_divergence_rbig():
    """kl_divergence_rbig returns a finite float."""
    from rbig import kl_divergence_rbig

    rng = np.random.default_rng(42)
    X_P = rng.standard_normal((200, 2))
    X_Q = rng.standard_normal((200, 2)) + 1.0  # shifted distribution

    model_P = AnnealedRBIG(n_layers=5).fit(X_P)
    kl = kl_divergence_rbig(model_P, X_Q)
    assert isinstance(kl, float)
    assert np.isfinite(kl)


def test_negentropy_kde_constant_feature():
    """negentropy_kde returns 0.0 for a constant column."""
    from rbig import negentropy_kde

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))
    X[:, 1] = 3.0  # constant column
    neg = negentropy_kde(X)
    assert neg.shape == (2,)
    assert neg[1] == 0.0


def test_entropy_quantile_spacing_constant():
    """entropy_quantile_spacing returns 0.0 for constant data."""
    x = np.full(100, 7.0)
    h = entropy_quantile_spacing(x)
    assert h == 0.0
