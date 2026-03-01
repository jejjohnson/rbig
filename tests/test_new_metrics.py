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
