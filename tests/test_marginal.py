"""Test marginal Gaussianization."""
import numpy as np
from scipy.stats import normaltest

from rbig import (
    entropy_marginal,
    fit_marginal_params,
    marginal_gaussianize,
    marginal_gaussianize_inverse,
)


def test_fit_marginal_params_keys(data_2d):
    col = data_2d[:, 0]
    params = fit_marginal_params(col)
    assert "uniform_cdf_support" in params
    assert "uniform_cdf" in params
    assert "empirical_pdf_support" in params
    assert "empirical_pdf" in params
    assert "_cdf_fn" in params
    assert "_ppf_fn" in params


def test_marginal_gaussianize_output_normal(data_2d):
    col = data_2d[:, 0]
    params = fit_marginal_params(col)
    z = marginal_gaussianize(col, params)
    assert z.shape == col.shape
    # Output should be approximately normal
    _, p_val = normaltest(z)
    assert p_val > 1e-3


def test_marginal_gaussianize_inverse_recovers(data_2d):
    col = data_2d[:, 0]
    params = fit_marginal_params(col)
    z = marginal_gaussianize(col, params)
    x_rec = marginal_gaussianize_inverse(z, params)
    np.testing.assert_allclose(x_rec, col, atol=1e-2)


def test_entropy_marginal_shape(data_2d):
    H = entropy_marginal(data_2d)
    assert H.shape == (data_2d.shape[1],)
    assert np.all(np.isfinite(H))


def test_entropy_marginal_nd(data_nd):
    H = entropy_marginal(data_nd)
    assert H.shape == (data_nd.shape[1],)
