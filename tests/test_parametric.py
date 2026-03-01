"""Test parametric transforms."""
import numpy as np

from rbig import (
    HistogramUniformization,
    KDEUniformization,
    QuantileUniformization,
    fit_parametric_marginal,
    parametric_gaussianize,
    parametric_gaussianize_inverse,
)


def test_histogram_uniformization(data_2d):
    col = data_2d[:, 0]
    model = HistogramUniformization().fit(col)
    u = model.cdf(col)
    assert u.shape == col.shape
    assert np.all(u > 0) and np.all(u < 1)
    log_p = model.logpdf(col)
    assert log_p.shape == col.shape


def test_kde_uniformization(data_2d):
    col = data_2d[:, 0]
    model = KDEUniformization().fit(col)
    u = model.cdf(col)
    assert np.all(u > 0) and np.all(u < 1)
    x_rec = model.ppf(u)
    assert x_rec.shape == col.shape


def test_quantile_uniformization(data_2d):
    col = data_2d[:, 0]
    model = QuantileUniformization().fit(col)
    u = model.cdf(col)
    assert np.all(u > 0) and np.all(u < 1)
    x_rec = model.ppf(u)
    assert x_rec.shape == col.shape
    np.testing.assert_allclose(x_rec, col, atol=0.01)


def test_fit_parametric_marginal_factory(data_2d):
    col = data_2d[:, 0]
    for method in ["histogram", "kde", "quantile", "normal", "logistic", "laplace"]:
        m = fit_parametric_marginal(col, method=method)
        assert hasattr(m, "cdf")
        assert hasattr(m, "ppf")


def test_parametric_gaussianize(data_2d):
    col = data_2d[:, 0]
    params = fit_parametric_marginal(col, method="histogram")
    z = parametric_gaussianize(col, params)
    assert z.shape == col.shape
    assert np.all(np.isfinite(z))


def test_parametric_gaussianize_inverse(data_2d):
    col = data_2d[:, 0]
    params = fit_parametric_marginal(col, method="histogram")
    z = parametric_gaussianize(col, params)
    x_rec = parametric_gaussianize_inverse(z, params)
    assert x_rec.shape == col.shape
    assert np.all(np.isfinite(x_rec))
