import numpy as np
import pytest
from rbig._src.marginal import (
    make_cdf_monotonic, entropy_marginal, bin_estimation,
    univariate_make_uniform, univariate_make_normal
)


def test_make_cdf_monotonic():
    cdf = np.array([0.0, 0.2, 0.15, 0.4, 0.9, 1.0])
    mono_cdf = make_cdf_monotonic(cdf)
    assert np.all(np.diff(mono_cdf) >= 0)


def test_entropy_marginal(rng):
    data = rng.standard_normal((500, 3))
    H = entropy_marginal(data)
    assert H.shape == (3,)
    assert np.all(np.isfinite(H))


def test_bin_estimation():
    n_bins = bin_estimation(1000)
    assert isinstance(n_bins, int)
    assert n_bins > 0


def test_univariate_make_uniform(rng):
    x = rng.standard_normal(200)
    x_uniform, params = univariate_make_uniform(x, extension=10, precision=1000)
    assert x_uniform.shape == x.shape
    assert np.all(x_uniform >= 0) and np.all(x_uniform <= 1)


def test_univariate_make_normal(rng):
    x = rng.standard_normal(200)
    x_gauss, params = univariate_make_normal(x, extension=10, precision=1000)
    assert x_gauss.shape == x.shape
    assert np.all(np.isfinite(x_gauss))
