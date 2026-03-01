import numpy as np
import pytest
from rbig import RBIG


def test_rbig_fit(sample_2d):
    model = RBIG(n_layers=10, zero_tolerance=5)
    model.fit(sample_2d)
    assert hasattr(model, 'gauss_params')
    assert hasattr(model, 'rotation_matrix')


def test_rbig_transform(sample_2d, fitted_rbig):
    Z = fitted_rbig.transform(sample_2d)
    assert Z.shape == sample_2d.shape


def test_rbig_inverse_transform(sample_2d, fitted_rbig):
    Z = fitted_rbig.transform(sample_2d)
    X_rec = fitted_rbig.inverse_transform(Z)
    np.testing.assert_allclose(X_rec, sample_2d, atol=1e-5)


def test_rbig_fit_transform(sample_2d):
    model = RBIG(n_layers=10, zero_tolerance=5)
    Z = model.fit_transform(sample_2d)
    assert Z.shape == sample_2d.shape
