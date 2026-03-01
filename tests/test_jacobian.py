import numpy as np
import pytest
from rbig import RBIG, compute_jacobian


def test_compute_jacobian_shape(sample_2d, fitted_rbig):
    jac = compute_jacobian(fitted_rbig, sample_2d)
    n_samples, n_features = sample_2d.shape
    assert jac.shape == (n_samples, n_features, n_features)


def test_compute_jacobian_with_transform(sample_2d, fitted_rbig):
    jac, X_t = compute_jacobian(fitted_rbig, sample_2d, return_X_transform=True)
    assert jac.shape[0] == sample_2d.shape[0]
    assert X_t.shape == sample_2d.shape


def test_compute_jacobian_3d(sample_3d):
    model = RBIG(n_layers=10, zero_tolerance=5)
    model.fit(sample_3d)
    jac = compute_jacobian(model, sample_3d)
    n_samples, n_features = sample_3d.shape
    assert jac.shape == (n_samples, n_features, n_features)
