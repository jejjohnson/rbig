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


def test_rbig_score_samples(sample_2d, fitted_rbig):
    """RBIG provides score_samples for log-probability."""
    log_prob = fitted_rbig.score_samples(sample_2d)
    assert log_prob.shape == (sample_2d.shape[0],)
    assert np.all(np.isfinite(log_prob))
    assert np.mean(log_prob) < 0


def test_rbig_residual_info(fitted_rbig):
    """RBIG stores residual_info and mutual_information."""
    assert hasattr(fitted_rbig, 'residual_info')
    assert hasattr(fitted_rbig, 'mutual_information')
    assert isinstance(fitted_rbig.mutual_information, float)
