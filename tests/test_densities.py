"""Test density estimation."""
import numpy as np
import pytest

from rbig import AnnealedRBIG, score_samples, log_prob


def test_score_samples_shape(data_2d):
    model = AnnealedRBIG(n_layers=30, zero_tolerance=10).fit(data_2d)
    log_p = score_samples(data_2d, model.gauss_params_, model.rotation_matrix_)
    assert log_p.shape == (data_2d.shape[0],)


def test_score_samples_finite(data_2d):
    model = AnnealedRBIG(n_layers=30, zero_tolerance=10).fit(data_2d)
    log_p = score_samples(data_2d, model.gauss_params_, model.rotation_matrix_)
    assert np.all(np.isfinite(log_p))


def test_log_prob_shape(data_2d):
    model = AnnealedRBIG(n_layers=30, zero_tolerance=10).fit(data_2d)
    log_p = log_prob(data_2d, model)
    assert log_p.shape == (data_2d.shape[0],)
    assert np.all(np.isfinite(log_p))
