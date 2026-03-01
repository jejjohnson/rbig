"""Test AnnealedRBIG model."""
import numpy as np

from rbig import AnnealedRBIG


def test_instantiation():
    model = AnnealedRBIG()
    assert model.n_layers == 1000


def test_fit(data_2d):
    model = AnnealedRBIG(n_layers=50, zero_tolerance=10).fit(data_2d)
    assert hasattr(model, "gauss_params_")
    assert hasattr(model, "rotation_matrix_")
    assert hasattr(model, "residual_info_")


def test_transform_shape(data_2d):
    model = AnnealedRBIG(n_layers=50, zero_tolerance=10).fit(data_2d)
    Z = model.transform(data_2d)
    assert Z.shape == data_2d.shape


def test_inverse_transform_recovers_input(data_2d):
    model = AnnealedRBIG(n_layers=50, zero_tolerance=10).fit(data_2d)
    Z = model.transform(data_2d)
    X_rec = model.inverse_transform(Z)
    np.testing.assert_allclose(X_rec, data_2d, atol=1e-3)


def test_score_samples_shape(data_2d):
    model = AnnealedRBIG(n_layers=50, zero_tolerance=10).fit(data_2d)
    log_p = model.score_samples(data_2d)
    assert log_p.shape == (data_2d.shape[0],)
    assert np.all(np.isfinite(log_p))


def test_total_correlation_float(data_2d):
    model = AnnealedRBIG(n_layers=50, zero_tolerance=10).fit(data_2d)
    tc = model.total_correlation()
    assert isinstance(tc, float)
    assert tc >= 0


def test_entropy_float(data_2d):
    model = AnnealedRBIG(n_layers=50, zero_tolerance=10).fit(data_2d)
    h = model.entropy()
    assert isinstance(h, float)
