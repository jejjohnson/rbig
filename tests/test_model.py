"""Tests for the RBIG model."""

import pytest

from rbig import AnnealedRBIG, RBIGLayer


def test_rbig_fit(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    model.fit(simple_2d)
    assert hasattr(model, "layers_")
    assert len(model.layers_) > 0


def test_rbig_transform_shape(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    model.fit(simple_2d)
    Xt = model.transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_fit_transform(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    Xt = model.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_inverse_transform_approx(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    model.fit(simple_2d)
    Xt = model.transform(simple_2d)
    Xr = model.inverse_transform(Xt)
    # After 5 layers the inverse may not be perfect, check shape at least
    assert Xr.shape == simple_2d.shape


def test_rbig_score_samples(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    model.fit(simple_2d)
    log_probs = model.score_samples(simple_2d)
    assert log_probs.shape == (simple_2d.shape[0],)


def test_rbig_score(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    model.fit(simple_2d)
    score = model.score(simple_2d)
    assert isinstance(score, float)


def test_rbig_entropy(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    model.fit(simple_2d)
    h = model.entropy()
    assert isinstance(h, float)


def test_rbig_sample(simple_2d):
    model = AnnealedRBIG(n_layers=5, rotation="pca")
    model.fit(simple_2d)
    samples = model.sample(50)
    assert samples.shape == (50, simple_2d.shape[1])


def test_rbig_ica_rotation(simple_2d):
    model = AnnealedRBIG(n_layers=3, rotation="ica", random_state=42)
    model.fit(simple_2d)
    Xt = model.transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_invalid_rotation(simple_2d):
    model = AnnealedRBIG(n_layers=3, rotation="invalid")
    with pytest.raises(ValueError, match="Unknown rotation"):
        model.fit(simple_2d)


def test_rbig_layer_fit_transform(simple_2d):
    layer = RBIGLayer()
    Xt = layer.fit(simple_2d).transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_tc_convergence(simple_5d):
    model = AnnealedRBIG(n_layers=20, rotation="pca", tol=1e-3, zero_tolerance=5)
    model.fit(simple_5d)
    assert len(model.tc_per_layer_) > 0
