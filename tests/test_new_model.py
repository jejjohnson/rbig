"""Tests for enhanced AnnealedRBIG with strategy pattern."""

import numpy as np

from rbig import AnnealedRBIG


def test_rbig_strategy_list(simple_2d):
    """Test AnnealedRBIG with list of (rotation, marginal) strategy tuples."""
    strategy = [("pca", "quantile"), ("random", "kde")]
    model = AnnealedRBIG(n_layers=4, strategy=strategy)
    Xt = model.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_strategy_single(simple_2d):
    """Test AnnealedRBIG with single strategy tuple repeated."""
    strategy = [("pca", "spline")]
    model = AnnealedRBIG(n_layers=3, strategy=strategy)
    Xt = model.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_strategy_multiple(simple_2d):
    """Test AnnealedRBIG cycling through multiple strategy tuples."""
    strategy = [("ica", "gmm"), ("pca", "quantile"), ("random", "spline")]
    model = AnnealedRBIG(n_layers=6, strategy=strategy, random_state=42)
    Xt = model.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_predict_proba(simple_2d):
    """Test predict_proba returns probabilities."""
    model = AnnealedRBIG(n_layers=5)
    model.fit(simple_2d)
    proba = model.predict_proba(simple_2d)
    assert proba.shape == (simple_2d.shape[0],)
    assert np.all(proba >= 0)


def test_rbig_calculate_negentropy(simple_2d):
    """Test _calculate_negentropy returns array."""
    J = AnnealedRBIG._calculate_negentropy(simple_2d)
    assert J.shape == (simple_2d.shape[1],)
    assert np.all(J >= -0.5)  # negentropy approximately non-negative


def test_rbig_get_component_rotation(simple_2d):
    """Test _get_component for rotation."""
    model = AnnealedRBIG(n_layers=5)
    r = model._get_component("pca", "rotation", seed=0)
    Xt = r.fit(simple_2d).transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_rbig_get_component_marginal(simple_2d):
    """Test _get_component for marginal."""
    model = AnnealedRBIG(n_layers=5)
    m = model._get_component("quantile", "marginal", seed=0)
    Xt = m.fit(simple_2d).transform(simple_2d)
    assert Xt.shape == simple_2d.shape
