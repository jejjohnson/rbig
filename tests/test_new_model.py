"""Tests for enhanced AnnealedRBIG with strategy pattern."""

import numpy as np
import pytest

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


class TestPredictProba:
    def test_input_domain_default(self, simple_2d):
        """Default domain='input' returns non-negative densities."""
        model = AnnealedRBIG(n_layers=5)
        model.fit(simple_2d)
        proba = model.predict_proba(simple_2d)
        assert proba.shape == (simple_2d.shape[0],)
        assert np.all(proba >= 0)

    def test_transform_domain(self, simple_2d):
        """domain='transform' returns Gaussian-space densities."""
        model = AnnealedRBIG(n_layers=5)
        model.fit(simple_2d)
        proba = model.predict_proba(simple_2d, domain="transform")
        assert proba.shape == (simple_2d.shape[0],)
        assert np.all(proba >= 0)

    def test_both_domain(self, simple_2d):
        """domain='both' returns tuple of (input, transform) densities."""
        model = AnnealedRBIG(n_layers=5)
        model.fit(simple_2d)
        result = model.predict_proba(simple_2d, domain="both")
        assert isinstance(result, tuple)
        p_input, p_transform = result
        assert p_input.shape == (simple_2d.shape[0],)
        assert p_transform.shape == (simple_2d.shape[0],)
        assert np.all(p_input >= 0)
        assert np.all(p_transform >= 0)

    def test_invalid_domain(self, simple_2d):
        """Unknown domain raises ValueError."""
        model = AnnealedRBIG(n_layers=5)
        model.fit(simple_2d)
        with pytest.raises(ValueError, match="Unknown domain"):
            model.predict_proba(simple_2d, domain="invalid")


class TestAutoTol:
    def test_auto_tol_fits(self, simple_2d):
        """tol='auto' should fit without error."""
        model = AnnealedRBIG(n_layers=10, tol="auto")
        model.fit(simple_2d)
        assert len(model.layers_) > 0
        assert model.tol_ > 0

    def test_auto_tol_scales_with_samples(self):
        """Larger datasets should get smaller tolerances."""
        tol_small = AnnealedRBIG._get_information_tolerance(200)
        tol_large = AnnealedRBIG._get_information_tolerance(10_000)
        assert tol_small > tol_large

    def test_auto_tol_stored(self, simple_2d):
        """Resolved tolerance is stored as tol_ attribute."""
        model = AnnealedRBIG(n_layers=5, tol="auto")
        model.fit(simple_2d)
        assert hasattr(model, "tol_")
        assert isinstance(model.tol_, float)

    def test_fixed_tol_stored(self, simple_2d):
        """Fixed tol is also stored as tol_ for consistency."""
        model = AnnealedRBIG(n_layers=5, tol=0.01)
        model.fit(simple_2d)
        assert model.tol_ == 0.01


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


def test_rbig_get_component_unknown_rotation(simple_2d):
    """_get_component raises for unknown rotation name."""
    model = AnnealedRBIG(n_layers=5)
    with pytest.raises(ValueError, match="Unknown rotation"):
        model._get_component("unknown_rot", "rotation", seed=0)


def test_rbig_get_component_unknown_marginal(simple_2d):
    """_get_component raises for unknown marginal name."""
    model = AnnealedRBIG(n_layers=5)
    with pytest.raises(ValueError, match="Unknown marginal"):
        model._get_component("unknown_marg", "marginal", seed=0)


# ── jacobian ───────────────────────────────────────────────────────────────


class TestJacobian:
    def test_shape(self, simple_2d):
        model = AnnealedRBIG(n_layers=5)
        model.fit(simple_2d)
        jac = model.jacobian(simple_2d)
        n, d = simple_2d.shape
        assert jac.shape == (n, d, d)

    def test_return_X_transform(self, simple_2d):
        model = AnnealedRBIG(n_layers=5)
        model.fit(simple_2d)
        result = model.jacobian(simple_2d, return_X_transform=True)
        assert isinstance(result, tuple)
        jac, Xt = result
        assert jac.shape == (simple_2d.shape[0], 2, 2)
        assert Xt.shape == simple_2d.shape
        # Xt should match model.transform
        np.testing.assert_allclose(Xt, model.transform(simple_2d), atol=1e-10)

    def test_log_det_matches_score_samples(self, simple_2d):
        """log|det(jac)| should approximate the accumulated log-det from score_samples."""
        model = AnnealedRBIG(n_layers=10)
        model.fit(simple_2d)
        jac = model.jacobian(simple_2d)

        # Log abs det from the Jacobian matrix
        sign, log_abs_det_jac = np.linalg.slogdet(jac)

        # Log-det from score_samples path
        Xt = simple_2d.copy()
        log_det_layers = np.zeros(simple_2d.shape[0])
        for layer in model.layers_:
            log_det_layers += layer.log_det_jacobian(Xt)
            Xt = layer.transform(Xt)

        # Allow a few outlier samples where the empirical density estimate
        # diverges at the tails; check that 95% of samples agree closely.
        abs_diff = np.abs(log_abs_det_jac - log_det_layers)
        pct_close = np.mean(abs_diff < 1.0)
        assert pct_close >= 0.95, (
            f"Only {pct_close:.1%} of samples have |diff| < 1.0"
        )

    def test_5d_shape(self, simple_5d):
        model = AnnealedRBIG(n_layers=5)
        model.fit(simple_5d)
        jac = model.jacobian(simple_5d)
        n, d = simple_5d.shape
        assert jac.shape == (n, d, d)
