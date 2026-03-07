"""Tests for density bijectors and migrated density utilities."""

import numpy as np
import pytest

from rbig import Cube, Exp, Tanh, bin_estimation, entropy_histogram, generate_batches

# ── Bijector tests ─────────────────────────────────────────────────────────


def test_tanh_shape(simple_2d):
    t = Tanh()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_tanh_range(simple_2d):
    t = Tanh()
    Xt = t.fit_transform(simple_2d)
    assert np.all(Xt > -1)
    assert np.all(Xt < 1)


def test_tanh_inverse(simple_2d):
    t = Tanh()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-6)


def test_tanh_log_det(simple_2d):
    t = Tanh()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)
    assert np.all(ldj <= 0)  # tanh is contractive


def test_exp_shape(simple_2d):
    t = Exp()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_exp_positive(simple_2d):
    t = Exp()
    Xt = t.fit_transform(simple_2d)
    assert np.all(Xt > 0)


def test_exp_inverse(simple_2d):
    t = Exp()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-10)


def test_exp_log_det(simple_2d):
    t = Exp()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)
    np.testing.assert_allclose(ldj, np.sum(simple_2d, axis=1))


def test_cube_shape(simple_2d):
    t = Cube()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_cube_inverse(simple_2d):
    t = Cube()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-10)


def test_cube_log_det(simple_2d):
    t = Cube()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)


# ── bin_estimation ──────────────────────────────────────────────────────────


class TestBinEstimation:
    def test_sturge_known_value(self):
        # n=256 → 1 + log2(256) = 1 + 8 = 9
        assert bin_estimation(256, rule="sturge") == 9

    def test_sqrt_known_value(self):
        # n=100 → sqrt(100) = 10
        assert bin_estimation(100, rule="sqrt") == 10

    def test_rice_known_value(self):
        # n=125 → 2 * 125^(1/3) = 2 * 5 = 10
        assert bin_estimation(125, rule="rice") == 10

    def test_sturge_rounds_up(self):
        # n=100 → 1 + log2(100) ≈ 7.64 → ceil = 8
        assert bin_estimation(100, rule="sturge") == 8

    def test_sqrt_rounds_up(self):
        # n=10 → sqrt(10) ≈ 3.16 → ceil = 4
        assert bin_estimation(10, rule="sqrt") == 4

    def test_minimum_one(self):
        assert bin_estimation(1, rule="sturge") >= 1
        assert bin_estimation(1, rule="sqrt") >= 1
        assert bin_estimation(1, rule="rice") >= 1

    def test_invalid_rule(self):
        with pytest.raises(ValueError, match="Unknown bin estimation rule"):
            bin_estimation(100, rule="bogus")

    def test_default_rule_is_sturge(self):
        assert bin_estimation(100) == bin_estimation(100, rule="sturge")

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            bin_estimation(0)


# ── generate_batches ────────────────────────────────────────────────────────


class TestGenerateBatches:
    def test_exact_division(self):
        batches = list(generate_batches(10, 5))
        assert batches == [(0, 5), (5, 10)]

    def test_remainder(self):
        batches = list(generate_batches(10, 3))
        assert batches == [(0, 3), (3, 6), (6, 9), (9, 10)]

    def test_single_batch(self):
        batches = list(generate_batches(5, 10))
        assert batches == [(0, 5)]

    def test_full_coverage(self):
        n = 17
        batches = list(generate_batches(n, 4))
        covered = set()
        for start, end in batches:
            covered.update(range(start, end))
        assert covered == set(range(n))

    def test_batch_size_one(self):
        batches = list(generate_batches(3, 1))
        assert batches == [(0, 1), (1, 2), (2, 3)]

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be >= 0"):
            list(generate_batches(-1, 5))

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            list(generate_batches(10, 0))


# ── entropy_histogram ──────────────────────────────────────────────────────


class TestEntropyHistogram:
    def test_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 3))
        H = entropy_histogram(X)
        assert H.shape == (3,)

    def test_gaussian_approximation(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((2000, 2))
        H = entropy_histogram(X)
        expected = 0.5 * (1 + np.log(2 * np.pi))  # ≈ 1.419 nats
        np.testing.assert_allclose(H, expected, atol=0.3)

    def test_correction_effect(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 1))
        H_corr = entropy_histogram(X, correction=True)
        H_no_corr = entropy_histogram(X, correction=False)
        # Correction adds a positive term
        assert np.all(H_corr > H_no_corr)

    def test_base_2_vs_base_e(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 1))
        H_e = entropy_histogram(X, base=np.e)
        H_2 = entropy_histogram(X, base=2.0)
        # H_e = H_2 * ln(2), so H_2 = H_e / ln(2)
        np.testing.assert_allclose(H_2, H_e / np.log(2), atol=0.05)
