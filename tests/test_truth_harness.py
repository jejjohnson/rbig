"""Tests for the analytic-truth harness itself (issue #132).

The harness must demonstrably catch the Jacobian-density-term bug class:
an entropy estimator that drops the ``log f(x)`` term reports
``log(2*pi*e)`` instead of ``0.5*log(2*pi*e)`` for standard-normal data,
and :func:`tests.truth.assert_close_stat` must fail on it with the
discrepancy visible.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tests.truth import (
    GAUSSIAN_1D_ENTROPY,
    assert_close_stat,
    gaussian_entropy,
    gaussian_kld,
    gaussian_mi,
)


def test_closed_forms():
    assert GAUSSIAN_1D_ENTROPY == pytest.approx(1.41894, abs=1e-5)
    assert gaussian_entropy(np.eye(2)) == pytest.approx(2 * GAUSSIAN_1D_ENTROPY)
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    # H(joint) = H(x1) + H(x2) - MI
    assert gaussian_entropy(cov) == pytest.approx(
        2 * GAUSSIAN_1D_ENTROPY - gaussian_mi(rho)
    )
    assert gaussian_mi(0.5) == pytest.approx(0.1438, abs=5e-5)
    assert gaussian_mi(0.9) == pytest.approx(0.8304, abs=5e-5)
    # KLD(N(0,1) || N(1,1)) = 0.5
    assert gaussian_kld([0.0], [[1.0]], [1.0], [[1.0]]) == pytest.approx(0.5)
    assert gaussian_kld([0.0], [[1.0]], [0.0], [[1.0]]) == pytest.approx(0.0)


def _gaussian_mle_entropy(seed: int, n: int = 5000) -> float:
    """Known-good 1-D entropy estimator: Gaussian MLE plug-in."""
    x = np.random.default_rng(seed).standard_normal(n)
    return 0.5 * np.log(2.0 * np.pi * np.e * x.var())


def _biased_jacobian_entropy(seed: int, n: int = 5000) -> float:
    """Known-bad estimator reproducing the missing-density-term bug.

    For the probit-CDF Gaussianization ``z = Phi^{-1}(F(x))`` the correct
    per-point log-derivative is ``log f(x) - log phi(z)``.  Dropping the
    ``log f(x)`` term and estimating ``H = -E[-log phi(z)]``-style yields
    ``log(2*pi*e)`` (double the truth) on standard-normal data.
    """
    x = np.random.default_rng(seed).standard_normal(n)
    u = (stats.rankdata(x) - 0.5) / x.size
    z = stats.norm.ppf(u)
    # BUG (deliberate): log|dz/dx| taken as -log phi(z) alone.
    biased_log_det = -stats.norm.logpdf(z)
    return GAUSSIAN_1D_ENTROPY - float(np.mean(-biased_log_det))


def test_assert_close_stat_accepts_known_good():
    assert_close_stat(
        _gaussian_mle_entropy, GAUSSIAN_1D_ENTROPY, tol=0.05, label="gaussian MLE"
    )


def test_assert_close_stat_rejects_biased_jacobian():
    """The harness must catch the 2x entropy bias with the values visible."""
    with pytest.raises(AssertionError, match=r"2\.8|mean"):
        assert_close_stat(
            _biased_jacobian_entropy,
            GAUSSIAN_1D_ENTROPY,
            tol=0.05,
            label="biased Jacobian entropy",
        )
    # And the bias is exactly the log(2*pi*e) vs 0.5*log(2*pi*e) discrepancy.
    assert _biased_jacobian_entropy(0) == pytest.approx(
        2 * GAUSSIAN_1D_ENTROPY, abs=0.02
    )


def test_no_retry_plugins_installed():
    """Flake policy: retries are banned — widen tolerance or raise n."""
    from pathlib import Path

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert "rerunfailures" not in pyproject
    with pytest.raises(ImportError):
        import pytest_rerunfailures  # noqa: F401


def test_golden_plumbing(golden):
    """Golden fixture pins and compares fixed-seed arrays."""
    rng = np.random.default_rng(7)
    golden("harness_selfcheck", rng.standard_normal(16))
