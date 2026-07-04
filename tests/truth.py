"""Analytic ground-truth helpers and the statistical test policy.

Closed-form quantities (nats) used as literal assertion targets across the
estimator-suite tests, plus :func:`assert_close_stat`, which implements the
seed/tolerance policy from issue #132: estimates are averaged over seeds,
the mean must fall within ``tol`` of the truth, and the worst single seed
within ``2 * tol``.  No retries — a flaky statistical test gets a wider
documented tolerance or a bigger ``n``, never a retry loop.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

GAUSSIAN_1D_ENTROPY = 0.5 * np.log(2.0 * np.pi * np.e)  # ~1.41894 nats


def gaussian_entropy(cov: np.ndarray) -> float:
    """Differential entropy of ``N(mu, cov)`` in nats: ``0.5*ln((2*pi*e)^d |cov|)``."""
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    d = cov.shape[0]
    _sign, logdet = np.linalg.slogdet(cov)
    return float(0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet))


def gaussian_mi(rho: float) -> float:
    """MI of a bivariate Gaussian with correlation ``rho``: ``-0.5*ln(1-rho^2)``."""
    return float(-0.5 * np.log(1.0 - rho**2))


def equicorr_tc(d: int, rho: float) -> float:
    """Total correlation of a d-dim equicorrelated Gaussian: ``-0.5*ln(det R)``."""
    cov = np.full((d, d), rho)
    np.fill_diagonal(cov, 1.0)
    _sign, logdet = np.linalg.slogdet(cov)
    return float(-0.5 * logdet)


def gaussian_kld(
    mu_p: np.ndarray, cov_p: np.ndarray, mu_q: np.ndarray, cov_q: np.ndarray
) -> float:
    """KL divergence ``KL(N(mu_p, cov_p) || N(mu_q, cov_q))`` in nats."""
    mu_p, mu_q = np.atleast_1d(mu_p).astype(float), np.atleast_1d(mu_q).astype(float)
    cov_p = np.atleast_2d(np.asarray(cov_p, dtype=float))
    cov_q = np.atleast_2d(np.asarray(cov_q, dtype=float))
    d = mu_p.shape[0]
    cov_q_inv = np.linalg.inv(cov_q)
    diff = mu_q - mu_p
    _s_p, logdet_p = np.linalg.slogdet(cov_p)
    _s_q, logdet_q = np.linalg.slogdet(cov_q)
    return float(
        0.5
        * (
            np.trace(cov_q_inv @ cov_p)
            + diff @ cov_q_inv @ diff
            - d
            + logdet_q
            - logdet_p
        )
    )


def assert_close_stat(
    estimator: Callable[[int], float],
    truth: float,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    tol: float = 0.05,
    label: str = "estimate",
) -> None:
    """Assert a statistical estimate matches an analytic truth.

    Policy: the mean over ``seeds`` must lie within ``tol`` of ``truth``
    and every individual seed within ``2 * tol``.  On failure the message
    reports every per-seed value so the discrepancy is visible.

    Parameters
    ----------
    estimator : callable
        ``estimator(seed) -> float``; called once per seed.
    truth : float
        The analytic target value.
    seeds : tuple of int, default (0, 1, 2, 3, 4)
        Seeds to average over.
    tol : float, default 0.05
        Tolerance on the seed-mean (worst case allowed ``2 * tol``).
    label : str, default "estimate"
        Name used in the failure message.
    """
    values = np.array([float(estimator(seed)) for seed in seeds])
    mean = float(values.mean())
    worst = float(np.abs(values - truth).max())
    detail = ", ".join(f"seed {s}: {v:.4f}" for s, v in zip(seeds, values, strict=True))
    assert abs(mean - truth) <= tol, (
        f"{label}: mean {mean:.4f} not within {tol} of truth {truth:.4f} ({detail})"
    )
    assert worst <= 2 * tol, (
        f"{label}: worst-seed deviation {worst:.4f} exceeds {2 * tol} "
        f"(truth {truth:.4f}; {detail})"
    )
