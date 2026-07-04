"""Synthetic distribution zoo shared by tests, benchmarks, and docs.

Seeded generators for every distribution shape used in the estimator-suite
acceptance criteria (see issue #132).  Each generator returns ``(X, meta)``
where ``meta`` is a dict carrying analytic quantities when they exist
(true entropy, mutual information, total correlation, labels, sensitive
attributes), so tests can assert against ground truth without re-deriving
it inline.

All generators are deterministic given ``seed``: two calls with the same
arguments return bit-identical arrays.
"""

from __future__ import annotations

import numpy as np


def make_banana(
    n_samples: int = 2000, curvature: float = 0.5, seed: int | None = 0
) -> tuple[np.ndarray, dict]:
    """Gaussian bent along a parabola (the classic "banana").

    ``x1 ~ N(0, 1)``, ``x2 = e + curvature * x1**2 - curvature`` with
    ``e ~ N(0, 1)``.

    Parameters
    ----------
    n_samples : int, default 2000
        Number of samples.
    curvature : float, default 0.5
        Strength of the parabolic bend.
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        Samples.
    meta : dict
        ``{"name": "banana", "entropy": float}`` — the true differential
        entropy.  The map ``(x1, e) -> (x1, x2)`` is volume-preserving
        (unit-triangular Jacobian), so ``H = 2 * H(N(0,1)) = log(2*pi*e)``.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n_samples)
    e = rng.standard_normal(n_samples)
    x2 = e + curvature * x1**2 - curvature
    X = np.stack([x1, x2], axis=1)  # X: (n_samples, 2)
    return X, {"name": "banana", "entropy": float(np.log(2.0 * np.pi * np.e))}


def make_rings(
    n_samples: int = 2000,
    radii: tuple[float, ...] = (1.0, 3.0),
    noise: float = 0.15,
    seed: int | None = 0,
) -> tuple[np.ndarray, dict]:
    """Concentric noisy rings with per-ring labels.

    Parameters
    ----------
    n_samples : int, default 2000
        Total number of samples, split evenly across rings.
    radii : tuple of float, default (1.0, 3.0)
        Ring radii.
    noise : float, default 0.15
        Radial Gaussian noise standard deviation.
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        Samples.
    meta : dict
        ``{"name": "rings", "labels": (n_samples,) int array}``.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(radii), size=n_samples)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    r = np.asarray(radii)[labels] + noise * rng.standard_normal(n_samples)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return X, {"name": "rings", "labels": labels}


def make_bimodal(
    n_samples: int = 2000,
    separation: float = 4.0,
    n_features: int = 2,
    seed: int | None = 0,
) -> tuple[np.ndarray, dict]:
    """Two well-separated Gaussian modes along the first axis.

    Parameters
    ----------
    n_samples : int, default 2000
        Number of samples.
    separation : float, default 4.0
        Distance between the two mode centers along dimension 0.
    n_features : int, default 2
        Dimensionality; dimensions beyond the first are standard normal.
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Samples.
    meta : dict
        ``{"name": "bimodal", "labels": (n_samples,) int array}`` — the
        mode assignments.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_samples)
    X = rng.standard_normal((n_samples, n_features))
    X[:, 0] += (labels - 0.5) * separation
    return X, {"name": "bimodal", "labels": labels}


def make_heavy_tail(
    n_samples: int = 2000, df: float = 3.0, n_features: int = 2, seed: int | None = 0
) -> tuple[np.ndarray, dict]:
    """Independent Student-t marginals (heavy tails).

    Parameters
    ----------
    n_samples : int, default 2000
        Number of samples.
    df : float, default 3.0
        Degrees of freedom of each Student-t marginal.
    n_features : int, default 2
        Dimensionality.
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Samples.
    meta : dict
        ``{"name": "heavy_tail", "entropy": float}`` — the true joint
        entropy (sum of independent Student-t marginal entropies).
    """
    from scipy import special, stats

    rng = np.random.default_rng(seed)
    X = stats.t.rvs(df, size=(n_samples, n_features), random_state=rng)
    # Student-t differential entropy (per dimension), in nats.
    half = (df + 1.0) / 2.0
    h1 = half * (special.digamma(half) - special.digamma(df / 2.0)) + np.log(
        np.sqrt(df) * special.beta(df / 2.0, 0.5)
    )
    return X, {"name": "heavy_tail", "entropy": float(n_features * h1)}


def make_cross(
    n_samples: int = 2000, width: float = 0.25, seed: int | None = 0
) -> tuple[np.ndarray, dict]:
    """Two orthogonal elongated Gaussian bars forming a cross.

    Parameters
    ----------
    n_samples : int, default 2000
        Total number of samples, split evenly across the two bars.
    width : float, default 0.25
        Standard deviation of each bar's thin axis (the long axis has
        standard deviation 1).
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        Samples.
    meta : dict
        ``{"name": "cross", "labels": (n_samples,) int array}`` — bar
        assignments.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_samples)
    long_axis = rng.standard_normal(n_samples)
    thin_axis = width * rng.standard_normal(n_samples)
    # Bar 0 is horizontal, bar 1 vertical.
    x = np.where(labels == 0, long_axis, thin_axis)
    y = np.where(labels == 0, thin_axis, long_axis)
    return np.stack([x, y], axis=1), {"name": "cross", "labels": labels}


def make_equicorrelated_gaussian(
    n_samples: int = 2000, d: int = 3, rho: float = 0.5, seed: int | None = 0
) -> tuple[np.ndarray, dict]:
    """Zero-mean Gaussian with equicorrelation matrix ``R = (1-rho)I + rho*J``.

    Parameters
    ----------
    n_samples : int, default 2000
        Number of samples.
    d : int, default 3
        Dimensionality.
    rho : float, default 0.5
        Common pairwise correlation (must satisfy ``-1/(d-1) < rho < 1``).
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, d)
        Samples.
    meta : dict
        ``{"name": ..., "cov": (d, d), "entropy": float, "tc": float}``
        and, when ``d == 2``, ``"mi"`` — all analytic values in nats.
    """
    cov = np.full((d, d), rho)
    np.fill_diagonal(cov, 1.0)
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n_samples)
    _sign, logdet = np.linalg.slogdet(cov)
    entropy = 0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet)
    # TC = sum of marginal entropies - joint entropy = -1/2 log det R
    tc = -0.5 * logdet
    meta = {
        "name": "equicorrelated_gaussian",
        "cov": cov,
        "entropy": float(entropy),
        "tc": float(tc),
    }
    if d == 2:
        meta["mi"] = float(-0.5 * np.log(1.0 - rho**2))
    return X, meta


def make_xor_labels(
    n_samples: int = 2000, n_noise: int = 3, noise: float = 0.1, seed: int | None = 0
) -> tuple[np.ndarray, dict]:
    """XOR classification data: ``y = sign(x1 * x2)`` plus noise features.

    Each of ``x1, x2`` is individually independent of ``y`` (univariate
    MI is zero), but the pair determines ``y`` — the canonical synergy
    benchmark for conditional-MI feature selection.

    Parameters
    ----------
    n_samples : int, default 2000
        Number of samples.
    n_noise : int, default 3
        Number of pure-noise features appended after the XOR pair.
    noise : float, default 0.1
        Label-flip corruption applied through a noisy margin.
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2 + n_noise)
        Features; columns 0 and 1 are the XOR pair.
    meta : dict
        ``{"name": "xor_labels", "labels": (n_samples,) int array,
        "informative": [0, 1]}``.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 2 + n_noise))
    margin = X[:, 0] * X[:, 1] + noise * rng.standard_normal(n_samples)
    labels = (margin > 0).astype(int)
    return X, {"name": "xor_labels", "labels": labels, "informative": [0, 1]}


def make_variance_leak(
    n_samples: int = 2000,
    n_features: int = 4,
    scale_ratio: float = 2.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, dict]:
    """Two groups with equal means but unequal variances (nonlinear leakage).

    Both groups are zero-mean Gaussians; group 1's features are scaled by
    ``scale_ratio``.  A linear projection cannot separate the groups (all
    means coincide) but the sensitive attribute ``A`` remains predictable
    from second-order statistics — the benchmark that motivates
    distribution-level fairness transforms.

    Parameters
    ----------
    n_samples : int, default 2000
        Total number of samples, split evenly between groups.
    n_features : int, default 4
        Dimensionality.
    scale_ratio : float, default 2.0
        Standard-deviation ratio between group 1 and group 0.
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Samples.
    meta : dict
        ``{"name": "variance_leak", "A": (n_samples,) int array}`` — the
        sensitive-group attribute.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=n_samples)
    scales = np.where(A == 1, scale_ratio, 1.0)[:, None]
    X = scales * rng.standard_normal((n_samples, n_features))
    return X, {"name": "variance_leak", "A": A}


def make_signal_plus_noise_dims(
    n_samples: int = 2000,
    k_signal: int = 2,
    m_noise: int = 4,
    snr: float = 0.5,
    seed: int | None = 0,
) -> tuple[np.ndarray, dict]:
    """Low-variance bimodal signal dims + high-variance Gaussian noise dims.

    The regime where variance-based reduction (PCA) provably ranks noise
    above signal: signal dimensions are bimodal mixtures with total
    standard deviation ``snr``, noise dimensions are ``N(0, 1)``.
    Negentropy-based criteria must rank all signal dims above all noise
    dims.

    Parameters
    ----------
    n_samples : int, default 2000
        Number of samples.
    k_signal : int, default 2
        Number of bimodal signal dimensions (placed first).
    m_noise : int, default 4
        Number of Gaussian noise dimensions (placed after the signal).
    snr : float, default 0.5
        Overall scale of the signal dimensions relative to unit-variance
        noise.
    seed : int or None, default 0
        Seed for the random generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, k_signal + m_noise)
        Samples.
    meta : dict
        ``{"name": ..., "signal_dims": list, "noise_dims": list,
        "labels": (n_samples,) int array}`` — labels are the mode
        assignments of the first signal dimension.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=(n_samples, k_signal))
    # Bimodal: modes at +/- 1, within-mode std 0.2, overall scale snr.
    signal = snr * (
        (labels * 2.0 - 1.0) + 0.2 * rng.standard_normal((n_samples, k_signal))
    )
    noise = rng.standard_normal((n_samples, m_noise))
    X = np.concatenate([signal, noise], axis=1)
    return X, {
        "name": "signal_plus_noise_dims",
        "signal_dims": list(range(k_signal)),
        "noise_dims": list(range(k_signal, k_signal + m_noise)),
        "labels": labels[:, 0],
    }
