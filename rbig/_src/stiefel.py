"""Stiefel-manifold optimization for projection directions.

The Sliced Iterative Gaussianization family finds, at each layer, a small
set of ``K`` orthonormal directions along which the data is *most*
non-Gaussian, then Gaussianizes those 1D slices.  "Most non-Gaussian" is
measured by the **K-Sliced Wasserstein Distance** (K-SWD): the mean
1-Wasserstein distance between the data projected onto each direction and
a Gaussian reference projected onto the same direction.

Maximizing the K-SWD over the set of orthonormal frames is an optimization
on the Stiefel manifold ``St(D, K) = {A in R^{D x K} : A^T A = I_K}``.  We
use the Cayley-transform curvilinear update of Wen & Yin (2013) with a
backtracking line search, which keeps every iterate exactly orthonormal.

References
----------
Wen, Z., & Yin, W. (2013). A feasible method for optimization with
orthogonality constraints. *Mathematical Programming*, 142, 397-434.

Dai, B., & Seljak, U. (2020). Sliced Iterative Normalizing Flows.
https://arxiv.org/abs/2007.00674
"""

from __future__ import annotations

import numpy as np


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """1-Wasserstein (earth-mover) distance between two 1D samples.

    Computed in closed form from the sorted order statistics; sample sizes
    need not match (quantiles are compared on a shared grid).

    Parameters
    ----------
    x, y : np.ndarray
        One-dimensional samples (flattened if not already 1D).

    Returns
    -------
    distance : float
        The 1-Wasserstein distance ``W_1(x, y) >= 0``.
    """
    x = np.sort(np.asarray(x, dtype=float).ravel())
    y = np.sort(np.asarray(y, dtype=float).ravel())
    if x.size == y.size:
        return float(np.mean(np.abs(x - y)))
    n = max(x.size, y.size)
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def random_orthogonal_directions(
    n_features: int, n_directions: int, random_state: int | None = None
) -> np.ndarray:
    """Draw ``K`` random orthonormal directions in ``R^D`` via QR.

    Parameters
    ----------
    n_features : int
        Ambient dimension ``D``.
    n_directions : int
        Number of directions ``K`` (capped at ``D``).
    random_state : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    A : np.ndarray of shape (n_features, n_directions)
        Matrix with orthonormal columns (``A.T @ A == I_K``).
    """
    rng = np.random.default_rng(random_state)
    k = min(n_directions, n_features)
    m = rng.standard_normal((n_features, k))
    q, r = np.linalg.qr(m)
    # Fix column signs so the result is deterministic given the seed.
    q = q * np.sign(np.diag(r))
    return q[:, :k]


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` rows of ``X`` (random subset without replacement)."""
    if X.shape[0] == n:
        return X
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx]


def _kswd_objective_and_grad(
    A: np.ndarray, X: np.ndarray, Z: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean per-direction ``W_1`` and its Euclidean gradient w.r.t. ``A``.

    ``X`` and ``Z`` must have the same number of rows ``n`` so the sorted
    projections pair up one-to-one.  The (sub)gradient of the 1D
    Wasserstein distance w.r.t. a projection direction ``a`` is
    ``(1/n) sum_i sign(p_i - q_i) (x_i - z_i)`` evaluated on the matched
    sorted samples.
    """
    n, k = X.shape[0], A.shape[1]
    P = X @ A
    Q = Z @ A
    obj = 0.0
    grad = np.zeros_like(A)
    for j in range(k):
        ox = np.argsort(P[:, j])
        oz = np.argsort(Q[:, j])
        diff = P[ox, j] - Q[oz, j]
        obj += np.mean(np.abs(diff))
        s = np.sign(diff)
        grad[:, j] = (X[ox].T @ s - Z[oz].T @ s) / n
    return obj / k, grad / k


def max_sliced_wasserstein_directions(
    X: np.ndarray,
    Z_target: np.ndarray,
    n_directions: int,
    max_iter: int = 100,
    lr: float = 1.0,
    tol: float = 1e-6,
    random_state: int | None = None,
    return_history: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    """Find ``K`` orthonormal directions maximizing the K-SWD.

    Maximizes the mean 1-Wasserstein distance between ``X`` and
    ``Z_target`` projected onto each of ``K`` orthonormal directions, using
    a Cayley-transform curvilinear search on the Stiefel manifold.  The
    returned directions are those along which the two distributions differ
    most -- i.e. the directions most worth Gaussianizing next.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Current data sample.
    Z_target : np.ndarray of shape (n_target, n_features)
        Reference sample (a Gaussian sample for GIS; data for SIG).
    n_directions : int
        Number of directions ``K`` to return (capped at ``n_features``).
    max_iter : int, default 100
        Maximum number of curvilinear-search iterations.
    lr : float, default 1.0
        Initial step size for the backtracking line search.
    tol : float, default 1e-6
        Stop when the objective improves by less than ``tol``.
    random_state : int or None, optional
        Seed controlling the initial frame and any subsampling.
    return_history : bool, default False
        If True, also return the list of objective values per iteration.

    Returns
    -------
    A : np.ndarray of shape (n_features, n_directions)
        Orthonormal projection directions (``A.T @ A == I_K``).
    history : list of float
        Objective value at each accepted iterate.  Only returned when
        ``return_history=True``.
    """
    rng = np.random.default_rng(random_state)
    d = X.shape[1]
    k = min(n_directions, d)

    n = min(X.shape[0], Z_target.shape[0])
    Xs = _subsample(X, n, rng)
    Zs = _subsample(Z_target, n, rng)

    A = random_orthogonal_directions(d, k, random_state=rng.integers(2**31))
    obj, grad = _kswd_objective_and_grad(A, Xs, Zs)
    history = [obj]
    eye = np.eye(d)

    for _ in range(max_iter):
        # Skew-symmetric generator; the "+" sign ascends the objective.
        W = grad @ A.T - A @ grad.T
        norm_w = np.linalg.norm(W)
        if norm_w < tol:
            break

        tau = lr
        accepted = False
        for _ls in range(25):
            cayley = np.linalg.solve(eye + (tau / 2.0) * W, eye - (tau / 2.0) * W)
            A_new = cayley @ A
            obj_new, grad_new = _kswd_objective_and_grad(A_new, Xs, Zs)
            if obj_new > obj:
                accepted = True
                break
            tau *= 0.5

        if not accepted:
            break

        improvement = obj_new - obj
        A, obj, grad = A_new, obj_new, grad_new
        history.append(obj)
        if improvement < tol:
            break

    # Re-orthonormalize to clean up any accumulated numerical drift.
    A, _ = np.linalg.qr(A)
    A = A[:, :k]

    if return_history:
        return A, history
    return A
