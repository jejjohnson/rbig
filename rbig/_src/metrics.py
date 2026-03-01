"""Information-theoretic metrics for RBIG."""
from typing import Optional

import numpy as np
from scipy.stats import norm

from rbig._src.marginal import entropy_marginal, bin_estimation


def information_reduction(
    x_data: np.ndarray,
    y_data: np.ndarray,
    tol_dimensions: Optional[float] = None,
    correction: bool = True,
) -> float:
    """Compute multi-information (total correlation) reduction I(X) - I(Y).

    Parameters
    ----------
    x_data : np.ndarray, shape (n_samples, n_features)
        Data before transformation.
    y_data : np.ndarray, shape (n_samples, n_features)
        Data after transformation.
    tol_dimensions : float or None
        Tolerance on the minimum multi-information difference.
    correction : bool
        Apply Shannon-Miller entropy correction.

    Returns
    -------
    I : float
        Information reduction (non-negative).
    """
    assert x_data.shape == y_data.shape, "x_data and y_data must have the same shape"
    n_samples, n_dimensions = x_data.shape

    if tol_dimensions is None or tol_dimensions == 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = float(np.interp(n_samples, xxx, yyy))

    hx = entropy_marginal(x_data, correction=correction)
    hy = entropy_marginal(y_data, correction=correction)

    I = float(np.sum(hy) - np.sum(hx))
    II = float(np.sqrt(np.sum((hy - hx) ** 2)))

    p = 0.25
    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0.0

    return I


def total_correlation(rbig_model) -> float:
    """Total correlation from a fitted AnnealedRBIG model.

    Parameters
    ----------
    rbig_model : AnnealedRBIG
        A fitted RBIG model.

    Returns
    -------
    tc : float
        Total correlation in nats (sum of residual info).
    """
    return float(np.sum(rbig_model.residual_info_))


def entropy_rbig(rbig_model) -> float:
    """Differential entropy from a fitted RBIG model.

    Computed as: H(X) = H(Z) + sum of log|det J| terms, where Z is Gaussian.

    Parameters
    ----------
    rbig_model : AnnealedRBIG
        A fitted RBIG model.

    Returns
    -------
    h : float
        Estimated differential entropy in bits.
    """
    n_features = rbig_model.gauss_data_.shape[1]
    # Gaussian entropy in bits: 0.5 * d * log2(2*pi*e)
    h_gauss = 0.5 * n_features * np.log2(2 * np.pi * np.e)
    # subtract total correlation (bits)
    tc = total_correlation(rbig_model)
    return float(h_gauss - tc)


def neg_entropy_normal(data: np.ndarray) -> np.ndarray:
    """Marginal negative entropy (negentropy) per dimension.

    J(X) = H(Gaussian with same variance) - H(X)

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    J : np.ndarray, shape (n_features,)
    """
    data = np.atleast_2d(data)
    n_samples, n_features = data.shape
    h_x = entropy_marginal(data)
    h_gauss = np.array([
        0.5 * np.log2(2 * np.pi * np.e * np.var(data[:, i]) + 1e-12)
        for i in range(n_features)
    ])
    return h_gauss - h_x


def histogram_entropy(data: np.ndarray, bins: str = "auto") -> float:
    """Histogram-based entropy estimate for 1-D data.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,)
    bins : str or int
        Bin specification for np.histogram.

    Returns
    -------
    h : float
        Entropy in nats.
    """
    data = np.asarray(data).ravel()
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_width = bin_edges[1] - bin_edges[0]
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)) + np.log(bin_width))


def mutual_information(X: np.ndarray, Y: np.ndarray, **rbig_kwargs) -> float:
    """Mutual information between X and Y estimated via RBIG.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, dx)
    Y : np.ndarray, shape (n_samples, dy)
    **rbig_kwargs
        Extra keyword arguments passed to AnnealedRBIG.

    Returns
    -------
    mi : float
        Estimated mutual information (>= 0).
    """
    from rbig._src.model import AnnealedRBIG

    X = np.atleast_2d(X) if X.ndim == 1 else X
    Y = np.atleast_2d(Y) if Y.ndim == 1 else Y
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    XY = np.concatenate([X, Y], axis=1)

    model_x = AnnealedRBIG(**rbig_kwargs).fit(X)
    model_y = AnnealedRBIG(**rbig_kwargs).fit(Y)
    model_xy = AnnealedRBIG(**rbig_kwargs).fit(XY)

    tc_x = total_correlation(model_x)
    tc_y = total_correlation(model_y)
    tc_xy = total_correlation(model_xy)

    h_x = entropy_rbig(model_x)
    h_y = entropy_rbig(model_y)
    h_xy = entropy_rbig(model_xy)

    mi = h_x + h_y - h_xy
    return float(max(mi, 0.0))
