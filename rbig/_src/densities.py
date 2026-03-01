"""Density estimation for RBIG models."""
import numpy as np
from scipy.stats import norm

from rbig._src.marginal import marginal_gaussianize, log_pdf_marginal


def score_samples_gaussian(Z: np.ndarray) -> np.ndarray:
    """Log probability under a standard Gaussian for each sample.

    Parameters
    ----------
    Z : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    log_p : np.ndarray, shape (n_samples,)
    """
    return np.sum(norm.logpdf(Z), axis=1)


def marginal_log_pdf(x: np.ndarray, params: dict) -> np.ndarray:
    """Log PDF of a single marginal at x using stored empirical PDF.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
    params : dict
        Fitted marginal params from fit_marginal_params.

    Returns
    -------
    log_p : np.ndarray, shape (n_samples,)
    """
    return log_pdf_marginal(x, params)


def score_samples(
    X: np.ndarray,
    gauss_params: list,
    rotation_matrix: list,
) -> np.ndarray:
    """Log probability of X under the RBIG density model.

    Uses the change-of-variables formula:
        log p_X(x) = log p_Z(z) + sum_layers sum_dims log|det J_i(x)|

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    gauss_params : list of list of dict
        Per-layer, per-dimension marginal params.
    rotation_matrix : list of np.ndarray
        Per-layer rotation matrices.

    Returns
    -------
    log_p : np.ndarray, shape (n_samples,)
    """
    X = np.array(X, dtype=float, copy=True)
    n_samples, n_features = X.shape
    log_det_jac = np.zeros(n_samples)

    for layer_params, R in zip(gauss_params, rotation_matrix):
        for dim in range(n_features):
            col = X[:, dim]
            params_d = layer_params[dim]

            # Log of marginal PDF (uniform density in original space)
            log_pdf = log_pdf_marginal(col, params_d)

            # Gaussianize
            z_dim = marginal_gaussianize(col, params_d)

            # Log of Gaussian PDF at z_dim
            log_gauss = norm.logpdf(z_dim)

            # Log |det J| contribution for this dim: log p_X - log p_Z
            log_det_jac += log_pdf - log_gauss

            X[:, dim] = z_dim

        # Apply rotation (orthogonal, so |det R| = 1, no Jacobian contribution)
        X = X @ R

    # Final Gaussian log-probability
    log_p_z = score_samples_gaussian(X)
    return log_p_z + log_det_jac


def log_prob(X: np.ndarray, rbig_model) -> np.ndarray:
    """Log probability of X using a fitted AnnealedRBIG model.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    rbig_model : AnnealedRBIG
        A fitted RBIG model.

    Returns
    -------
    log_p : np.ndarray, shape (n_samples,)
    """
    return score_samples(X, rbig_model.gauss_params_, rbig_model.rotation_matrix_)
