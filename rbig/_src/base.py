"""RBIG: Rotation-Based Iterative Gaussianization using ECDF-based marginals."""

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm, ortho_group
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils import check_array

from rbig._src.metrics import information_reduction

logger = logging.getLogger(__name__)

_BOUNDS_THRESHOLD = 1e-6


# ---------------------------------------------------------------------------
# Marginal Gaussianization helpers (ECDF-based)
# ---------------------------------------------------------------------------


def fit_marginal_params(X, extension=10, n_quantiles=1000):
    """Fit ECDF-based marginal Gaussianization parameters for a 1-D array.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
    extension : int, optional
        Percentage extension beyond the data range.
    n_quantiles : int, optional
        Number of support points for the interpolated CDF.

    Returns
    -------
    params : dict
    """
    n = len(X)

    x_min, x_max = float(X.min()), float(X.max())
    x_range = max(x_max - x_min, 1e-10)
    ext = (extension / 100.0) * x_range
    lb, ub = x_min - ext, x_max + ext

    sorted_x = np.sort(X)
    ecdf_y = np.arange(1, n + 1) / n

    x_ext = np.concatenate([[lb], sorted_x, [ub]])
    y_ext = np.concatenate([[0.0], ecdf_y, [1.0]])

    n_pts = max(min(n_quantiles, 2 * n), 50)
    support = np.linspace(lb, ub, n_pts)
    cdf_vals = np.interp(support, x_ext, y_ext)
    cdf_vals = np.clip(cdf_vals, _BOUNDS_THRESHOLD, 1.0 - _BOUNDS_THRESHOLD)

    pdf_vals = np.gradient(cdf_vals, support)
    pdf_vals = np.maximum(pdf_vals, 1e-15)

    dx = support[1] - support[0]
    pdf_support = np.concatenate([[support[0] - dx], support, [support[-1] + dx]])
    empirical_pdf = np.concatenate([[0.0], pdf_vals, [0.0]])

    return {
        "uniform_cdf_support": support,
        "uniform_cdf": cdf_vals,
        "empirical_pdf_support": pdf_support,
        "empirical_pdf": empirical_pdf,
        "_cdf_fn": interp1d(support, cdf_vals, fill_value="extrapolate"),
        "_ppf_fn": interp1d(cdf_vals, support, fill_value="extrapolate"),
    }


def marginal_gaussianize(X, params):
    """Apply marginal Gaussianization: X → Φ⁻¹(CDF(X))."""
    u = np.clip(params["_cdf_fn"](X), _BOUNDS_THRESHOLD, 1.0 - _BOUNDS_THRESHOLD)
    return norm.ppf(u)


def marginal_gaussianize_inverse(Z, params):
    """Invert marginal Gaussianization: Z → CDF⁻¹(Φ(Z))."""
    u = np.clip(norm.cdf(Z), _BOUNDS_THRESHOLD, 1.0 - _BOUNDS_THRESHOLD)
    return params["_ppf_fn"](u)


# ---------------------------------------------------------------------------
# RBIG — primary class
# ---------------------------------------------------------------------------


class RBIG(BaseEstimator, TransformerMixin):
    """Rotation-Based Iterative Gaussianization (RBIG).

    Iteratively applies marginal Gaussianization (via ECDF) and a linear
    rotation (PCA or random orthogonal) until the joint distribution is
    approximately Gaussian.

    Parameters
    ----------
    n_layers : int, optional (default=1000)
    rotation_type : {'PCA', 'random'}, optional (default='PCA')
    n_quantiles : int, optional (default=1000)
        Number of support points used when fitting the marginal CDF.
    pdf_extension : int, optional (default=10)
        Percentage extension beyond the data range for the CDF support.
    random_state : int or None, optional
    verbose : int, optional
    tolerance : float or None, optional
    zero_tolerance : int, optional (default=60)
    entropy_correction : bool, optional (default=True)
    """

    def __init__(
        self,
        n_layers=1000,
        rotation_type="PCA",
        n_quantiles=1000,
        pdf_extension=10,
        random_state=None,
        verbose=0,
        tolerance=None,
        zero_tolerance=60,
        entropy_correction=True,
    ):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.n_quantiles = n_quantiles
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
        self.entropy_correction = entropy_correction

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X):
        """Fit the RBIG model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = check_array(X, ensure_2d=True)
        self.X_fit_ = X.copy()
        gauss_data = X.copy()
        n_samples, n_dimensions = X.shape

        if self.tolerance is None:
            self.tolerance = self._get_information_tolerance(n_samples)

        self.gauss_params = []
        self.rotation_matrix = []
        residual_info = []

        for layer in range(self.n_layers):
            if self.verbose > 1:
                logger.debug("RBIG layer %d", layer + 1)

            layer_params = []
            for idim in range(n_dimensions):
                params = fit_marginal_params(
                    gauss_data[:, idim],
                    extension=self.pdf_extension,
                    n_quantiles=self.n_quantiles,
                )
                gauss_data[:, idim] = marginal_gaussianize(gauss_data[:, idim], params)
                layer_params.append(params)
            self.gauss_params.append(layer_params)

            gauss_data_prerotation = gauss_data.copy()

            if self.rotation_type.lower() == "pca":
                pca = PCA(random_state=self.random_state)
                gauss_data = pca.fit_transform(gauss_data)
                self.rotation_matrix.append(pca.components_.T)
            elif self.rotation_type == "random":
                rng = np.random.default_rng(self.random_state)
                R = ortho_group.rvs(n_dimensions, random_state=rng)
                gauss_data = np.dot(gauss_data, R)
                self.rotation_matrix.append(R)
            else:
                raise ValueError(
                    f"Unknown rotation_type: {self.rotation_type!r}. "
                    "Choose 'PCA' or 'random'."
                )

            ri = information_reduction(
                gauss_data, gauss_data_prerotation, self.tolerance
            )
            residual_info.append(ri)

            if self._stopping_criteria(layer, residual_info):
                break

        self.residual_info = np.array(residual_info)
        self.gauss_data = gauss_data
        self.mutual_information = float(self.residual_info.sum())
        self.n_layers = len(self.gauss_params)
        return self

    # ------------------------------------------------------------------
    # Transform / inverse
    # ------------------------------------------------------------------

    def transform(self, X):
        """Transform X to the Gaussian domain."""
        X = check_array(X, ensure_2d=True, copy=True)
        for layer_params, rotation in zip(self.gauss_params, self.rotation_matrix):
            for idim, params in enumerate(layer_params):
                X[:, idim] = marginal_gaussianize(X[:, idim], params)
            X = np.dot(X, rotation)
        return X

    def inverse_transform(self, X):
        """Transform X from the Gaussian domain back to input space."""
        X = check_array(X, ensure_2d=True, copy=True)
        for layer_params, rotation in zip(
            reversed(self.gauss_params), reversed(self.rotation_matrix)
        ):
            X = np.dot(X, rotation.T)
            for idim, params in enumerate(layer_params):
                X[:, idim] = marginal_gaussianize_inverse(X[:, idim], params)
        return X

    # ------------------------------------------------------------------
    # Density estimation
    # ------------------------------------------------------------------

    def score_samples(self, X):
        """Compute log-probability for each sample via change of variables.

        log p_X(x) = log p_Z(Z) + Σ_{l,i} [log φ(z_i^l) − log p̂(x_i^l)]

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples,)
        """
        X = check_array(X, ensure_2d=True, copy=True)
        n_samples = X.shape[0]
        log_det_jac = np.zeros(n_samples)

        for layer_params, rotation in zip(self.gauss_params, self.rotation_matrix):
            for idim, params in enumerate(layer_params):
                x_col = X[:, idim]

                log_px = np.log(
                    np.maximum(
                        interp1d(
                            params["empirical_pdf_support"],
                            params["empirical_pdf"],
                            fill_value="extrapolate",
                        )(x_col),
                        1e-15,
                    )
                )

                u = np.clip(
                    params["_cdf_fn"](x_col), _BOUNDS_THRESHOLD, 1.0 - _BOUNDS_THRESHOLD
                )
                z = norm.ppf(u)
                log_pz = norm.logpdf(z)

                log_det_jac += log_pz - log_px
                X[:, idim] = z

            X = np.dot(X, rotation)

        log_prob_z = norm.logpdf(X).sum(axis=1)
        return log_prob_z + log_det_jac

    # ------------------------------------------------------------------
    # Information-theoretic helpers
    # ------------------------------------------------------------------

    def total_correlation(self):
        """Return the total correlation accumulated across layers."""
        return float(self.residual_info.sum())

    def entropy(self, correction=None):
        """Return the differential entropy of X_fit_."""
        from rbig._src.marginal import entropy_marginal

        if correction is None:
            correction = self.entropy_correction
        return (
            entropy_marginal(self.X_fit_, correction=correction).sum()
            - self.mutual_information
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_information_tolerance(self, n_samples):
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        return float(interp1d(xxx, yyy, fill_value="extrapolate")(n_samples))

    def _stopping_criteria(self, layer, residual_info):
        if layer > self.zero_tolerance:
            aux = np.array(residual_info)
            if np.abs(aux[-self.zero_tolerance :]).sum() == 0:
                self.rotation_matrix = self.rotation_matrix[:-50]
                self.gauss_params = self.gauss_params[:-50]
                return True
        return False
