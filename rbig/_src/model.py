"""AnnealedRBIG: Rotation-Based Iterative Gaussianization model."""
from typing import Optional

import numpy as np

from rbig._src.marginal import (
    fit_marginal_params,
    marginal_gaussianize,
    marginal_gaussianize_inverse,
)
from rbig._src.metrics import information_reduction
from rbig._src.rotation import fit_rotation


class AnnealedRBIG:
    """Rotation-Based Iterative Gaussianization with annealed convergence.

    Parameters
    ----------
    n_layers : int, optional (default=1000)
        Maximum number of RBIG layers.
    rotation_type : {'PCA', 'random'}, optional (default='PCA')
        Rotation applied at each layer.
    n_quantiles : int, optional (default=1000)
        Number of quantile points for the marginal CDF/PPF.
    pdf_extension : int, optional (default=10)
        Percentage to extend the PDF support beyond data range.
    random_state : int or None, optional
        Seed for random rotations.
    verbose : int, optional (default=0)
        Verbosity level.
    tolerance : float or None, optional
        Stopping tolerance for information reduction.
    zero_tolerance : int, optional (default=60)
        Number of consecutive near-zero information layers before stopping.
    entropy_correction : bool, optional (default=True)
        Apply Shannon-Miller entropy correction.
    """

    def __init__(
        self,
        n_layers: int = 1000,
        rotation_type: str = "PCA",
        n_quantiles: int = 1000,
        pdf_extension: int = 10,
        random_state: Optional[int] = None,
        verbose: int = 0,
        tolerance: Optional[float] = None,
        zero_tolerance: int = 60,
        entropy_correction: bool = True,
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

    def fit(self, X: np.ndarray) -> "AnnealedRBIG":
        """Fit the RBIG model to data X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float, copy=True)
        n_samples, n_features = X.shape

        tolerance = self.tolerance
        if tolerance is None:
            xxx = np.logspace(2, 8, 7)
            yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
            tolerance = float(np.interp(n_samples, xxx, yyy))

        if self.zero_tolerance is not None:
            zero_tolerance = self.zero_tolerance
        else:
            zero_tolerance = self.n_layers + 1

        gauss_data = X.copy()
        gauss_params = []
        rotation_matrix = []
        residual_info = []

        for layer in range(self.n_layers):
            if self.verbose > 1:
                print(f"Layer {layer}")

            # -- Marginal Gaussianization --
            layer_params = []
            for dim in range(n_features):
                params_d = fit_marginal_params(
                    gauss_data[:, dim],
                    extension=self.pdf_extension,
                    n_quantiles=self.n_quantiles,
                )
                gauss_data[:, dim] = marginal_gaussianize(gauss_data[:, dim], params_d)
                layer_params.append(params_d)
            gauss_params.append(layer_params)

            # -- Rotation --
            R, _ = fit_rotation(
                gauss_data,
                rotation_type=self.rotation_type,
                random_state=self.random_state,
            )
            rotation_matrix.append(R)

            x_after = gauss_data @ R

            # -- Information reduction --
            info = information_reduction(
                gauss_data,
                x_after,
                correction=self.entropy_correction,
                tol_dimensions=tolerance,
            )
            residual_info.append(info)

            gauss_data = x_after

            # -- Stopping criteria --
            if layer >= zero_tolerance:
                recent = np.array(residual_info[-zero_tolerance:])
                if np.all(np.abs(recent) < tolerance):
                    # Trim last 50 layers
                    trim = min(50, len(gauss_params))
                    gauss_params = gauss_params[:-trim]
                    rotation_matrix = rotation_matrix[:-trim]
                    residual_info = residual_info[:-trim]
                    # Rebuild gauss_data without those last layers
                    gauss_data = self._apply_layers(X, gauss_params, rotation_matrix)
                    break

        self.gauss_params_ = gauss_params
        self.rotation_matrix_ = rotation_matrix
        self.residual_info_ = np.array(residual_info)
        self.gauss_data_ = gauss_data
        self.mutual_information_ = float(np.sum(self.residual_info_))
        self.n_layers_ = len(gauss_params)
        return self

    @staticmethod
    def _apply_layers(X: np.ndarray, gauss_params: list, rotation_matrix: list) -> np.ndarray:
        """Apply fitted layers to X."""
        X = np.array(X, dtype=float, copy=True)
        for layer_params, R in zip(gauss_params, rotation_matrix):
            for dim in range(X.shape[1]):
                X[:, dim] = marginal_gaussianize(X[:, dim], layer_params[dim])
            X = X @ R
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to Gaussian domain.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        Z : np.ndarray, shape (n_samples, n_features)
        """
        return self._apply_layers(X, self.gauss_params_, self.rotation_matrix_)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Transform Z back to the original domain.

        Parameters
        ----------
        Z : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        """
        Z = np.array(Z, dtype=float, copy=True)
        for layer_params, R in zip(reversed(self.gauss_params_), reversed(self.rotation_matrix_)):
            Z = Z @ R.T
            for dim in range(Z.shape[1]):
                Z[:, dim] = marginal_gaussianize_inverse(Z[:, dim], layer_params[dim])
        return Z

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log probability of each sample under the fitted model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        log_p : np.ndarray, shape (n_samples,)
        """
        from rbig._src.densities import log_prob
        return log_prob(X, self)

    def total_correlation(self) -> float:
        """Total correlation captured by the model (bits)."""
        from rbig._src.metrics import total_correlation as _tc
        return _tc(self)

    def entropy(self) -> float:
        """Differential entropy estimate (bits)."""
        from rbig._src.metrics import entropy_rbig as _h
        return _h(self)
