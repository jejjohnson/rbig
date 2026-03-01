"""RBIG Model - Rotation-Based Iterative Gaussianization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from rbig._src.marginal import MarginalGaussianize
from rbig._src.rotation import PCARotation


@dataclass
class RBIGLayer:
    """Single RBIG layer: marginal Gaussianization + rotation."""

    marginal: MarginalGaussianize = field(default_factory=MarginalGaussianize)
    rotation: PCARotation = field(default_factory=PCARotation)

    def fit(self, X: np.ndarray) -> RBIGLayer:
        Xm = self.marginal.fit_transform(X)
        self.rotation.fit(Xm)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xm = self.marginal.transform(X)
        return self.rotation.transform(Xm)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log |det J| for this layer at input X."""
        Xm = self.marginal.transform(X)
        return self.marginal.log_det_jacobian(X) + self.rotation.log_det_jacobian(Xm)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        Xr = self.rotation.inverse_transform(X)
        return self.marginal.inverse_transform(Xr)


class AnnealedRBIG:
    """Rotation-Based Iterative Gaussianization (RBIG).

    Iteratively Gaussianizes data via marginal Gaussianization and rotation
    (PCA or ICA) until the total correlation converges.

    Parameters
    ----------
    n_layers : int
        Number of RBIG layers. If -1, run until convergence.
    rotation : str
        Rotation method: "pca" or "ica".
    zero_tolerance : int
        Number of consecutive layers with no TC improvement before stopping.
    tol : float
        Tolerance for convergence check.
    random_state : int or None
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_layers: int = 100,
        rotation: str = "pca",
        zero_tolerance: int = 60,
        tol: float = 1e-5,
        random_state: int | None = None,
        strategy: list | None = None,
    ):
        self.n_layers = n_layers
        self.rotation = rotation
        self.zero_tolerance = zero_tolerance
        self.tol = tol
        self.random_state = random_state
        self.strategy = strategy

    def fit(self, X: np.ndarray) -> AnnealedRBIG:
        """Fit RBIG model to data X."""
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.layers_: list[RBIGLayer] = []
        self.tc_per_layer_: list[float] = []

        Xt = X.copy()
        self.log_det_train_ = np.zeros(n_samples)
        zero_count = 0

        for i in range(self.n_layers):
            layer = RBIGLayer(
                marginal=self._make_marginal(layer_index=i),
                rotation=self._make_rotation(layer_index=i),
            )
            layer.fit(Xt)
            self.log_det_train_ += layer.log_det_jacobian(Xt)
            Xt = layer.transform(Xt)
            self.layers_.append(layer)

            tc = self._total_correlation(Xt)
            self.tc_per_layer_.append(tc)

            if i > 0:
                delta = abs(self.tc_per_layer_[-2] - tc)
                if delta < self.tol:
                    zero_count += 1
                else:
                    zero_count = 0

            if zero_count >= self.zero_tolerance:
                break

        self.X_transformed_ = Xt
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to Gaussian space."""
        Xt = X.copy()
        for layer in self.layers_:
            Xt = layer.transform(Xt)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform from Gaussian space back to data space."""
        Xt = X.copy()
        for layer in reversed(self.layers_):
            Xt = layer.inverse_transform(Xt)
        return Xt

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log-likelihood of samples X using the change-of-variables formula.

        log p(x) = log p_Z(f(x)) + log|det J_f(x)|
        """
        Xt = X.copy()
        log_det_jac = np.zeros(X.shape[0])
        for layer in self.layers_:
            log_det_jac += layer.log_det_jacobian(Xt)
            Xt = layer.transform(Xt)
        log_pz = np.sum(stats.norm.logpdf(Xt), axis=1)
        return log_pz + log_det_jac

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood of samples X."""
        return float(np.mean(self.score_samples(X)))

    def entropy(self) -> float:
        """Entropy of the fitted distribution in nats.

        H(X) = -E_X[log p(x)] estimated from training data.
        """
        return float(-np.mean(self.score_samples_raw_()))

    def score_samples_raw_(self) -> np.ndarray:
        """Log-likelihood for stored training data (avoids recomputing layers)."""
        log_pz = np.sum(stats.norm.logpdf(self.X_transformed_), axis=1)
        return log_pz + self.log_det_train_

    def sample(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
        """Sample from the learned distribution."""
        rng = np.random.default_rng(random_state)
        Z = rng.standard_normal((n_samples, self.n_features_in_))
        return self.inverse_transform(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates exp(score_samples(X))."""
        return np.exp(self.score_samples(X))

    def _make_rotation(self, layer_index: int = 0):
        if self.strategy is not None:
            idx = layer_index % len(self.strategy)
            entry = self.strategy[idx]
            rotation_name = entry[0] if isinstance(entry, (list, tuple)) else entry
            return self._get_component(rotation_name, "rotation", layer_index)
        if self.rotation == "pca":
            return PCARotation(whiten=True)
        elif self.rotation == "ica":
            from rbig._src.rotation import ICARotation

            return ICARotation(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown rotation: {self.rotation}. Use 'pca' or 'ica'.")

    def _make_marginal(self, layer_index: int = 0):
        if self.strategy is not None:
            idx = layer_index % len(self.strategy)
            entry = self.strategy[idx]
            marginal_name = (
                entry[1] if isinstance(entry, (list, tuple)) else "gaussianize"
            )
            return self._get_component(marginal_name, "marginal", layer_index)
        return MarginalGaussianize()

    def _get_component(self, name: str, kind: str, seed: int = 0):
        """Instantiate a rotation or marginal component by name."""
        rng_seed = (self.random_state or 0) + seed
        if kind == "rotation":
            return self._make_rotation_by_name(name, rng_seed)
        return self._make_marginal_by_name(name, rng_seed)

    def _make_rotation_by_name(self, name: str, seed: int):
        if name == "pca":
            return PCARotation(whiten=True)
        if name == "ica":
            from rbig._src.rotation import ICARotation

            return ICARotation(random_state=seed)
        if name == "random":
            from rbig._src.rotation import RandomRotation

            return RandomRotation(random_state=seed)
        raise ValueError(f"Unknown rotation: {name!r}. Use 'pca', 'ica', or 'random'.")

    def _make_marginal_by_name(self, name: str, seed: int):
        if name in ("gaussianize", "empirical", None):
            return MarginalGaussianize()
        if name == "quantile":
            from rbig._src.marginal import QuantileGaussianizer

            return QuantileGaussianizer(random_state=seed)
        if name == "kde":
            from rbig._src.marginal import KDEGaussianizer

            return KDEGaussianizer()
        if name == "gmm":
            from rbig._src.marginal import GMMGaussianizer

            return GMMGaussianizer(random_state=seed)
        if name == "spline":
            from rbig._src.marginal import SplineGaussianizer

            return SplineGaussianizer()
        raise ValueError(
            f"Unknown marginal: {name!r}. Use 'gaussianize', 'quantile', 'kde', 'gmm', or 'spline'."
        )

    @staticmethod
    def _calculate_negentropy(X: np.ndarray) -> np.ndarray:
        """Negentropy of each marginal: J(x) = H(Gauss) - H(x) >= 0."""
        from rbig._src.densities import marginal_entropy

        gauss_h = 0.5 * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.var(X, axis=0))
        marg_h = marginal_entropy(X)
        return gauss_h - marg_h

    @staticmethod
    def _total_correlation(X: np.ndarray) -> float:
        """TC = sum of marginal entropies - joint entropy."""
        from rbig._src.densities import joint_entropy_gaussian, marginal_entropy

        marg_h = marginal_entropy(X)
        joint_h = joint_entropy_gaussian(X)
        return float(np.sum(marg_h) - joint_h)
