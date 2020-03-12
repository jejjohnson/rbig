from abc import abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_array, check_random_state
from typing import Optional, Union


class UniformMixin(BaseEstimator, TransformerMixin):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError


class GaussMixin:
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError


class ScoreMixin(object):
    """Mixin for :func:`score` that returns mean of :func:`score_samples`."""

    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Return the mean log likelihood (or log(det(Jacobian))).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples and n_features
            is the number of features.
        y : None, default=None
            Not used but kept for compatibility.
        Returns
        -------
        log_likelihood : float
            Mean log likelihood data points in X.
        """
        return np.mean(self.score_samples(X, y))


class DensityMixin(object):
    """Mixin for :func:`sample` that returns the """

    @abstractmethod
    def abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def log_abs_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.log(self.abs_det_jacobian(X))

    @abstractmethod
    def pdf(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, X: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(X))

    @abstractmethod
    def sample(
        self, n_samples: int, random_state: Optional[Union[RandomState, int]] = None
    ) -> np.ndarray:
        raise NotImplementedError


def get_n_features(model):

    # Check if RBIG block
    if hasattr(model, "d_dimensions"):
        return model.d_dimensions

    # Check if Model with RBIG block as attribute
    elif hasattr(model.transforms[0], "d_dimensions"):
        return model.transforms[0].d_dimensions

    else:
        raise ValueError("No model density (or block density) has been found.")
