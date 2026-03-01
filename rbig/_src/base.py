from abc import ABC, abstractmethod

import numpy as np


class BaseTransform(ABC):
    """Abstract base class for all RBIG transforms."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseTransform":
        """Fit the transform to data X."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transform to data X."""
        ...

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply inverse transform to data X."""
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data X."""
        return self.fit(X).transform(X)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log absolute determinant of Jacobian at X."""
        raise NotImplementedError


class BaseITMeasure(ABC):
    """Abstract base class for IT measures."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray | None = None) -> "BaseITMeasure": ...

    @abstractmethod
    def score(self) -> float: ...


class Bijector(ABC):
    """Abstract bijector with fit/transform/inverse_transform/get_log_det_jacobian."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Bijector": ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Alias for get_log_det_jacobian for compatibility with RBIGLayer."""
        return self.get_log_det_jacobian(X)


class MarginalBijector(Bijector):
    """Abstract bijector for marginal (per-dimension) transforms."""

    # inherits all abstract methods


class RotationBijector(Bijector):
    """Abstract bijector for rotation transforms (log det = 0 by default)."""

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class CompositeBijector(Bijector):
    """Chains a list of bijectors."""

    def __init__(self, bijectors: list):
        self.bijectors = bijectors

    def fit(self, X: np.ndarray) -> "CompositeBijector":
        Xt = X.copy()
        for b in self.bijectors:
            Xt = b.fit_transform(Xt)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.copy()
        for b in self.bijectors:
            Xt = b.transform(Xt)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.copy()
        for b in reversed(self.bijectors):
            Xt = b.inverse_transform(Xt)
        return Xt

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        Xt = X.copy()
        log_det = np.zeros(X.shape[0])
        for b in self.bijectors:
            log_det += b.get_log_det_jacobian(Xt)
            Xt = b.transform(Xt)
        return log_det
