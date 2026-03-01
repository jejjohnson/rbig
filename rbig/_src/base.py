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
