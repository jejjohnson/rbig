"""Base classes for RBIG bijectors and transforms."""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Bijector(ABC):
    """Abstract base class for invertible transforms (bijectors)."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward transform: x → y."""

    @abstractmethod
    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform: y → x."""

    @abstractmethod
    def log_abs_det_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Log absolute value of the Jacobian determinant at x."""

    def forward_and_ladj(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward transform and log |det J|."""
        y = self.forward(x)
        ladj = self.log_abs_det_jacobian(x)
        return y, ladj


class MarginalBijector(Bijector):
    """Bijector applied independently to each dimension."""

    @abstractmethod
    def fit(self, x: np.ndarray) -> "MarginalBijector":
        """Fit the bijector to 1-D data x."""


class RotationTransform(ABC):
    """Abstract base class for rotation transforms (not invertible via PPF)."""

    @abstractmethod
    def fit(self, x: np.ndarray) -> "RotationTransform":
        """Fit the rotation to data x."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation."""

    @abstractmethod
    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse rotation."""


class RBIGLayer(ABC):
    """One layer of RBIG: marginal Gaussianization + rotation."""

    @abstractmethod
    def fit(self, x: np.ndarray) -> "RBIGLayer":
        """Fit the layer to data x."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward transform."""

    @abstractmethod
    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform."""

    @abstractmethod
    def log_abs_det_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Log |det J| summed over dimensions."""
