"""RBIG: Rotation-Based Iterative Gaussianization."""

from rbig._src.base import BaseITMeasure, BaseTransform
from rbig._src.densities import (
    entropy_reduction,
    gaussian_entropy,
    joint_entropy_gaussian,
    marginal_entropy,
    total_correlation,
)
from rbig._src.marginal import (
    MarginalGaussianize,
    MarginalKDEGaussianize,
    MarginalUniformize,
)
from rbig._src.metrics import (
    entropy_normal_approx,
    kl_divergence_rbig,
    mutual_information_rbig,
    negentropy,
    total_correlation_rbig,
)
from rbig._src.model import AnnealedRBIG, RBIGLayer
from rbig._src.parametric import BoxCoxTransform, LogitTransform, QuantileTransform
from rbig._src.rotation import ICARotation, PCARotation
from rbig._version import __version__

__all__ = [
    "AnnealedRBIG",
    "BaseITMeasure",
    "BaseTransform",
    "BoxCoxTransform",
    "ICARotation",
    "LogitTransform",
    "MarginalGaussianize",
    "MarginalKDEGaussianize",
    "MarginalUniformize",
    "PCARotation",
    "QuantileTransform",
    "RBIGLayer",
    "__version__",
    "entropy_normal_approx",
    "entropy_reduction",
    "gaussian_entropy",
    "joint_entropy_gaussian",
    "kl_divergence_rbig",
    "marginal_entropy",
    "mutual_information_rbig",
    "negentropy",
    "total_correlation",
    "total_correlation_rbig",
]
