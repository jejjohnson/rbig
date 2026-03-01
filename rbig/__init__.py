"""RBIG: Rotation-Based Iterative Gaussianization."""
from rbig._version import __version__
from rbig._src.model import AnnealedRBIG
from rbig._src.marginal import (
    fit_marginal_params,
    marginal_gaussianize,
    marginal_gaussianize_inverse,
    entropy_marginal,
)
from rbig._src.rotation import (
    PCARotation,
    RandomRotation,
    fit_rotation,
    apply_rotation,
    apply_rotation_inverse,
)
from rbig._src.metrics import (
    information_reduction,
    total_correlation,
    entropy_rbig,
    mutual_information,
)
from rbig._src.densities import (
    score_samples,
    log_prob,
)
from rbig._src.parametric import (
    HistogramUniformization,
    KDEUniformization,
    QuantileUniformization,
    fit_parametric_marginal,
    parametric_gaussianize,
    parametric_gaussianize_inverse,
)
from rbig._src.image import (
    ImageRBIG,
    image_gaussianize,
    extract_patches_2d,
    reconstruct_from_patches_2d,
)

__all__ = [
    "__version__",
    "AnnealedRBIG",
    "fit_marginal_params",
    "marginal_gaussianize",
    "marginal_gaussianize_inverse",
    "entropy_marginal",
    "PCARotation",
    "RandomRotation",
    "fit_rotation",
    "apply_rotation",
    "apply_rotation_inverse",
    "information_reduction",
    "total_correlation",
    "entropy_rbig",
    "mutual_information",
    "score_samples",
    "log_prob",
    "HistogramUniformization",
    "KDEUniformization",
    "QuantileUniformization",
    "fit_parametric_marginal",
    "parametric_gaussianize",
    "parametric_gaussianize_inverse",
    "ImageRBIG",
    "image_gaussianize",
    "extract_patches_2d",
    "reconstruct_from_patches_2d",
]
