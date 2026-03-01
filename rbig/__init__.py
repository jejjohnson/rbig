from rbig._src.base import RBIG as LegacyRBIG
from rbig._src.model import AnnealedRBIG
from rbig._src.mi import RBIGMI
from rbig._src.kld import RBIGKLD
from rbig._src.metrics import information_reduction, neg_entropy_normal
from rbig._src.marginal import entropy_marginal
from rbig._src.jacobian import compute_jacobian
from rbig._version import __version__

# AnnealedRBIG is the primary RBIG implementation (from the snippets).
# LegacyRBIG (histogram-based) is retained for salvaged features such as
# compute_jacobian on older models.
RBIG = AnnealedRBIG

__all__ = [
    "RBIG",
    "AnnealedRBIG",
    "LegacyRBIG",
    "RBIGMI",
    "RBIGKLD",
    "information_reduction",
    "neg_entropy_normal",
    "entropy_marginal",
    "compute_jacobian",
    "__version__",
]
