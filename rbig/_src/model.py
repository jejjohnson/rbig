import numpy as np
from rbig._src.base import RBIG as _RBIG


class AnnealedRBIG(_RBIG):
    """RBIG model with score_samples method."""

    def score_samples(self, X):
        """Compute log probability for each sample."""
        proba = self.predict_proba(X)
        return np.log(proba + 1e-10)


RBIG = AnnealedRBIG
