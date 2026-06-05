"""Shared convergence / early-stopping criteria for iterative Gaussianization.

Iterative Gaussianization models (``AnnealedRBIG``, ``GIS``, ``SIG``) stack
layers greedily until the data is "Gaussian enough".  Different signals can
drive that decision, trading principle against cost:

* ``"log_likelihood"`` -- validation log-likelihood under the
  change-of-variables formula.  Most principled; needs the accumulated
  log-determinant each iteration.
* ``"swd"`` -- K-sliced Wasserstein distance between the current
  representation and a standard Gaussian.  Cheap proxy for the remaining
  non-Gaussianity.
* ``"total_correlation"`` -- residual total correlation of the current
  representation (the classic RBIG signal).

:class:`StoppingCriterion` tracks the chosen metric across iterations and
signals when training should stop, applying a ``patience`` window so that a
few non-improving layers do not halt training prematurely.

References
----------
Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
537-549.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from rbig._src.stiefel import wasserstein_1d

_VALID_METRICS = ("log_likelihood", "swd", "total_correlation")
# Whether a larger metric value indicates a better (more converged) model.
_GREATER_IS_BETTER = {
    "log_likelihood": True,
    "swd": False,
    "total_correlation": False,
}


class StoppingCriterion:
    """Track a convergence metric and signal early stopping.

    Parameters
    ----------
    metric : {"log_likelihood", "swd", "total_correlation"}, default "log_likelihood"
        Which signal drives stopping.
    patience : int, default 10
        Number of consecutive non-improving updates tolerated before
        :meth:`update` returns ``True``.
    min_delta : float, default 1e-4
        Minimum change in the metric counted as an improvement.
    validation_fraction : float, default 0.2
        Fraction of data held out by :meth:`split` for monitoring.  Set to
        ``0`` to monitor on the training data itself.
    n_projections : int, default 50
        Number of random 1D projections used by the ``"swd"`` metric.
    random_state : int or None, optional
        Seed controlling the validation split and the ``"swd"`` reference
        sample / projections.

    Attributes
    ----------
    history_ : list of float
        Metric value recorded at each :meth:`update`.
    best_score_ : float or None
        Best metric value seen so far (in the metric's natural orientation).
    best_iter_ : int
        Index of the update that achieved ``best_score_``.
    """

    def __init__(
        self,
        metric: str = "log_likelihood",
        patience: int = 10,
        min_delta: float = 1e-4,
        validation_fraction: float = 0.2,
        n_projections: int = 50,
        random_state: int | None = None,
    ):
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Choose from {_VALID_METRICS}."
            )
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.n_projections = n_projections
        self.random_state = random_state
        self.reset()

    def reset(self) -> None:
        """Clear all tracked state so the criterion can be reused."""
        self.history_: list[float] = []
        self.best_score_: float | None = None
        self.best_iter_: int = -1
        self._n_bad_: int = 0
        self._iter_: int = -1
        self._rng_ = np.random.default_rng(self.random_state)

    def split(
        self, X: np.ndarray, random_state: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split ``X`` into train / validation parts.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to split.
        random_state : int or None, optional
            Overrides the criterion's ``random_state`` for this split.

        Returns
        -------
        X_train, X_val : np.ndarray
            The two partitions.  When ``validation_fraction == 0`` both are
            the full array.
        """
        if self.validation_fraction <= 0:
            return X, X
        from sklearn.model_selection import train_test_split

        seed = self.random_state if random_state is None else random_state
        return train_test_split(
            X, test_size=self.validation_fraction, random_state=seed
        )

    def _score(self, X_val: np.ndarray, log_det: np.ndarray | None) -> float:
        """Compute the configured metric on the current representation."""
        if self.metric == "log_likelihood":
            log_pz = np.sum(stats.norm.logpdf(X_val), axis=1)
            if log_det is None:
                log_det = np.zeros(X_val.shape[0])
            return float(np.mean(log_pz + log_det))

        if self.metric == "swd":
            d = X_val.shape[1]
            ref = self._rng_.standard_normal(X_val.shape)
            dirs = self._rng_.standard_normal((d, self.n_projections))
            dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
            proj_x = X_val @ dirs
            proj_z = ref @ dirs
            return float(
                np.mean(
                    [
                        wasserstein_1d(proj_x[:, j], proj_z[:, j])
                        for j in range(self.n_projections)
                    ]
                )
            )

        # total_correlation
        from rbig._src.densities import joint_entropy_gaussian, marginal_entropy

        marg = marginal_entropy(X_val)
        joint = joint_entropy_gaussian(X_val)
        return float(np.sum(marg) - joint)

    def update(self, X_val: np.ndarray, log_det: np.ndarray | None = None) -> bool:
        """Record the metric for the current iterate and test for stopping.

        Parameters
        ----------
        X_val : np.ndarray of shape (n_samples, n_features)
            The validation data **after** passing through all layers fitted
            so far (i.e. the current latent representation).
        log_det : np.ndarray of shape (n_samples,) or None, optional
            Accumulated per-sample log-determinant on ``X_val``.  Required
            only for the ``"log_likelihood"`` metric.

        Returns
        -------
        should_stop : bool
            ``True`` when the metric has failed to improve for ``patience``
            consecutive updates.
        """
        self._iter_ += 1
        score = self._score(X_val, log_det)
        self.history_.append(score)

        greater_is_better = _GREATER_IS_BETTER[self.metric]
        if self.best_score_ is None:
            improved = True
        elif greater_is_better:
            improved = score > self.best_score_ + self.min_delta
        else:
            improved = score < self.best_score_ - self.min_delta

        if improved:
            self.best_score_ = score
            self.best_iter_ = self._iter_
            self._n_bad_ = 0
        else:
            self._n_bad_ += 1

        return self._n_bad_ >= self.patience

    def best_score(self) -> float:
        """Return the best metric value observed so far."""
        if self.best_score_ is None:
            raise ValueError("No updates recorded yet; call update() first.")
        return self.best_score_
