"""Generative Bayes classification on per-class RBIG densities (issue #129).

:class:`RBIGBayesClassifier` is nonlinear QDA: one
:class:`~rbig._src.model.AnnealedRBIG` flow per class provides an
arbitrarily shaped class-conditional density, and the decision rule is
``argmax_c [log p(x | y=c) + log π_c]`` on the exact ``score_samples``.

Cross-class comparison happens deep in each class's *tail* near decision
boundaries, so the per-class flows fit Gaussian-extended marginals
(``marginal_kwargs={"tail": "gaussian"}``) by default — with plain
clipping marginals the log-density plunges then flattens outside a
class's support (boundary notebooks 07/18) and the comparison
degenerates.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.model import AnnealedRBIG


class RBIGBayesClassifier(ClassifierMixin, BaseEstimator):
    """Generative Bayes classifier with RBIG class-conditional densities.

    Where LDA/QDA assume Gaussian classes, each class here gets an exact
    normalizing-flow density: curved and multi-modal class boundaries
    (rings, bananas) come for free, and ``predict_proba`` is derived
    from actual densities rather than a discriminative surface.  A low
    maximum class probability additionally signals an off-manifold input
    (the sample is in no class's high-density region).

    Parameters
    ----------
    n_layers : int, default 40
        Layer budget per class flow.
    tol : float or "auto", default 1e-5
        Convergence tolerance of the class flows.
    priors : {"empirical", "uniform"} or array-like, default "empirical"
        Class priors π_c: empirical frequencies, uniform, or explicit
        probabilities (must be positive and sum to 1).
    min_samples_per_class : int or None, default None
        Small-class guard threshold; ``None`` means ``20 · n_features``.
        Classes below it are fitted with a reduced layer budget and
        smooth KDE marginals (empirical marginals need n ≳ 20·d), with a
        warning.  A class needs at least 2 samples (the flow's own
        floor for estimating a marginal CDF).
    random_state : int or None, optional
        Seed for the class flows.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted unique class labels.
    class_models_ : list of AnnealedRBIG
        Per-class fitted flows, aligned with ``classes_``.
    log_priors_ : np.ndarray of shape (n_classes,)
        Log prior probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import RBIGBayesClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack(
    ...     [rng.standard_normal((80, 2)) + 3, rng.standard_normal((80, 2)) - 3]
    ... )
    >>> y = np.array(["a"] * 80 + ["b"] * 80)
    >>> clf = RBIGBayesClassifier(n_layers=5, random_state=0).fit(X, y)
    >>> clf.predict([[3.0, 3.0], [-3.0, -3.0]]).tolist()
    ['a', 'b']
    """

    def __init__(
        self,
        n_layers: int = 40,
        tol: float | str = 1e-5,
        priors: str | np.ndarray = "empirical",
        min_samples_per_class: int | None = None,
        random_state: int | None = None,
    ):
        self.n_layers = n_layers
        self.tol = tol
        self.priors = priors
        self.min_samples_per_class = min_samples_per_class
        self.random_state = random_state

    def _resolve_log_priors(self, counts: np.ndarray) -> np.ndarray:
        n_classes = counts.size
        if isinstance(self.priors, str):
            if self.priors == "empirical":
                return np.log(counts / counts.sum())
            if self.priors == "uniform":
                return np.full(n_classes, -np.log(n_classes))
            raise ValueError(
                f"priors must be 'empirical', 'uniform', or an array; "
                f"got {self.priors!r}."
            )
        priors = np.asarray(self.priors, dtype=float)
        if priors.shape != (n_classes,):
            raise ValueError(
                f"priors has shape {priors.shape}, expected ({n_classes},)."
            )
        if (priors <= 0).any() or not np.isclose(priors.sum(), 1.0):
            raise ValueError("priors must be positive and sum to 1.")
        return np.log(priors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RBIGBayesClassifier:
        """Fit one flow per class and the class priors.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Class labels (any dtype; stored in ``classes_``).

        Returns
        -------
        self : RBIGBayesClassifier
            The fitted classifier.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError(
                f"RBIGBayesClassifier needs samples from at least 2 classes; "
                f"y contains one class: {self.classes_[0]!r}."
            )
        d = X.shape[1]
        counts = np.bincount(y_idx, minlength=self.classes_.size)
        self.log_priors_ = self._resolve_log_priors(counts)

        min_n = (
            20 * d
            if self.min_samples_per_class is None
            else int(self.min_samples_per_class)
        )
        self.class_models_: list[AnnealedRBIG] = []
        for c, n_c in enumerate(counts):
            kwargs: dict = {
                "n_layers": self.n_layers,
                "tol": self.tol,
                "random_state": self.random_state,
                "marginal_kwargs": {"tail": "gaussian"},
            }
            if n_c < min_n:
                # KDE marginals smooth the sparse empirical CDF; the
                # shallow budget avoids overfitting n_c points.  But
                # gaussian_kde cannot handle zero-variance columns, so a
                # class with constant features keeps the empirical
                # marginals (which have a constant-feature fallback).
                has_constant = bool((X[y_idx == c].var(axis=0) == 0).any())
                marginal = "empirical" if has_constant else "KDE"
                warnings.warn(
                    f"Class {self.classes_[c]!r} has {n_c} samples "
                    f"(< {min_n}); using {marginal} marginals and a "
                    f"reduced layer budget for its density.",
                    UserWarning,
                    stacklevel=2,
                )
                kwargs = {
                    "n_layers": min(self.n_layers, 10),
                    "tol": self.tol,
                    "random_state": self.random_state,
                }
                if has_constant:
                    kwargs["marginal_kwargs"] = {"tail": "gaussian"}
                else:
                    kwargs["strategy"] = [["pca", "kde"]]
            self.class_models_.append(AnnealedRBIG(**kwargs).fit(X[y_idx == c]))
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """``log p(x | y=c) + log π_c`` for every class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        jll : np.ndarray of shape (n_samples, n_classes)
            Unnormalized class log-posteriors.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        scores = np.column_stack([m.score_samples(X) for m in self.class_models_])
        return scores + self.log_priors_[None, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most probable class per sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y : np.ndarray of shape (n_samples,)
            Predicted labels from ``classes_``.
        """
        jll = self._joint_log_likelihood(X)  # raises NotFittedError first
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Log class posteriors via a numerically stable log-softmax.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        log_proba : np.ndarray of shape (n_samples, n_classes)
            Rows log-sum-exp to 0.
        """
        jll = self._joint_log_likelihood(X)
        log_proba = jll - logsumexp(jll, axis=1, keepdims=True)
        # Floor at log(tiny): flow log-densities can differ by > 745 nats
        # across classes, and exp() must round-trip so that
        # log(predict_proba(X)) == predict_log_proba(X) exactly.
        return np.maximum(log_proba, np.log(np.finfo(np.float64).tiny))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class posterior probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Rows sum to 1.
        """
        return np.exp(self.predict_log_proba(X))

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean log posterior of the true labels — a model-selection metric.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.
        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        ll : float
            ``mean_i log p(y_i | x_i)`` in nats (higher is better).
        """
        log_proba = self.predict_log_proba(X)
        y = np.asarray(y).ravel()
        if y.shape[0] != log_proba.shape[0]:
            raise ValueError(
                f"y has {y.shape[0]} entries, X has {log_proba.shape[0]} rows."
            )
        idx = np.searchsorted(self.classes_, y)
        # Short-circuit keeps classes_[idx] safe: it only runs when every
        # idx is in range.
        if (idx >= self.classes_.size).any() or (self.classes_[idx] != y).any():
            raise ValueError("y contains labels not seen during fit.")
        return float(log_proba[np.arange(y.size), idx].mean())
