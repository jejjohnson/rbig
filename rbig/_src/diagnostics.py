"""Information-theoretic regression residual diagnostics (issue #130).

:class:`ResidualDiagnostics` wraps any regressor and quantifies the three
signatures of misspecification in nats:

- **Non-Gaussian residuals** ``J(ε)`` — negentropy via
  :func:`~rbig._src.metrics.entropy_quantile_spacing` (never bespoke
  entropy code in this module).
- **Missed structure** ``I(ε; X_j)`` per feature via
  :func:`~rbig._src.rbig_measures.estimate_mi` — the top offender names
  the feature needing an interaction/nonlinear term.
- **Heteroskedasticity** ``I(ε²; X)``.

The composite ``specification_score_`` is a *relative* measure for
ranking candidate models on the same data — never an absolute threshold.

Two estimator-level corrections (both bias fixes, not new theory):

1. **Quadratic block augmentation.** The RBIG MI estimator harvests
   only correlation-visible structure per rotation, so it is blind to
   even-symmetric dependence — exactly the missed-``x²``-term and
   ``|x|``-heteroskedasticity signatures these diagnostics exist for
   (the same mechanism as the pinned XOR limitation of
   ``RBIGMISelector``).  Each MI block therefore includes standardized
   squares: ``I(ε; (X_j, X_j²)) = I(ε; X_j)`` *exactly* (a deterministic
   function adds no information), but the squared channel makes the
   even dependence visible to covariance-driven rotations.
2. **Bias-matched negentropy baseline.** The spacing estimator's finite-
   sample bias (~0.1 nats at n ≈ 1500) is removed by evaluating the
   *same estimator* on a seeded Gaussian reference sample of equal size
   and variance, instead of using the analytic Gaussian entropy.
   Negative estimates (MI and J) are clamped to 0.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.metrics import entropy_quantile_spacing
from rbig._src.rbig_measures import estimate_mi


class ResidualDiagnostics(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """Wrap a regressor and quantify residual misspecification in nats.

    ``predict``/``score`` delegate to the wrapped estimator, so this is
    a drop-in replacement whose extra fitted attributes turn "eyeball
    the residual plot" into comparable numbers.

    Parameters
    ----------
    estimator : object
        The regressor to wrap (cloned at fit).
    n_layers_rbig : int, default 30
        Layer budget of the internal MI flows.
    tol_rbig : float or "auto", default 1e-4
        Convergence tolerance of the internal flows.
    cv : int or None, default None
        ``None`` diagnoses in-sample residuals (fast).  An integer uses
        out-of-fold residuals via ``cross_val_predict`` — recommended
        (``cv=5``) for flexible models, whose in-sample residuals are
        optimistically Gaussian.
    random_state : int or None, optional
        Seed for the internal flows.

    Attributes
    ----------
    estimator_ : object
        The fitted wrapped regressor.
    residuals_ : np.ndarray of shape (n_samples,)
        The diagnosed residuals (in-sample or out-of-fold per ``cv``).
    residual_negentropy_ : float
        ``J(ε) = H_gauss(Var ε) − H(ε)`` in nats (0 for Gaussian noise).
    residual_mi_ : np.ndarray of shape (n_features,)
        ``I(ε; X_j)`` per feature in nats.
    residual_mi_max_ : float
        The worst per-feature MI.
    heteroskedasticity_ : float
        ``I(ε²; X)`` in nats.
    specification_score_ : float
        ``0.3·J(ε) + 0.5·max_j I(ε;X_j) + 0.2·I(ε²;X)`` — relative
        composite for ranking models on the same data.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from rbig import ResidualDiagnostics
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 2))
    >>> y = X[:, 0] + 0.1 * rng.standard_normal(200)
    >>> diag = ResidualDiagnostics(
    ...     LinearRegression(), n_layers_rbig=3, random_state=0
    ... ).fit(X, y)
    >>> diag.predict(X).shape
    (200,)
    >>> diag.residual_negentropy_ >= 0.0
    True
    """

    def __init__(
        self,
        estimator,
        n_layers_rbig: int = 30,
        tol_rbig: float | str = 1e-4,
        cv: int | None = None,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.n_layers_rbig = n_layers_rbig
        self.tol_rbig = tol_rbig
        self.cv = cv
        self.random_state = random_state

    def _rbig_kwargs(self) -> dict:
        return {
            "n_layers": self.n_layers_rbig,
            "tol": self.tol_rbig,
            "patience": 5,
            "random_state": self.random_state,
        }

    @staticmethod
    def _augment(v: np.ndarray) -> np.ndarray:
        """Append standardized squares: makes even dependence visible.

        ``I(ε; (V, V²)) = I(ε; V)`` exactly, but the squared channel is
        correlated with even-symmetric structure the rotation-based MI
        estimator cannot otherwise see (module docstring, point 1).
        """
        sq = v**2
        std = sq.std(axis=0)
        sq = (sq - sq.mean(axis=0)) / np.where(std > 0, std, 1.0)
        return np.hstack([v, sq])  # (n, d) -> (n, 2d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ResidualDiagnostics:
        """Fit the wrapped regressor and compute the three diagnostics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features.
        y : np.ndarray of shape (n_samples,)
            Continuous target.

        Returns
        -------
        self : ResidualDiagnostics
            The fitted meta-estimator.
        """
        X, y = validate_data(self, X, y, y_numeric=True)
        self.estimator_ = clone(self.estimator).fit(X, y)
        if self.cv is None:
            eps = y - self.estimator_.predict(X)
        else:
            from sklearn.model_selection import cross_val_predict

            # Out-of-fold residuals: flexible models fit in-sample noise,
            # making their in-sample residuals optimistically Gaussian.
            eps = y - cross_val_predict(clone(self.estimator), X, y, cv=self.cv)
        self.residuals_ = eps

        var = float(eps.var())
        if var > 0:
            # Bias-matched baseline: same estimator on a same-size Gaussian
            # reference cancels the spacing estimator's finite-sample bias
            # (module docstring, point 2).
            rng = np.random.default_rng(self.random_state)
            ref = rng.standard_normal(eps.size) * np.sqrt(var)
            h_ref = entropy_quantile_spacing(ref)
            self.residual_negentropy_ = float(
                max(h_ref - entropy_quantile_spacing(eps), 0.0)
            )
        else:
            self.residual_negentropy_ = 0.0

        kw = self._rbig_kwargs()
        eps_col = eps[:, None]  # (n,) -> (n, 1) block for estimate_mi
        self.residual_mi_ = np.array(
            [
                max(float(estimate_mi(self._augment(X[:, [j]]), eps_col, **kw)), 0.0)
                for j in range(X.shape[1])
            ]
        )
        self.residual_mi_max_ = float(self.residual_mi_.max())
        self.heteroskedasticity_ = max(
            float(estimate_mi(self._augment(X), eps_col**2, **kw)), 0.0
        )
        self.specification_score_ = float(
            0.3 * self.residual_negentropy_
            + 0.5 * self.residual_mi_max_
            + 0.2 * self.heteroskedasticity_
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Delegate to the wrapped estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            The wrapped estimator's predictions.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Delegate to the wrapped estimator's ``score``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.
        y : np.ndarray of shape (n_samples,)
            True targets.

        Returns
        -------
        score : float
            The wrapped estimator's score (R² for regressors).
        """
        check_is_fitted(self)
        return self.estimator_.score(X, y)

    def diagnostic_report(self, feature_names=None) -> str:
        """Formatted diagnostic table, per-feature MI sorted descending.

        Parameters
        ----------
        feature_names : sequence of str or None, optional
            Names for the feature columns; defaults to ``x0, x1, ...``.

        Returns
        -------
        report : str
            Human-readable multi-line report.
        """
        check_is_fitted(self)
        d = self.residual_mi_.size
        names = (
            [f"x{j}" for j in range(d)]
            if feature_names is None
            else list(feature_names)
        )
        if len(names) != d:
            raise ValueError(f"feature_names has {len(names)} entries, expected {d}.")
        order = np.argsort(self.residual_mi_)[::-1]
        lines = [
            "Residual diagnostics (nats)",
            f"  negentropy J(eps):        {self.residual_negentropy_:.4f}",
            f"  heteroskedasticity:       {self.heteroskedasticity_:.4f}",
            f"  specification score:      {self.specification_score_:.4f}",
            "  per-feature I(eps; X_j), worst first:",
        ]
        lines += [f"    {names[j]:<20s} {self.residual_mi_[j]:.4f}" for j in order]
        return "\n".join(lines)
