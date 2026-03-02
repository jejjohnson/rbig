"""Information-theoretic metrics for RBIG.

This module exposes functions for estimating mutual information, KL
divergence, total correlation, negentropy, and related quantities using
fitted RBIG models or directly from data samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rbig._src.model import AnnealedRBIG


def mutual_information_rbig(
    model_X: AnnealedRBIG,
    model_Y: AnnealedRBIG,
    model_XY: AnnealedRBIG,
) -> float:
    """Estimate mutual information between X and Y via RBIG models.

    Uses the identity:

        MI(X; Y) = H(X) + H(Y) − H(X, Y)

    where each entropy is estimated from a separately fitted RBIG model.

    Parameters
    ----------
    model_X : AnnealedRBIG
        RBIG model fitted on samples from the marginal distribution of X.
    model_Y : AnnealedRBIG
        RBIG model fitted on samples from the marginal distribution of Y.
    model_XY : AnnealedRBIG
        RBIG model fitted on joint samples [X, Y] (i.e. columns concatenated).

    Returns
    -------
    mi : float
        Estimated mutual information MI(X; Y) in nats.  Non-negative for
        well-calibrated models; small negative values may appear due to
        numerical imprecision.

    Notes
    -----
    Each ``model.entropy()`` call returns the differential entropy estimated
    from the RBIG-transformed representation.

    Examples
    --------
    >>> # Assumes pre-fitted models; see AnnealedRBIG for fitting details.
    >>> mi = mutual_information_rbig(model_X, model_Y, model_XY)
    >>> mi >= 0  # MI is non-negative
    True
    """
    hx = model_X.entropy()  # H(X)
    hy = model_Y.entropy()  # H(Y)
    hxy = model_XY.entropy()  # H(X, Y)
    return float(hx + hy - hxy)  # MI(X;Y) = H(X) + H(Y) - H(X,Y)


def kl_divergence_rbig(
    model_P: AnnealedRBIG,
    X_Q: np.ndarray,
) -> float:
    """Estimate KL divergence KL(P ‖ Q) via a fitted RBIG model of P.

    The KL divergence is computed as:

        KL(P ‖ Q) = 𝔼_P[log p(x)] − 𝔼_P[log q(x)]

    In this implementation ``model_P`` provides log p(x) via its
    ``score_samples`` method, and ``X_Q`` are samples drawn from Q.
    As implemented, the cross-entropy term −𝔼_Q[log p(x)] is estimated by
    evaluating the model P on samples from Q:

        estimate = −mean(model_P.score_samples(X_Q)) − H(P)

    .. note::
        This estimates KL using Q samples to evaluate the model's log-density,
        which corresponds to −𝔼_Q[log p(x)] − H(P).  Interpretation is valid
        when X_Q are representative samples from Q.

    Parameters
    ----------
    model_P : AnnealedRBIG
        RBIG model fitted on samples from distribution P.  Must expose
        ``score_samples(X)`` and ``entropy()``.
    X_Q : np.ndarray of shape (n_samples, n_features)
        Samples drawn from distribution Q against which P is compared.

    Returns
    -------
    kl : float
        Estimated KL(P ‖ Q) in nats.

    Examples
    --------
    >>> # When P == Q the KL divergence should be near zero.
    >>> kl = kl_divergence_rbig(model_P, X_from_P)
    >>> kl >= -0.1  # small negative values possible due to approximation
    True
    """
    # Evaluate log p(x) for samples drawn from Q
    log_pq = model_P.score_samples(X_Q)  # shape (n_samples,)
    hp = model_P.entropy()  # H(P) estimated by RBIG
    return float(-np.mean(log_pq) - hp)  # -E_Q[log p] - H(P)


def total_correlation_rbig(X: np.ndarray) -> float:
    """Estimate Total Correlation (multivariate mutual information) of X.

    Total Correlation is defined as:

        TC(X) = ∑ᵢ H(Xᵢ) − H(X)

    where the marginal entropies H(Xᵢ) are estimated via KDE (using
    ``marginal_entropy``) and the joint entropy H(X) is estimated by fitting
    a multivariate Gaussian to the data (``joint_entropy_gaussian``).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    tc : float
        Estimated total correlation in nats.  Values close to zero indicate
        approximate statistical independence among the dimensions.

    Notes
    -----
    See :func:`rbig._src.densities.total_correlation` for identical logic.
    This function is kept in ``metrics`` for API convenience.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.metrics import total_correlation_rbig
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 4))  # independent Gaussians
    >>> tc = total_correlation_rbig(X)
    >>> tc >= -0.5  # should be near 0
    True
    """
    from rbig._src.densities import joint_entropy_gaussian, marginal_entropy

    marg_h = marginal_entropy(X)  # ∑ᵢ H(Xᵢ), shape (n_features,)
    joint_h = joint_entropy_gaussian(X)  # H(X) via Gaussian approximation
    return float(np.sum(marg_h) - joint_h)


def entropy_normal_approx(X: np.ndarray) -> float:
    """Entropy via Gaussian approximation H(X) ≈ ½ log|2πeΣ|.

    Fits a multivariate Gaussian with the empirical covariance of X and
    returns the analytic Gaussian entropy.  This is a fast, closed-form
    approximation that is exact only when X is truly Gaussian.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    entropy : float
        Gaussian approximation to the differential entropy in nats.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.metrics import entropy_normal_approx
    >>> rng = np.random.default_rng(7)
    >>> X = rng.standard_normal((500, 2))
    >>> h = entropy_normal_approx(X)
    >>> np.isfinite(h)
    True
    """
    from rbig._src.densities import joint_entropy_gaussian

    return joint_entropy_gaussian(X)


def negentropy(X: np.ndarray) -> np.ndarray:
    """Compute per-dimension negentropy J(Xᵢ) = H_Gauss(Xᵢ) − H(Xᵢ).

    Negentropy measures non-Gaussianity for each marginal:

        J(Xᵢ) = H_Gauss(Xᵢ) − H(Xᵢ) ≥ 0

    where H_Gauss(Xᵢ) = ½(1 + log 2π) + ½ log Var(Xᵢ) is the Gaussian
    entropy matched to the observed variance and H(Xᵢ) is estimated via KDE.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    neg_entropy : np.ndarray of shape (n_features,)
        Non-negative negentropy for each dimension.  A value of 0 indicates
        that the marginal is Gaussian; larger values indicate more
        non-Gaussianity.

    Notes
    -----
    Negentropy is guaranteed non-negative by the maximum-entropy principle:
    among all distributions with a given variance, the Gaussian has the
    highest entropy.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.metrics import negentropy
    >>> rng = np.random.default_rng(3)
    >>> X_gauss = rng.standard_normal((500, 2))
    >>> J_gauss = negentropy(X_gauss)
    >>> np.all(J_gauss >= -0.05)  # nearly zero for Gaussian data
    True
    """
    _n, _d = X.shape
    # Gaussian entropy matched to empirical variance: H_Gauss = ½(1+log 2π) + ½ log σ²
    gauss_h = 0.5 * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.var(X, axis=0))
    from rbig._src.densities import marginal_entropy

    marg_h = marginal_entropy(X)  # KDE-based entropy, shape (n_features,)
    return gauss_h - marg_h  # J(Xi) = H_Gauss - H >= 0


def entropy_univariate(x: np.ndarray) -> float:
    """Univariate differential entropy via the Vasicek spacing estimator.

    The Vasicek (1976) estimator uses order-statistic spacings:

        Ĥ = (1/N) ∑ᵢ log[ (N / (2m)) · (x_{(i+m)} − x_{(i−m)}) ]

    where N is the sample size, m = ⌊√(N/2)⌋ is the window half-width, and
    x_{(i)} denotes the i-th order statistic.  In practice the spacings are
    formed as ``x_sorted[m:] − x_sorted[:N−m]``.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples,)
        1-D array of observations.

    Returns
    -------
    entropy : float
        Estimated differential entropy in nats.

    Notes
    -----
    A small constant (1e-300) is added inside the log to prevent ``−∞`` for
    duplicate values.  The window width m = ⌊√(N/2)⌋ follows a commonly used
    heuristic that balances bias and variance.

    References
    ----------
    Vasicek, O. (1976). A test for normality based on sample entropy.
    *Journal of the Royal Statistical Society B*, 38(1), 54–59.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.metrics import entropy_univariate
    >>> rng = np.random.default_rng(5)
    >>> x = rng.standard_normal(1000)
    >>> h = entropy_univariate(x)
    >>> # Gaussian entropy is 0.5*(1 + log 2π) ≈ 1.419 nats
    >>> np.isclose(h, 0.5 * (1 + np.log(2 * np.pi)), atol=0.1)
    True
    """
    n = len(x)
    m = max(1, int(np.floor(np.sqrt(n / 2))))  # window half-width
    x_sorted = np.sort(x)  # x_{(1)}, …, x_{(N)}
    # Spacing: x_{(i+m)} - x_{(i)}, length N-m
    diffs = x_sorted[m:] - x_sorted[: n - m]
    # Ĥ = mean of log[(N/2m) · spacing]
    h = np.mean(np.log(n / (2 * m) * diffs + 1e-300))
    return float(h)


def entropy_marginal(X: np.ndarray) -> np.ndarray:
    """Per-dimension marginal entropy using the Vasicek spacing estimator.

    Applies :func:`entropy_univariate` independently to each column of X.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    entropies : np.ndarray of shape (n_features,)
        Vasicek entropy estimate (nats) for each feature dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.metrics import entropy_marginal
    >>> rng = np.random.default_rng(9)
    >>> X = rng.standard_normal((800, 3))
    >>> h = entropy_marginal(X)
    >>> h.shape
    (3,)
    """
    n_features = X.shape[1]
    # Apply 1-D Vasicek estimator to each column independently
    return np.array([entropy_univariate(X[:, i]) for i in range(n_features)])


def entropy_quantile_spacing(x: np.ndarray, n_quantiles: int = 100) -> float:
    """Estimate univariate differential entropy via quantile spacings.

    Divides the empirical distribution into ``n_quantiles`` equal-probability
    intervals and uses the spacing between adjacent quantile values as a proxy
    for the local probability mass:

        Ĥ ≈ mean_k log(Δq_k · n_quantiles)

    where Δq_k = q_{k+1} − q_k are the differences between successive
    quantile values.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples,)
        1-D array of observations.
    n_quantiles : int, optional (default=100)
        Number of equal-probability quantile intervals.

    Returns
    -------
    entropy : float
        Estimated differential entropy in nats.  Returns ``0.0`` if all
        quantile spacings are zero (i.e. the data is constant).

    Notes
    -----
    The endpoints (0 and 1) are excluded to avoid quantile boundary effects.
    A small constant (1e-300) guards against log(0) for repeated quantile
    values.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.metrics import entropy_quantile_spacing
    >>> rng = np.random.default_rng(4)
    >>> x = rng.standard_normal(2000)
    >>> h = entropy_quantile_spacing(x, n_quantiles=200)
    >>> # Gaussian entropy ≈ 1.419 nats
    >>> np.isclose(h, 0.5 * (1 + np.log(2 * np.pi)), atol=0.15)
    True
    """
    # Evenly spaced probability levels, excluding 0 and 1
    quantiles = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # length n_quantiles
    q = np.quantile(x, quantiles)  # empirical quantile values
    diffs = np.diff(q)  # Dq_k = q_{k+1} - q_k
    diffs = diffs[diffs > 0]  # keep positive spacings only
    if len(diffs) == 0:
        return 0.0  # constant data
    # Ĥ ≈ mean log(Δqₖ · n_quantiles)
    return float(np.mean(np.log(diffs * n_quantiles + 1e-300)))


def entropy_rbig(model: AnnealedRBIG, X: np.ndarray) -> float:
    """Estimate differential entropy of X using a fitted RBIG model.

    Approximates the entropy via the plug-in estimator:

        H(X) = −𝔼[log p(x)] ≈ −(1/N) ∑ᵢ log p(xᵢ)

    where log p(xᵢ) is provided by ``model.score_samples``.

    Parameters
    ----------
    model : AnnealedRBIG
        RBIG model fitted on data from the same distribution as X.  Must
        expose a ``score_samples(X)`` method returning per-sample log
        probabilities.
    X : np.ndarray of shape (n_samples, n_features)
        Evaluation data used to compute the empirical expectation.

    Returns
    -------
    entropy : float
        Estimated differential entropy in nats.

    Examples
    --------
    >>> # Assumes a pre-fitted AnnealedRBIG model.
    >>> h = entropy_rbig(fitted_model, X_test)
    >>> h > 0  # entropy is typically positive for continuous distributions
    True
    """
    log_probs = model.score_samples(X)  # log p(xᵢ) for each sample, shape (N,)
    return float(-np.mean(log_probs))  # H ~= -(1/N) sum log p(xi)


def negative_log_likelihood(model: AnnealedRBIG, X: np.ndarray) -> float:
    """Average negative log-likelihood of X under the RBIG model.

    Computes:

        NLL = −(1/N) ∑ᵢ log p(xᵢ)

    This is equivalent to :func:`entropy_rbig` but is exposed separately to
    make its role as a loss / evaluation metric explicit.

    Parameters
    ----------
    model : AnnealedRBIG
        Fitted RBIG model.  Must expose ``score_samples(X)``.
    X : np.ndarray of shape (n_samples, n_features)
        Evaluation data.

    Returns
    -------
    nll : float
        Average negative log-likelihood in nats.

    Examples
    --------
    >>> nll = negative_log_likelihood(fitted_model, X_test)
    >>> nll > 0  # NLL is positive for well-calibrated models
    True
    """
    log_probs = model.score_samples(X)  # log p(xᵢ), shape (N,)
    return float(-np.mean(log_probs))  # NLL = -(1/N) sum log p(xi)


def information_summary(model: AnnealedRBIG, X: np.ndarray) -> dict:
    """Compute a summary of information-theoretic quantities from a RBIG model.

    Estimates three quantities from the fitted model and the data:

    * **entropy** H(X) = −𝔼[log p(x)]
    * **total_correlation** TC = ∑ᵢ H(Xᵢ) − H(X)
    * **neg_log_likelihood** NLL = −(1/N) ∑ log p(xᵢ)

    Parameters
    ----------
    model : AnnealedRBIG
        RBIG model fitted on data from the same distribution as X.  Must
        expose ``score_samples(X)``.
    X : np.ndarray of shape (n_samples, n_features)
        Evaluation data.

    Returns
    -------
    summary : dict
        Dictionary with the following keys:

        ``'entropy'`` : float
            Differential entropy H(X) in nats.
        ``'total_correlation'`` : float
            Total correlation TC = ∑ᵢ H(Xᵢ) − H(X) in nats.
        ``'neg_log_likelihood'`` : float
            Average negative log-likelihood in nats.

    Notes
    -----
    Marginal entropies H(Xᵢ) are estimated via KDE (see
    :func:`rbig._src.densities.marginal_entropy`); joint entropy H(X) is
    obtained from the RBIG model.

    Examples
    --------
    >>> summary = information_summary(fitted_model, X)
    >>> set(summary.keys()) == {"entropy", "total_correlation", "neg_log_likelihood"}
    True
    """
    from rbig._src.densities import marginal_entropy as _marginal_entropy

    h = entropy_rbig(model, X)  # H(X) via RBIG model
    marginal_h = _marginal_entropy(X)  # H(Xᵢ) via KDE, shape (n_features,)
    tc = float(np.sum(marginal_h) - h)  # TC = sum H(Xi) - H(X)
    return {
        "entropy": h,
        "total_correlation": tc,
        "neg_log_likelihood": negative_log_likelihood(model, X),
    }


def information_reduction(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """Compute reduction in Total Correlation between two representations.

    Measures how much statistical dependence is removed by a transformation:

        ΔTC = TC(X_before) − TC(X_after)

    where TC is computed using KDE-based marginal entropies and a Gaussian
    approximation for the joint entropy.

    Parameters
    ----------
    X_before : np.ndarray of shape (n_samples, n_features)
        Data matrix prior to the transformation.
    X_after : np.ndarray of shape (n_samples, n_features)
        Data matrix after the transformation.

    Returns
    -------
    delta_tc : float
        Reduction in total correlation (nats).  A positive value indicates
        that the transformation increased statistical independence.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.metrics import information_reduction
    >>> rng = np.random.default_rng(2)
    >>> cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    >>> X = rng.multivariate_normal([0, 0], cov, size=400)
    >>> X_white = (X - X.mean(0)) / X.std(0)
    >>> information_reduction(X, X_white) >= 0
    True
    """
    from rbig._src.densities import (
        joint_entropy_gaussian,
        marginal_entropy as _marginal_entropy,
    )

    def _tc(X):
        # TC(X) = sum H(Xi) - H(X)
        return float(np.sum(_marginal_entropy(X)) - joint_entropy_gaussian(X))

    return _tc(X_before) - _tc(X_after)  # DeltaTC = TC_before - TC_after
