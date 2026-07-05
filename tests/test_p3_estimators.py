"""P3 estimator batch: RBIGKMeans (#127), RBIGBayesClassifier (#129),
ResidualDiagnostics (#130), RBIGFairTransformer (#131).

Fast tests cover API contracts and behavior; ``@pytest.mark.statistical``
tests pin the quality claims (and the honest limitations) with seed-mean
gates.  Every numeric gate documents the observed values it was derived
from.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, brier_score_loss, silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from rbig import (
    RBIGBayesClassifier,
    RBIGFairTransformer,
    RBIGKMeans,
    ResidualDiagnostics,
    make_bimodal,
    make_variance_leak,
)

# ── zoo helpers ──────────────────────────────────────────────────────────────


def _elongated(
    n_per: int, seed: int, gap: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Two parallel elongated clusters raw KMeans cuts across."""
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.standard_normal((n_per, 2)) * [3.0, 0.3] + [0, +gap],
            rng.standard_normal((n_per, 2)) * [3.0, 0.3] + [0, -gap],
        ]
    )
    return X, np.repeat([0, 1], n_per)


def _three_elongated(n_per: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.standard_normal((n_per, 2)) * [2.0, 0.3] + [0, 4.0 * k] for k in range(3)]
    )


def _rings2(n_per: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two concentric rings — equal means, LDA is at chance."""
    rng = np.random.default_rng(seed)
    r = np.concatenate(
        [
            1.0 + 0.15 * rng.standard_normal(n_per),
            2.2 + 0.15 * rng.standard_normal(n_per),
        ]
    )
    th = rng.uniform(0, 2 * np.pi, 2 * n_per)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return X, np.repeat([0, 1], n_per)


def _bananas2(n_per: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two opposed curved (banana) classes — breaks the QDA quadric."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n_per)
    y0 = 0.5 * x0**2 - 1.5 + 0.4 * rng.standard_normal(n_per)
    x1 = rng.standard_normal(n_per)
    y1 = -0.5 * x1**2 + 1.5 + 0.4 * rng.standard_normal(n_per)
    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    return X, np.repeat([0, 1], n_per)


def _util_fair_data(n_per: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """I(A; Y) = 0 by construction: Y from A-independent signal dims,
    A leaking only through the variance of separate leak dims."""
    rng = np.random.default_rng(seed)
    A = np.repeat([0, 1], n_per)
    S = rng.standard_normal((2 * n_per, 2))  # signal dims: (2n, 2)
    y = (S[:, 0] + 0.5 * S[:, 1] + 0.5 * rng.standard_normal(2 * n_per) > 0).astype(int)
    L = np.where(A[:, None] == 1, 2.0, 1.0) * rng.standard_normal((2 * n_per, 2))
    return np.hstack([S, L]), y, A


def _a_pred_auc(X: np.ndarray, A: np.ndarray) -> float:
    """A-predictability: GBM ROC-AUC for recovering A (0.5 = removed)."""
    return float(
        cross_val_score(
            GradientBoostingClassifier(random_state=0), X, A, cv=3, scoring="roc_auc"
        ).mean()
    )


# ── RBIGKMeans (#127) ────────────────────────────────────────────────────────


def test_kmeans_fit_predict_consistency():
    X, _ = _elongated(120, 0)
    km = RBIGKMeans(n_clusters=2, n_layers_rbig=3, n_init=2, random_state=0).fit(X)
    assert (km.predict(X) == km.labels_).all()
    assert (
        RBIGKMeans(n_clusters=2, n_layers_rbig=3, n_init=2, random_state=0).fit_predict(
            X
        )
        == km.labels_
    ).all()
    assert km.centroids_z_.shape == (2, 2)
    assert km.centroids_x_.shape == (2, 2)
    assert np.isfinite(km.centroids_x_).all()
    assert km.transform(X).shape == X.shape


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
def test_kmeans_proba_contract(method):
    X, _ = _elongated(150, 1)
    km = RBIGKMeans(
        n_clusters=2, n_layers_rbig=3, method=method, n_init=2, random_state=0
    ).fit(X)
    p = km.predict_proba(X)
    assert p.shape == (X.shape[0], 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == km.predict(X)).all()
    scores = km.score_samples(X)
    assert scores.shape == (X.shape[0],)
    assert np.isfinite(scores).all()


def test_kmeans_validation():
    X, _ = _elongated(30, 0)
    with pytest.raises(ValueError, match="method"):
        RBIGKMeans(method="dbscan").fit(X)
    with pytest.raises(ValueError, match="n_clusters"):
        RBIGKMeans(n_clusters=100).fit(X)


def test_kmeans_pipeline():
    X, _ = _elongated(120, 2)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("km", RBIGKMeans(n_clusters=2, n_layers_rbig=3, n_init=2, random_state=0)),
        ]
    )
    labels = pipe.fit_predict(X)
    assert labels.shape == (X.shape[0],)


@pytest.mark.statistical
def test_kmeans_ari_improvement_on_elongated():
    """RBIGKMeans beats raw KMeans by >= 0.15 ARI on elongated clusters.

    Observed (5-seed means, n=200/cluster, n_layers_rbig=3): rbig ~ 0.48,
    raw ~ 0.001 — raw KMeans cuts across the long axis; the flow rounds
    the clusters first.
    """
    aris, raws = [], []
    for seed in range(5):
        X, y = _elongated(200, seed)
        km = RBIGKMeans(n_clusters=2, n_layers_rbig=3, n_init=3, random_state=0).fit(X)
        aris.append(adjusted_rand_score(y, km.labels_))
        raws.append(
            adjusted_rand_score(y, KMeans(2, n_init=3, random_state=0).fit_predict(X))
        )
    assert np.mean(aris) >= np.mean(raws) + 0.15, (aris, raws)
    assert np.mean(aris) >= 0.25, aris


@pytest.mark.statistical
def test_kmeans_over_gaussianization_limitation_pin():
    """Pinned limitation: Gaussianize-then-cluster LOSES to raw KMeans on
    well-separated axis-aligned modes.

    The issue #127 sketch expected ARI(n_layers=50) < ARI(n_layers=5);
    in reality the damage is immediate, not depth-driven: the very first
    marginal Gaussianization maps the bimodal marginal to a unimodal
    Gaussian (compressing the mode gap), and early stopping makes deep
    budgets converge to the same flow as shallow ones.  Observed 5-seed
    mean ARI on ``make_bimodal(separation=5)``: raw ~ 0.97, rbig ~ 0.2-0.5
    at every depth in {1, 2, 3, 5, 10}.  This is exactly why
    ``n_layers_rbig`` defaults small and the docstring warns about it.
    """
    rbig_ari, raw_ari = [], []
    for seed in range(5):
        X, meta = make_bimodal(n_samples=600, separation=5.0, seed=seed)
        y = meta["labels"]
        km = RBIGKMeans(n_clusters=2, n_layers_rbig=10, n_init=3, random_state=0).fit(X)
        rbig_ari.append(adjusted_rand_score(y, km.labels_))
        raw_ari.append(
            adjusted_rand_score(y, KMeans(2, n_init=3, random_state=0).fit_predict(X))
        )
    assert np.mean(raw_ari) >= 0.9, raw_ari
    assert np.mean(rbig_ari) <= 0.7, rbig_ari  # documents the failure mode


@pytest.mark.statistical
def test_kmeans_silhouette_in_z_prefers_true_k():
    """Z-space silhouette prefers k=3 over k=2 on 3 elongated clusters.

    Observed margin ~ 0.02-0.03 per seed (5-seed mean gate 0.005); the
    k=4 comparison is within noise and deliberately not asserted.
    """
    margins = []
    for seed in range(5):
        X = _three_elongated(150, seed)
        sils = {}
        for k in (2, 3):
            km = RBIGKMeans(
                n_clusters=k, n_layers_rbig=3, n_init=3, random_state=0
            ).fit(X)
            sils[k] = silhouette_score(km.transform(X), km.labels_)
        margins.append(sils[3] - sils[2])
    assert np.mean(margins) >= 0.005, margins


# ── RBIGBayesClassifier (#129) ───────────────────────────────────────────────


def test_classifier_basic_contract():
    X, y = _rings2(150, 0)
    clf = RBIGBayesClassifier(n_layers=5, random_state=0).fit(X, y)
    assert list(clf.classes_) == [0, 1]
    assert clf.score(X, y) >= 0.95  # in-sample rings, observed 1.0
    p = clf.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.exp(clf.predict_log_proba(X)), p)
    assert clf.log_likelihood(X, y) <= 0.0


def test_classifier_string_labels():
    X, y = _rings2(120, 1)
    ys = np.where(y == 0, "inner", "outer")
    clf = RBIGBayesClassifier(n_layers=4, random_state=0).fit(X, ys)
    assert set(clf.predict(X)) <= {"inner", "outer"}
    with pytest.raises(ValueError, match="not seen"):
        clf.log_likelihood(X, np.array(["nope"] * X.shape[0]))


def test_classifier_priors():
    X, y = _rings2(120, 2)
    uni = RBIGBayesClassifier(n_layers=3, priors="uniform", random_state=0).fit(X, y)
    np.testing.assert_allclose(uni.log_priors_, np.log([0.5, 0.5]))
    arr = RBIGBayesClassifier(n_layers=3, priors=[0.3, 0.7], random_state=0).fit(X, y)
    np.testing.assert_allclose(arr.log_priors_, np.log([0.3, 0.7]))
    with pytest.raises(ValueError, match="sum to 1"):
        RBIGBayesClassifier(priors=[0.5, 0.9]).fit(X, y)
    with pytest.raises(ValueError, match="shape"):
        RBIGBayesClassifier(priors=[1.0]).fit(X, y)
    with pytest.raises(ValueError, match="priors"):
        RBIGBayesClassifier(priors="jeffreys").fit(X, y)


def test_classifier_class_guards():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 2))
    with pytest.raises(ValueError, match="2 classes"):
        RBIGBayesClassifier().fit(X, np.zeros(40))
    y = np.array([0] * 39 + [1])
    with pytest.raises(ValueError, match="at least 2 samples"):
        # A 1-sample class hits the flow's own marginal-CDF floor.
        RBIGBayesClassifier(min_samples_per_class=1).fit(X, y)


def test_classifier_small_class_kde_fallback_warns():
    X, y = _rings2(150, 3)
    idx = np.concatenate([np.where(y == 0)[0][:15], np.where(y == 1)[0]])
    with pytest.warns(UserWarning, match="KDE"):
        clf = RBIGBayesClassifier(n_layers=5, random_state=0).fit(X[idx], y[idx])
    assert clf.predict(X).shape == (X.shape[0],)


@pytest.mark.statistical
def test_classifier_nonlinear_boundaries_beat_lda_qda():
    """Held-out accuracy on rings/bananas vs LDA/QDA.

    Observed (2 seeds, n_layers=15): rings rbig ~ 1.0 with LDA ~ 0.5 and
    QDA ~ 1.0 (concentric rings are QDA-friendly — equal means, scaled
    covariances); bananas rbig ~ 0.95 vs QDA ~ 0.90 (the curved boundary
    breaks the single quadric).
    """
    ring_rbig, ring_lda, ring_qda = [], [], []
    ban_rbig, ban_qda = [], []
    for seed in range(3):
        X, y = _rings2(600, seed)
        Xt, yt = _rings2(300, 100 + seed)
        ring_rbig.append(
            RBIGBayesClassifier(n_layers=15, random_state=0).fit(X, y).score(Xt, yt)
        )
        ring_lda.append(LinearDiscriminantAnalysis().fit(X, y).score(Xt, yt))
        ring_qda.append(QuadraticDiscriminantAnalysis().fit(X, y).score(Xt, yt))
        Xb, yb = _bananas2(600, seed)
        Xbt, ybt = _bananas2(300, 100 + seed)
        ban_rbig.append(
            RBIGBayesClassifier(n_layers=15, random_state=0).fit(Xb, yb).score(Xbt, ybt)
        )
        ban_qda.append(QuadraticDiscriminantAnalysis().fit(Xb, yb).score(Xbt, ybt))
    assert np.mean(ring_rbig) >= 0.9, ring_rbig
    assert np.mean(ring_lda) <= 0.6, ring_lda
    assert np.mean(ring_rbig) >= np.mean(ring_qda) - 0.02, (ring_rbig, ring_qda)
    assert np.mean(ban_rbig) >= 0.9, ban_rbig
    assert np.mean(ban_rbig) >= np.mean(ban_qda) + 0.02, (ban_rbig, ban_qda)


@pytest.mark.statistical
def test_classifier_calibration_on_bananas():
    """Brier score beats QDA and the reliability slope is near 1.

    Observed (n=2000 train / 2000 test): Brier rbig 0.034 vs QDA 0.071;
    reliability slope 0.97.
    """
    X, y = _bananas2(1000, 0)
    Xt, yt = _bananas2(1000, 7)
    clf = RBIGBayesClassifier(n_layers=15, random_state=0).fit(X, y)
    qda = QuadraticDiscriminantAnalysis().fit(X, y)
    p = clf.predict_proba(Xt)[:, 1]
    assert brier_score_loss(yt, p) <= brier_score_loss(yt, qda.predict_proba(Xt)[:, 1])
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    obs = [yt[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
    pred = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
    slope = np.polyfit(pred, obs, 1)[0]
    assert 0.8 <= slope <= 1.2, slope


@pytest.mark.statistical
def test_classifier_gaussian_sanity_matches_lda():
    """On truly Gaussian shared-covariance classes, stays close to LDA.

    Observed per-seed LDA-minus-RBIG gaps of 0.000-0.013; seed-mean gate
    0.015, worst-seed 0.03 (the issue's "within 1 point" holds on the
    mean, not pointwise — small-sample flow noise).
    """
    gaps = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        X = np.vstack(
            [
                rng.standard_normal((500, 2)) + np.array([1.5, 0.0]),
                rng.standard_normal((500, 2)) - np.array([1.5, 0.0]),
            ]
        )
        y = np.repeat([0, 1], 500)
        Xt = np.vstack(
            [
                rng.standard_normal((300, 2)) + np.array([1.5, 0.0]),
                rng.standard_normal((300, 2)) - np.array([1.5, 0.0]),
            ]
        )
        yt = np.repeat([0, 1], 300)
        rbig = RBIGBayesClassifier(n_layers=15, random_state=0).fit(X, y).score(Xt, yt)
        lda = LinearDiscriminantAnalysis().fit(X, y).score(Xt, yt)
        gaps.append(max(lda - rbig, 0.0))
    assert np.mean(gaps) <= 0.015, gaps
    assert max(gaps) <= 0.03, gaps


@pytest.mark.statistical
def test_classifier_small_class_beats_chance():
    """The KDE-fallback path still classifies far above chance.

    n_c = 5·d = 10 for class 0 triggers the fallback; observed held-out
    accuracy 0.79 on the bimodal task (gate: chance + 15 points).
    """
    X, meta = make_bimodal(n_samples=800, separation=4.0, seed=0)
    y = meta["labels"]
    idx = np.concatenate([np.where(y == 0)[0][:10], np.where(y == 1)[0]])
    with pytest.warns(UserWarning, match="KDE"):
        clf = RBIGBayesClassifier(n_layers=15, random_state=0).fit(X[idx], y[idx])
    Xt, meta_t = make_bimodal(n_samples=400, separation=4.0, seed=9)
    assert clf.score(Xt, meta_t["labels"]) >= 0.65


@pytest.mark.statistical
def test_classifier_log_likelihood_monotone_in_depth():
    """Held-out mean log p(y|x) improves with depth up to convergence.

    Observed on rings: -0.091 (L=1) -> -0.029 (L=4) -> -0.014 (L=16);
    tolerance 0.005 absorbs flow-fit noise.
    """
    X, y = _rings2(600, 3)
    Xt, yt = _rings2(300, 42)
    lls = [
        RBIGBayesClassifier(n_layers=layers, random_state=0)
        .fit(X, y)
        .log_likelihood(Xt, yt)
        for layers in (1, 4, 16)
    ]
    assert lls[1] >= lls[0] - 0.005, lls
    assert lls[2] >= lls[1] - 0.005, lls


# ── ResidualDiagnostics (#130) ───────────────────────────────────────────────


def _linear_data(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    return X, X @ np.array([1.0, -0.5, 0.2]) + 0.5 * rng.standard_normal(n)


def test_diagnostics_delegation_and_attributes():
    X, y = _linear_data(0)
    diag = ResidualDiagnostics(LinearRegression(), n_layers_rbig=3, random_state=0).fit(
        X, y
    )
    np.testing.assert_allclose(diag.predict(X), diag.estimator_.predict(X))
    assert diag.score(X, y) == pytest.approx(diag.estimator_.score(X, y))
    assert diag.residual_mi_.shape == (3,)
    assert diag.residual_negentropy_ >= 0.0
    assert diag.heteroskedasticity_ >= 0.0
    assert diag.residual_mi_max_ == diag.residual_mi_.max()
    assert diag.residuals_.shape == (X.shape[0],)


def test_diagnostics_report():
    X, y = _linear_data(1)
    diag = ResidualDiagnostics(LinearRegression(), n_layers_rbig=3, random_state=0).fit(
        X, y
    )
    report = diag.diagnostic_report(feature_names=["alpha", "beta", "gamma"])
    assert "alpha" in report and "negentropy" in report
    with pytest.raises(ValueError, match="feature_names"):
        diag.diagnostic_report(feature_names=["only-one"])


def test_diagnostics_cv_path():
    X, y = _linear_data(2)
    diag = ResidualDiagnostics(
        LinearRegression(), n_layers_rbig=3, cv=3, random_state=0
    ).fit(X, y)
    assert np.isfinite(diag.specification_score_)


def test_diagnostics_nested_params_gridsearch():
    X, y = _linear_data(3, n=200)
    grid = GridSearchCV(
        ResidualDiagnostics(Ridge(), n_layers_rbig=2, random_state=0),
        {"estimator__alpha": [0.1, 1.0]},
        cv=2,
    )
    grid.fit(X, y)
    assert set(grid.best_params_) == {"estimator__alpha"}


@pytest.mark.statistical
def test_diagnostics_detects_quadratic_and_ranks_models():
    """Missed x0² structure: named, quantified, and gone under GBM.

    Observed (n=1000, n_layers_rbig=10): I(eps; x0) ~ 0.55 for the linear
    fit with argmax 0; GBM mi_max ~ 0.01; spec ordering
    linear ~ 0.40 > poly-ridge ~ 0.02 > gbm ~ 0.005 on both probe seeds.
    """
    kw = {"n_layers_rbig": 10, "tol_rbig": 1e-4, "random_state": 0}
    for seed in range(2):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((1000, 3))
        y = X[:, 0] ** 2 + X[:, 1] + 0.3 * rng.standard_normal(1000)
        lin = ResidualDiagnostics(LinearRegression(), **kw).fit(X, y)
        poly = ResidualDiagnostics(
            make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0)), **kw
        ).fit(X, y)
        gbm = ResidualDiagnostics(GradientBoostingRegressor(random_state=0), **kw).fit(
            X, y
        )
        assert lin.residual_mi_.argmax() == 0
        assert lin.residual_mi_[0] >= 0.1, lin.residual_mi_
        assert gbm.residual_mi_max_ < 0.03, gbm.residual_mi_
        assert (
            lin.specification_score_
            > poly.specification_score_
            > gbm.specification_score_
        ), (
            lin.specification_score_,
            poly.specification_score_,
            gbm.specification_score_,
        )


@pytest.mark.statistical
def test_diagnostics_heteroskedasticity():
    """y = x + |x|·eps flags heteroskedasticity; the control does not.

    Observed (n=1000): het 0.16-0.20 vs control <= 0.003.
    """
    kw = {"n_layers_rbig": 10, "tol_rbig": 1e-4, "random_state": 0}
    hets, controls = [], []
    for seed in range(2):
        rng = np.random.default_rng(10 + seed)
        X = rng.standard_normal((1000, 3))
        yh = X[:, 0] + np.abs(X[:, 0]) * rng.standard_normal(1000)
        yc = X[:, 0] + rng.standard_normal(1000)
        hets.append(
            ResidualDiagnostics(LinearRegression(), **kw).fit(X, yh).heteroskedasticity_
        )
        controls.append(
            ResidualDiagnostics(LinearRegression(), **kw).fit(X, yc).heteroskedasticity_
        )
    assert np.mean(hets) >= 0.1, hets
    assert np.mean(controls) <= 0.03, controls


@pytest.mark.statistical
def test_diagnostics_well_specified_null():
    """OLS on linear-Gaussian data: all three diagnostics near zero.

    Observed (n=1000, 3 seeds): J = 0 and het = 0 exactly (bias-matched
    baseline / clamping); per-feature MI up to 0.06 — the augmented
    2-dim blocks carry more small-sample MI bias than the issue's 0.03
    figure anticipated, so the MI gate is 0.08 on the seed mean.
    """
    kw = {"n_layers_rbig": 10, "tol_rbig": 1e-4, "random_state": 0}
    js, mis, hets = [], [], []
    for seed in range(3):
        X, y = _linear_data(20 + seed, n=1000)
        d0 = ResidualDiagnostics(LinearRegression(), **kw).fit(X, y)
        js.append(d0.residual_negentropy_)
        mis.append(d0.residual_mi_max_)
        hets.append(d0.heteroskedasticity_)
    assert np.mean(js) <= 0.03, js
    assert np.mean(hets) <= 0.03, hets
    assert np.mean(mis) <= 0.08, mis


# ── RBIGFairTransformer (#131) ───────────────────────────────────────────────


def test_fair_projection_and_subspace_shapes():
    X, meta = make_variance_leak(n_samples=300, seed=0)
    A = meta["A"]
    for strategy in ("projection", "subspace"):
        ft = RBIGFairTransformer(strategy=strategy, n_layers=3, random_state=0).fit(
            X, A=A
        )
        out = ft.transform(X)  # A not needed at transform for linear modes
        assert out.shape == X.shape
    assert ft.sensitive_subspace_.shape[1] == X.shape[1]


def test_fair_projection_removes_linear_leak():
    """A mean-shifted group is fully removed by the linear projection."""
    rng = np.random.default_rng(0)
    A = np.repeat([0, 1], 200)
    X = rng.standard_normal((400, 3))
    X[:, 0] += 2.0 * A  # linear leak in one direction
    ft = RBIGFairTransformer(strategy="projection", n_layers=3, random_state=0).fit(
        X, A=A
    )
    out = ft.transform(X)
    gap = out[A == 1].mean(axis=0) - out[A == 0].mean(axis=0)
    assert np.abs(gap).max() < 0.15, gap


def test_fair_sensitive_col_mode():
    X, meta = make_variance_leak(n_samples=300, seed=1)
    XA = np.column_stack([meta["A"], X])
    ft = RBIGFairTransformer(
        strategy="transport", sensitive_col=0, n_layers=3, random_state=0
    ).fit(XA)
    out = ft.transform(XA)
    assert out.shape == (300, X.shape[1])  # A column consumed and dropped
    assert list(ft.get_feature_names_out()) == [f"rbigfair{i}" for i in range(4)]


def test_fair_validation():
    X, meta = make_variance_leak(n_samples=200, seed=2)
    A = meta["A"]
    with pytest.raises(ValueError, match="strategy"):
        RBIGFairTransformer(strategy="adversarial").fit(X, A=A)
    with pytest.raises(ValueError, match="alpha"):
        RBIGFairTransformer(alpha=1.5).fit(X, A=A)
    with pytest.raises(ValueError, match="sensitive attribute"):
        RBIGFairTransformer().fit(X)
    with pytest.raises(ValueError, match="single group"):
        RBIGFairTransformer().fit(X, A=np.zeros(200))
    with pytest.raises(ValueError, match="discrete"):
        RBIGFairTransformer(strategy="transport").fit(X, A=np.arange(200.0))
    with pytest.raises(ValueError, match="out of range"):
        RBIGFairTransformer(sensitive_col=10).fit(X)
    with pytest.raises(ValueError, match="requires y"):
        RBIGFairTransformer(strategy="conditional", n_layers=2).fit(X, A=A)
    ft = RBIGFairTransformer(strategy="transport", n_layers=2, random_state=0).fit(
        X, A=A
    )
    with pytest.raises(ValueError, match="needs the sensitive attribute"):
        ft.transform(X)
    with pytest.raises(ValueError, match="unseen"):
        ft.transform(X, A=np.full(200, 7))


def test_fair_stratum_guard_merges_with_warning():
    X, meta = make_variance_leak(n_samples=400, seed=3)
    A = meta["A"]
    y = np.zeros(400, dtype=int)
    y[:15] = 1  # a 15-sample stratum, below any reasonable threshold
    with pytest.warns(UserWarning, match="falls back to the global"):
        ft = RBIGFairTransformer(
            strategy="conditional",
            n_layers=3,
            min_samples_per_stratum=30,
            random_state=0,
        ).fit(X, y, A=A)
    out = ft.transform(X, A=A, y=y)  # merged cells route through transport
    assert out.shape == X.shape


def test_fair_conditional_without_y_falls_back():
    X, meta = make_variance_leak(n_samples=400, seed=4)
    A = meta["A"]
    y = (np.arange(400) % 2).astype(int)
    ft = RBIGFairTransformer(
        strategy="conditional", n_layers=3, min_samples_per_stratum=10, random_state=0
    ).fit(X, y, A=A)
    with pytest.warns(UserWarning, match="falling back"):
        out = ft.transform(X, A=A)
    assert out.shape == X.shape


def test_fair_pipeline_gridsearch_over_alpha():
    X, meta = make_variance_leak(n_samples=300, seed=5)
    XA = np.column_stack([meta["A"], X])
    y = (np.arange(300) % 2).astype(int)
    pipe = Pipeline(
        [
            (
                "fair",
                RBIGFairTransformer(
                    strategy="projection", sensitive_col=0, n_layers=2, random_state=0
                ),
            ),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    grid = GridSearchCV(pipe, {"fair__alpha": [0.5, 1.0]}, cv=2)
    grid.fit(XA, y)
    assert grid.best_estimator_.predict(XA).shape == y.shape


def test_fair_metadata_routing_roundtrip():
    """A routes through a Pipeline via set_fit_request (sklearn >= 1.4)."""
    sklearn = pytest.importorskip("sklearn")
    from packaging.version import Version

    if Version(sklearn.__version__) < Version("1.4"):
        pytest.skip("metadata routing requires sklearn >= 1.4")
    from sklearn import config_context

    X, meta = make_variance_leak(n_samples=300, seed=6)
    A = meta["A"]
    with config_context(enable_metadata_routing=True):
        fair = (
            RBIGFairTransformer(strategy="projection", n_layers=2, random_state=0)
            .set_fit_request(A=True)
            .set_transform_request(A=True)  # fit_transform composition
        )
        pipe = Pipeline([("fair", fair), ("clf", LogisticRegression(max_iter=200))])
        y = (np.arange(300) % 2).astype(int)
        pipe.fit(X, y, A=A)
        assert pipe.predict(X).shape == y.shape  # projection needs no A here


def _mmd_rbf(X0: np.ndarray, X1: np.ndarray, gamma: float) -> float:
    """Biased RBF-kernel MMD² between two samples."""

    def k(a, b):
        sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * sq)

    return float(k(X0, X0).mean() + k(X1, X1).mean() - 2 * k(X0, X1).mean())


@pytest.mark.statistical
def test_fair_variance_leak_projection_vs_transport():
    """The reason this estimator exists (issue #131 headline criterion).

    Equal group means, unequal variances: the linear projection cannot
    remove second-order leakage (A-predictability stays high) while
    transport drives it to chance.  Observed (n=800): raw 0.87,
    projection 0.81, transport 0.48.
    """
    X, meta = make_variance_leak(n_samples=800, seed=0)
    A = meta["A"]
    proj = RBIGFairTransformer(strategy="projection", n_layers=5, random_state=0).fit(
        X, A=A
    )
    tran = RBIGFairTransformer(strategy="transport", n_layers=10, random_state=0).fit(
        X, A=A
    )
    auc_proj = _a_pred_auc(proj.transform(X), A)
    auc_tran = _a_pred_auc(tran.transform(X, A=A), A)
    assert auc_proj >= 0.65, auc_proj
    assert auc_tran <= 0.55, auc_tran


@pytest.mark.statistical
def test_fair_transport_distributional_guarantee():
    """After transport the groups match distributionally: per-dimension
    KS p > 0.01 and MMD below the 95th permutation percentile.

    Observed KS p-values 0.7-1.0 on all dims of the variance-leak data.
    """
    from scipy.stats import ks_2samp

    X, meta = make_variance_leak(n_samples=800, seed=1)
    A = meta["A"]
    ft = RBIGFairTransformer(strategy="transport", n_layers=10, random_state=0).fit(
        X, A=A
    )
    out = ft.transform(X, A=A)
    for j in range(out.shape[1]):
        assert ks_2samp(out[A == 0, j], out[A == 1, j]).pvalue > 0.01, j

    rng = np.random.default_rng(0)
    sub0 = out[A == 0][:200]
    sub1 = out[A == 1][:200]
    pooled = np.vstack([sub0, sub1])
    gamma = 1.0 / np.median(((pooled[:50, None] - pooled[None, :50]) ** 2).sum(axis=2))
    observed = _mmd_rbf(sub0, sub1, gamma)
    null = []
    for _ in range(100):
        perm = rng.permutation(pooled.shape[0])
        null.append(_mmd_rbf(pooled[perm[:200]], pooled[perm[200:]], gamma))
    assert observed <= np.quantile(null, 0.95), (observed, np.quantile(null, 0.95))


@pytest.mark.statistical
def test_fair_utility_preservation():
    """On I(A;Y)=0 data, removal costs (almost) no task signal.

    Observed (n=1400): baseline task AUC 0.932; transport retains 0.999
    of it at A-pred 0.44; conditional retains 1.007 at A-pred 0.49.  The
    shared-rotation group flows are what makes this hold — see the
    module docstring and the coherence regression test below.
    """
    X, y, A = _util_fair_data(700, 0)
    base = cross_val_score(
        GradientBoostingClassifier(random_state=0), X, y, cv=3, scoring="roc_auc"
    ).mean()
    for strategy, retain in (("transport", 0.90), ("conditional", 0.95)):
        ft = RBIGFairTransformer(strategy=strategy, n_layers=10, random_state=0)
        ft.fit(X, y, A=A)
        out = (
            ft.transform(X, A=A, y=y)
            if strategy == "conditional"
            else ft.transform(X, A=A)
        )
        task = cross_val_score(
            GradientBoostingClassifier(random_state=0), out, y, cv=3, scoring="roc_auc"
        ).mean()
        assert task >= retain * base, (strategy, task, base)
        assert _a_pred_auc(out, A) <= 0.55, strategy


@pytest.mark.statistical
def test_fair_transport_pointwise_coherence():
    """Shared rotation frames keep transport ≈ identity on matching dims.

    Regression for the honest finding that motivated the design:
    independently fitted group flows match distributions but scramble
    per-sample coordinates (PCA sign/order flips), destroying task
    signal in dimensions where the groups already agree.  Here the
    signal dims are identically distributed across groups, so transport
    must correlate strongly with the input on them (observed r > 0.95).
    """
    X, _y, A = _util_fair_data(700, 1)
    ft = RBIGFairTransformer(strategy="transport", n_layers=10, random_state=0).fit(
        X, A=A
    )
    out = ft.transform(X, A=A)
    for j in (0, 1):  # the A-independent signal dims
        r = np.corrcoef(X[:, j], out[:, j])[0, 1]
        assert r >= 0.9, (j, r)


@pytest.mark.statistical
def test_fair_alpha_sweep_monotone():
    """A-predictability decreases monotonically in alpha; task AUC holds.

    Observed on I(A;Y)=0 data: A-pred 0.76 / 0.56 / 0.44 at alpha
    0 / 0.5 / 1 with task AUC flat at 0.93.  (The issue text phrased the
    direction as "alpha -> 0"; the removal strength grows with alpha,
    so leakage must be non-increasing IN alpha — asserted with tol 0.02.)
    """
    X, y, A = _util_fair_data(700, 0)
    apreds, tasks = [], []
    for alpha in (0.0, 0.5, 1.0):
        ft = RBIGFairTransformer(
            strategy="transport", alpha=alpha, n_layers=10, random_state=0
        ).fit(X, A=A)
        out = ft.transform(X, A=A)
        apreds.append(_a_pred_auc(out, A))
        tasks.append(
            cross_val_score(
                GradientBoostingClassifier(random_state=0),
                out,
                y,
                cv=3,
                scoring="roc_auc",
            ).mean()
        )
    assert apreds[1] <= apreds[0] + 0.02, apreds
    assert apreds[2] <= apreds[1] + 0.02, apreds
    assert min(tasks) >= max(tasks) - 0.02, tasks


def test_fair_deterministic_given_seed():
    X, meta = make_variance_leak(n_samples=300, seed=7)
    A = meta["A"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = (
            RBIGFairTransformer(strategy="transport", n_layers=3, random_state=0)
            .fit(X, A=A)
            .transform(X, A=A)
        )
        b = (
            RBIGFairTransformer(strategy="transport", n_layers=3, random_state=0)
            .fit(X, A=A)
            .transform(X, A=A)
        )
    np.testing.assert_allclose(a, b)
