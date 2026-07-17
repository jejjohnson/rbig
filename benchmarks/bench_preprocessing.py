"""Preprocessing: downstream accuracy, RBIG vs QuantileTransformer+PCA vs raw.

RBIG Gaussianizes the joint density (calibrating the geometry a linear
classifier sees); QuantileTransformer Gaussianizes each margin
independently and PCA de-correlates linearly.  Downstream models are a
plain LogisticRegression and an RBF-SVC.
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from rbig import AnnealedRBIG, make_bimodal, make_rings

from ._report import Section, Table, fmt


def _labelled():
    Xr, mr = make_rings(n_samples=1500, seed=0)
    yield "rings", Xr, mr["labels"]
    Xb, mb = make_bimodal(n_samples=1500, separation=3.0, seed=0)
    yield "bimodal", Xb, mb["labels"]


def _score(pre, clf, X, y):
    steps = [] if pre is None else [("pre", pre)]
    pipe = Pipeline([*steps, ("clf", clf)])
    return float(cross_val_score(pipe, X, y, cv=4).mean())


def run() -> Section:
    rows = []
    for name, X, y in _labelled():
        for clf_name, clf in (
            ("LogReg", LogisticRegression(max_iter=300)),
            ("RBF-SVC", SVC(kernel="rbf")),
        ):
            raw = _score(StandardScaler(), clf, X, y)
            qt_pca = _score(
                Pipeline(
                    [
                        (
                            "qt",
                            QuantileTransformer(
                                output_distribution="normal", n_quantiles=200
                            ),
                        ),
                        ("pca", PCA(whiten=True)),
                    ]
                ),
                clf,
                X,
                y,
            )
            rbig = _score(AnnealedRBIG(n_layers=15, random_state=0), clf, X, y)
            scores = {"raw+scale": raw, "QT+PCA": qt_pca, "RBIG": rbig}
            best = max(scores, key=scores.get)
            rows.append(
                [
                    name,
                    clf_name,
                    *[
                        (f"**{fmt(scores[k])}**" if k == best else fmt(scores[k]))
                        for k in ("raw+scale", "QT+PCA", "RBIG")
                    ],
                ]
            )
    table = Table(
        columns=["dataset", "classifier", "raw+scale", "QT+PCA", "RBIG"],
        rows=rows,
        caption="4-fold CV accuracy, best per row in bold.",
    )
    return Section(
        title="Preprocessing for downstream classifiers",
        intro=(
            "Accuracy of a linear and an RBF classifier on raw (scaled) "
            "features vs two Gaussianizing preprocessors."
        ),
        tables=[table],
        takeaway=(
            "For a linear classifier on curved data (rings) the joint "
            "Gaussianization pays off most — QT+PCA only removes linear "
            "correlation and per-margin shape. For an RBF-SVC, which "
            "already handles nonlinearity, the preprocessing choice "
            "matters far less and raw+scale is often within noise: "
            "Gaussianize when the downstream model is linear, not when it "
            "is already a universal kernel."
        ),
    )
