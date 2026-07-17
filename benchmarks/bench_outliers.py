"""Outlier detection: RBIGOutlierDetector ROC-AUC vs IF / LOF / OneClassSVM.

Zoo shapes (rings, banana, bimodal) plus two real sklearn tabular sets
(breast-cancer, wine — no network download).  Outliers are injected as a
uniform "background" contamination so every method sees the same task.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from rbig import RBIGOutlierDetector, make_banana, make_bimodal, make_rings

from ._report import Section, Table, fmt


def _inject(X_inlier, rng, frac=0.15, spread=4.0):
    """Append uniform-background outliers spanning ``spread``× the range."""
    n_out = int(len(X_inlier) * frac)
    lo, hi = X_inlier.min(0), X_inlier.max(0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * spread
    outliers = rng.uniform(mid - half, mid + half, size=(n_out, X_inlier.shape[1]))
    X = np.vstack([X_inlier, outliers])
    y = np.r_[np.ones(len(X_inlier)), -np.ones(n_out)]  # +1 inlier, -1 outlier
    return X, y


def _auc(scores, y_true):
    """ROC-AUC with the convention: higher score = more normal."""
    return roc_auc_score((y_true == 1).astype(int), scores)


def _zoo():
    yield "rings", make_rings(n_samples=800, seed=0)[0]
    yield "banana", make_banana(n_samples=800, seed=0)[0]
    yield "bimodal", make_bimodal(n_samples=800, seed=0)[0]


def _real():
    Xb = StandardScaler().fit_transform(load_breast_cancer().data[:, :10])
    yield "breast-cancer(10d)", Xb
    Xw = StandardScaler().fit_transform(load_wine().data)
    yield "wine(13d)", Xw


def run() -> Section:
    rng = np.random.default_rng(0)
    rows = []
    for name, X_inlier in [*_zoo(), *_real()]:
        X, y = _inject(X_inlier, rng)
        d = X.shape[1]
        # RBIG (higher score_samples = more normal).
        rbig = RBIGOutlierDetector(n_layers=30, random_state=0).fit(X)
        aucs = {"RBIG": _auc(rbig.score_samples(X), y)}
        aucs["IForest"] = _auc(
            IsolationForest(random_state=0).fit(X).score_samples(X), y
        )
        aucs["LOF"] = _auc(
            LocalOutlierFactor(novelty=False).fit(X).negative_outlier_factor_, y
        )
        aucs["OneClassSVM"] = _auc(
            OneClassSVM(gamma="scale").fit(X).score_samples(X), y
        )
        best = max(aucs, key=aucs.get)
        rows.append(
            [
                name,
                str(d),
                *[
                    (f"**{fmt(aucs[m])}**" if m == best else fmt(aucs[m]))
                    for m in ("RBIG", "IForest", "LOF", "OneClassSVM")
                ],
            ]
        )
    table = Table(
        columns=["dataset", "d", "RBIG", "IForest", "LOF", "OneClassSVM"],
        rows=rows,
        caption=(
            "ROC-AUC (higher better), 15% uniform-background contamination, "
            "seed 0. Best per row in bold."
        ),
    )
    return Section(
        title="Outlier detection vs IsolationForest / LOF / OneClassSVM",
        intro=(
            "RBIG scores anomalies by exact log-density; the baselines are "
            "depth-based (IForest), local-density (LOF), and boundary "
            "(OneClassSVM)."
        ),
        tables=[table],
        takeaway=(
            "On the low-dimensional curved zoo shapes the density score is "
            "competitive-to-best. On the higher-dimensional real tables "
            "IsolationForest is the more robust default — log-density "
            "scores concentrate as d grows (the d>30 warning in "
            "`RBIGOutlierDetector` points at exactly this), so reduce "
            "dimensionality first when d is large."
        ),
    )
