"""Sampling quality: two-sample classifier AUC, RBIG vs GaussianMixture.

A GradientBoosting classifier is trained to tell real held-out data from
generated samples.  AUC → 0.5 means the samples are indistinguishable
from real data (better); AUC → 1.0 means the classifier separates them
trivially (worse).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from rbig import AnnealedRBIG, make_banana, make_bimodal, make_rings

from ._report import Section, Table, fmt


def _c2st(real_test, fake):
    """Classifier two-sample test AUC (0.5 = indistinguishable)."""
    X = np.vstack([real_test, fake])
    y = np.r_[np.ones(len(real_test)), np.zeros(len(fake))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    clf = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def run() -> Section:
    rows = []
    for name, (X, _meta) in [
        ("banana", make_banana(n_samples=2000, seed=0)),
        ("rings", make_rings(n_samples=2000, seed=0)),
        ("bimodal", make_bimodal(n_samples=2000, seed=0)),
    ]:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        train, test = X[idx[:1500]], X[idx[1500:]]
        rbig = AnnealedRBIG(n_layers=40, random_state=0).fit(train)
        rbig_fake = rbig.sample(len(test), random_state=0)
        gmm = GaussianMixture(n_components=5, random_state=0).fit(train)
        gmm_fake = gmm.sample(len(test))[0]
        auc_rbig = _c2st(test, rbig_fake)
        auc_gmm = _c2st(test, gmm_fake)
        best = "RBIG" if abs(auc_rbig - 0.5) <= abs(auc_gmm - 0.5) else "GMM(5)"
        rows.append(
            [
                name,
                (f"**{fmt(auc_rbig)}**" if best == "RBIG" else fmt(auc_rbig)),
                (f"**{fmt(auc_gmm)}**" if best == "GMM(5)" else fmt(auc_gmm)),
            ]
        )
    table = Table(
        columns=["dataset", "RBIG C2ST-AUC", "GMM(5) C2ST-AUC"],
        rows=rows,
        caption=(
            "Classifier two-sample-test AUC, closer to 0.5 is better "
            "(samples indistinguishable from held-out real data), seed 0."
        ),
    )
    return Section(
        title="Sampling quality vs GaussianMixture",
        intro=(
            "Both models are fit on the same 1500 training points and asked "
            "to generate 500 samples; a boosted classifier then tries to "
            "separate them from 500 held-out real points."
        ),
        tables=[table],
        takeaway=(
            "A 5-component GMM is a strong parametric baseline on these "
            "low-dimensional shapes and is hard to beat on the multi-modal "
            "ones; RBIG is competitive without choosing a component count, "
            "and its edge grows on the curved single-manifold shapes "
            "(banana/rings) where a small GMM under-fits the geometry."
        ),
    )
