"""Fairness: alpha Pareto curves (A-predictability vs task AUC) per strategy.

The ``alpha`` knob trades utility for sensitive-attribute removal.  A good
strategy pushes A-predictability toward 0.5 (chance) while keeping task
AUC high.  Data is constructed with ``I(A; Y) = 0`` so full removal is
achievable in principle.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from rbig import RBIGFairTransformer

from ._report import Section, Table, fmt


def _data(n_per=600, seed=0):
    rng = np.random.default_rng(seed)
    A = np.repeat([0, 1], n_per)
    S = rng.standard_normal((2 * n_per, 2))  # A-independent signal
    y = (S[:, 0] + 0.5 * S[:, 1] + 0.5 * rng.standard_normal(2 * n_per) > 0).astype(int)
    L = np.where(A[:, None] == 1, 2.0, 1.0) * rng.standard_normal((2 * n_per, 2))
    return np.hstack([S, L]), y, A


def _auc(X, target):
    return float(
        cross_val_score(
            GradientBoostingClassifier(random_state=0),
            X,
            target,
            cv=3,
            scoring="roc_auc",
        ).mean()
    )


def run() -> Section:
    X, y, A = _data()
    base_task = _auc(X, y)
    base_apred = _auc(X, A)
    rows = []
    for strategy in ("projection", "transport", "conditional"):
        for alpha in (0.0, 0.5, 1.0):
            ft = RBIGFairTransformer(
                strategy=strategy, alpha=alpha, n_layers=15, random_state=0
            )
            if strategy == "conditional":
                ft.fit(X, y, A=A)
                out = ft.transform(X, A=A, y=y)
            else:
                ft.fit(X, A=A)
                out = ft.transform(X, A=A)
            rows.append(
                [
                    strategy,
                    f"{alpha:.1f}",
                    fmt(_auc(out, A)),
                    fmt(_auc(out, y)),
                ]
            )
    table = Table(
        columns=["strategy", "alpha", "A-pred AUC ↓", "task AUC ↑"],
        rows=rows,
        caption=(
            f"A-predictability (want 0.5) and task AUC (want high) across "
            f"the alpha sweep. Baselines: raw A-pred {base_apred:.3f}, "
            f"raw task {base_task:.3f}. Data has I(A;Y)=0."
        ),
    )
    return Section(
        title="Fairness — the alpha Pareto frontier",
        intro=(
            "Each strategy is swept over alpha ∈ {0, 0.5, 1}; alpha=1 is "
            "full removal, alpha=0 is the identity (raw baseline)."
        ),
        tables=[table],
        takeaway=(
            "`projection` cannot remove the second-order (variance) leak — "
            "its A-predictability stays high even at alpha=1. `transport` "
            "drives A-predictability to chance but, on this I(A;Y)=0 data, "
            "so does `conditional` while additionally preserving task AUC. "
            "Use `conditional` when labels are available at fit; otherwise "
            "`transport`. `projection` is only appropriate for purely "
            "linear leakage."
        ),
    )
