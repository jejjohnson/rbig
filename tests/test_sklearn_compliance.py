"""Full scikit-learn compliance for the transformer estimators (issue #122).

Runs the complete ``parametrize_with_checks`` suite over every public
transformer estimator with a single sanctioned XFAIL table, plus
``set_output`` / feature-name and Pipeline / GridSearchCV integration.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import parametrize_with_checks

from rbig import (
    GIS,
    SIG,
    AnnealedRBIG,
    RBIGMISelector,
    RBIGOutlierDetector,
    RBIGReducer,
    make_banana,
    make_rings,
)

# The registry of public estimators under full compliance, and the only
# sanctioned place to skip a check: map check name -> documented reason.
ESTIMATOR_REGISTRY = [
    AnnealedRBIG(n_layers=3, patience=2),
    GIS(n_layers=3, patience=2),
    SIG(n_layers=3, patience=2),
    RBIGOutlierDetector(n_layers=3, contamination=0.1, random_state=0),
    RBIGReducer(n_components=1),
    RBIGMISelector(
        n_features_to_select=1, strategy="filter", n_layers_rbig=3, random_state=0
    ),
]
# Per-class check name -> documented reason; non-strict xfails.
XFAIL: dict[str, dict[str, str]] = {
    "AnnealedRBIG": {
        "check_methods_subset_invariance": (
            "Environment-conditional (observed on macOS 3.10/3.13 and "
            "ubuntu 3.10 CI; passes on batch-consistent numpy/BLAS builds "
            "such as ubuntu 3.13): training points sit exactly on the "
            "empirical-CDF nodes, and matmul results that depend on batch "
            "size resolve those ties differently for full-batch vs subset "
            "inputs after the first rotation — a one-rank jump at small n "
            "is a visible probit step. GIS/SIG use continuous splines and "
            "are unaffected."
        ),
    },
    "RBIGOutlierDetector": {
        "check_methods_subset_invariance": (
            "Inherited from AnnealedRBIG (decision_function is its "
            "score_samples shifted by a constant): empirical-CDF ties "
            "resolve differently on numpy/BLAS builds whose matmul "
            "results depend on batch size (observed on macOS CI). See the "
            "AnnealedRBIG entry above for the mechanism."
        ),
    },
}


def _expected_failed_checks(estimator):
    return XFAIL.get(type(estimator).__name__, {})


@parametrize_with_checks(
    ESTIMATOR_REGISTRY, expected_failed_checks=_expected_failed_checks
)
def test_full_sklearn_compliance(estimator, check):
    check(estimator)


# ── Feature names & set_output ───────────────────────────────────────────────


def test_get_feature_names_out():
    X, _ = make_banana(n_samples=300, seed=0)
    model = AnnealedRBIG(n_layers=3, random_state=0).fit(X)
    assert list(model.get_feature_names_out()) == ["rbig0", "rbig1"]
    gis = GIS(n_layers=2, random_state=0).fit(X)
    assert list(gis.get_feature_names_out()) == ["gis0", "gis1"]


def test_set_output_pandas():
    pd = pytest.importorskip("pandas")
    X, _ = make_banana(n_samples=300, seed=0)
    df = pd.DataFrame(X, columns=["a", "b"])
    model = AnnealedRBIG(n_layers=3, random_state=0)
    model.set_output(transform="pandas")
    out = model.fit(df).transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["rbig0", "rbig1"]
    assert out.shape == df.shape


# ── Pipeline / GridSearchCV integration ──────────────────────────────────────


def _binary_banana(n=600, seed=0):
    X, _ = make_banana(n_samples=n, seed=seed)
    y = (X[:, 1] > np.median(X[:, 1])).astype(int)
    return X, y


def test_pipeline_gridsearch_clone():
    X, y = _binary_banana()
    pipe = Pipeline(
        [
            ("rbig", AnnealedRBIG(n_layers=3, random_state=0)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    grid = GridSearchCV(pipe, {"rbig__n_layers": [2, 3]}, cv=2)
    grid.fit(X, y)
    assert grid.best_estimator_.predict(X).shape == y.shape


def test_transform_removes_joint_dependence():
    """Residual Gaussian-proxy TC of transformed held-out data is tiny."""
    X_train, _ = make_banana(n_samples=2000, seed=0)
    X_test, _ = make_banana(n_samples=1000, seed=1)
    model = AnnealedRBIG(n_layers=30, random_state=0).fit(X_train)
    Z = model.transform(X_test)
    # Gaussian-proxy TC: sum of marginal Gaussian entropies - joint.
    _sign, logdet = np.linalg.slogdet(np.corrcoef(Z.T))
    residual_tc = -0.5 * logdet
    assert residual_tc < 0.02


@pytest.mark.statistical
def test_downstream_svc_improves_on_rings():
    """RBF-SVC accuracy on rings improves materially after Gaussianization.

    Sanity check, not a claim: the rings' radial structure is easier for
    an RBF kernel once the density is Gaussianized (fixed seed; threshold
    chosen with slack below the observed improvement).
    """
    X, meta = make_rings(n_samples=1200, seed=0)
    y = meta["labels"]
    X_test, meta_test = make_rings(n_samples=600, seed=1)
    y_test = meta_test["labels"]
    raw = SVC(kernel="rbf", random_state=0).fit(X, y).score(X_test, y_test)
    pipe = Pipeline(
        [
            ("rbig", AnnealedRBIG(n_layers=10, random_state=0)),
            ("svc", SVC(kernel="rbf", random_state=0)),
        ]
    ).fit(X, y)
    gaussianized = pipe.score(X_test, y_test)
    assert gaussianized >= raw + 0.05 or gaussianized > 0.97
