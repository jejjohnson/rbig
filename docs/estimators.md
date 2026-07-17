# The scikit-learn estimator suite

One fitted RBIG flow wears **four identities** — it is simultaneously a
density model, a generator, an information estimator, and a feature
transformer. Every estimator in this suite is a thin, `check_estimator`-
compliant wrapper that exposes one of those identities through the
scikit-learn API you already know.

## The four identities of a fitted RBIG

| Identity | Core method | What it gives you |
| --- | --- | --- |
| **Density** | `score_samples(X)` | exact `log p(x)` via change-of-variables |
| **Generator** | `sample(n)` | draws `z ~ N(0, I)` and inverts the flow |
| **Information** | `estimate_mi`, `estimate_tc`, … | entropy / MI / TC through total-correlation reduction |
| **Transformer** | `transform` / `inverse_transform` | an invertible map to a Gaussianized latent space |

Because the *same* fitted object serves all four, the estimators below
compose these identities rather than reimplementing anything.

## The suite

| Estimator | sklearn role | Built on | Example |
| --- | --- | --- | --- |
| [`RBIGOutlierDetector`](api/reference.md) | `OutlierMixin` | density | [19](notebooks/19_outlier_detection.ipynb) |
| [`RBIGReducer`](api/reference.md) | `TransformerMixin` | information (negentropy) | [20](notebooks/20_negentropy_reduction.ipynb) |
| [`RBIGMISelector`](api/reference.md) | `SelectorMixin` | information (MI) | [21](notebooks/21_mi_feature_selection.ipynb) |
| [`RBIGKMeans`](api/reference.md) | `ClusterMixin` | transformer + inverse | [22](notebooks/22_gaussianize_then_cluster.ipynb) |
| [`RBIGBayesClassifier`](api/reference.md) | `ClassifierMixin` | density (one per class) | [23](notebooks/23_bayes_classifier.ipynb) |
| [`RBIGFairTransformer`](api/reference.md) | `TransformerMixin` | transformer + inverse (transport) | [24](notebooks/24_fairness_transform.ipynb) |
| [`ResidualDiagnostics`](api/reference.md) | meta `RegressorMixin` | information (residual MI) | [25](notebooks/25_residual_diagnostics.ipynb) |

All estimators pass scikit-learn's `parametrize_with_checks` suite and
work inside `Pipeline` and `GridSearchCV`.

## Where they win — and where they don't

The [benchmark report](benchmarks.md) compares every estimator against
its standard baseline on the same task, **including the rows where the
baseline wins**. In brief:

- **MI** — for genuinely joint, multivariate MI there is no sklearn
  baseline; for scalar–scalar screening, KSG is competitive and often
  more accurate.
- **Outliers** — competitive-to-best on low-dimensional curved shapes;
  IsolationForest is the more robust default as dimensionality grows.
- **Clustering** — a large win on elongated clusters; raw k-means wins on
  well-separated axis-aligned modes (which is why `n_layers_rbig`
  defaults small).
- **Classification** — nonlinear boundaries (rings, bananas) where LDA is
  at chance; on genuinely Gaussian classes it matches LDA, no better.
- **Preprocessing** — pays off most for a *linear* downstream model on
  curved data; for an RBF-SVC the choice barely matters.
- **Fairness** — `transport`/`conditional` remove second-order leakage
  that a linear `projection` cannot.

See the individual notebooks for the mechanism behind each result.
