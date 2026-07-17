# Fairness API guide

`RBIGFairTransformer` removes information about a sensitive attribute `A`
from your features at a chosen strength. This guide covers the two things
you have to decide: **which strategy** and **how `A` reaches the
transformer** at fit and transform time.

See the [worked example](../notebooks/24_fairness_transform.ipynb) for the
variance-leak case and the [benchmarks](../benchmarks.md) for the
alpha-Pareto frontier.

## Choosing a strategy

| Strategy | Removes | Needs `A` at transform? | Needs `y`? |
| --- | --- | --- | --- |
| `projection` | the single linear A-direction | no | no |
| `subspace` | top-`k` linear A-directions | no | no |
| `transport` | the full per-group distribution | **yes** | no |
| `conditional` | `I(X; A \| Y)`, preserving Y-signal | **yes** | at fit |

- Use **`projection`/`subspace`** only for *linear* leakage — they need
  `A` just at fit and are the friction-free entry point.
- Use **`transport`** for *distributional* leakage (e.g. equal means but
  unequal variances) — a linear projection provably cannot remove it.
- Use **`conditional`** when you have labels and want to preserve the
  task signal: it transports within Y-strata, removing `I(X; A | Y)`
  while leaving `I(X; Y)` intact.

The `alpha` knob blends `α·X_fair + (1−α)·X`. **No free lunch:** when
`I(A; Y) > 0`, removing A-information necessarily removes some Y-signal;
sweep `alpha` to find your point on the trade-off.

## Getting `A` to the transformer

`transport` and `conditional` need `A` at *transform* time, but
`Pipeline.transform(X)` passes only `X`. There are two supported ways in.

### 1. `sensitive_col` (Pipeline-friendly default)

Designate `A` as a column of `X`; it is consumed and dropped from the
output. Fully compatible with `Pipeline` and `GridSearchCV`:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from rbig import RBIGFairTransformer, make_variance_leak

X, meta = make_variance_leak(n_samples=800, seed=0)
XA = np.column_stack([meta["A"], X])          # A is column 0

pipe = Pipeline([
    ("fair", RBIGFairTransformer(strategy="transport", sensitive_col=0)),
    ("clf", LogisticRegression(max_iter=200)),
])
pipe.fit(XA, y=(np.arange(len(XA)) % 2))       # A rides inside X
```

### 2. Metadata routing (sklearn ≥ 1.4)

Pass `A` as a keyword and let sklearn route it through the pipeline:

```python
from sklearn import config_context

with config_context(enable_metadata_routing=True):
    fair = (
        RBIGFairTransformer(strategy="projection")
        .set_fit_request(A=True)
        .set_transform_request(A=True)
    )
    pipe = Pipeline([("fair", fair), ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(X, y, A=meta["A"])
```

Outside a pipeline you can always call `fit(X, A=A)` /
`transform(X, A=A)` directly.

## Guards you should know about

- **Discrete `A` only** (≤ 20 groups); each group needs ≥ 10 samples to
  fit its flow.
- **Stratum guard** (`conditional`): an `(A, y)` cell below
  `min_samples_per_stratum` (default `20·d`) merges into the global
  transport flow with a warning — a tiny flow is never fit silently.
- **Integer inputs are coerced to float** so transported values are not
  truncated.

## Measuring leakage

The honest metric is **A-predictability**: train a strong classifier
(e.g. `GradientBoostingClassifier`) to recover `A` from the transformed
features and read its ROC-AUC. `0.5` means `A` is gone; anything above
means residual leakage. The [benchmarks](../benchmarks.md) report this
for every strategy across the `alpha` sweep.
