# Choosing a marginal

Every RBIG layer Gaussianizes each dimension with a **marginal
transform** before rotating. Which one you pick changes what happens
*outside the training support* — and that boundary behavior is a feature
for some tasks and a bug for others. This guide is the decision.

## The three marginals

| Marginal | How the tail behaves | Reach for it when |
| --- | --- | --- |
| **Empirical** (`MarginalGaussianize`, default) | clips: out-of-range points map to the most extreme rank | anomaly **ranking**, robustness to outliers |
| **KDE** (`MarginalKDEGaussianize`) | smooth, but flattens far out | small samples where the empirical CDF is too steppy |
| **Tail-extended** (`MarginalGaussianize(tail="gaussian"|"pareto")`) | parametric tail continues past the seam | **likelihood** comparison, **sampling**, cross-model scoring |

## Why the tail matters

The empirical CDF is only defined on `[min, max]` of the training data.
Past that range the clipping marginal assigns the same extreme quantile
to every point — so the log-density **plunges and then flattens**. For
*ranking* anomalies that is exactly right: anything past the support is
"maximally weird," and the ordering is preserved. See
[Boundary Issues](../notebooks/07_boundary_issues.ipynb) and
[SIG Boundary Behavior](../notebooks/18_sig_boundary_behavior.ipynb) for
the plunge-then-flatten picture.

But for anything that **compares** log-densities across models — a Bayes
classifier scoring a point under each class's flow, an
optimal-transport-style inverse mapping into a region sparse in one
group — the flattened tail is a bug: two very different points get the
same degenerate score. There you want the tail to *keep going*, which is
what the parametric extension provides:

- `tail="gaussian"` matches a normal through two empirical quantiles per
  side — exact for genuinely Gaussian tails, and the sensible default.
- `tail="pareto"` fits a generalized Pareto to the seam exceedances — for
  genuinely heavy-tailed data.

Both are C⁰-continuous at the seam with a small, documented C¹ kink.

## What the estimators pick for you

You rarely set this by hand — the composed estimators choose the right
marginal for their task:

- `RBIGOutlierDetector` uses **empirical** marginals on purpose (ranking).
- `RBIGBayesClassifier` and `RBIGFairTransformer` use **tail-extended**
  marginals on purpose (cross-model / cross-group comparison).

If you are driving `AnnealedRBIG` directly, pass
`marginal_kwargs={"tail": "gaussian"}` when you need finite, comparable
log-densities beyond the training support, and leave the default when you
are ranking.

## Memory note

The default marginal stores the full sorted training columns
(`O(n_layers · n · d)`). For large `n`, cap it with
`MarginalGaussianize(n_quantiles=1000)`, which stores a fixed quantile
grid instead (`O(n_layers · n_quantiles · d)`) at negligible accuracy
cost when `n ≫ n_quantiles`. See the runtime table in the
[benchmarks](../benchmarks.md).
