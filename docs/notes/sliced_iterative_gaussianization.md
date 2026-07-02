# Sliced Iterative Gaussianization

This page derives the **Sliced Iterative Gaussianization** family of
Dai & Seljak (2020) — the `GIS` and `SIG` estimators in `rbig` — and compares
it in detail with classic **Rotation-Based Iterative Gaussianization** (RBIG,
`AnnealedRBIG`).

**Paper**: [Sliced Iterative Normalizing Flows (Dai & Seljak, 2020)](https://arxiv.org/abs/2007.00674)

**Reference implementations**:
[SINF (official, PyTorch)](https://github.com/biweidai/SINF) ·
[sinflow (NumPy/SciPy)](https://github.com/minaskar/sinflow)

---

## Setup: Gaussianization as a Normalizing Flow

All iterative Gaussianization methods construct an invertible map
$f: \mathbb{R}^D \to \mathbb{R}^D$ such that

$$\mathbf{z} = f(\mathbf{x}) \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D)$$

for data $\mathbf{x} \sim p_\text{data}$. Once fitted, the map defines a
**normalizing flow**: the model density follows from the
change-of-variables formula

$$\log p(\mathbf{x}) = \log \mathcal{N}\big(f(\mathbf{x}); \mathbf{0}, \mathbf{I}\big) + \log \left| \det \mathbf{J}_f(\mathbf{x}) \right|$$

and sampling follows from the inverse map

$$\mathbf{x} = f^{-1}(\mathbf{z}), \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D).$$

The map is built greedily as a composition of $L$ simple layers,

$$f = f_L \circ f_{L-1} \circ \cdots \circ f_1,$$

where each layer removes some of the remaining non-Gaussianity. The methods
differ in **what a layer does** and **how it is chosen**:

| Method | Layer | How directions are chosen |
|--------|-------|---------------------------|
| RBIG | Rotate **all** $D$ dims, then marginally Gaussianize **all** $D$ dims | Fixed family (PCA / ICA / random) |
| GIS / SIG | Gaussianize only $K \le D$ **projected slices**, leave the rest untouched | **Learned** — maximize sliced non-Gaussianity on the Stiefel manifold |

---

## The Sliced Layer

A single sliced layer (class `SIGLayer`) is parameterized by an orthonormal
frame $\mathbf{A} \in \mathbb{R}^{D \times K}$ with $\mathbf{A}^\top \mathbf{A} = \mathbf{I}_K$
and $K$ monotone scalar maps $g_1, \dots, g_K$ (rational-quadratic splines,
see below). Decompose $\mathbf{x}$ into its component in the span of
$\mathbf{A}$ and the orthogonal residual:

$$\mathbf{x} = \underbrace{\mathbf{A}\mathbf{A}^\top \mathbf{x}}_{\text{sliced part}} + \underbrace{(\mathbf{I} - \mathbf{A}\mathbf{A}^\top)\,\mathbf{x}}_{\text{residual (untouched)}}.$$

The layer Gaussianizes each projected coordinate and leaves the residual
alone:

$$f(\mathbf{x}) = (\mathbf{I} - \mathbf{A}\mathbf{A}^\top)\,\mathbf{x} + \mathbf{A}\, g\!\big(\mathbf{A}^\top \mathbf{x}\big), \qquad g(\mathbf{u}) = \big(g_1(u_1), \dots, g_K(u_K)\big).$$

**Log-determinant.** Complete $\mathbf{A}$ to an orthonormal basis
$\mathbf{Q} = [\mathbf{A}, \mathbf{A}_\perp]$. In that basis the layer acts as
$(g_1, \dots, g_K, \text{id}, \dots, \text{id})$, and orthogonal changes of
basis contribute nothing to the Jacobian, so

$$\log \left| \det \mathbf{J}_f(\mathbf{x}) \right| = \sum_{k=1}^{K} \log g_k'\!\big(\mathbf{a}_k^\top \mathbf{x}\big).$$

Each $g_k$ is strictly monotone, so the layer is exactly invertible:

$$f^{-1}(\mathbf{z}) = (\mathbf{I} - \mathbf{A}\mathbf{A}^\top)\,\mathbf{z} + \mathbf{A}\, g^{-1}\!\big(\mathbf{A}^\top \mathbf{z}\big).$$

RBIG is recovered as the special case $K = D$ with $\mathbf{A}$ a full
rotation matrix chosen by PCA/ICA/random instead of by optimization.

---

## Finding Directions: Max K-SWD on the Stiefel Manifold

The core novelty of SIG/GIS is *where to slice*. The layer should
Gaussianize the directions along which the current data is **most
non-Gaussian**. Non-Gaussianity along a direction $\mathbf{a}$ is measured by
the 1-Wasserstein distance between the projected data and a projected
Gaussian reference.

### 1D Wasserstein distance

For two 1D samples with sorted order statistics $x_{(i)}$ and $z_{(i)}$
($i = 1, \dots, N$), the 1-Wasserstein (earth mover's) distance has the
closed form

$$W_1(x, z) = \frac{1}{N} \sum_{i=1}^{N} \left| x_{(i)} - z_{(i)} \right| = \int_0^1 \left| F_x^{-1}(q) - F_z^{-1}(q) \right| dq,$$

i.e. sorting both samples and averaging the absolute quantile differences
(`wasserstein_1d` in `rbig._src.stiefel`).

### The K-SWD objective

Given the current data $\mathbf{X} \in \mathbb{R}^{N \times D}$ and a target
sample $\mathbf{Z}$ (standard Gaussian draws for GIS), the **K-Sliced
Wasserstein Distance** of a frame $\mathbf{A} = [\mathbf{a}_1, \dots, \mathbf{a}_K]$ is

$$\mathcal{F}(\mathbf{A}) = \frac{1}{K} \sum_{k=1}^{K} W_1\!\big(\mathbf{X}\mathbf{a}_k,\; \mathbf{Z}\mathbf{a}_k\big),$$

maximized subject to the orthonormality constraint

$$\mathbf{A} \in \mathrm{St}(D, K) = \left\{ \mathbf{A} \in \mathbb{R}^{D \times K} : \mathbf{A}^\top \mathbf{A} = \mathbf{I}_K \right\},$$

the **Stiefel manifold**. Orthonormality matters: it keeps the layer's
log-determinant simple (previous section) and prevents the $K$ slices from
collapsing onto one direction.

The Euclidean (sub)gradient of the matched-sorted objective w.r.t. one
column is

$$\nabla_{\mathbf{a}_k} W_1 = \frac{1}{N} \sum_{i=1}^{N} \operatorname{sign}\!\big(p_{(i)} - q_{(i)}\big)\big(\mathbf{x}_{(i)} - \mathbf{z}_{(i)}\big),$$

where $p_{(i)} = \mathbf{x}_{(i)}^\top \mathbf{a}_k$ and
$q_{(i)} = \mathbf{z}_{(i)}^\top \mathbf{a}_k$ are the sorted projections and
$\mathbf{x}_{(i)}, \mathbf{z}_{(i)}$ the correspondingly re-ordered samples.

### Cayley retraction

To ascend $\mathcal{F}$ while staying exactly on $\mathrm{St}(D, K)$, `rbig`
uses the curvilinear search of Wen & Yin (2013). With Euclidean gradient
$\mathbf{G} = \nabla_\mathbf{A} \mathcal{F}$, form the skew-symmetric generator

$$\mathbf{W} = \mathbf{A}\mathbf{G}^\top - \mathbf{G}\mathbf{A}^\top \qquad (\mathbf{W}^\top = -\mathbf{W}),$$

and move along the **Cayley transform** path

$$\mathbf{A}(\tau) = \left( \mathbf{I} + \tfrac{\tau}{2} \mathbf{W} \right)^{-1} \left( \mathbf{I} - \tfrac{\tau}{2} \mathbf{W} \right) \mathbf{A}.$$

Because the Cayley transform of a skew-symmetric matrix is orthogonal,
$\mathbf{A}(\tau)^\top \mathbf{A}(\tau) = \mathbf{I}_K$ holds **exactly** for every
step size $\tau$ — no re-projection needed. The path satisfies
$\frac{d}{d\tau} \mathcal{F}(\mathbf{A}(\tau))\big|_{\tau=0} = \tfrac{1}{2}\|\mathbf{W}\|_F^2 \ge 0$,
so it is an ascent direction. The step size $\tau$ is chosen by backtracking
line search (halve $\tau$ until the objective improves), and iteration stops
when $\|\mathbf{W}\|_F$ or the improvement falls below tolerance
(`max_sliced_wasserstein_directions`).

A cheaper alternative (`direction_method="random"`) draws a random
orthonormal frame by QR decomposition of a Gaussian matrix — the sliced
analogue of RBIG's random rotations.

---

## The 1D Transform: Rational-Quadratic Spline Gaussianization

Each slice is Gaussianized by the classic composition of an estimated
marginal CDF with the Gaussian quantile function (probit):

$$z = g(x) = \Phi^{-1}\!\big(\hat{F}(x)\big),$$

which maps $x \sim \hat{F}$ to $z \sim \mathcal{N}(0, 1)$ exactly when
$\hat{F}$ is the true CDF. Its derivative — the quantity that enters the
log-determinant — is

$$g'(x) = \frac{\hat{f}(x)}{\phi\big(g(x)\big)},$$

with $\hat{f}$ the density estimate and $\phi$ the standard normal PDF.
Since $\hat f > 0$, the map is strictly increasing and hence invertible.

Rather than evaluating this transform pointwise (and inverting it
numerically), `rbig` bakes it into a **monotonic rational-quadratic (RQ)
spline** (Durkan et al. 2019; class `RQSpline`):

1. Estimate $\hat{F}$ with a Gaussian KDE and place $M$ knots $x_m$ on a
   regular grid over the data support.
2. Set the latent knots to the exact Gaussianization targets
   $y_m = \Phi^{-1}(\hat{F}(x_m))$ (CDF values clipped away from $\{0, 1\}$
   to keep the probit finite).
3. Set the knot derivatives to the analytic values
   $d_m = \hat{f}(x_m) / \phi(y_m)$, clipped below by `alpha`
   (`min_derivative`) so slopes never vanish.

Within bin $m$, with width $w = x_{m+1} - x_m$, height $h = y_{m+1} - y_m$,
slope $s = h/w$, and local coordinate $\xi = (x - x_m)/w \in [0, 1]$, the RQ
interpolant is

$$g(x) = y_m + \frac{h\left[ s\xi^2 + d_m \xi(1 - \xi) \right]}{s + (d_{m+1} + d_m - 2s)\,\xi(1 - \xi)},$$

with derivative

$$g'(x) = \frac{s^2 \left[ d_{m+1}\xi^2 + 2s\xi(1 - \xi) + d_m(1 - \xi)^2 \right]}{\left[ s + (d_{m+1} + d_m - 2s)\,\xi(1-\xi) \right]^2} > 0.$$

The inverse solves a quadratic in $\xi$ **analytically**, so
$g^{-1}(g(x)) = x$ to numerical precision — no iterative root finding.
Outside the outermost knots the spline extends **linearly** with the
boundary derivative, so both $g$ and $g^{-1}$ are defined and smooth on all
of $\mathbb{R}$. This is what gives GIS/SIG graceful out-of-support
behavior: a test point beyond the training range gets a finite, linearly
extrapolated latent value and a finite log-density, whereas an empirical-CDF
transform saturates at $\Phi^{-1}(0^+/1^-) \to \mp\infty$ and must be clipped.

The same primitive backs `SplineGaussianizer`, so raw feature dimensions
(RBIG-style marginal steps) and projected slices (GIS/SIG) share one
implementation — only the 1D data fed in differs.

---

## GIS: Gaussianization via Iterative Slicing

`GIS` maps **data → Gaussian**, oriented toward density estimation. The fit
is greedy:

1. Split $\mathbf{X}$ into train/validation parts.
2. Optionally **whiten** (`PCARotation(whiten=True)`) — centers, rotates to
   principal axes and rescales to unit variance. Whitening removes all
   second-order structure up front so layers spend capacity on higher-order
   structure only; its log-determinant ($-\tfrac{1}{2}\sum_j \log \lambda_j$,
   constant in $\mathbf{x}$) is included in `score_samples`.
3. Repeat, **appending** layers ($\ell = 1, 2, \dots$):
    - Draw a fresh Gaussian target $\mathbf{Z}_\ell$, find
      $\mathbf{A}_\ell = \arg\max \mathcal{F}$ (Stiefel) or draw it at random.
    - Fit $K$ RQ splines on the projections of the current train
      representation; transform train and validation, accumulating the
      validation log-determinant.
    - Update the stopping criterion on the validation representation; stop
      when it fails to improve for `patience` consecutive layers.
4. Keep only the layer prefix up to the best validation iterate.

Each layer strictly reduces (in expectation) the sliced non-Gaussianity: it
makes $K$ one-dimensional marginals of the representation exactly standard
normal, while an orthogonal $(D - K)$-dimensional subspace passes through
untouched, to be attacked by later layers along new learned directions.

**Density estimation** then follows from the change of variables through the
whitener and all layers:

$$\log p(\mathbf{x}) = \sum_{j=1}^{D} \log \phi\big(z_j\big) + \log|\det \mathbf{J}_\text{whiten}| + \sum_{\ell=1}^{L} \sum_{k=1}^{K} \log g'_{\ell,k}\big(\mathbf{a}_{\ell,k}^\top \mathbf{x}^{(\ell-1)}\big),$$

where $\mathbf{x}^{(\ell)}$ is the representation after $\ell$ layers and
$\mathbf{z} = \mathbf{x}^{(L)}$.

## SIG: Sliced Iterative Gaussianization (generative orientation)

`SIG` targets the **Gaussian → data** direction: draw
$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and push it through the
inverse flow (`sample` / `inverse_transform`) to generate data-like samples.

In `rbig`, `SIG` builds the *same* invertible flow as `GIS` (every fitted
layer is exactly invertible, so the two directions are two views of one
bijection) and differs in configuration and intent:

- **Stopping signal.** `SIG` defaults to `stopping_metric="swd"` — the
  sliced Wasserstein distance between the current representation and a
  Gaussian reference. Sample quality is what a generative model cares
  about, and SWD tracks it directly and cheaply. `GIS` defaults to
  `stopping_metric="log_likelihood"`, the density-estimation criterion.
- **Headline methods.** `SIG.sample(n)` and `SIG.inverse_transform(Z)`
  versus `GIS.transform(X)` and `GIS.score_samples(X)`.

In the original paper the two algorithms are also distinguished by the
direction in which layers are *constructed* (SIG prepends layers, matching
samples to data; GIS appends layers, matching data to Gaussian). Because
each `rbig` layer is an exact bijection fitted on the current data
representation, constructing in the forward direction and inverting yields
the same family of flows; `rbig` therefore shares one fitting path
(`_BaseSliced`) for both estimators.

---

## Stopping Criteria

All three iterative models share `StoppingCriterion`
(`rbig._src.convergence`), which tracks a validation metric with a
`patience` window:

| Metric | Definition | Cost | Notes |
|--------|------------|------|-------|
| `"log_likelihood"` | $\frac{1}{N}\sum_i \big[\log \mathcal{N}(\mathbf{z}_i) + \log\|\det \mathbf{J}(\mathbf{x}_i)\|\big]$ on held-out data | needs accumulated log-det | Most principled; directly the density-estimation objective. Default for `GIS`. |
| `"swd"` | Mean $W_1$ over random 1D projections between the representation and a Gaussian sample | cheap ($O(N \log N)$ per projection) | Measures remaining non-Gaussianity; tracks sample quality. Default for `SIG`. |
| `"total_correlation"` | $\sum_j H(z_j) - H(\mathbf{z})$ residual redundancy | histogram entropies | The classic RBIG signal ($\Delta$ multi-information per layer). |

Early stopping keeps the prefix of layers up to the best validation score —
layers added during the patience window are discarded.

---

## RBIG vs GIS vs SIG

| | **RBIG** (`AnnealedRBIG`) | **GIS** (`GIS`) | **SIG** (`SIG`) |
|---|---|---|---|
| **Reference** | Laparra et al. (2011) | Dai & Seljak (2020) | Dai & Seljak (2020) |
| **Direction of construction** | data → Gaussian | data → Gaussian | Gaussian → data (paper); shared flow in `rbig` |
| **Primary use** | Density / IT measures | Density estimation | Sampling / generation |
| **Layer** | Full rotation + $D$ marginal Gaussianizations | $K \le D$ learned slices + RQ splines | same as GIS |
| **Direction choice** | PCA / ICA / random (fixed family) | Max K-SWD on Stiefel manifold (learned) | learned (K-SWD) |
| **Dims transformed per layer** | all $D$ | $K$ (default $D/2$) | $K$ (default $D/2$) |
| **1D transform** | Histogram/KDE CDF + probit | RQ spline (analytic inverse, linear tails) | RQ spline |
| **Out-of-support behavior** | CDF clipping → saturated probit | Linear tail extrapolation, finite log-density | same as GIS |
| **Per-layer cost** | $O(D^3)$ (PCA/ICA) $+ O(ND)$ marginals | Stiefel iterations: $O(T\,(N \log N)K + T\,D^3)$ Cayley solves | same as GIS |
| **Layers needed** | Many cheap layers (50–100+) | Few expensive layers (targeted) | few expensive layers |
| **Stopping (default)** | Total-correlation change | Validation log-likelihood | Sliced Wasserstein distance |
| **Whitening** | Via first PCA rotation | Explicit `whiten=True` pre-transform | `whiten=True` |
| **Strengths** | Simple, robust, well-studied IT estimators; exact rotations are free (log-det 0) | Targeted directions → faster convergence per layer in high-D; smooth analytic inverse; finite OOD densities | Direct control of sample quality; cheap SWD stopping |
| **Weaknesses** | Untargeted rotations waste layers in high-D; boundary saturation | Stiefel optimization adds per-layer cost; K-SWD is non-convex (local optima) | same as GIS; log-likelihood not the stopping target |
| **When to use** | IT measures (entropy, TC, MI); low/moderate D; well-tested baselines | Density estimation, especially higher-D or when few layers desired | Generative sampling, sample-quality-driven workflows |

**Rule of thumb.** RBIG spends many cheap, untargeted layers; GIS/SIG spend
few expensive, targeted ones. When $D$ is small, RBIG's simplicity often
wins. As $D$ grows, random/PCA rotations touch informative directions ever
more rarely (a random direction is nearly orthogonal to any fixed structure
in high-D), while the Stiefel search finds them directly — this is where
sliced methods shine. For information-theoretic measures, prefer RBIG: its
estimators (`rbig_measures`) are built on the total-correlation reduction
accounting.

---

## API Summary

| Concept | Class / function |
|---------|------------------|
| Shared 1D RQ spline | `rbig.RQSpline` |
| 1-Wasserstein distance | `rbig._src.stiefel.wasserstein_1d` |
| Stiefel K-SWD optimizer | `rbig._src.stiefel.max_sliced_wasserstein_directions` |
| Random orthonormal frame | `rbig._src.stiefel.random_orthogonal_directions` |
| One sliced layer | `rbig.SIGLayer` |
| Data → Gaussian model | `rbig.GIS` |
| Generative model | `rbig.SIG` |
| Shared early stopping | `rbig._src.convergence.StoppingCriterion` |
| RBIG baseline | `rbig.AnnealedRBIG` |

See the hands-on notebooks:
[GIS demo](../notebooks/14_sliced_gaussianization.ipynb) ·
[SIG vs RBIG convergence](../notebooks/15_sig_vs_rbig_convergence.ipynb) ·
[density estimation quality](../notebooks/16_gis_density_estimation.ipynb) ·
[sampling quality](../notebooks/17_sig_sampling_quality.ipynb) ·
[boundary behavior](../notebooks/18_sig_boundary_behavior.ipynb).

---

## References

- Dai, B., & Seljak, U. (2020). *Sliced Iterative Normalizing Flows.*
  [arXiv:2007.00674](https://arxiv.org/abs/2007.00674)
- Laparra, V., Camps-Valls, G., & Malo, J. (2011). *Iterative Gaussianization:
  from ICA to Random Rotations.* IEEE Transactions on Neural Networks, 22(4), 537–549.
- Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). *Neural
  Spline Flows.* NeurIPS. [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)
- Wen, Z., & Yin, W. (2013). *A feasible method for optimization with
  orthogonality constraints.* Mathematical Programming, 142, 397–434.
- Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). *Sliced and Radon
  Wasserstein Barycenters of Measures.* Journal of Mathematical Imaging and
  Vision, 51, 22–45.
