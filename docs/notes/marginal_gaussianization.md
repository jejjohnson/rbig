# Marginal Gaussianization — Quick Reference

> For theory, code examples, and a comparison of all 7 available methods, see
> the [Marginal Transforms notebook](../notebooks/01_marginal_transforms.ipynb).

---

## Pipeline

$$x_d \xrightarrow{F_d} u_d \xrightarrow{\Phi^{-1}} z_d$$

1. Estimate the CDF $F_d$ for each feature independently
2. Apply the CDF to obtain $u_d \in [0, 1]$
3. Apply the probit $\Phi^{-1}$ to map to $\mathcal{N}(0,1)$

The Jacobian is diagonal: $\log|\det J| = \sum_d \log|dz_d / dx_d|$

---

## Available Methods

| Class | CDF Method | Best for |
|-------|-----------|----------|
| `MarginalUniformize` | Empirical (rank) | Uniformization step |
| `MarginalGaussianize` | Empirical + probit | Default in `RBIGLayer` |
| `MarginalKDEGaussianize` | KDE + probit | Smooth CDF, small data |
| `QuantileGaussianizer` | Quantile (sklearn) | Large data |
| `KDEGaussianizer` | KDE + probit (bijector) | Analytic Jacobian |
| `GMMGaussianizer` | GMM CDF + probit | Multimodal marginals |
| `SplineGaussianizer` | PCHIP spline | Smooth + fast + accurate |

---

## Log Determinant Jacobian

$$\frac{dF^{-1}}{dx} = \frac{1}{f(F^{-1}(x))} \implies \log\left|\frac{dF^{-1}}{dx}\right| = -\log f(F^{-1}(x))$$

See [Uniformization](uniformization.md) for the full derivation.

---

## Resources

* Quantile Transformation — [sklearn](https://stats.stackexchange.com/questions/325570/quantile-transformation-with-gaussian-distribution-sklearn-implementation) | [PyTorch](https://github.com/MilesCranmer/differentiable_quantile_transform)
* KDE — [Density Estimation](density_estimation.md#kernel-density-estimation)
* Splines — [Neural Spline Flows](https://github.com/bayesiains/nsf) | [TF Probability](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/rational_quadratic_spline.py)
