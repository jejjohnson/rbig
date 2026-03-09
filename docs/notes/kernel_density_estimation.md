# Kernel Density Estimation

Kernel Density Estimation (KDE) is a non-parametric method for estimating the probability density function of a random variable. Given $N$ samples $\{x_1, \ldots, x_N\}$, the KDE estimate is:

$$\hat{p}(x) = \frac{1}{Nh} \sum_{i=1}^{N} K\left(\frac{x - x_i}{h}\right)$$

where $K(\cdot)$ is a kernel function (typically Gaussian) and $h > 0$ is the bandwidth parameter.

### Bandwidth Selection

The bandwidth $h$ controls the smoothness of the estimate:

* **Too small**: overfitting, noisy estimate
* **Too large**: oversmoothing, loss of detail

Common selection methods include Scott's rule ($h = 1.06 \hat{\sigma} N^{-1/5}$) and Silverman's rule.

### Role in RBIG

In the RBIG pipeline, KDE is one option for estimating the marginal CDF $F_d(x_d)$ during the [uniformization](uniformization.md) step. See also [PDF Estimation](pdf_estimation.md) for alternative approaches.

## Resources

### Built-In

* Jake Vanderplas - [In Depth: Kernel Density Estimation](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html)
