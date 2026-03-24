# Uniformization

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Date Updated: 11-03-2020



## Forward Transformation

In this step, we estimate the forward transformation of samples from $\mathcal{X}$ to the uniform distribution $\mathcal{U}$. The relation is:

$$
u = F_\theta(x)
$$

where $F_\theta(\cdot)$ is the empirical Cumulative distribution function (CDF) for $\mathcal{X}$, and $u$ is drawn from a uniform distribution, $u\sim \mathcal{U}([0,1])$.

!!! note "Visual walkthrough"
    See the [Marginal Uniformization notebook](../notebooks/01_marginal_uniformization.ipynb) for interactive plots of data distribution, empirical CDF, and resulting uniform samples.



### Boundary Issues

The bounds for $\mathcal{U}$ are $[0,1]$ and the bounds for $\mathcal{X}$ are `X.min()` and `X.max()`. So function $F_\theta$ will be between 0 and 1 and the support $F_\theta$ will be between the limits for $\mathcal{X}$. We have two options for dealing with this:

**Map Outliers to Boundaries**

The simplest method is to map all points outside the limits to the boundaries. This allows us to deal with points that lie outside the support of the estimated distribution.


!!! note "Boundary visualization"
    See the [Boundary Issues notebook](../notebooks/07_boundary_issues.ipynb) for plots comparing boundary handling strategies.

**Widen the Limits of the Support**

This is the harder option. This will essentially squish the CDF function near the middle and widen the tails.


## Reverse Transformation

This isn't really useful because we don't really want to draw samples from our distribution $x \sim \mathcal{X}$ only to project them to a uniform distribution $\mathcal{U}$. What we really want to draw samples from the uniform distribution $u \sim \mathcal{U}$ and then project them into our data distribution $\mathcal{X}$.

We can simply take the inverse of our function $F(\cdot)$ to go from $\mathcal{U}$ to $\mathcal{X}$.

$$
x = F^{-1}(u)
$$

where $u \sim \mathcal{U}[0,1]$. This is the inverse of the CDF, known in probability theory as the quantile function or inverse distribution function.

Assuming that $F$ is differentiable and invertible, we should be able to generate data points for our data distribution from a uniform distribution. We need to be careful of the bounds as we are mapping data from $[0,1]$ to the support $[x_\text{min}, x_\text{max}]$, which can cause issues at the boundaries.


## Derivative

The key property we exploit is that the derivative of the CDF $F$ is the PDF $f$. Recall the uniformization relationship:

$$
u = F_\theta(x)
$$

where $F_\theta(\cdot)$ is the empirical cumulative distribution function (ECDF) of $\mathcal{X}$.

**Proof**:

Let $F(x) = \int_{-\infty}^{x}f(t) \, dt$ from the fundamental theorem of calculus. The derivative is $f(x)=\frac{d F(x)}{dx}$. Then that means

$$
F(b)-F(a)=\int_a^b f(t) dt
$$

Since $\lim_{a \rightarrow -\infty} F(a) = 0$, we have $F(x) = \int_{-\infty}^{x} f(t) \, dt$.

So the derivative of $F(x)$ is:

$$
\begin{aligned}
\frac{d F(x)}{dx}
&= \frac{d}{dx} \left[ F(x) - \lim_{a \rightarrow - \infty} F(a)  \right] \\
&= f(x)
\end{aligned}
$$

### Log Abs Determinant Jacobian

Since the Jacobian of the uniformization transform $u = F_\theta(x)$ is just $f_\theta(x)$ (the PDF), the log absolute determinant Jacobian is:

$$\log \left| \frac{du}{dx} \right| = \log f_\theta(x)$$

This decomposition is useful for computing composite transformations and optimizing the negative log-likelihood. In practice, we add a small regularization constant $\alpha$ to avoid $\log(0)$ when estimated probabilities are exactly zero.


## Probability (Computing the Density)

So now, we can take it a step further and estimate densities. We don't inherently know the density of our dataset $\mathcal{X}$ but we do know the density of $\mathcal{U}$. So we can use this information by means of the **change of variables** formula.

$$
p_{\mathcal{X}}(x) = p_{\mathcal{U}}(u) \; \left| \frac{d u}{d x} \right|
$$

There are a few things we can do to this equation that simplify this expression. Firstly, because we are doing a uniform distribution, the probability is 1 everywhere. So the first term $p_{\mathcal{U}}(u)$ can cancel. So we're left with just:

$$
p_{\mathcal{X}}(x) =  \left| \frac{d u}{d x} \right|
$$

The second thing is that we explicitly assigned $u$ to be equal to the CDF of $x$, $u = F(x)$. So we can plug this term into the equation to obtain

$$
p_{\mathcal{X}}(x) =  \left| \frac{d F(x)}{d x} \right|
$$

But we know by definition that the derivative of $F(x)$ (the CDF) is the PDF $f(x)$. So we actually have the equation:

$$
p_{\mathcal{X}}(x) =  f_\theta(x)
$$

So they are equivalent. This is very redundant as we actually don't know the PDF so saying that you can find the PDF of $\mathcal{X}$ by knowing the PDF is meaningless. However, we do this transformation in order to obtain a nice property of uniform distributions in general which we will use in the next section.
