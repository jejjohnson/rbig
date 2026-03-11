# Gaussianization

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Notebooks:
  * [1D Gaussianization](https://colab.research.google.com/drive/1C-hP2XCii-DLwLmK1wyvET095ZLQTdON)

---

- [Why Gaussianization?](#why-gaussianization)
- [Main Idea](#main-idea)
  - [Loss Function](#loss-function)
    - [Negentropy](#negentropy)
  - [Methods](#methods)
    - [Projection Pursuit](#projection-pursuit)
    - [Gaussianization](#gaussianization)
    - [RBIG](#rbig)
- [References](#references)

## Why Gaussianization?

> **Gaussianization**: Transforms multidimensional data into multivariate Gaussian data.

It is notorious that we say "assume our data is Gaussian". We do this all of the time in practice. It's because Gaussian data typically has nice properties, e.g. closed-form solutions, linear dependence structure, and analytical expressions for entropy and KL-divergence. But as sensors get better, data gets bigger and algorithms get better, this assumption does not always hold.

However, what if we could make our data Gaussian? If it were possible, then all of the nice properties of Gaussians can be used as our data is actually Gaussian. How is this possible? Well, we use a series of invertible transformations to transform our data $\mathcal X$ to the Gaussian domain $\mathcal Z$. The logic is that by independently transforming each dimension of the data followed by some rotation will eventually converge to a multivariate dataset that is completely Gaussian.

We can achieve statistical independence of data components. This is useful for the following reasons:

* We can process dimensions independently
* We can alleviate the curse of dimensionality
* We can tackle the PDF estimation problem directly
  * With PDF estimation, we can sample and assign probabilities. It really is the holy grail of ML models.
* We can apply and design methods that assume Gaussianity of the data
* Get insight into the data characteristics

---

## Main Idea

The idea of the Gaussianization frameworks is to transform some data distribution $\mathcal{D}$ to an approximate Gaussian distribution $\mathcal{N}$. Let $x$ be some data from our original distribution, $x\sim \mathcal{D}$ and $\mathcal{G}_{\theta}(\cdot)$ be the transformation to the Normal distribution $\mathcal{N}(0, \mathbf{I})$.
$$z=\mathcal{G}_{\theta}(x)$$

where:
* $x\sim$Data Distribution
* $\theta$ - Parameters of transformation
* $\mathcal{G}$ - family of transformations from Data Distribution to Normal Distribution, $\mathcal{N}$.
* $z\sim\mathcal{N}(0, \mathbf{I})$


If the transformation is differentiable, we have a clear relationship between the input and output variables by means of the **change of variables transformation**:

$$
\begin{aligned}
p_\mathbf{x}(\mathbf{x})
&= p_\mathbf{y} \left[ \mathcal{G}_\theta(\mathbf{x})  \right] \left| \nabla_\mathbf{x} \mathcal{G}_\theta(\mathbf{x}) \right|
\end{aligned}
$$

where:

* $\left| \cdot \right|$ - absolute value of the matrix determinant
* $P_z \sim \mathcal{N}(0, \mathbf{I})$
* $\mathcal{P}_x$ - determined solely by the transformation of variables.


We can say that $\mathcal{G}_{\theta}$ provides an implicit density model on $x$ given the parameters $\theta$.

---

### Loss Function

The objective is to minimize the total correlation (or equivalently maximize the negentropy), as described in the original [paper][1].

#### Negentropy

Negentropy measures the departure of a distribution from Gaussianity. It is defined as $J(\mathbf{x}) = H(\mathbf{x}_\text{gauss}) - H(\mathbf{x})$ where $H(\mathbf{x}_\text{gauss})$ is the entropy of a Gaussian with the same covariance as $\mathbf{x}$.

### Methods

#### Projection Pursuit

Projection pursuit seeks low-dimensional projections of multivariate data that reveal interesting structure by iteratively finding directions that maximize non-Gaussianity.

#### Gaussianization

See [Marginal Gaussianization](marginal_gaussianization.md) for the dimension-wise Gaussianization transform.

#### RBIG

See [RBIG](rbig.md) for the full Rotation-Based Iterative Gaussianization algorithm.

---

## References

* Iterative Gaussianization: From ICA to Random Rotations - Laparra et al. - [Paper][1]

[1]: https://www.uv.es/lapeva/papers/Laparra11.pdf "Iterative Gaussianization: From ICA to Random Rotations - Laparra et al. - IEEE TNNs (2011)"
