# Uniform Distribution

The continuous uniform distribution on $[a, b]$ is defined by:

**PDF**

$$p(x) = \frac{1}{b - a}, \quad a \leq x \leq b$$

**CDF**

$$F(x) = \frac{x - a}{b - a}, \quad a \leq x \leq b$$

**Entropy**

For the multivariate case with independent marginals on $[a_d, b_d]$:

$$H(x) = \log \left[ \prod_{d=1}^{D}(b_d - a_d) \right]$$

For the standard uniform on $[0, 1]^D$, this simplifies to $H(x) = 0$.

### Role in RBIG

The uniform distribution is the intermediate target in the [uniformization](uniformization.md) step of RBIG. Applying the marginal CDF $F_d$ to each dimension maps the data to $[0, 1]$, producing a uniform marginal distribution before the subsequent [Gaussianization](marginal_gaussianization.md) step.
