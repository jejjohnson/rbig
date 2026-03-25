# RBIG — Rotation-Based Iterative Gaussianization

Rotation-Based Iterative Gaussianization — density estimation, IT measures, and generative modeling.

??? info "Abstract From Paper"
    > Most signal processing problems involve the challenging task of multidimensional probability density function (PDF) estimation. In this work, we propose a solution to this problem by using a family of Rotation-based Iterative Gaussianization (RBIG) transforms. The general framework consists of the sequential application of a univariate marginal Gaussianization transform followed by an orthonormal transform. The proposed procedure looks for differentiable transforms to a known PDF so that the unknown PDF can be estimated at any point of the original domain. In particular, we aim at a zero mean unit covariance Gaussian for convenience. RBIG is formally similar to classical iterative Projection Pursuit (PP) algorithms. However, we show that, unlike in PP methods, the particular class of rotations used has no special qualitative relevance in this context, since looking for interestingness is not a critical issue for PDF estimation. The key difference is that our approach focuses on the univariate part (marginal Gaussianization) of the problem rather than on the multivariate part (rotation). This difference implies that one may select the most convenient rotation suited to each practical application. The differentiability, invertibility and convergence of RBIG are theoretically and experimentally analyzed. Relation to other methods, such as Radial Gaussianization (RG), one-class support vector domain description (SVDD), and deep neural networks (DNN) is also pointed out. The practical performance of RBIG is successfully illustrated in a number of multidimensional problems such as image synthesis, classification, denoising, and multi-information estimation.

---

## Installation

=== "pip"

    ```bash
    pip install rbig
    ```

=== "uv"

    ```bash
    uv add rbig
    ```

=== "extras"

    ```bash
    pip install "rbig[image]"   # wavelet/DCT image support
    pip install "rbig[xarray]"  # spatiotemporal xarray integration
    pip install "rbig[all]"     # everything
    ```

---

## Quick Links

- [Quickstart](quickstart.md) — From install to results in 5 minutes
- [RBIG Walk-Through](notebooks/03_rbig_walkthrough.ipynb) — Theory + hands-on tutorial
- [Notes](notes/rbig.md) — Mathematical reference
- [API Reference](api/reference.md) — Full API documentation

---

??? info "Other Resources"

    * Original Webpage - [isp.uv.es](http://isp.uv.es/rbig.html)
    * Original MATLAB Code - [webpage](http://isp.uv.es/code/featureextraction/RBIG_toolbox.zip)
    * Original Python Code - [spencerkent/pyRBIG](https://github.com/spencerkent/pyRBIG)
    * [Iterative Gaussianization: from ICA to Random Rotations](https://arxiv.org/abs/1602.00229) - Laparra et al (2011)
