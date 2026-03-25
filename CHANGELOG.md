# Changelog

## [0.1.12](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.11...rbig-v0.1.12) (2026-03-25)


### Features

* add orthogonal mode to ICARotation and PicardRotation ([ca49a71](https://github.com/jejjohnson/rbig/commit/ca49a713560d2795fc30c3031b5f8659c1f63e73))


### Bug Fixes

* address PR review comments ([f8c5a3b](https://github.com/jejjohnson/rbig/commit/f8c5a3b8ee5bbed10fe964b0a4fd706b5ae6e25b))
* change PCARotation default to whiten=False for orthogonality ([649466b](https://github.com/jejjohnson/rbig/commit/649466bdae05dd6e32d9b88b880e4684d3629ea8))
* correct HSIC normalization and guard ICC against negative MI ([6376186](https://github.com/jejjohnson/rbig/commit/637618658b1fd686a7b604395af6177ac849cd0e))
* correct pseudoinverse transpose in GaussianRandomProjection.inverse_transform ([ad8c0c4](https://github.com/jejjohnson/rbig/commit/ad8c0c4f3d9d0614bc2162e8d2fb4a3054421c5f))
* enable MathJax rendering in notebook pages ([9f02a73](https://github.com/jejjohnson/rbig/commit/9f02a733c5b8cdd174b8c28d7e95413a071dc6ae))
* replace biased spacing-based density with KDE in log_det_jacobian ([74b758c](https://github.com/jejjohnson/rbig/commit/74b758c6510725eacae652c8f2b5c994068f370c))


### Documentation

* absorb notebook 08 (INNF demo) into 03 and 04, use hex plots ([a67f846](https://github.com/jejjohnson/rbig/commit/a67f84612277cc47e7ad803928bb595c4b78e136))
* add Colab badges and install helpers to all notebooks ([a52e53d](https://github.com/jejjohnson/rbig/commit/a52e53ddee2b479187ea0628694b7cf86adde924))
* add configuration guide for AnnealedRBIG hyperparameters ([2073940](https://github.com/jejjohnson/rbig/commit/2073940da550ce87854eb14f442ed1e3ece44c6b))
* add cross-links between notebooks and notes ([3ec57c5](https://github.com/jejjohnson/rbig/commit/3ec57c596dcdd2a7574235795604d818280f0863))
* add image rotations and dimensionality reduction notebooks ([8471241](https://github.com/jejjohnson/rbig/commit/847124137bd8ff07fb7fdc087b6f20a6de6e401e))
* add quickstart guide with density estimation and IT measures ([f76e810](https://github.com/jejjohnson/rbig/commit/f76e8103277ede193feb974d6300a2520905daa6))
* add RBF kernel HSIC alongside linear CKA in dependence notebook ([4aa4070](https://github.com/jejjohnson/rbig/commit/4aa40708b51799259e6a17801c4c91e924e5c496))
* add rotation choices notebook comparing PCA, ICA, and random ([8b4b8fc](https://github.com/jejjohnson/rbig/commit/8b4b8fc91c30c90dbac1223e926792bc6420cad2))
* add visual roundtrip demos to image rotations notebook ([3510d2c](https://github.com/jejjohnson/rbig/commit/3510d2c7cc439595a2d61eaf0a6c1f5bd3ea8874))
* consolidate duplicate literature files into notes/literature.md ([f20550d](https://github.com/jejjohnson/rbig/commit/f20550de0bc4f7ed6d0783f23f8acbf98a4dfd54))
* consolidate notes, enrich notebooks, fix bugs ([5924a1d](https://github.com/jejjohnson/rbig/commit/5924a1d03c998b0036fbfae347bf127123e606b8))
* cross-link uniformization and marginal_gaussianization notes ([d80685d](https://github.com/jejjohnson/rbig/commit/d80685d2b2a89e48840dbae92dc24e2a10458b11))
* execute all notebooks and fix runtime errors ([206085f](https://github.com/jejjohnson/rbig/commit/206085f4d46bba9145512c0e5aae489fff8c33e5))
* merge normalizing_flows and deep_density_destructors into related_methods ([d9753da](https://github.com/jejjohnson/rbig/commit/d9753da3e19076fc77974a6aa470b39b72095420))
* merge notebooks 01+02 into comprehensive marginal transforms notebook ([df17e8e](https://github.com/jejjohnson/rbig/commit/df17e8edd50c9a28683bffac43b19edb56b757d6))
* merge pdf_estimation and kernel_density_estimation into density_estimation ([8d22d74](https://github.com/jejjohnson/rbig/commit/8d22d74f5678f551ec4c5655b63f142f10db0e6c))
* merge RBIG theory into walkthrough notebook, delete intro.md and gaussianization.md ([8d9eea8](https://github.com/jejjohnson/rbig/commit/8d9eea856627de76019719b776d3d4cff72ce6ca))
* merge uniform_distribution into uniformization and remove exponential_distribution ([bb31f96](https://github.com/jejjohnson/rbig/commit/bb31f961711339d45f5eb6ac3c29c96b017d4e33))
* organize notes into Algorithm, Components, Foundations, Context ([307f34c](https://github.com/jejjohnson/rbig/commit/307f34cf65b5d766b2a6b91e8fe1db92c86f226f))
* re-execute all notebooks after PCA/ICA orthogonality changes ([a8cedb3](https://github.com/jejjohnson/rbig/commit/a8cedb35dad268ff334bb1d0dee6b0d0f4ec6b64))
* use sklearn digits dataset in image rotations notebook ([b6a99a0](https://github.com/jejjohnson/rbig/commit/b6a99a0e059c2cbb69c1c2063535f735ccb91bb4))

## [0.1.11](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.10...rbig-v0.1.11) (2026-03-24)


### Bug Fixes

* **docs:** correct AnnealedRBIG kwargs and misleading notebook title ([4b793fe](https://github.com/jejjohnson/rbig/commit/4b793fecb8295586957356414beaeb0104a6d51e))


### Documentation

* clean up stale notes, expand API reference, add 3 new notebooks ([749cc5e](https://github.com/jejjohnson/rbig/commit/749cc5e505e7f25d7754606a286b173d2bfc7608))
* clean up stale notes, expand API reference, add dependence & IT notebooks ([fdb85db](https://github.com/jejjohnson/rbig/commit/fdb85db178731b6516ad11ccf3578d96cb93cb6a))
* execute new notebooks to include cell outputs ([453c39f](https://github.com/jejjohnson/rbig/commit/453c39f063ec5bf2e7b497f6635fb273008816e3))
* improve scatter plot visibility across all notebooks ([aa6a870](https://github.com/jejjohnson/rbig/commit/aa6a870e0f629e91b712f8f457b1ae531e6383c0))
* improve scatter plot visibility across notebooks ([4d19040](https://github.com/jejjohnson/rbig/commit/4d19040f4e03ee026f177f2a0bc9d58c015ea83b))
* use log p(x) instead of p(x) for density scatter plots ([4cf64b2](https://github.com/jejjohnson/rbig/commit/4cf64b233edb377bb9a75f78a9491d053e854176))

## [0.1.10](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.9...rbig-v0.1.10) (2026-03-24)


### Bug Fixes

* **docs:** render pre-executed ipynb notebooks instead of py sources ([cbf47ac](https://github.com/jejjohnson/rbig/commit/cbf47ac17f9ff5331b25ea9f1dc7b4a32d6a0923))
* **docs:** render pre-executed ipynb notebooks with figures ([446c167](https://github.com/jejjohnson/rbig/commit/446c16707e33ced90cbb6b8ffd635b1d0123f5e1))
* use np.linalg.solve instead of np.linalg.inv in KLD example ([bb05ab3](https://github.com/jejjohnson/rbig/commit/bb05ab37f676027cee4c6c1897be63c61c2a20f7))


### Documentation

* reduce example notebook runtimes ([34b2e27](https://github.com/jejjohnson/rbig/commit/34b2e277c2d65cd9866e7840923077a0015ab382))

## [0.1.9](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.8...rbig-v0.1.9) (2026-03-24)


### Features

* add tqdm progress bars to iterative methods ([020efb0](https://github.com/jejjohnson/rbig/commit/020efb09ba9813270654a4b73a918bf88c300f88))
* add tqdm progress bars to iterative methods ([5035934](https://github.com/jejjohnson/rbig/commit/5035934d87c0626b2ac20437196fa679befb2a6f))


### Bug Fixes

* address PR review comments for tqdm progress bars ([bd45e1f](https://github.com/jejjohnson/rbig/commit/bd45e1fcc44bfd01f410e61fb7881f76e9052230))
* use property for zero_tolerance deprecation (sklearn-compatible) ([891ebf0](https://github.com/jejjohnson/rbig/commit/891ebf09dc7f0b40eb7628e3ad8ab9bc048fa1e2))

## [0.1.8](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.7...rbig-v0.1.8) (2026-03-24)


### Features

* add scikit-learn compatibility to all RBIG estimators ([db86a38](https://github.com/jejjohnson/rbig/commit/db86a38b596a0a59cde1c20d58f49dc8fd1ade9d))
* add scikit-learn compatibility to all RBIG estimators ([cab9249](https://github.com/jejjohnson/rbig/commit/cab9249cfe670c1377f85b25a0d8723c8b3dfaea))


### Bug Fixes

* address PR [#59](https://github.com/jejjohnson/rbig/issues/59) review comments ([2bdf1bb](https://github.com/jejjohnson/rbig/commit/2bdf1bbd27ea667235a41845d6e6ad9c86fab39e))
* address PR review comments on Makefile and README ([d0d65e8](https://github.com/jejjohnson/rbig/commit/d0d65e84c57b8e04225b0cda67113df4ba0ee0d2))
* skip flaky check_methods_subset_invariance in CI ([cbb12d1](https://github.com/jejjohnson/rbig/commit/cbb12d1f4a3306a13dada54aca430f62e84f9058))


### Documentation

* reduce example notebook runtimes from minutes to seconds ([f124f84](https://github.com/jejjohnson/rbig/commit/f124f84e83c92b5a62f3f8e78a0b9eed3dfa26c9))

## [0.1.7](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.6...rbig-v0.1.7) (2026-03-24)


### Bug Fixes

* address PR [#51](https://github.com/jejjohnson/rbig/issues/51) review comments ([cf6f9ab](https://github.com/jejjohnson/rbig/commit/cf6f9abbd18085b47bbe23818f8748474341ad4f))
* address PR [#52](https://github.com/jejjohnson/rbig/issues/52) review comments ([bc287fe](https://github.com/jejjohnson/rbig/commit/bc287fe49ba80318d868febd4821c140cad5e539))
* **build:** use --group docs for mkdocs and --all-groups for install ([026c095](https://github.com/jejjohnson/rbig/commit/026c095102bf092f1c65d0c9aa18425166b84996))


### Documentation

* add AI agent instructions and code review standards ([68527d2](https://github.com/jejjohnson/rbig/commit/68527d23cadd9c84bd5909b924bd8016c728ed97))
* add AI agent instructions and code review standards ([40d36bd](https://github.com/jejjohnson/rbig/commit/40d36bd6edc36a92b766af625f211c53d5821f03)), closes [#39](https://github.com/jejjohnson/rbig/issues/39) [#40](https://github.com/jejjohnson/rbig/issues/40) [#41](https://github.com/jejjohnson/rbig/issues/41) [#42](https://github.com/jejjohnson/rbig/issues/42)
* add plans policy to AGENTS.md and copilot-instructions ([4582d8b](https://github.com/jejjohnson/rbig/commit/4582d8b92fed8b3861061428ddf2dd1c75340e58))
* add PR review comment resolution policy ([dafee04](https://github.com/jejjohnson/rbig/commit/dafee04c222f46acb62f73c54ef951ca22687770))
* clean up notebooks, remove matplotlib.use("Agg") and reduce n_layers ([61c5be0](https://github.com/jejjohnson/rbig/commit/61c5be0d28a6f9dab4412e64fddae4a841ac4b10))
* clean up notes formatting, fix typos, and improve coherence ([94b87b0](https://github.com/jejjohnson/rbig/commit/94b87b0c7eea2d578a394fa9a1ed99572fdad2c0))
* clean up notes, notebooks, and build config ([e94aa17](https://github.com/jejjohnson/rbig/commit/e94aa17a2305a2c7414231adda516eb99e753b69))
* fix duplicates, notation consistency, and fill stub sections ([68bd1b6](https://github.com/jejjohnson/rbig/commit/68bd1b6cf9e69f778fcea56383e90e676d121fc2))
* fix text coherence, notation consistency, and fill empty sections ([e0fbac8](https://github.com/jejjohnson/rbig/commit/e0fbac8c9549a33f9aa89bf5ba53fcc4f0baf223))
* fix trailing whitespace and missing EOF newlines in notes ([2ab8267](https://github.com/jejjohnson/rbig/commit/2ab826795fb81c2ef9fe13c522b6d14627d41f32))
* port literature review and refactoring notes from Unsorted ([24da85d](https://github.com/jejjohnson/rbig/commit/24da85d36967ea9e16f9e030f68e0069258e4303))
* remove broken #algorithms anchor from normalizing_flows TOC ([2c0c123](https://github.com/jejjohnson/rbig/commit/2c0c123d9e09d5887e123e0e8b80241124dfc4ab))
* remove legacy .ipynb notebooks superseded by .py versions ([84bf255](https://github.com/jejjohnson/rbig/commit/84bf2554e3fc05a61133f3d993efaeed23038e8a))
* remove legacy Notes/Unsorted directory ([095e42c](https://github.com/jejjohnson/rbig/commit/095e42c46c7a11a6c6bd30a10bcc1bd6890708cd))

## [0.1.6](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.5...rbig-v0.1.6) (2026-03-07)


### Features

* add bin_estimation, generate_batches, and entropy_histogram ([04252ca](https://github.com/jejjohnson/rbig/commit/04252ca0fb1ac68ee492ef9ae5b7c5a412701613))
* add jacobian, predict_proba domain, and auto_tol to AnnealedRBIG ([b336c80](https://github.com/jejjohnson/rbig/commit/b336c806c7ac315b2e335b7d890bedc1627f0fe7))
* add make_cdf_monotonic and histogram CDF to MarginalUniformize ([ca57971](https://github.com/jejjohnson/rbig/commit/ca57971d954de22409116cee61d54210cb0916f2))
* add negentropy_kde for KDE-based negentropy estimation ([53f141c](https://github.com/jejjohnson/rbig/commit/53f141c09feccba5606aba43cdad71c33856c004)), closes [#18](https://github.com/jejjohnson/rbig/issues/18)
* add rotation="random" as first-class option in AnnealedRBIG ([2f2ad71](https://github.com/jejjohnson/rbig/commit/2f2ad71fd17bd3b805adfa05291b19756783a622))
* complete legacy RBIG migration ([f9b0dea](https://github.com/jejjohnson/rbig/commit/f9b0dea50d45c0dbd1df83adcada90c0c8a746a3))
* export new legacy migration symbols from public API ([8eaf44c](https://github.com/jejjohnson/rbig/commit/8eaf44c35f15d43cac8a90e13b6758061455db04))


### Bug Fixes

* address PR [#24](https://github.com/jejjohnson/rbig/issues/24) review comments ([be2290d](https://github.com/jejjohnson/rbig/commit/be2290dc1fd4427f933b77a06001fd7eb8bcafe8))
* resolve ruff lint warnings in test_new_model ([a6b3375](https://github.com/jejjohnson/rbig/commit/a6b3375e1be91a40d8c3029ec1eb8ed40042e1cb))

## [0.1.5](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.4...rbig-v0.1.5) (2026-03-02)


### Documentation

* add comprehensive NumPy-style docstrings to base.py and model.py ([037afec](https://github.com/jejjohnson/rbig/commit/037afecaa9a1b4bbbc2c15c3e5a0f15a4fea372e))
* add comprehensive NumPy-style docstrings to densities, metrics, parametric ([1a57171](https://github.com/jejjohnson/rbig/commit/1a571717bebd530f504c5b20585f2e5ad58c755d))
* add comprehensive NumPy-style docstrings to image, xarray_st, xarray_image modules ([ad8264e](https://github.com/jejjohnson/rbig/commit/ad8264e3ac283137470af52105852e97a7d55aa1))
* add comprehensive NumPy-style docstrings to marginal.py and rotation.py ([f4dfad2](https://github.com/jejjohnson/rbig/commit/f4dfad2c222981515e019c3dd3e7dc673e2733f1))
* address review feedback on docstring accuracy ([cfc1d32](https://github.com/jejjohnson/rbig/commit/cfc1d32a7408e7c2f68c7d36b0b6b0ade64c180b))
* comprehensive NumPy-style docstring overhaul for rbig/_src/ ([d98c8ab](https://github.com/jejjohnson/rbig/commit/d98c8abb9a367b3b75dfa38d0e326904b8b96c5d))

## [0.1.4](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.3...rbig-v0.1.4) (2026-03-02)


### Bug Fixes

* address all review comments — typos, math equations, MathJax config, notebook paths, gh-pages workflow ([debb07d](https://github.com/jejjohnson/rbig/commit/debb07ddaca199019576ea20e31f8640fde369c3))


### Documentation

* migrate Notes/Unsorted → docs/notes, add mkdocs.yml, overhaul README, add gh-pages deployment ([e79c5e6](https://github.com/jejjohnson/rbig/commit/e79c5e6a1ad3009f6045714e0b87d94afb94adc6))
* migrate Notes/Unsorted to docs/notes, add mkdocs.yml, update README ([e6b106c](https://github.com/jejjohnson/rbig/commit/e6b106cae07fcdce189bb3b7c881ff3658bf530e))

## [0.1.3](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.2...rbig-v0.1.3) (2026-03-01)


### Features

* port demo notebooks to Jupytext percent-format with new rbig API ([6fa6ebc](https://github.com/jejjohnson/rbig/commit/6fa6ebcef5c887588527161b3a599a9d0d1ef5e3))
* port demo notebooks to Jupytext percent-format with new rbig API ([a82a559](https://github.com/jejjohnson/rbig/commit/a82a5599a87736bedacf68df42017e4d4f3dcf19))


### Bug Fixes

* address review comments on notebook backend, assertions, and IT estimates ([2c5a729](https://github.com/jejjohnson/rbig/commit/2c5a729f95ec7ac3079296ffd0f6d49f404de055))

## [0.1.2](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.1...rbig-v0.1.2) (2026-03-01)


### Features

* port new bijectors, Gaussianizers, rotations, image transforms, metrics, and parametric utilities from jej_vc_snippets ([06061b3](https://github.com/jejjohnson/rbig/commit/06061b3f131ed03323c0a39f8a06ec3503b086ab))

## [0.1.1](https://github.com/jejjohnson/rbig/compare/rbig-v0.1.0...rbig-v0.1.1) (2026-03-01)


### Features

* migrate RBIG code to new package structure ([31cb293](https://github.com/jejjohnson/rbig/commit/31cb29383143bf6b47a11c01d42b2eb5bc66d08a))
* migrate RBIG repository to new package structure ([e0a977a](https://github.com/jejjohnson/rbig/commit/e0a977a86b92373b8b2a76b76b7b8c8d2a7b045f))
* migrate RBIG to modern package structure with _src layout ([f79ae1a](https://github.com/jejjohnson/rbig/commit/f79ae1a5015aab03c588aa1b4a65460ca2e92671))


### Bug Fixes

* address PR review comments on Jacobian, Makefile, typing, and imports ([6a65f3e](https://github.com/jejjohnson/rbig/commit/6a65f3e2aac390cf814c64c414a811e53131b825))

## [0.1.0](https://github.com/jejjohnson/rbig/releases/tag/v0.1.0)

### Features

* Initial release with migrated code from jej_vc_snippets
