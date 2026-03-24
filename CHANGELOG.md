# Changelog

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
