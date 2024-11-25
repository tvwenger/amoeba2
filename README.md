# amoeba2 <!-- omit in toc -->

![publish](https://github.com/tvwenger/amoeba2/actions/workflows/publish.yml/badge.svg)
![tests](https://github.com/tvwenger/amoeba2/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/amoeba2/badge/?version=latest)](https://amoeba2.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/tvwenger/amoeba2/graph/badge.svg?token=QCDHJB3KWR)](https://codecov.io/gh/tvwenger/amoeba2)

Automated Molecular Excitation Bayesian line-fitting Algorithm

`amoeba2` is a Bayesian model of the 1612, 1665, 1667, and 1720 MHz hyperfine transitions of OH written in the [`bayes_spec`](https://github.com/tvwenger/bayes_spec) spectral line modeling framework. `amoeba2` is inspired by [AMOEBA](https://github.com/AnitaPetzler/AMOEBA) and [Petzler et al. (2021)](https://iopscience.iop.org/article/10.3847/1538-4357/ac2f42).

Read below to get started, and check out the tutorials here: https://amoeba2.readthedocs.io

- [Installation](#installation)
  - [Basic Installation](#basic-installation)
  - [Development Installation](#development-installation)
- [Notes on Physics \& Radiative Transfer](#notes-on-physics--radiative-transfer)
- [Models](#models)
  - [`AbsorptionModel`](#absorptionmodel)
    - [`mainline_pos_tau`](#mainline_pos_tau)
  - [`EmissionAbsorptionModel`](#emissionabsorptionmodel)
    - [`mainline_pos_tau`](#mainline_pos_tau-1)
  - [`ordered`](#ordered)
- [Syntax \& Examples](#syntax--examples)
- [Issues and Contributing](#issues-and-contributing)
- [License and Copyright](#license-and-copyright)


# Installation

## Basic Installation

Install with `pip` in a `conda` virtual environment:
```
conda create --name amoeba2 -c conda-forge pymc pytensor pip
conda activate amoeba2
pip install amoeba2
```

## Development Installation

Alternatively, download and unpack the [latest release](https://github.com/tvwenger/amoeba2/releases/latest), or [fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and contribute to the development of `amoeba2`!

Install in a `conda` virtual environment:
```
cd /path/to/amoeba2
conda env create -f environment.yml
conda activate amoeba2-dev
pip install -e .
```

# Notes on Physics & Radiative Transfer

All models in `amoeba2` apply the same physics and equations of radiative transfer. 

The transition optical depth is taken from [Magnum & Shirley (2015) equation 29](https://ui.adsabs.harvard.edu/abs/2015PASP..127..266M/abstract). The excitation temperature is allowed to vary between transitions (a non-LTE assumption) and clouds. The excitation temperatures of the 1612, 1665, and 1667 MHz transitions are free, whereas that of the 1720 MHz transition is derived from the excitation temperature sum rule.

The radiative transfer is calculated explicitly assuming an off-source background temperature `bg_temp` (see below) similar to [Magnum & Shirley (2015) equation 23](https://ui.adsabs.harvard.edu/abs/2015PASP..127..266M/abstract). By default, the clouds are ordered from *nearest* to *farthest*, so optical depth effects (i.e., self-absorption) may be present.

Notably, since these are *forward models*, we do not make assumptions regarding the optical depth or the Rayleigh-Jeans limit. These effects are *predicted* by the model. There is one exception: the `ordered` argument, [described below](#ordered).

# Models

The models provided by `amoeba2` are implemented in the [`bayes_spec`](https://github.com/tvwenger/bayes_spec) framework. `bayes_spec` assumes that the source of spectral line emission can be decomposed into a series of "clouds", each of which is defined by a set of model parameters. Here we define the models available in `amoeba2`.

## `AbsorptionModel`

`AbsorptionModel` is a model that predicts the OH hyperfine absorption (`1-exp(-tau)`) spectra. The `SpecData` keys for this model must be "absorption_1612", "absorption_1665", "absorption_1667", and "absorption_1720". The following diagram demonstrates the relationship between the free parameters (empty ellipses), deterministic quantities (rectangles), model predictions (filled ellipses), and observations (filled, round rectangles). Many of the parameters are internally normalized (and thus have names like `_norm`). The subsequent tables describe the model parameters in more detail.

![absorption model graph](docs/source/notebooks/absorption_model.png)

| Cloud Parameter<br>`variable` | Parameter                 | Units    | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}` | Default<br>`prior_{variable}` |
| :---------------------------- | :------------------------ | :------- | :------------------------------------------------------- | :---------------------------- |
| `tau`                         | Line-center optical depth | ``       | $\tau \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$            | `[0.1, 0.1]`                  |
| `log10_depth`                 | log10 line-of-sight depth | `pc`     | $\log_{10} d \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$     | `[0.0, 0.25]`                 |
| `log10_Tkin`                  | log10 kinetic temperature | `K`      | $\log_{10} T_K \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$   | `[2.0, 1.0]`                  |
| `velocity`                    | Velocity                  | `km s-1` | $V \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$               | `[0.0, 10.0]`                 |

| Hyper Parameter<br>`variable` | Parameter                                        | Units    | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}`              | Default<br>`prior_{variable}` |
| :---------------------------- | :----------------------------------------------- | :------- | :-------------------------------------------------------------------- | :---------------------------- |
| `log10_nth_fwhm_1pc`          | Non-thermal broadening at 1 pc                   | `km s-1` | $\log_{10}\Delta V_{\rm 1 pc} \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$ | `[0.2, 0.1]`                  |
| `log10_depth_nth_fwhm_power`  | Non-thermal broadening vs. depth power law index | ``       | $\alpha \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$                       | `[0.4, 0.1]`                  |
| `baseline_coeffs`             | Normalized polynomial baseline coefficients      | ``       | $\beta_i \sim {\rm Normal}(mu=0, \sigma=p_i)$                         | `[1.0]*(baseline_degree + 1)` |

### `mainline_pos_tau`

An additional parameter to `AbsorptionModel` is `mainline_pos_tau`. If `True`, then the mainline (1665 MHz and 1667 MHz) optical depths are required to be positive by changing the prior distribution as follows.

| Cloud Parameter<br>`variable` | Parameter                 | Units | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}` | Default<br>`prior_{variable}` |
| :---------------------------- | :------------------------ | :---- | :------------------------------------------------------- | :---------------------------- |
| `tau`                         | Line-center optical depth | ``    | $\tau \sim {\rm HalfNormal}(\sigma=p_1)$                 | `[0.1, 0.1]`                  |


## `EmissionAbsorptionModel`

`EmissionAbsorptionModel` is a more physically motivated model that also predicts the brightness temperature spectra assuming a given background source brightness temperature (where `bg_temp` is in `K` and is supplied during model initialization; `EmissionAbsorptionModel(bg_temp=3.77)` is the default). The `SpecData` keys for this model must be "absorption_1612", "absorption_1665", "absorption_1667", "absorption_1720", "emission_1612", "emission_1665", "emission_1667", and "emission_1720". The following diagram demonstrates the model, and the subsequent table describe the additional model parameters.

![emission absorption model graph](docs/source/notebooks/emission_absorption_model.png)


| Cloud Parameter<br>`variable` | Parameter                                   | Units    | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}` | Default<br>`prior_{variable}` |
| :---------------------------- | :------------------------------------------ | :------- | :------------------------------------------------------- | :---------------------------- |
| `log10_N0`                    | log10 column density in lowest energy state | `cm-2`   | $\log_{10} N_0 \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$   | `[13.0, 1.0]`                 |
| `log_boltz_factor`            | log Boltzmann factor (`-h*freq/(k*Tex)`)    | ``       | $\ln B \sim {\rm Normal}(\mu=p_0, \sigma=p_1)            | `[-0.1, 0.1]`                 |
| `log10_depth`                 | log10 line-of-sight depth                   | `pc`     | $\log_{10} d \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$     | `[0.0, 0.25]`                 |
| `log10_Tkin`                  | log10 kinetic temperature                   | `K`      | $\log_{10} T_K \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$   | `[2.0, 1.0]`                  |
| `velocity`                    | Velocity                                    | `km s-1` | $V \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$               | `[0.0, 10.0]`                 |

| Hyper Parameter<br>`variable` | Parameter                                        | Units    | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}`              | Default<br>`prior_{variable}` |
| :---------------------------- | :----------------------------------------------- | :------- | :-------------------------------------------------------------------- | :---------------------------- |
| `log10_nth_fwhm_1pc`          | Non-thermal broadening at 1 pc                   | `km s-1` | $\log_{10}\Delta V_{\rm 1 pc} \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$ | `[0.2, 0.1]`                  |
| `log10_depth_nth_fwhm_power`  | Non-thermal broadening vs. depth power law index | ``       | $\alpha \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$                       | `[0.4, 0.1]`                  |
| `baseline_coeffs`             | Normalized polynomial baseline coefficients      | ``       | $\beta_i \sim {\rm Normal}(mu=0, \sigma=p_i)$                         | `[1.0]*(baseline_degree + 1)` |

### `mainline_pos_tau`

An additional parameter to `EmissionAbsorptionModel` is `mainline_pos_tau`. If `True`, then the mainline (1665 MHz and 1667 MHz) optical depths are required to be positive by changing the prior distribution as follows.

| Cloud Parameter<br>`variable` | Parameter                               | Units | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}` | Default<br>`prior_{variable}` |
| :---------------------------- | :-------------------------------------- | :---- | :------------------------------------------------------- | :---------------------------- |
| `log_boltz_factor`            | log Boltzmann factor (`h*freq/(k*Tex)`) | ``    | $\ln B \sim {\rm HalfNormal}(\sigma=p_1)                 | `[0.1, 0.1]`                  |


## `ordered`

An additional parameter to `set_priors` for both the `AbsorptionModel` and `EmissionAbsorptionModel` is `ordered`. By default, this parameter is `False`, in which case the order of the clouds is from *nearest* to *farthest*. Sampling from these models can be challenging due to the labeling degeneracy: if the order of clouds does not matter (i.e., the emission is optically thin), then each Markov chain could decide on a different, equally-valid order of clouds.

If we assume that the emission is optically thin, then we can set `ordered=True`, in which case the order of clouds is restricted to be increasing with velocity. This assumption can *drastically* improve sampling efficiency. When `ordered=True`, the `velocity` prior is defined differently:

| Cloud Parameter<br>`variable` | Parameter | Units    | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}`                 | Default<br>`prior_{variable}` |
| :---------------------------- | :-------- | :------- | :----------------------------------------------------------------------- | :---------------------------- |
| `velocity`                    | Velocity  | `km s-1` | $V_i \sim p_0 + \sum_0^{i-1} V_i + {\rm Gamma}(\alpha=2, \beta=1.0/p_1)$ | `[0.0, 1.0]`                  |

# Syntax & Examples

See the various tutorial notebooks under [docs/source/notebooks](https://github.com/tvwenger/amoeba2/tree/main/docs/source/notebooks). Tutorials and the full API are available here: https://amoeba2.readthedocs.io.

# Issues and Contributing

Anyone is welcome to submit issues or contribute to the development of this software via [Github](https://github.com/tvwenger/amoeba2).

# License and Copyright

Copyright(C) 2024 by Trey V. Wenger; tvwenger@gmail.com. This code is licensed under MIT license (see LICENSE for details)