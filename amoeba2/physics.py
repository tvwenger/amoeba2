"""physics.py
Molecular spectroscopy physics utilities

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import numpy as np
import pytensor.tensor as pt

import astropy.constants as c

from bayes_spec.utils import gaussian

_K_B = c.k_B.to("erg K-1").value
_H = c.h.to("erg MHz-1").value
_C = c.c.to("km s-1").value
_C_CM_MHZ = c.c.to("cm MHz").value


def calc_thermal_fwhm(kinetic_temp: float, weight: float = 17.0) -> float:
    """Calculate the thermal line broadening assuming a Maxwellian velocity distribution
    (Condon & Ransom eq. 7.35)

    Parameters
    ----------
    kinetic_temp : float
        Kinetic temperature (K)
    weight : float
        Molecular weight (number of protons). By default, 17 (OH molecule)

    Returns
    -------
    float
        Thermal FWHM line width (km s-1)
    """
    # constant = sqrt(8*ln(2)*k_B/m_p)
    const = 0.21394418  # km/s K-1/2
    return const * pt.sqrt(kinetic_temp / weight)


def calc_nonthermal_fwhm(depth: float, nth_fwhm_1pc: float, depth_nth_fwhm_power: float) -> float:
    """Calculate the non-thermal line broadening assuming a power-law size-linewidth relationship.

    Parameters
    ----------
    depth : float
        Line-of-sight depth (pc)
    nth_fwhm_1pc : float
        Non-thermal broadening at 1 pc (km s-1)
    depth_nth_fwhm_power : float
        Power law index

    Returns
    -------
    float
        Non-thermal FWHM line width (km s-1)
    """
    return nth_fwhm_1pc * depth**depth_nth_fwhm_power


def calc_Tex(freq: float, log_boltz_factor: float) -> float:
    """Evaluate the excitation temperature from a given Boltzmann factor.

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    log_boltz_factor : float
        log Boltzmann factor = -h*freq/(k*Tex)

    Returns
    -------
    float
        Excitation temperature
    """
    return -_H * freq / (_K_B * log_boltz_factor)


def calc_line_profile(velo_axis: Iterable[float], velocity: Iterable[float], fwhm: Iterable[float]) -> Iterable[float]:
    """Evaluate the Gaussian line profile, ensuring normalization.

    Parameters
    ----------
    velo_axis : Iterable[float]
        Observed velocity axis (km s-1; length S)
    velocity : Iterable[float]
        Cloud center velocities (km s-1; length C)
    fwhm : Iterable[float]
        Cloud FWHM line widths (km s-1; length C)

    Returns
    -------
    Iterable[float]
        Line profile (km-1 s; shape S x C)
    """
    amp = pt.sqrt(4.0 * pt.log(2.0) / (np.pi * fwhm**2.0))
    profile = gaussian(velo_axis[:, None], amp, velocity, fwhm)

    # normalize
    channel_size = pt.abs(velo_axis[1] - velo_axis[0])
    profile_int = pt.sum(profile, axis=0)
    norm = pt.switch(pt.lt(profile_int, 1.0e-6), 1.0, profile_int * channel_size)
    return profile / norm


def calc_optical_depth(
    gu: int,
    gl: int,
    Nl: Iterable[float],
    boltz_factor: Iterable[float],
    line_profile: Iterable[float],
    freq: float,
    Aul: float,
) -> Iterable[float]:
    """Evaluate the optical depth spectra, from Mangum & Shirley eq. 29

    Parameters
    ----------
    gu : int
        Upper state degeneracy
    gl : int
        Lower state degeneracy
    Nl : Iterable[float]
        Cloud lower state column densities (cm-2; length C)
    boltz_factor : Iterable[float]
        Boltzmann factor = exp(-h*freq/(k*Tex)) (length C)
    line_profile : Iterable[float]
        Line profile (km-1 s; shape S x C)
    freq : float
        Transition frequency (MHz)
    Aul : float
        Transition Einstein A coefficient (s-1)

    Returns
    -------
    Iterable[float]
        Optical depth spectral (shape S x C)
    """
    return (
        _C_CM_MHZ**2.0  # cm2 MHz2
        / (8.0 * np.pi * freq**2.0)  # MHz-2
        * (gu / gl)
        * Aul  # s-1
        * (_C * line_profile / (1e6 * freq))  # Hz-1
        * Nl  # cm-2
        * (1.0 - boltz_factor)
    )


def rj_temperature(freq: float, temp: float) -> float:
    """Calculate the Rayleigh-Jeans equivalent temperature (AKA the brightness temperature)

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    temp : float
        Temperature (K)

    Returns
    -------
    float
        R-J equivalent temperature (K)
    """
    const = _H * freq / _K_B
    return const / (pt.exp(const / temp) - 1.0)


def radiative_transfer(
    freq: float,
    tau: Iterable[float],
    Tex: Iterable[float],
    bg_temp: float,
) -> Iterable[float]:
    """Evaluate the radiative transfer to predict the emission spectrum. The emission
    spectrum is ON - OFF, where ON includes the attenuated emission of the background and
    the clouds, and the OFF is the emission of the background. Order of N clouds is
    assumed to be [nearest, ..., farthest].

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    tau : Iterable[float]
        Optical depth spectra (shape S x N)
    Tex : Iterable[float]
        Cloud excitation temperature (K; length N)
    bg_temp : float
        Assumed background temperature

    Returns
    -------
    Iterable[float]
        Predicted emission brightness temperature spectrum (K; length S)
    """
    front_tau = pt.zeros_like(tau[:, 0:1])
    sum_tau = pt.concatenate([front_tau, pt.cumsum(tau, axis=1)], axis=1)

    # radiative transfer, assuming filling factor = 1.0
    emission_bg = rj_temperature(freq, bg_temp)
    emission_bg_attenuated = emission_bg * pt.exp(-sum_tau[:, -1])
    emission_clouds = rj_temperature(freq, Tex) * (1.0 - pt.exp(-tau))
    emission_clouds_attenuated = emission_clouds * pt.exp(-sum_tau[:, :-1])
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=1)

    # ON - OFF
    return emission - emission_bg
