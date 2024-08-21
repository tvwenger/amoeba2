"""physics.py
Molecular spectroscopy physics utilities

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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


def calc_boltzmann(gu: int, gl: int, freq: float, inv_Tex: float) -> float:
    """Evaluate the Boltzmann factor Nu/Nl, which is the ratio of column densities
    in the upper and lower energy state.

    Parameters
    ----------
    gu : int
        Upper state degeneracy
    gl : int
        Lower state degeneracy
    freq : float
        Frequency (MHz)
    inv_Tex : float
        Inverse excitation temperature (K-1)

    Returns
    -------
    float
        Boltzmann factor
    """
    return gu / gl * pt.exp(-_H * freq * inv_Tex / _K_B)


def calc_line_profile(
    velo_axis: Iterable[float], velocity: Iterable[float], fwhm: Iterable[float]
) -> Iterable[float]:
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
    N_u: Iterable[float],
    inv_Tex: Iterable[float],
    line_profile: Iterable[float],
    freq: float,
    Aul: float,
) -> Iterable[float]:
    """Evaluate the optical depth spectra, from Mangum & Shirley eq. 29

    Parameters
    ----------
    N_u : Iterable[float]
        Cloud upper state column densities (cm-2; length C)
    inv_Tex : Iterable[float]
        Cloud inverse excitation temperatures (K-1; length C)
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
    const = _H * freq * inv_Tex / _K_B
    return (
        _C_CM_MHZ**2.0  # cm2 MHz2
        / (8.0 * np.pi * freq**2.0)  # MHz-2
        * (pt.exp(const[None, :]) - 1.0)
        * Aul  # s-1
        * (_C * line_profile / (1e6 * freq))  # Hz-1
        * N_u[None, :]  # cm-2
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
    inv_Tex: Iterable[float],
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
    inv_Tex : Iterable[float]
        Cloud inverse excitation temperature (K-1; length N)
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
    emission_clouds = rj_temperature(freq, 1.0 / inv_Tex) * (1.0 - pt.exp(-tau))
    emission_clouds_attenuated = emission_clouds * pt.exp(-sum_tau[:, :-1])
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=1)

    # ON - OFF
    return emission - emission_bg
