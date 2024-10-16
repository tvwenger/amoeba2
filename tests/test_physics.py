"""
test_physics.py - tests for physics.py

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

import numpy as np
from numpy.testing import assert_allclose

from amoeba2 import physics


def test_calc_thermal_fwhm():
    fwhm = physics.calc_thermal_fwhm(100.0).eval()
    assert not np.isnan(fwhm)


def test_calc_nonthermal_fwhm():
    fwhm = physics.calc_nonthermal_fwhm(1.0, 10.0, 0.5)
    assert not np.isnan(fwhm)


def test_calc_boltzmann():
    boltzmann = physics.calc_boltzmann(2, 1, 1800.0, 1.0).eval()
    assert not np.isnan(boltzmann)
    boltzmann = physics.calc_boltzmann(2, 1, 1800.0, 0.0)
    assert boltzmann == 0.0


def test_calc_Tex():
    Tex = physics.calc_Tex(2, 1, 1800.0, 1e12, 1e11).eval()
    assert not np.isnan(Tex)
    Tex = physics.calc_Tex(2, 1, 1800.0, 1e12, 0.0)
    assert np.isinf(Tex)


def test_calc_line_ratio():
    velo_axis = np.linspace(-10.0, 10.0, 1001)
    velocity = np.array([0.0, 1.0])
    fwhm = np.array([1.0, 2.0])
    line_profile = physics.calc_line_profile(velo_axis, velocity, fwhm).eval()
    assert line_profile.shape == (1001, 2)
    assert_allclose(line_profile.sum(axis=0) * (velo_axis[1] - velo_axis[0]), np.ones(2))
    exp_line_profile = np.array(
        [
            np.sqrt(4.0 * np.log(2.0) / np.pi),
            0.5 * np.sqrt(np.log(2.0) / np.pi),
        ]
    )
    assert_allclose(line_profile[500, :], exp_line_profile)


def test_calc_optical_depth():
    velo_axis = np.linspace(-10.0, 10.0, 1001)
    velocity = np.array([0.0, 1.0, 2.0])
    fwhm = np.array([1.0, 2.0, 3.0])
    line_profile = physics.calc_line_profile(velo_axis, velocity, fwhm).eval()

    gu = 3
    gl = 5
    Nu = np.array([1.0e12, 1.0e12, 1.0e12])
    Nl = np.array([1.0e11, 0.0, 1.0e13])
    freq = 1800.0
    Aul = 1.0e-13
    optical_depth = physics.calc_optical_depth(gu, gl, Nu, Nl, line_profile, freq, Aul).eval()
    assert optical_depth.shape == (1001, 3)
    assert np.all(optical_depth[:, 0] < 0)
    assert np.all(optical_depth[:, 1] == 0)
    assert np.all(optical_depth[:, 2] > 0)


def test_rj_temperature():
    Tex = np.array([-10.0, 10.0, 0.0])
    rj_temp = physics.rj_temperature(1800.0, Tex).eval()
    assert rj_temp[0] < rj_temp[1]
    assert rj_temp[2] == 0.0


def test_radiative_transfer():
    velo_axis = np.linspace(-10.0, 10.0, 1001)
    velocity = np.array([0.0, 1.0, 2.0])
    fwhm = np.array([1.0, 2.0, 3.0])
    line_profile = physics.calc_line_profile(velo_axis, velocity, fwhm).eval()

    gu = 3
    gl = 5
    Nu = np.array([1.0e12, 1.0e12, 1.0e12])
    Nl = np.array([1.0e11, 0.0, 1.0e13])
    freq = 1800.0
    Aul = 1.0e-13
    optical_depth = physics.calc_optical_depth(gu, gl, Nu, Nl, line_profile, freq, Aul).eval()

    Tex = np.array([-10.0, 10.0, 0.0])
    bg_temp = 2.7
    tb = physics.radiative_transfer(freq, optical_depth, Tex, bg_temp).eval()
    assert tb.shape == (1001,)
