"""
test_emission_absorption_model.py - tests for EmissionAbsorptionModel

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

from bayes_spec import SpecData
from amoeba2 import EmissionAbsorptionModel


def test_emission_absorption_model():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "absorption_1612": SpecData(velocity, brightness, 1.0),
        "absorption_1665": SpecData(velocity, brightness, 1.0),
        "absorption_1667": SpecData(velocity, brightness, 1.0),
        "absorption_1720": SpecData(velocity, brightness, 1.0),
        "emission_1612": SpecData(velocity, brightness, 1.0),
        "emission_1665": SpecData(velocity, brightness, 1.0),
        "emission_1667": SpecData(velocity, brightness, 1.0),
        "emission_1720": SpecData(velocity, brightness, 1.0),
    }
    model = EmissionAbsorptionModel(data, 2, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors()
    model.add_likelihood()
    assert model._validate()


def test_emission_absorption_model_ordered():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "absorption_1612": SpecData(velocity, brightness, 1.0),
        "absorption_1665": SpecData(velocity, brightness, 1.0),
        "absorption_1667": SpecData(velocity, brightness, 1.0),
        "absorption_1720": SpecData(velocity, brightness, 1.0),
        "emission_1612": SpecData(velocity, brightness, 1.0),
        "emission_1665": SpecData(velocity, brightness, 1.0),
        "emission_1667": SpecData(velocity, brightness, 1.0),
        "emission_1720": SpecData(velocity, brightness, 1.0),
    }
    model = EmissionAbsorptionModel(data, 2, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()


def test_emission_absorption_model_mainline_pos_tau():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "absorption_1612": SpecData(velocity, brightness, 1.0),
        "absorption_1665": SpecData(velocity, brightness, 1.0),
        "absorption_1667": SpecData(velocity, brightness, 1.0),
        "absorption_1720": SpecData(velocity, brightness, 1.0),
        "emission_1612": SpecData(velocity, brightness, 1.0),
        "emission_1665": SpecData(velocity, brightness, 1.0),
        "emission_1667": SpecData(velocity, brightness, 1.0),
        "emission_1720": SpecData(velocity, brightness, 1.0),
    }
    model = EmissionAbsorptionModel(data, 2, baseline_degree=1)
    assert isinstance(model.mol_data, dict)
    model.add_priors(mainline_pos_tau=True)
    model.add_likelihood()
    assert model._validate()
