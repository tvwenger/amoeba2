"""
test_utils.py - tests for utils.py

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

import pytest
from amoeba2 import utils


def test_get_molecule_data():
    with pytest.raises(ValueError):
        utils.get_molecule_data(molecule="BAD MOLECULE NAME")
    data = utils.get_molecule_data()
    assert "freq" in data.keys()
    assert "Aul" in data.keys()
    assert "degu" in data.keys()
    assert "Eu" in data.keys()
    assert "relative_int" in data.keys()
    assert "log10_Q_terms" in data.keys()
