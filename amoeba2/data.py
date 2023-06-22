"""
data.py - AMOEBA data structure definition

MIT License

Copyright (c) 2023 Trey Wenger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Changelog:
Trey V. Wenger - June 2023 - v2.0
    Based on amoeba.py, complete re-write in pymc
"""

import numpy as np


class AmoebaSpectrum:
    """
    Data structure for AMOEBA spectrum
    """

    def __init__(
        self,
        transition: str,
        velocity: list[float],
        spectrum: list[float],
        rms: float,
        background=0.0,
    ):
        """
        Initialize a new AmoebaSpectrum instance

        Inputs:
            transition :: string
                Transition name
            velocity :: 1-D array of scalars
                Spectral axis definition
            spectrum :: 1-D array of scalars
                Data
            rms :: scalar
                Spectral rms
            background :: scalar
                For brightness temperature spectra, the background brightness

        Returns: spectrum
            spectrum :: AmoebaSpectrum
                New AmoebaSpectrum instance
        """
        self.transition = transition
        self.velocity = velocity
        self.spectrum = spectrum
        self.rms = rms
        self.background = background


class AmoebaData:
    """
    Data structure for AMOEBA
    """

    def __init__(self):
        """
        Initialize a new AmoebaData instances

        Inputs: None
        Returns: data
            data :: AmoebaData
                New AmoebaData instance
        """
        self._transitions = ["1612", "1665", "1667", "1720"]
        self.spectra = {transition: None for transition in self._transitions}

    def set_spectrum(
        self,
        transition: str,
        velocity: list[float],
        spectrum: list[float],
        rms: float,
    ):
        """
        Add a spectrum to the structure.

        Inputs:
            transition :: string
                Transition name
            velocity :: 1-D array of scalars
                Spectral axis definition
            spectrum :: 1-D array of scalars
                Data
            rms :: scalar
                Spectral rms
        """
        if transition not in self._transitions:
            raise ValueError(f"Transition must be one of {self._transitions}")

        self.spectra[transition] = AmoebaSpectrum(transition, velocity, spectrum, rms)

    def tau_sum_rule_rms(self):
        """
        Evaluate the optical depth sum rule rms.

        Inputs: None

        Returns: rms
            rms :: scalar
                Optical depth sum rule rms
        """
        return np.sqrt(
            self.spectra["1612"].rms ** 2.0
            + (self.spectra["1665"].rms / 5.0) ** 2.0
            + (self.spectra["1667"].rms / 9.0) ** 2.0
            + self.spectra["1720"].rms ** 2.0
        )
