"""
simulate.py - Simulate observations

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
Trey V. Wenger - June 2023
"""

import numpy as np
from .model import predict_tau_spectrum


def simulate_tau_spectra(
    velocity_axes: list[list[float]],
    rms: list[float],
    truths: dict,
    seed=1234,
):
    """
    Generate synthetic OH optical depth spectra. The 1720 MHz optical
    depth spectrum is set by the optical depth sum rule.

    Inputs:
        velocity :: list of four 1-D arrays of scalars
            Velocity axes for the 1612, 1665, 1667, and 1720 MHz spectra
        rms :: 4-element array of scalars
            Optical depth rms in each spectrum
        truths :: dictionary
            Dictionary with keys "center", "log10_fwhm", "peak_tau_1612",
            "peak_tau_1665", and "peak_tau_1667". Each value must be
            an array of scalars, with each element representing a different
            Gaussian component.
        seed :: integer
            Random state seed. Default = 1234

    Returns: tau_spectra, truths
        tau_spectra :: list of four 1-D arrays of scalars
            Simulated optical depth spectra for each transition.
        truths :: dictionary
            Same as supplied, but with additional "peak_tau_1720"
            key added and set by the optical depth sum rule.
    """
    transitions = ["1612", "1665", "1667", "1720"]

    # optical depth sum rule
    truths["peak_tau_1720"] = (
        -truths["peak_tau_1612"]
        + truths["peak_tau_1665"] / 5.0
        + truths["peak_tau_1667"] / 9.0
    )

    # generate spectra
    tau_spectra = [
        predict_tau_spectrum(
            velocity[:, None],
            truths[f"peak_tau_{transition}"],
            truths["center"],
            10.0 ** truths["log10_fwhm"],
        )
        for velocity, transition in zip(velocity_axes, transitions)
    ]

    # add noise
    rng = np.random.default_rng(seed)
    for i in range(len(transitions)):
        tau_spectra[i] += rng.normal(loc=0, scale=rms[i], size=len(tau_spectra[i]))
    return tau_spectra, truths
