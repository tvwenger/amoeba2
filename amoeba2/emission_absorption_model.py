"""emission_absorption_model.py
EmissionAbsorptionModel definition

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

import pymc as pm
import pytensor.tensor as pt

from amoeba2 import AbsorptionModel
from amoeba2 import physics
from amoeba2.utils import _G, _OH


class EmissionAbsorptionModel(AbsorptionModel):
    """Definition of the TBTauModel. SpecData keys must be "absorption_1612", "absorption_1665",
    "absorption_1667", "absorption_1720", "emission_1612", "emission_1665", "emission_1667",
    and "emission_1720".
    """

    def __init__(self, *args, bg_temp: float = 2.7, **kwargs):
        """Initialize a new EmissionAbsorptionModel instance.

        Parameters
        ----------
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        """
        # Initialize AbsorptionModel
        super().__init__(*args, **kwargs)

        # Save background temperature
        self.bg_temp = bg_temp

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "rms_emission": r"rms$_{T}$",
            }
        )

    def add_priors(
        self,
        prior_rms_emission: float = 1.0,
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model.

        Parameters
        ----------
        prior_rms_emission : float, optional
            Prior distribution on emission spectral rms, by default 1.0, where
            rms_emission ~ HalfNormal(sigma=prior)
        """
        # Add priors to AbsorptionModel
        super().add_priors(*args, **kwargs)

        with self.model:
            # Brightness temperature rms (K)
            rms_emission_norm = pm.HalfNormal("rms_emission_norm", sigma=1.0, dims="component")
            _ = pm.Deterministic("rms_emission", rms_emission_norm * prior_rms_emission, dims="component")

    def predict_emission_absorption(self) -> dict:
        """Predict the emission and absorption spectra from the model parameters.

        Returns
        -------
        dict
            Emission spectra (brightness temp; K) for 1612, 1665, 1667, and 1720 MHz transitions
        dict
            Absorption spectra (1 - exp(-tau)) for 1612, 1665, 1667, and 1720 MHz transitions
        """
        emission = {}
        absorption = {}
        for i, label in enumerate(self.model.coords["component"]):
            # Line profile (km-1 s; shape: spectral, cloud)
            line_profile_tau = physics.calc_line_profile(
                self.data[f"absorption_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )
            line_profile_TB = physics.calc_line_profile(
                self.data[f"emission_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )

            # Optical depth spectrum (shape: spectral, cloud)
            tau_spectrum_tau = physics.calc_optical_depth(
                _G[_OH[label][0]],
                _G[_OH[label][1]],
                10.0 ** self.model["log10_N"][_OH[label][0]],
                10.0 ** self.model["log10_N"][_OH[label][1]],
                line_profile_tau,
                self.mol_data["freq"][i],
                self.mol_data["Aul"][i],
            )
            tau_spectrum_TB = physics.calc_optical_depth(
                _G[_OH[label][0]],
                _G[_OH[label][1]],
                10.0 ** self.model["log10_N"][_OH[label][0]],
                10.0 ** self.model["log10_N"][_OH[label][1]],
                line_profile_TB,
                self.mol_data["freq"][i],
                self.mol_data["Aul"][i],
            )

            # Sum over clouds
            absorption[label] = 1.0 - pt.exp(-tau_spectrum_tau.sum(axis=1))

            # Radiative transfer
            emission[label] = physics.radiative_transfer(
                self.mol_data["freq"][i],
                tau_spectrum_TB,
                self.model["Tex"][i],
                self.bg_temp,
            )
        return emission, absorption

    def add_likelihood(self):
        """Add likelihood to the model."""
        # Predict emission and absorption spectra
        emission, absorption = self.predict_emission_absorption()

        # Get baseline models
        baseline_models = self.predict_baseline()

        with self.model:
            for i, label in enumerate(self.model.coords["component"]):
                _ = pm.Normal(
                    f"absorption_{label}",
                    mu=absorption[label] + baseline_models[f"absorption_{label}"],
                    sigma=self.model["rms_absorption"][i],
                    observed=self.data[f"absorption_{label}"].brightness,
                )
                _ = pm.Normal(
                    f"emission_{label}",
                    mu=emission[label] + baseline_models[f"emission_{label}"],
                    sigma=self.model["rms_emission"][i],
                    observed=self.data[f"emission_{label}"].brightness,
                )
