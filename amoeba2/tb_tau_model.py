"""tb_tau_model.py
TBTauModel definition

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

from amoeba2 import TauModel
from amoeba2 import physics


class TBTauModel(TauModel):
    """Definition of the TBTauModel. SpecData keys must be "tau_1612", "tau_1665", "tau_1667", "tau_1720",
    "TB_1612", "TB_1665", "TB_1667", and "TB_1720".
    """

    def __init__(self, *args, bg_temp: float = 2.7, **kwargs):
        """Initialize a new TBTauModel instance.

        Parameters
        ----------
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        """
        # Initialize TauModel
        super().__init__(*args, **kwargs)

        # Save background temperature
        self.bg_temp = bg_temp

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "rms_TB": r"rms$_{T}$",
            }
        )

    def add_priors(
        self,
        prior_rms_TB: float = 1.0,
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model.

        Parameters
        ----------
        prior_rms_TB : float, optional
            Prior distribution on brightness temperature spectral rms, by default 1.0, where
            rms_TB ~ HalfNormal(sigma=prior)
        """
        # Add priors to TauModel
        super().add_priors(*args, **kwargs)

        with self.model:
            # Brightness temperature rms (K)
            rms_TB_norm = pm.HalfNormal("rms_TB_norm", sigma=1.0, dims="component")
            _ = pm.Deterministic("rms_TB", rms_TB_norm * prior_rms_TB, dims="component")

    def predict_TB_tau(self) -> dict:
        """Predict the brightness temperature and optical dpeth spectra from the model parameters.

        Returns
        -------
        dict
            Brightness temperature spectra (K) for 1612, 1665, 1667, and 1720 MHz transitions
        dict
            Optical depth temperature spectra for 1612, 1665, 1667, and 1720 MHz transitions
        """
        TB = {}
        tau = {}
        for i, label in enumerate(self.model.coords["component"]):
            # Line profile (km-1 s; shape: spectral, cloud)
            line_profile = physics.calc_line_profile(
                self.data[f"tau_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )

            # Optical depth spectrum (shape: spectral, cloud)
            if "component" in ["1612", "1665"]:
                N_u = 10.0 ** self.model["log10_N"][2]
            else:
                N_u = 10.0 ** self.model["log10_N"][3]
            tau_spectrum = physics.calc_optical_depth(
                N_u,
                self.model["inv_Tex"][i],
                line_profile,
                self.mol_data["freq"][i],
                self.mol_data["Aul"][i],
            )

            # Sum over clouds
            tau[label] = tau_spectrum.sum(axis=1)

            # Radiative transfer
            TB[label] = physics.radiative_transfer(
                self.mol_data["freq"][i],
                tau_spectrum,
                self.model["inv_Tex"][i],
                self.bg_temp,
            )
        return TB, tau

    def add_likelihood(self):
        """Add likelihood to the model."""
        # Predict brightness temperature and optical depth spectra
        TB, tau = self.predict_TB_tau()

        # Get baseline models
        baseline_models = self.predict_baseline()

        with self.model:
            for i, label in enumerate(self.model.coords["component"]):
                _ = pm.Normal(
                    f"tau_{label}",
                    mu=tau[label] + baseline_models[f"tau_{label}"],
                    sigma=self.model["rms_tau"][i],
                    observed=self.data[f"tau_{label}"].brightness,
                )
                _ = pm.Normal(
                    f"TB_{label}",
                    mu=TB[label] + baseline_models[f"TB_{label}"],
                    sigma=self.model["rms_TB"][i],
                    observed=self.data[f"TB_{label}"].brightness,
                )
