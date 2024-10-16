"""absorption_model.py
AbsorptionModel definition

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

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel

from amoeba2.utils import _G, _OH, get_molecule_data
from amoeba2 import physics


class AbsorptionModel(BaseModel):
    """Definition of the AbsorptionModel. SpecData keys must be "absorption_1612", "absorption_1665",
    "absorption_1667", and "absorption_1720"."""

    def __init__(self, *args, mol_data: Optional[dict] = None, **kwargs):
        """Initialize a new AbsorptionModel instance.

        Parameters
        ----------
        mol_data : Optional[dict], optional
            OH molecular data dictionary returned by get_molecule_data(). If None, it will
            be downloaded. Default is None.
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # get OH data
        if mol_data is None:
            self.mol_data = get_molecule_data()
        else:
            self.mol_data = mol_data

        # Add states, components to model
        self.model.add_coords(
            {
                "state": [0, 1, 2, 3],
                "component_Tex_free": ["1612", "1665", "1667"],
                "component": ["1612", "1665", "1667", "1720"],
            }
        )

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N_0",
            "velocity",
            "fwhm",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_N_0": r"$\log_{10} N_0$ (cm$^{-2}$)",
                "log10_depth": r"log$_{10}$ $d$ (pc)",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "Tex": r"$T_{\rm ex}$ (K)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "log10_nth_fwhm_1pc": r"log$_{10}$ $\Delta V_{\rm 1 pc}$ (km s$^{-1}$)",
                "depth_nth_fwhm_power": r"$\alpha$",
                "log10_N": r"$\log_{10} N$ (cm$^{-2}$)",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nth}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "rms_absorption": r"rms$_{\tau}$",
            }
        )

    def add_priors(
        self,
        prior_log10_N_0: Iterable[float] = [13.0, 1.0],
        prior_log10_depth: Iterable[float] = [1.0, 1.0],
        prior_log10_Tkin: Iterable[float] = [2.0, 1.0],
        prior_Tex: Iterable[float] = [10.0, 10.0],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_log10_nth_fwhm_1pc: Iterable[float] = [0.2, 0.1],
        prior_depth_nth_fwhm_power: Iterable[float] = [0.4, 0.1],
        prior_rms_absorption: float = 0.1,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        ordered: bool = False,
        mainline_pos_Tex: bool = False,
    ):
        """Add priors and deterministics to the model.

        Parameters
        ----------
        prior_log10_N_0 : Iterable[float], optional
            Prior distribution on log10 column density (cm-2) in lowest energy state, by default [13.0, 1.0], where
            log10_N_0 ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_depth : Iterable[float], optional
            Prior distribution on log10 depth (pc), by default [1.0, 1.0], where
            log10_depth ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_Tkin : Iterable[float], optional
            Prior distribution on log10 kinetic temperature (K), by default [2.0, 1.0], where
            log10_Tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_Tex : Iterable[float], optional
            Prior distribution on excitation temperature (K), by default [10.0, 10.0], where
            Tex ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_nth_fwhm_1pc : Iterable[float], optional
            Prior distribution on non-thermal line width at 1 pc, by default [0.2, 0.1], where
            log10_nth_fwhm_1pc ~ Normal(mu=prior[0], sigma=prior[1])
        prior_depth_nth_fwhm_power : Iterable[float], optional
            Prior distribution on depth vs. non-thermal line width power law index, by default [0.4, 0.1], where
            depth_nth_fwhm_power ~ Normal(mu=prior[0], sigma=prior[1])
        prior_rms_absorption : float, optional
            Prior distribution on absorption spectral rms, by default 0.1, where
            rms_absorption ~ HalfNormal(sigma=prior)
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False. If True, the prior
            distribution on the velocity becomes
            velocity(cloud = n) ~ prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        mainline_pos_Tex: bool, optional
            If True, assume positive main line excitation temperatures, by default False. If True, the
            prior distribution on the excitation temperature for the main lines (1665 and 1667) becomes
            Tex ~ Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # catch bad prior
        if prior_Tex[0] == 0.0:
            raise ValueError("prior_Tex[0] must not be zero!")

        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # lowest energy state column density (cm-2; shape: clouds)
            log10_N_0_norm = pm.Normal("log10_N_0_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_N_0 = pm.Deterministic(
                "log10_N_0", prior_log10_N_0[0] + prior_log10_N_0[1] * log10_N_0_norm, dims="cloud"
            )

            # depth (pc; shape: clouds)
            log10_depth_norm = pm.Normal("log10_depth_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_depth = pm.Deterministic(
                "log10_depth",
                prior_log10_depth[0] + prior_log10_depth[1] * log10_depth_norm,
                dims="cloud",
            )

            # kinetic temperature (K; shape: clouds)
            log10_Tkin_norm = pm.Normal("log10_Tkin_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_Tkin = pm.Deterministic(
                "log10_Tkin",
                prior_log10_Tkin[0] + prior_log10_Tkin[1] * log10_Tkin_norm,
                dims="cloud",
            )

            # excitation temperature (K; shape: free components, clouds)
            if mainline_pos_Tex:
                Tex_1612_norm = pm.Normal("Tex_1612_norm", mu=0.0, sigma=1.0, dims="cloud")
                Tex_1612 = prior_Tex[0] + prior_Tex[1] * Tex_1612_norm

                Tex_1665_norm = pm.Gamma("Tex_1665_norm", alpha=2.0, beta=1.0, dims="cloud")
                Tex_1665 = prior_Tex[1] * Tex_1665_norm

                Tex_1667_norm = pm.Gamma("Tex_1667_norm", alpha=2.0, beta=1.0, dims="cloud")
                Tex_1667 = prior_Tex[1] * Tex_1667_norm

                Tex_free = pm.Deterministic(
                    "Tex_free", pt.stack([Tex_1612, Tex_1665, Tex_1667]), dims=["component_Tex_free", "cloud"]
                )
            else:
                Tex_free_norm = pm.Normal("Tex_free_norm", mu=0.0, sigma=1.0, dims=["component_Tex_free", "cloud"])
                Tex_free = pm.Deterministic(
                    "Tex_free", prior_Tex[0] + prior_Tex[1] * Tex_free_norm, dims=["component_Tex_free", "cloud"]
                )

            # Center velocity (km s-1; shape: clouds)
            if ordered:
                velocity_offset_norm = pm.Gamma("velocity_norm", alpha=2.0, beta=1.0, dims="cloud")
                velocity_offset = velocity_offset_norm * prior_velocity[1]
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + pm.math.cumsum(velocity_offset),
                    dims="cloud",
                )
            else:
                velocity_norm = pm.Normal(
                    "velocity_norm",
                    mu=0.0,
                    sigma=1.0,
                    dims="cloud",
                    initval=np.linspace(-1.0, 1.0, self.n_clouds),
                )
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + prior_velocity[1] * velocity_norm,
                    dims="cloud",
                )

            # Other state column densities (cm-2; shape: cloud)
            # 2 -> 0 == 1665
            log10_N_2 = log10_N_0 + pt.log10(
                physics.calc_boltzmann(_G[_OH["1665"][0]], _G[_OH["1665"][1]], self.mol_data["freq"][1], Tex_free[1])
            )
            # 2 -> 1 == 1612
            log10_N_1 = log10_N_2 - pt.log10(
                physics.calc_boltzmann(_G[_OH["1612"][0]], _G[_OH["1612"][1]], self.mol_data["freq"][0], Tex_free[0])
            )
            # 3 -> 1 == 1667
            log10_N_3 = log10_N_1 + pt.log10(
                physics.calc_boltzmann(_G[_OH["1667"][0]], _G[_OH["1667"][1]], self.mol_data["freq"][2], Tex_free[2])
            )
            log10_N = pm.Deterministic(
                "log10_N",
                pt.stack([log10_N_0, log10_N_1, log10_N_2, log10_N_3]),
                dims=["state", "cloud"],
            )

            # 1720 MHz excitation temperature (K; shape: clouds)
            Tex_1720 = physics.calc_Tex(
                _G[_OH["1720"][0]],
                _G[_OH["1720"][1]],
                self.mol_data["freq"][3],
                10.0 ** log10_N[_OH["1720"][0]],
                10.0 ** log10_N[_OH["1720"][1]],
            )

            # all excitation temperatures
            _ = pm.Deterministic("Tex", pt.concatenate([Tex_free, Tex_1720[None]], axis=0), dims=["component", "cloud"])

            # Non-thermal FWHM at 1 pc (km s-1; shape: clouds)
            log10_nth_fwhm_1pc_norm = pm.Normal("log10_nth_fwhm_1pc_norm", mu=0.0, sigma=1.0)
            log10_nth_fwhm_1pc = pm.Deterministic(
                "log10_nth_fwhm_1pc",
                prior_log10_nth_fwhm_1pc[0] + prior_log10_nth_fwhm_1pc[1] * log10_nth_fwhm_1pc_norm,
            )

            # Non-thermal FWHM vs. depth power law index (shape: clouds)
            depth_nth_fwhm_power_norm = pm.Normal("depth_nth_fwhm_power_norm", mu=0.0, sigma=1.0)
            depth_nth_fwhm_power = pm.Deterministic(
                "depth_nth_fwhm_power",
                prior_depth_nth_fwhm_power[0] + prior_depth_nth_fwhm_power[1] * depth_nth_fwhm_power_norm,
            )

            # Thermal FWHM (km/s; shape: clouds)
            fwhm_thermal = pm.Deterministic("fwhm_thermal", physics.calc_thermal_fwhm(10.0**log10_Tkin), dims="cloud")

            # Non-thermal FWHM (km/s; shape: clouds)
            fwhm_nonthermal = pm.Deterministic(
                "fwhm_nonthermal",
                physics.calc_nonthermal_fwhm(10.0**log10_depth, 10.0**log10_nth_fwhm_1pc, depth_nth_fwhm_power),
                dims="cloud",
            )

            # FWHM (km/s; shape: clouds)
            _ = pm.Deterministic("fwhm", pt.sqrt(fwhm_thermal**2.0 + fwhm_nonthermal**2.0), dims="cloud")

            # Optical depth rms
            rms_absorption_norm = pm.HalfNormal("rms_absorption_norm", sigma=1.0, dims="component")
            _ = pm.Deterministic("rms_absorption", rms_absorption_norm * prior_rms_absorption, dims="component")

    def predict_absorption(self) -> dict:
        """Predict the absorption spectra from the model parameters.

        Returns
        -------
        dict
            Optical depth spectra for 1612, 1665, 1667, and 1720 MHz transitions
        """
        absorption = {}
        for i, label in enumerate(self.model.coords["component"]):
            # Line profile (km-1 s; shape: spectral, cloud)
            line_profile = physics.calc_line_profile(
                self.data[f"absorption_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )

            # Optical depth spectrum (shape: spectral, cloud)
            tau_spectrum = physics.calc_optical_depth(
                _G[_OH[label][0]],
                _G[_OH[label][1]],
                10.0 ** self.model["log10_N"][_OH[label][0]],
                10.0 ** self.model["log10_N"][_OH[label][1]],
                line_profile,
                self.mol_data["freq"][i],
                self.mol_data["Aul"][i],
            )

            # Sum over clouds
            absorption[label] = 1.0 - pt.exp(-tau_spectrum.sum(axis=1))
        return absorption

    def add_likelihood(self):
        """Add likelihood to the model."""
        # Predict optical depth spectra
        absorption = self.predict_absorption()

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
