"""tau_model.py
TauModel definition

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

from amoeba2.utils import get_molecule_data
from amoeba2 import physics


class TauModel(BaseModel):
    """Definition of the TauModel. SpecData keys must be "tau_1612", "tau_1665", "tau_1667", and "tau_1720"."""

    def __init__(self, *args, mol_data: Optional[dict] = None, **kwargs):
        """Initialize a new TauModel instance.

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
                "log10_N": r"$\log_{10} N$ (cm$^{-2}$)",
                "inv_Tex": r"$T_{\rm ex}^{-1}$ (K$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "rms_tau": r"rms$_{\tau}$",
            }
        )

    def add_priors(
        self,
        prior_log10_N_0: Iterable[float] = [13.0, 1.0],
        prior_inv_Tex: Iterable[float] = [0.1, 1.0],
        prior_fwhm: float = 1.0,
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_rms_tau: float = 0.1,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model.

        Parameters
        ----------
        prior_log10_N_0 : Iterable[float], optional
            Prior distribution on log10 column density (cm-2) in lowest energy state, by default [13.0, 1.0], where
            log10_N_0 ~ Normal(mu=prior[0], sigma=prior[1])
        prior_inv_Tex : Iterable[float], optional
            Prior distribution on inverse excitation temperature (K-1), by default [0.0, 1.0], where
            inv_Tex ~ Normal(mu=prior[0], sigma=prior[1])
        prior_fwhm : float, optional
            Prior distribution on FWHM line width (km s-1), by default 1.0, where
            fwhm ~ Gamma(alpha=2.0, beta=1.0/prior)
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_rms_tau : float, optional
            Prior distribution on optical depth rms, by default 0.1, where
            rms_tau ~ HalfNormal(sigma=prior)
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False. If True, the prior
            distribution on the velocity becomes
            velocity(cloud = n) ~ prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # lowest energy state column density (cm-2; shape: clouds)
            log10_N_0_norm = pm.Normal("log10_N_0_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_N_0 = pm.Deterministic(
                "log10_N_0", prior_log10_N_0[0] + prior_log10_N_0[1] * log10_N_0_norm, dims="cloud"
            )

            # inverse excitation temperature (K-1; shape: components, clouds)
            inv_Tex_free_norm = pm.Normal("inv_Tex_free_norm", mu=0.0, sigma=1.0, dims=["component_Tex_free", "cloud"])
            inv_Tex_free = pm.Deterministic(
                "inv_Tex_free",
                prior_inv_Tex[0] + prior_inv_Tex[1] * inv_Tex_free_norm,
                dims=["component_Tex_free", "cloud"],
            )

            # excitation temperature sum rule
            inv_Tex_1720 = (
                inv_Tex_free[1] * self.mol_data["freq"][1]
                + inv_Tex_free[2] * self.mol_data["freq"][2]
                - inv_Tex_free[0] * self.mol_data["freq"][0]
            ) / self.mol_data["freq"][3]
            inv_Tex = pt.concatenate([inv_Tex_free, inv_Tex_1720[None]], axis=0)
            _ = pm.Deterministic(
                "inv_Tex",
                inv_Tex,
                dims=["component", "cloud"],
            )

            # Other state column densities (cm-2; shape: cloud)
            # 2 -> 0 == 1665
            log10_N_2 = log10_N_0 + pt.log10(physics.calc_boltzmann(3, 3, self.mol_data["freq"][1], inv_Tex[1]))
            # 2 -> 1 == 1612
            log10_N_1 = log10_N_2 - pt.log10(physics.calc_boltzmann(3, 5, self.mol_data["freq"][0], inv_Tex[0]))
            # 3 -> 1 == 1667
            log10_N_3 = log10_N_1 + pt.log10(physics.calc_boltzmann(5, 5, self.mol_data["freq"][2], inv_Tex[2]))
            log10_N = pt.concatenate(
                [log10_N_0[None], log10_N_1[None], log10_N_2[None], log10_N_3[None]],
                axis=0,
            )
            _ = pm.Deterministic(
                "log10_N",
                log10_N,
                dims=["state", "cloud"],
            )

            # FWHM line width (km s-1; shape: clouds)
            fwhm_norm = pm.Gamma("fwhm_norm", alpha=2.0, beta=1.0, dims="cloud")
            _ = pm.Deterministic(
                "fwhm",
                prior_fwhm * fwhm_norm,
                dims="cloud",
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

            # Optical depth rms
            rms_tau_norm = pm.HalfNormal("rms_tau_norm", sigma=1.0, dims="component")
            _ = pm.Deterministic("rms_tau", rms_tau_norm * prior_rms_tau, dims="component")

    def predict_tau(self) -> dict:
        """Predict the optical depth spectra from the model parameters.

        Returns
        -------
        dict
            Optical depth spectra for 1612, 1665, 1667, and 1720 MHz transitions
        """
        tau = {}
        for i, label in enumerate(self.model.coords["component"]):
            # Line profile (km-1 s; shape: spectral, cloud)
            line_profile = physics.calc_line_profile(
                self.data[f"tau_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )

            # Optical depth spectrum (shape: spectral, cloud)
            if label in ["1612", "1665"]:
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
        return tau

    def add_likelihood(self):
        """Add likelihood to the model."""
        # Predict optical depth spectra
        tau = self.predict_tau()

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
