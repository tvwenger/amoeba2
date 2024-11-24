"""emission_absorption_model.py
EmissionAbsorptionModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt
import numpy as np

from bayes_spec import BaseModel

from amoeba2.utils import _G, _OH, get_molecule_data
from amoeba2 import physics


class EmissionAbsorptionModel(BaseModel):
    """Definition of the TBTauModel. SpecData keys must be "absorption_1612", "absorption_1665",
    "absorption_1667", "absorption_1720", "emission_1612", "emission_1665", "emission_1667",
    and "emission_1720".
    """

    def __init__(self, *args, mol_data: Optional[dict] = None, bg_temp: float = 3.77, **kwargs):
        """Initialize a new EmissionAbsorptionModel instance.

        Parameters
        ----------
        mol_data : Optional[dict], optional
            OH molecular data dictionary returned by get_molecule_data(). If None, it will
            be downloaded. Default is None.
        bg_temp : float, optional
            Assumed background temperature (K), by default 3.77
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # get OH data
        if mol_data is None:
            self.mol_data = get_molecule_data()
        else:
            self.mol_data = mol_data

        # Save background temperature
        self.bg_temp = bg_temp

        # Add states, transitions to model
        self.model.add_coords(
            {
                "state": [0, 1, 2, 3],
                "transition": ["1612", "1665", "1667", "1720"],
            }
        )

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_N0",
            "velocity",
            "fwhm",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_N0": r"$\log_{10} N_0$ (cm$^{-2}$)",
                "log_boltz_factor": r"$\ln B$",
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
        prior_log10_N0: Iterable[float] = [13.0, 1.0],
        prior_log_boltz_factor: Iterable[float] = [-0.1, 0.1],
        prior_log10_depth: Iterable[float] = [0.0, 0.25],
        prior_log10_Tkin: Iterable[float] = [2.0, 1.0],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_log10_nth_fwhm_1pc: Iterable[float] = [0.2, 0.1],
        prior_depth_nth_fwhm_power: Iterable[float] = [0.4, 0.1],
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        ordered: bool = False,
        mainline_pos_tau: bool = False,
    ):
        """Add priors and deterministics to the model.

        Parameters
        ----------
        prior_log10_N0 : Iterable[float], optional
            Prior distribution on log10 column density (cm-2) in lowest energy state, by default [13.0, 1.0], where
            log10_N0 ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log_boltz_factor : Iterable[float], optional
            Prior distribution on log Boltzmann factor = -h*freq/(k*Tex), by default [-0.1, 0.1], where
            log_boltz_factor ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_depth : Iterable[float], optional
            Prior distribution on log10 depth (pc), by default [0.0, 0.25], where
            log10_depth ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_Tkin : Iterable[float], optional
            Prior distribution on log10 kinetic temperature (K), by default [2.0, 1.0], where
            log10_Tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_nth_fwhm_1pc : Iterable[float], optional
            Prior distribution on non-thermal line width at 1 pc, by default [0.2, 0.1], where
            log10_nth_fwhm_1pc ~ Normal(mu=prior[0], sigma=prior[1])
        prior_depth_nth_fwhm_power : Iterable[float], optional
            Prior distribution on depth vs. non-thermal line width power law index, by default [0.4, 0.1], where
            depth_nth_fwhm_power ~ Normal(mu=prior[0], sigma=prior[1])
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False. If True, the prior
            distribution on the velocity becomes
            velocity(cloud = n) ~ prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        mainline_pos_tau: bool, optional
            If True, assume positive main line excitation temperatures, by default False. If True, the
            prior distribution on the Boltzmann factor for the main lines (1665 and 1667) becomes
            log_boltz_factor ~ HalfNormal(sigma=prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # state 0 column density (cm-2; shape: clouds)
            log10_N0_norm = pm.Normal("log10_N0_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_N0 = pm.Deterministic("log10_N0", prior_log10_N0[0] + prior_log10_N0[1] * log10_N0_norm, dims="cloud")

            # 2 -> 1 (1612 MHz) Boltzmann factor (shape: clouds)
            log_boltz_factor_1612_norm = pm.Normal("log_boltz_factor_1612_norm", mu=0.0, sigma=1.0, dims="cloud")
            log_boltz_factor_1612 = pm.Deterministic(
                "log_boltz_factor_1612",
                prior_log_boltz_factor[0] + prior_log_boltz_factor[1] * log_boltz_factor_1612_norm,
                dims="cloud",
            )

            if mainline_pos_tau:
                # 0 -> 2 (1665 MHz) Boltzmann factor must be positive (shape: clouds)
                log_boltz_factor_1665_norm = pm.HalfNormal("log_boltz_factor_1665_norm", sigma=1.0, dims="cloud")
                log_boltz_factor_1665 = pm.Deterministic(
                    "log_boltz_factor_1665", -prior_log_boltz_factor[1] * log_boltz_factor_1665_norm, dims="cloud"
                )

                # 1 -> 3 (1667 MHz) Boltzmann factor must be positive (shape: clouds)
                log_boltz_factor_1667_norm = pm.HalfNormal("log_boltz_factor_1667_norm", sigma=1.0, dims="cloud")
                log_boltz_factor_1667 = pm.Deterministic(
                    "log_boltz_factor_1667", -prior_log_boltz_factor[1] * log_boltz_factor_1667_norm, dims="cloud"
                )
            else:
                # 0 -> 2 (1665 MHz) Boltzmann factor (shape: clouds)
                log_boltz_factor_1665_norm = pm.Normal("log_boltz_factor_1665_norm", mu=0.0, sigma=1.0, dims="cloud")
                log_boltz_factor_1665 = pm.Deterministic(
                    "log_boltz_factor_1665",
                    prior_log_boltz_factor[0] + prior_log_boltz_factor[1] * log_boltz_factor_1665_norm,
                    dims="cloud",
                )

                # 1 -> 3 (1667 MHz) Boltzmann factor (shape: clouds)
                log_boltz_factor_1667_norm = pm.Normal("log_boltz_factor_1667_norm", mu=0.0, sigma=1.0, dims="cloud")
                log_boltz_factor_1667 = pm.Deterministic(
                    "log_boltz_factor_1667",
                    prior_log_boltz_factor[0] + prior_log_boltz_factor[1] * log_boltz_factor_1667_norm,
                    dims="cloud",
                )

            # 3 -> 0 (1720 MHz) Boltzmann factor (shape: clouds)
            # Boltzmann factor sum rule
            _ = pm.Deterministic(
                "log_boltz_factor_1720", log_boltz_factor_1665 + log_boltz_factor_1667 - log_boltz_factor_1612
            )

            # State column density (cm-2; shape: state, cloud)
            log10_N2 = pm.Deterministic(
                "log10_N2", log10_N0 + log_boltz_factor_1665 * np.log10(np.e) - pt.log10(_G[0] / _G[2]), dims="cloud"
            )
            log10_N1 = pm.Deterministic(
                "log10_N1", log10_N2 - log_boltz_factor_1612 * np.log10(np.e) + pt.log10(_G[1] / _G[2]), dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_N3", log10_N1 + log_boltz_factor_1667 * np.log10(np.e) - pt.log10(_G[1] / _G[3]), dims="cloud"
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
                )
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + prior_velocity[1] * velocity_norm,
                    dims="cloud",
                )

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

            # Excitation temperatures (K; shape: clouds)
            for freq, label in zip(self.mol_data["freq"], self.model.coords["transition"]):
                _ = pm.Deterministic(
                    f"Tex_{label}",
                    physics.calc_Tex(freq, self.model[f"log_boltz_factor_{label}"]),
                    dims="cloud",
                )

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
        for freq, Aul, label in zip(self.mol_data["freq"], self.mol_data["Aul"], self.model.coords["transition"]):
            # Line profile (km-1 s; shape: spectral, cloud)
            line_profile_absorption = physics.calc_line_profile(
                self.data[f"absorption_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )
            line_profile_emission = physics.calc_line_profile(
                self.data[f"emission_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )

            # Optical depth spectrum (shape: spectral, cloud)
            tau_spectrum_absorption = physics.calc_optical_depth(
                _G[_OH[label][0]],
                _G[_OH[label][1]],
                10.0 ** self.model[f"log10_N{_OH[label][1]}"],
                pt.exp(self.model[f"log_boltz_factor_{label}"]),
                line_profile_absorption,
                freq,
                Aul,
            )
            tau_spectrum_emission = physics.calc_optical_depth(
                _G[_OH[label][0]],
                _G[_OH[label][1]],
                10.0 ** self.model[f"log10_N{_OH[label][1]}"],
                pt.exp(self.model[f"log_boltz_factor_{label}"]),
                line_profile_emission,
                freq,
                Aul,
            )

            # Sum over clouds
            absorption[label] = 1.0 - pt.exp(-tau_spectrum_absorption.sum(axis=1))

            # Radiative transfer
            emission[label] = physics.radiative_transfer(
                freq,
                tau_spectrum_emission,
                self.model[f"Tex_{label}"],
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
            for label in self.model.coords["transition"]:
                _ = pm.Normal(
                    f"absorption_{label}",
                    mu=absorption[label] + baseline_models[f"absorption_{label}"],
                    sigma=self.data[f"absorption_{label}"].noise,
                    observed=self.data[f"absorption_{label}"].brightness,
                )
                _ = pm.Normal(
                    f"emission_{label}",
                    mu=emission[label] + baseline_models[f"emission_{label}"],
                    sigma=self.data[f"emission_{label}"].noise,
                    observed=self.data[f"emission_{label}"].brightness,
                )
