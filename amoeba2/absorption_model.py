"""absorption_model.py
AbsorptionModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import pytensor.tensor as pt

from bayes_spec import BaseModel

from amoeba2.utils import get_molecule_data
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

        # Select features used for posterior clustering
        self._cluster_features += [
            "tau_1612",
            "tau_1665",
            "tau_1667",
            "velocity",
            "fwhm",
        ]

        # Add transitions to model
        self.model.add_coords(
            {
                "transition": ["1612", "1665", "1667", "1720"],
            }
        )

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "tau_1612": r"$\tau_{1612}$",
                "tau_1665": r"$\tau_{1665}$",
                "tau_1667": r"$\tau_{1667}$",
                "tau_1720": r"$\tau_{1720}$",
                "log10_depth": r"log$_{10}$ $d$ (pc)",
                "log10_Tkin": r"$\log_{10} T_{\rm kin}$ (K)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "log10_nth_fwhm_1pc": r"log$_{10}$ $\Delta V_{\rm 1 pc}$ (km s$^{-1}$)",
                "depth_nth_fwhm_power": r"$\alpha$",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nth}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_tau: Iterable[float] = [0.1, 0.1],
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
        prior_tau : float, optional
            Prior distribution on optical depth, by default [0.1, 0.1], where
            tau ~ Normal(mu=prior[0], sigma=prior[1])
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
            If True, assume positive main line optical depths, by default False. If True, the
            prior distribution on the main line (1665 and 1667 MHz) optical depths becomes
            tau ~ HalfNormal(sigma=prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # optical depth
            tau_1612_norm = pm.Normal("tau_1612_norm", mu=0.0, sigma=1.0, dims="cloud")
            tau_1612 = pm.Deterministic("tau_1612", prior_tau[0] + tau_1612_norm * prior_tau[1], dims="cloud")

            if mainline_pos_tau:
                tau_1665_norm = pm.HalfNormal("tau_1665_norm", sigma=1.0, dims="cloud")
                tau_1665 = pm.Deterministic("tau_1665", tau_1665_norm * prior_tau[1], dims="cloud")

                tau_1667_norm = pm.HalfNormal("tau_1667_norm", sigma=1.0, dims="cloud")
                tau_1667 = pm.Deterministic("tau_1667", tau_1667_norm * prior_tau[1], dims="cloud")
            else:
                tau_1665_norm = pm.Normal("tau_1665_norm", mu=0.0, sigma=1.0, dims="cloud")
                tau_1665 = pm.Deterministic("tau_1665", prior_tau[0] + tau_1665_norm * prior_tau[1], dims="cloud")

                tau_1667_norm = pm.Normal("tau_1667_norm", mu=0.0, sigma=1.0, dims="cloud")
                tau_1667 = pm.Deterministic("tau_1667", prior_tau[0] + tau_1667_norm * prior_tau[1], dims="cloud")

            # optical depth sum rule
            _ = pm.Deterministic("tau_1720", tau_1665 / 5.0 + tau_1667 / 9.0 - tau_1612, dims="cloud")

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

    def predict_absorption(self) -> dict:
        """Predict the absorption spectra from the model parameters.

        Returns
        -------
        dict
            Optical depth spectra for 1612, 1665, 1667, and 1720 MHz transitions
        """
        absorption = {}
        for label in self.model.coords["transition"]:
            # Line profile (km-1 s; shape: spectral, cloud)
            line_profile = physics.calc_line_profile(
                self.data[f"absorption_{label}"].spectral,
                self.model["velocity"],
                self.model["fwhm"],
            )

            # Optical depth spectrum (shape: spectral, cloud)
            tau_spectrum = self.model[f"tau_{label}"] * line_profile

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
            for label in self.model.coords["transition"]:
                _ = pm.Normal(
                    f"absorption_{label}",
                    mu=absorption[label] + baseline_models[f"absorption_{label}"],
                    sigma=self.data[f"absorption_{label}"].noise,
                    observed=self.data[f"absorption_{label}"].brightness,
                )
