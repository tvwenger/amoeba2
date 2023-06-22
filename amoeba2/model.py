"""
model.py - AMOEBA model definitions

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
Trey V. Wenger - May 2023 - v2.0
    Based on amoeba.py, complete re-write in pymc
"""

import os
import numpy as np
import pymc as pm
import arviz as az
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

import matplotlib.pyplot as plt
import corner

from .data import AmoebaData

# Conversion from Gaussian standard deviation to FWHM
_SIG_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def predict_tau_spectrum(
    velocity: list[float],
    peak_tau: list[float],
    center: list[float],
    fwhm: list[float],
):
    """
    Return optical depth spectrum for given model parameters

    Inputs:
        velocity :: 1-D array of scalars
            N-length array of velocities
        peak_tau :: N-D array of scalars
            Peak optical depths, with component along last axis
        center :: N-D array of scalars
            Gaussian centroids, with component along last axis
        fwhm :: N-D array of scalars
            Gaussian FWHMs, with component along last axis

    Returns: tau_spectrum
        tau_spectra :: 1-D array of scalars
            N-length predicted optical depth spectrum
    """
    sigma = fwhm / _SIG_FWHM
    tau_spectra = peak_tau * np.exp(-0.5 * (velocity - center) ** 2.0 / sigma**2.0)
    # sum over components
    return tau_spectra.sum(axis=-1)


class _AmoebaBaseModel:
    """
    AMOEBA base model definition
    """

    def __init__(self, n_gauss: int, seed: int, verbose=False):
        """
        Initialize a new AmoebaBaseModel instance.

        Inputs:
            n_gauss :: integer
                Number of Gaussian components
            seed :: integer
                Random state seed
            verbose :: boolean
                If True, print helpful information. Default = False
        """
        self._transitions = ["1612", "1665", "1667", "1720"]
        self.data = None
        self.n_gauss = n_gauss
        self.seed = seed
        self.verbose = verbose

        # initialize model
        self.model = pm.Model()

        # storage for results
        self.trace = None
        self.posterior_samples = None

        # reset convergence checks
        self._reset()

    def _reset(self):
        """
        Reset convergence checks (i.e., after running fit() again)

        Inputs: None
        Returns: Nothing
        """
        self._gmm_n_gauss = None
        self._cluster_converged = None
        self._chains_converged = None
        self._good_chains = None
        self._has_divergences = None

    def cluster_converged(self):
        """
        Check if GMM clusters appear converged.

        Inputs: None

        Returns: converged
            converged :: boolean
                True if clusters are converged
        """
        if self._cluster_converged is None:
            raise ValueError("Cluster converge not checked. try model.fit()")
        return self._cluster_converged

    def chains_converged(self, frac_good_chains=0.6, mad_threshold=5.0):
        """
        Check if chains appear converged.

        Inputs:
            frac_good_chains :: scalar
                Chains are converged if the number of good chains exceeds
                {frac_good_chains} * {num_chains}
                Default = 0.6
            mad_threshold :: scalar
                Chains are converged if all have BICs within
                {mad_threshold} * MAD of the clustered BIC
                Default = 5.0

        Returns: converged
            converged :: boolean
                True if chains appear converged
        """
        if self.trace is None:
            raise ValueError("No trace. try model.fit()")

        # check if already determined
        if self._chains_converged is not None:
            return self._chains_converged

        good_chains = self.good_chains()
        num_good_chains = len(good_chains)
        num_chains = len(self.trace.posterior.chain)
        if num_good_chains <= frac_good_chains * num_chains:
            return False

        # per-chain BIC
        bics = np.array([self.bic(chain=chain) for chain in good_chains])
        mad = np.median(np.abs(bics - np.median(bics)))

        # BIC of clustered chains
        clustered_bic = self.bic()

        self._chains_converged = np.all(
            np.abs(bics - clustered_bic) < mad_threshold * mad
        )
        return self._chains_converged

    def has_divergences(self):
        """
        Check if there are any divergences in the good chains.

        Inputs: None
        Returns: divergences
            divergences :: boolean
                True if the model has any divergences
        """
        # check if already determined
        if self._has_divergences is not None:
            return self._has_divergences

        self._has_divergences = (
            self.trace.sample_stats.diverging.sel(chain=self.good_chains()).data.sum()
            > 0
        )
        return self._has_divergences

    def good_chains(self, mad_threshold=5.0):
        """
        Identify bad chains as those with deviant BICs.

        Inputs:
            mad_threshold :: scalar
                Chains are good if they have BICs within
                {mad_threshold} * MAD of the median BIC.
                Default = 5.0

        Returns: good_chains
            good_chains :: 1-D array of integers
                Chains that appear converged
        """
        if self.trace is None:
            raise ValueError("No trace. try model.fit()")

        # check if already determined
        if self._good_chains is not None:
            return self._good_chains

        # per-chain BIC
        bics = np.array(
            [self.bic(chain=chain) for chain in self.trace.posterior.chain.data]
        )
        mad = np.median(np.abs(bics - np.median(bics)))
        good = np.abs(bics - np.median(bics)) < mad_threshold * mad

        self._good_chains = self.trace.posterior.chain.data[good]
        return self._good_chains

    def _set_prior(
        self,
        parameters: list[str],
        distribution: str,
        scale: float,
        offset: float,
    ):
        """
        Add a prior on a parameter to the model.

        Inputs:
            parameters :: list of strings
                Parameters for which to set a prior.
            distribution :: string
                Shape of prior distribution. One of "uniform" or "normal"
            scale :: scalar
            offset :: scalar
                Transformation from scaled parameter space to true parameter space
                true_parameter = scale * scaled_parameter + offset

        Returns: Nothing
        """
        with self.model:
            for param in parameters:
                if distribution == "uniform":
                    _ = pm.Uniform(
                        f"scaled_{param}",
                        lower=-1.0,
                        upper=1.0,
                        shape=self.n_gauss,
                    )

                elif distribution == "normal":
                    _ = pm.Normal(
                        f"scaled_{param}",
                        mu=0.0,
                        sigma=1.0,
                        shape=self.n_gauss,
                    )

                else:
                    raise ValueError(f"uncaught distribution {distribution}")

                # add deterministic quantity
                determ = scale * self.model[f"scaled_{param}"] + offset
                _ = pm.Deterministic(param, determ)

    def _cluster_posterior(self, cluster_more=3, rel_bic_threshold=0.1):
        """
        Break the labeling degeneracy (each chain could have a different
        order of components) by identifying clusters in posterior samples.
        Determine if optimal number of components matches n_gauss as
        a convergence test.

        Inputs:
            cluster_more :: integer
                Try fitting Gaussian Mixture Model with n_components between
                1 and n_gauss + cluster_more
            rel_bic_threshold :: scalar
                Identify optimal number of components, n, when
                (BIC[n+1] - BIC[n]) < rel_bic_threshold * BIC[n]

        Returns: Nothing
        """
        # Get posterior samples, flatten
        good_chains = self.good_chains()
        features = np.array(
            [
                self.trace.posterior[param].sel(chain=good_chains).data.flatten()
                for param in self._parameters
            ]
        ).T

        # Use a Gaussian Mixture Model to cluster posterior samples. Test
        # different number of clusters as a convergence check.
        if self.verbose:
            print("Clustering posterior samples...")
        gmms = {}
        max_clusters = self.n_gauss + cluster_more
        n_clusters = [i for i in range(1, max_clusters + 1)]
        for n_cluster in n_clusters:
            gmms[n_cluster] = GaussianMixture(
                n_components=n_cluster,
                max_iter=100,
                init_params="kmeans",
                n_init=10,
                verbose=False,
                random_state=self.seed,
            )
            gmms[n_cluster].fit(features)

        # identify knee in BIC distribution
        self._gmm_bics = np.array(
            [gmms[n_cluster].bic(features) for n_cluster in n_clusters]
        )
        # first time when relative BIC change is less than threshold
        rel_bic_diff = np.abs(np.diff(self._gmm_bics) / self._gmm_bics[:-1])
        best = np.where(rel_bic_diff < rel_bic_threshold)[0]
        if len(best) == 0:
            self._gmm_n_gauss = -1
        else:
            self._gmm_n_gauss = n_clusters[best[0]]

        # check if posterior clusters matches n_gauss
        self._cluster_converged = self._gmm_n_gauss == self.n_gauss

        if self.verbose:
            if self._cluster_converged:
                print(f"GMM converged at n_gauss = {self._gmm_n_gauss}")
            elif self._gmm_n_gauss == -1:
                print(f"GMM prefers n_gauss > {self.n_gauss + cluster_more}")
            else:
                print(f"GMM prefers n_gauss = {self._gmm_n_gauss}")

        # Save clustered posterior samples for n_clusters = n_gauss
        self.posterior_samples = {}
        labels = gmms[self.n_gauss].predict(features)
        for param in self._parameters + self._deterministics:
            posterior = (
                self.trace.posterior[param].sel(chain=good_chains).data.flatten()
            )
            self.posterior_samples[param] = {}
            for i in range(self.n_gauss):
                self.posterior_samples[param][i] = posterior[labels == i]

    def _plot_predictive(
        self,
        predictive: az.InferenceData,
        prefix: str,
        plot_fname: str,
        xlabel: str,
        ylabels: list[str],
    ):
        """
        Generate plots of predictive checks.

        Inputs:
            predictive :: InferenceData
                Predictive samples
            prefix :: string
                Predictive data like {prefix}_{transition}
                e.g., prefix = "tau_spectra"
            plot_fname :: string
                Plot filename
            xlabel :: string
                x-axis label
            ylabels :: list of strings
                y-axis labels

        Returns: Nothing
        """
        fig, axes = plt.subplots(
            len(self._transitions), sharex=True, sharey=True, figsize=(8, 11)
        )
        num_chains = len(predictive.chain)
        for ax, transition, ylabel in zip(axes, self._transitions, ylabels):
            color = iter(plt.cm.rainbow(np.linspace(0, 1, num_chains)))
            for chain in predictive.chain:
                c = next(color)
                outcomes = predictive[f"{prefix}_{transition}"].sel(chain=chain).data
                ax.plot(
                    self.data.spectra[transition].velocity,
                    outcomes.T,
                    linestyle="-",
                    color=c,
                    alpha=0.1,
                )
            ax.plot(
                self.data.spectra[transition].velocity,
                self.data.spectra[transition].spectrum,
                "k-",
            )
            ax.set_ylabel(ylabel)
        axes[-1].set_xlabel(xlabel)
        fig.subplots_adjust(hspace=0)
        fig.savefig(plot_fname, bbox_inches="tight")
        plt.close(fig)

    def set_data(self, data: AmoebaData):
        """
        Set or update the data.

        Inputs:
            data :: AmoebaData
                The data

        Returns: Nothing
        """
        self.data = data
        self._update_data()

    def fit(
        self,
        draws=1000,
        tune=1000,
        target_accept=0.8,
        chains: int = 4,
        cores: int = None,
        cluster_more=3,
        rel_bic_threshold=0.1,
    ):
        """
        Sample posterior distribution.

        Inputs:
            draws :: integer
                Number of non-tuning samples
            tune :: integer
                Number of tuning samples
            target_accept :: scalar
                Target sampling acceptance rate. Default = 0.8
            chains :: integer
                Number of chains. Default = 4
            cores :: integer
                Number of cores to run chains in parallel.
                If None, then cores = min(4, num_cpus)
                where num_cpus is the number of CPUs in the system
            cluster_more :: integer
                Try fitting Gaussian Mixture Model with n_components between
                1 and n_gauss + cluster_more
            rel_bic_threshold :: scalar
                Identify optimal number of components, n, when
                (BIC[n+1] - BIC[n]) < rel_bic_threshold * BIC[n]

        Returns: Nothing
        """
        # check that we have enough chains for convergence checks
        if chains < 4:
            raise ValueError("You should use at least 4 chains!")

        # reset convergence checks
        self._reset()

        rng = np.random.default_rng(self.seed)
        with self.model:
            self.trace = pm.sample(
                init="jitter+adapt_diag",
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                progressbar=self.verbose,
                target_accept=target_accept,
                discard_tuned_samples=False,
                compute_convergence_checks=False,
                random_seed=rng,
            )

        # check how many chains converged
        if self.verbose:
            good_chains = self.good_chains()
            if len(good_chains) < chains:
                print(f"Only {len(good_chains)} chains appear converged.")

        # check if there were any divergences
        if self.verbose:
            num_divergences = self.trace.sample_stats.diverging.data.sum()
            if num_divergences > 0:
                print(f"There were {num_divergences} divergences.")

        # cluster posterior samples
        self._cluster_posterior(
            cluster_more=cluster_more,
            rel_bic_threshold=rel_bic_threshold,
        )

    def point_estimate(
        self,
        stats=["mean", "std", "hdi"],
        hdi_prob=0.68,
        chain=None,
    ):
        """
        Get point estimate and other statistics from trace

        Inputs:
            stats :: list of strings
                Statstics to return. Options include "mean", "median",
                "std" (standard deviation), "mad" (median absolute deviation),
                "hdi" (highest density interval)
            hdi_prob :: scalar
                Highest density interval probability to evaluate
                (e.g., stats=['hdi'], hdi_prob=0.68 will calculate
                the 68% highest density interval)
            chain :: None or integer
                If None (default), evaluate statistics across all chains using
                clustered posterior samples. Otherwise, evaluate statistics for
                this chain only.

        Returns: point_estimate
            point_estimate :: dictionary
                Statistics for each parameter
        """
        if chain is None and self.posterior_samples is None:
            raise ValueError("Model has no posterior samples. try model.fit()")

        point_estimate = {}
        for param in self._parameters + self._deterministics:
            if chain is None:
                posterior = self.posterior_samples[param]
            else:
                posterior = self.trace.posterior[param].sel(chain=chain).data.T

            point_estimate[param] = {}
            for stat in stats:
                if stat == "mean":
                    point_estimate[param][stat] = [
                        np.mean(posterior[i], axis=0) for i in range(self.n_gauss)
                    ]
                elif stat == "median":
                    point_estimate[param][stat] = [
                        np.median(posterior[i], axis=0) for i in range(self.n_gauss)
                    ]
                elif stat == "std":
                    point_estimate[param][stat] = [
                        np.std(posterior[i], axis=0) for i in range(self.n_gauss)
                    ]
                elif stat == "mad":
                    median = [
                        np.median(posterior[i], axis=0) for i in range(self.n_gauss)
                    ]
                    point_estimate[param][stat] = [
                        np.median(np.abs(posterior[i] - median[i]), axis=0)
                        for i in range(self.n_gauss)
                    ]
                elif stat == "hdi":
                    point_estimate[param][stat] = [
                        az.hdi(posterior[i], hdi_prob=hdi_prob)
                        for i in range(self.n_gauss)
                    ]

        return point_estimate

    def lnlike_mean_point_estimate(self, chain=None):
        """
        Evaluate model log-likelihood at the mean point estimate.

        Inputs:
            chain :: None or integer
                If None (default), determine point estimate across all chains using
                clustered posterior samples. Otherwise, get point estimate for
                this chain only.

        Returns: lnlike
            lnlike :: scalar
                Log likelihood at point
        """
        # scale parameters for model
        point = self.point_estimate(stats=["mean"], chain=chain)
        params = {param: point[param]["mean"] for param in self._parameters}

        return float(self.model.observedlogp.eval(params))

    def bic(self, chain=None):
        """
        Calculate the Bayesian information criterion at the mean point
        estimate.

        Inputs:
            chain :: integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only.

        Returns: bic
            bic :: scalar
                Bayesian information criterion
        """
        lnlike = self.lnlike_mean_point_estimate(chain=chain)
        return self._n_params * np.log(self._n_data) - 2.0 * lnlike

    def plot_traces(self, plot_fname: str, warmup=False):
        """
        Plot traces.

        Inputs:
            plot_fname :: string
                Plot filename
            warmup :: boolean
                If True, plot warmup samples instead

        Returns: Nothing
        """
        posterior = self.trace.warmup_posterior if warmup else self.trace.posterior
        axes = az.plot_trace(posterior, var_names=self._deterministics)
        fig = axes.ravel()[0].figure
        fig.savefig(plot_fname, bbox_inches="tight")
        plt.close(fig)

    def plot_corner(self, plot_fname: str, truths=None):
        """
        Generate corner plots with optional truths

        Inputs:
            plot_fname :: string
                Figure filename that includes extension. One corner plot has *all*
                of the samples per parameter. Additionally, one corner plot is generated
                per Gaussian component, and the component number is appended like "_0"
                before the extension. Out filename is like {plot_fname}_0.pdf, etc.
            truths :: dictionary
                Dictionary of "truths" for each determinstic parameter

        Returns: Nothing
        """
        if truths is not None and len(truths["center"]) != self.n_gauss:
            print(f"Expected {self.n_gauss} components in truths")

        # point estimate
        point_estimate = self.point_estimate(stats=["mean"])

        # First plot all samples
        labels = []
        posteriors = []
        for param in self._deterministics:
            labels += [f"{param}"]
            posteriors += [
                np.concatenate(
                    [self.posterior_samples[param][n] for n in range(self.n_gauss)]
                )
            ]
        fig = corner.corner(np.array(posteriors).T, labels=labels)
        # Add truths
        if truths is not None:
            for n in range(self.n_gauss):
                values = [truths[param][n] for param in self._deterministics]
                corner.overplot_lines(fig, values)
        fig.savefig(plot_fname, bbox_inches="tight")
        plt.close(fig)

        # Generate plots for individual components
        for i in range(self.n_gauss):
            labels = []
            posteriors = []
            for param in self._deterministics:
                labels += [f"{param}[{i}]"]
                posteriors += [self.posterior_samples[param][i]]

            truth_values = None
            if truths is not None:
                # get closest center to this component
                closest = np.argmin(
                    np.abs(point_estimate["center"]["mean"][i] - truths["center"])
                )
                truth_values = [
                    truths[param][closest] for param in self._deterministics
                ]

            fig = corner.corner(
                np.array(posteriors).T, labels=labels, truths=truth_values
            )
            fname = os.path.splitext(plot_fname)
            fname = fname[0] + f"_{i}" + fname[1]
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)


class AmoebaTauModel(_AmoebaBaseModel):
    """
    AMOEBA model definition for optical-depth only scenario
    """

    def __init__(self, n_gauss: int, seed=5391, verbose=False):
        """
        Initialize a new AmoebaTauModel instance.

        Inputs:
            n_gauss :: integer
                Number of Gaussian components
            seed :: integer
                Random state seed
            verbose :: boolean
                If True, print helpful information
        """
        super().__init__(n_gauss, seed=seed, verbose=verbose)

        # model parameters
        self._parameters = [
            "scaled_center",
            "scaled_log10_fwhm",
            "scaled_peak_tau_1612",
            "scaled_peak_tau_1665",
            "scaled_peak_tau_1667",
            "scaled_peak_tau_1720",
        ]

        # useful deterministic quantities
        self._deterministics = [
            "center",
            "log10_fwhm",
            "peak_tau_1612",
            "peak_tau_1665",
            "peak_tau_1667",
            "peak_tau_1720",
        ]

        # Number of model parameters per gaussian = center + fwhm + 3 * peak
        # (N.B. One peak is not independent because of sum rule)
        self._n_params = 5 * self.n_gauss
        self._n_data = 0

        # Add mutable data storage to model
        with self.model:
            for transition in self._transitions:
                _ = pm.MutableData(f"{transition}_velocity", [])
                _ = pm.MutableData(f"{transition}_spectrum", [])
                _ = pm.MutableData(f"{transition}_rms", 0.0)

        # arrays of zeros that represent the expected tau_sum_rule likelihoods
        with self.model:
            _ = pm.ConstantData("tau_sum_rule_zeros", np.zeros(n_gauss))

    def _update_data(self):
        """
        Update model data in case the data have changed.

        Inputs: None
        Returns: Nothing
        """
        with self.model:
            for transition in self._transitions:
                pm.set_data(
                    {f"{transition}_velocity": self.data.spectra[transition].velocity}
                )
                pm.set_data(
                    {f"{transition}_spectrum": self.data.spectra[transition].spectrum}
                )
                pm.set_data({f"{transition}_rms": self.data.spectra[transition].rms})

        # Number of data points
        self._n_data = np.sum(
            [
                len(self.data.spectra[transition].velocity)
                for transition in self._transitions
            ]
        )

    def set_prior(
        self, parameter: str, distribution: str, shape_parameters: list[float]
    ):
        """
        Add a prior on a parameter to the model.

        Inputs:
            parameter :: string
                Parameter for which to set a prior. One of "center", "log10_fwhm", or "peak_tau".
            distribution :: string
                Shape of prior distribution. One of "uniform" or "normal"
            shape_parameters :: list
                List containing shape parameters for prior distributions. For
                distribution ==
                    "uniform" -> [lower_bound, upper_bound]
                    "normal" ->[mu, sigma]

        Returns: Nothing
        """
        _distributions = ["uniform", "normal"]
        _parameters = ["center", "log10_fwhm", "peak_tau"]

        if distribution not in _distributions:
            raise ValueError(f"distribution must be one of {_distributions}")
        if parameter not in _parameters:
            raise ValueError(f"parameter must be one of {_parameters}")

        # expand peak_tau to include each transition
        if parameter == "peak_tau":
            parameters = [f"peak_tau_{transition}" for transition in self._transitions]
        else:
            parameters = [parameter]

        # set data scaling factors
        if distribution in ["normal"]:
            offset, scale = shape_parameters
        else:
            lower, upper = shape_parameters
            scale = 0.5 * (upper - lower)
            offset = lower + scale

        super()._set_prior(parameters, distribution, scale, offset)

    def add_likelihood(self, distribution: str):
        """
        Add the optical depth sum rule and data likelihoods to the model.

        Inputs:
            distribution :: string
                Likelihood distribution shape. One of "normal"

        Returns: Nothing
        """
        with self.model:
            tau_sum_rule_mu = (
                self.model["peak_tau_1612"]
                - self.model["peak_tau_1665"] / 5.0
                - self.model["peak_tau_1667"] / 9.0
                + self.model["peak_tau_1720"]
            )
            tau_sum_rule_rms = np.sqrt(
                self.model["1612_rms"] ** 2.0
                + (self.model["1665_rms"] / 5.0) ** 2.0
                + (self.model["1667_rms"] / 9.0) ** 2.0
                + self.model["1720_rms"] ** 2.0
            )

            if distribution == "normal":
                _ = pm.Normal(
                    "tau_sum_rule",
                    mu=tau_sum_rule_mu,
                    sigma=tau_sum_rule_rms,
                    observed=self.model["tau_sum_rule_zeros"],
                )
            else:
                raise ValueError(f"uncaught distribution {distribution}")

            for transition in self._transitions:
                tau_spectrum_mu = predict_tau_spectrum(
                    self.model[f"{transition}_velocity"][:, None],
                    self.model[f"peak_tau_{transition}"],
                    self.model["center"],
                    10.0 ** self.model["log10_fwhm"],
                )
                if distribution == "normal":
                    _ = pm.Normal(
                        f"tau_spectrum_{transition}",
                        mu=tau_spectrum_mu,
                        sigma=self.model[f"{transition}_rms"],
                        observed=self.model[f"{transition}_spectrum"],
                    )
                else:
                    raise ValueError(f"uncaught distribution {distribution}")

    def null_bic(self):
        """
        Evaluate the BIC for the null hypothesis (no components)

        Inputs: None
        Returns: Nothing
        """
        null_lnlike = 0.0
        for transition in self._transitions:
            null_lnlike += norm.logpdf(
                self.data.spectra[transition].spectrum,
                scale=self.data.spectra[transition].rms,
            ).sum()
        return -2.0 * null_lnlike

    def prior_predictive_check(self, samples=50, plot_fname: str = None):
        """
        Generate prior predictive samples, and optionally plot the outcomes.

        Inputs:
            samples :: integer
                Number of prior predictive samples to generate
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing prior predictive samples
        """
        rng = np.random.default_rng(self.seed)
        with self.model:
            trace = pm.sample_prior_predictive(samples=samples, random_seed=rng)

        if plot_fname is not None:
            xlabel = "Velocity"
            ylabels = [
                r"$\tau($" + transition + r"$)$" for transition in self._transitions
            ]
            super()._plot_predictive(
                trace.prior_predictive, "tau_spectrum", plot_fname, xlabel, ylabels
            )

        return trace.prior_predictive

    def posterior_predictive_check(self, thin=1, plot_fname: str = None):
        """
        Generate posterior predictive samples, and optionally plot the outcomes.

        Inputs:
            thin :: integer
                Thin posterior samples by keeping one in {thin}
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing posterior predictive samples
        """
        rng = np.random.default_rng(self.seed)
        with self.model:
            thinned_trace = self.trace.sel(draw=slice(None, None, thin))
            trace = pm.sample_posterior_predictive(
                thinned_trace, extend_inferencedata=True, random_seed=rng
            )

        if plot_fname is not None:
            xlabel = "Velocity"
            ylabels = [
                r"$\tau($" + transition + r"$)$" for transition in self._transitions
            ]
            super()._plot_predictive(
                trace.posterior_predictive.sel(chain=self.good_chains()),
                "tau_spectrum",
                plot_fname,
                xlabel,
                ylabels,
            )

        return trace.posterior_predictive
