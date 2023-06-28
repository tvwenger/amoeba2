"""
amoeba.py - Automated Molecular Excitation Bayesian line-fitting Algorithm

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

from .model import AmoebaTauModel
from .data import AmoebaData


class Amoeba:
    """
    AMOEBA class definition
    """

    def __init__(
        self,
        max_n_gauss=10,
        seed=5391,
        verbose=False,
    ):
        """
        Initialize a new Amoeba instance

        Inputs:
            max_n_gauss :: integer
                Maximum number of Gaussian components to consider. Default = 10
            seed :: integer
                Random state seed. Default = 1234
            verbose :: boolean
                If True, print info. Default = False

        Returns: spectrum
            spectrum :: AmoebaSpectrum
                New AmoebaSpectrum instance
        """
        self.max_n_gauss = max_n_gauss
        self.verbose = verbose
        self.n_gauss = [i for i in range(1, max_n_gauss + 1)]
        self.seed = seed
        self.ready = False

        # initialize models
        self.models = {}
        for n_gauss in self.n_gauss:
            model = AmoebaTauModel(n_gauss, seed=seed, verbose=self.verbose)
            self.models[n_gauss] = model
        self.best_model = None

    def set_data(self, data: AmoebaData):
        """
        Set or update the data for each model.

        Inputs:
            data :: AmoebaData
                The data

        Returns: Nothing
        """
        for n_gauss in self.n_gauss:
            self.models[n_gauss].set_data(data)

    def set_prior(
        self, parameter: str, distribution: str, shape_parameters: list[float]
    ):
        """
        Add a prior on a parameter to the models.

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
        for n_gauss in self.n_gauss:
            self.models[n_gauss].set_prior(parameter, distribution, shape_parameters)

    def add_likelihood(self, distribution: str):
        """
        Add the optical depth sum rule and data likelihoods to the model.

        Inputs:
            distribution :: string
                Likelihood distribution shape. One of "normal"

        Returns: Nothing
        """
        for n_gauss in self.n_gauss:
            self.models[n_gauss].add_likelihood(distribution)
        self.ready = True

    def fit_all(
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
        Fit all of the models until clusters and chains converge, or
        until divergences occur, or until the BIC increases twice in a row.

        Inputs:
            draws :: integer
                Number of non-tuning samples. Default = 1000
            tune :: integer
                Number of tuning samples. Default = 1000
            target_accept :: scalar
                Target sampling acceptance rate. Default = 0.8
            chains :: integer
                Number of chains. You should use as many chains
                as reasonable for convergence checks. amoeba2 requires
                at least 4. Default = 4.
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

        Returns:
            Nothing
        """
        if not self.ready:
            raise RuntimeError(
                "You must first add the priors and likelihood to the models"
            )

        # reset best model
        self.best_model = None

        minimum_bic = self.models[self.n_gauss[0]].null_bic()
        last_bic = minimum_bic
        print(f"Null hypothesis BIC = {minimum_bic}")
        print()

        num_increase = 0
        for n_gauss in self.n_gauss:
            print(f"Fitting n_gauss = {n_gauss}")
            self.models[n_gauss].fit(
                draws,
                tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                cluster_more=cluster_more,
                rel_bic_threshold=rel_bic_threshold,
            )
            current_bic = self.models[n_gauss].bic()
            if self.verbose:
                print(f"Current BIC = {current_bic}")

            # update minimum BIC
            if current_bic < minimum_bic:
                minimum_bic = current_bic
                self.best_model = self.models[n_gauss]
                num_increase = 0

            # Check if BIC is increasing
            if current_bic > last_bic:
                num_increase += 1
            else:
                num_increase = 0

            # check stopping conditions
            if self.models[n_gauss].has_divergences():
                if self.verbose:
                    print("Model divergences. Stopping.")
                break

            if (
                self.models[n_gauss].cluster_converged()
                and self.models[n_gauss].chains_converged()
            ):
                if num_increase < 2:
                    if self.verbose:
                        print("Model converged, but BIC might decrease. Continuing.")
                else:
                    print("Model converged and BIC increasing. Stopping.")
                    break

            if num_increase > 1:
                if self.verbose:
                    print("BIC increasing. Stopping.")
                break

            last_bic = current_bic
            print()
        else:
            print("Reached maximum n_gauss. Stopping.")

    def fit_best(
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
        Starting with n_gauss = 1, fit with n_gauss suggested by the
        Gaussian Mixture Model until clusters and chains converge, or
        until divergences occur, or until the BIC increases twice in a row.

        Inputs:
            draws :: integer
                Number of non-tuning samples. Default = 1000
            tune :: integer
                Number of tuning samples. Default = 1000
            target_accept :: scalar
                Target sampling acceptance rate. Default = 0.8
            chains :: integer
                Number of chains. You should use as many chains
                as reasonable for convergence checks. amoeba2 requires
                at least 4. Default = 4.
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

        Returns:
            Nothing
        """
        if not self.ready:
            raise RuntimeError(
                "You must first add the priors and likelihood to the models"
            )

        # reset best model
        self.best_model = None

        minimum_bic = self.models[self.n_gauss[0]].null_bic()
        last_bic = minimum_bic
        print(f"Null hypothesis BIC = {minimum_bic}")
        print()

        num_increase = 0
        n_gauss = 1
        fit_n_gauss = {n: False for n in self.n_gauss}
        while True:
            current_best = "0" if self.best_model is None else self.best_model.n_gauss
            if self.verbose:
                print(f"Current best model: n_gauss = {current_best}")

            print(f"Fitting n_gauss = {n_gauss}")
            self.models[n_gauss].fit(
                draws,
                tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                cluster_more=cluster_more,
                rel_bic_threshold=rel_bic_threshold,
            )
            fit_n_gauss[n_gauss] = True
            current_bic = self.models[n_gauss].bic()
            if self.verbose:
                print(f"Current BIC = {current_bic}")

            # update minimum BIC
            if current_bic < minimum_bic:
                minimum_bic = current_bic
                self.best_model = self.models[n_gauss]

            # Check if BIC is increasing
            if current_bic > last_bic:
                num_increase += 1
            else:
                num_increase = 0

            # check stopping conditions
            if self.models[n_gauss].has_divergences():
                if self.verbose:
                    print("Model divergences. Stopping.")
                break

            if (
                self.models[n_gauss].cluster_converged()
                and self.models[n_gauss].chains_converged()
            ):
                if num_increase < 2:
                    if self.verbose:
                        print("Model converged, but BIC might decrease. Continuing.")
                else:
                    print("Model converged and BIC increasing. Stopping.")
                    break

            if num_increase > 1:
                if self.verbose:
                    print("BIC increasing. Stopping.")
                break

            # update n_gauss to GMM guess
            new_n_gauss = self.models[n_gauss]._gmm_n_gauss

            # check if GMM failed to identify n_gauss
            if new_n_gauss == -1:
                new_n_gauss = n_gauss + 1

            # check if we're over the limit
            if new_n_gauss > self.max_n_gauss:
                # check if we have fewer n_gauss we can still try
                for new_n_gauss in fit_n_gauss.keys():
                    if not fit_n_gauss[new_n_gauss]:
                        break
                else:
                    print("Exceeded max n_gauss. Stopping.")
                    break

            # check if we've already fit this one
            while new_n_gauss in fit_n_gauss.keys() and fit_n_gauss[new_n_gauss]:
                new_n_gauss += 1

            # now check that we're not over
            if new_n_gauss > self.max_n_gauss:
                print("Exceeded max n_gauss. Stopping.")
                break

            n_gauss = new_n_gauss
            print()
