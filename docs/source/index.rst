.. amoeba2 documentation master file, created by
   sphinx-quickstart on Tue Aug 20 12:17:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

amoeba2
=======

``amoeba2`` is a Bayesian model of OH emission and absorption spectra. Written in the ``bayes_spec`` Bayesian modeling framework, ``amoeba2`` implements two models, one for predicting optical depth spectral only, and one for predicting both optical depth spectra and emission spectra. The ``bayes_spec`` framework provides methods to fit these models to data using Monte Carlo Markov Chain methods.

Useful information can be found in the `amoeba2 Github repository <https://github.com/tvwenger/amoeba2>`_, the `bayes_spec Github repository <https://github.com/tvwenger/bayes_spec>`_, and in the tutorials below.

============
Installation
============
.. code-block::

    pip install amoeba2

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/tau_model
   notebooks/tb_tau_model
   notebooks/optimization
   notebooks/real_data

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules
