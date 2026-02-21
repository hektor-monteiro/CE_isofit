# CE_IsoFit

1. Codebase Overview
The repository contains a pipeline for fitting isochrones to Gaia DR3 star cluster data using a Cross-Entropy (CE) optimization method. It estimates cluster parameters such as Age, Distance, Metallicity (FeH), and Extinction (Av).

2. Architecture
OCFit-9.0.py: The main driver script. It iterates over cluster data files, preprocesses data, sets up priors (using external tables or gradients), runs the optimization loop, and generates logs/plots.
oc_tools_padova_edr3.py: A library containing domain-specific logic:
Loading and interpolating isochrone grids (Padova/Parsec).
Generating synthetic clusters (IMF sampling, binarity, photometric errors).
Likelihood functions (lnlikelihoodCE).
gaia_edr3_tools.py: A library containing statistical and optimization tools:
run_isoc_CE: The implementation of the Cross-Entropy method.
fit_iso_GAIA: Wrapper for the CE optimization.
Other fitting routines (MCMC, Differential Evolution) which seem unused in the main workflow but available.
