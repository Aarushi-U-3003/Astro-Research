"""
This file conducts the postprocessing pipeline for the 4 cases of .json outputs from bilby_cbc_param_estimation.py.
By applying Bayesian re-weighting on the posteriors for each case, it constrains the parameter probabilities
even further. There will be 3 main postprocessing lines: sGRBs, host galaxy redshift, and both. 
"""
import bilby
import numpy as np 
from scipy.stats import entropy


np.random.seed(42)

# Load the result object using bilby -- can also be done with the json library
gw_results = bilby.result.read_in_result(filename="outdir/Case1_result.json")

# Extract luminosity_distance and theta_jn as a numpy arrays
dl_array = gw_results.posterior["luminosity_distance"].values
thetajn_array = gw_results.posterior["theta_jn"].values
# The above will be the GW-only probability distribution. 

"""def em_likelihood_thetajn(thetajn_array, em_estimate, em_sigma):
  return np.exp(-0.5 * ((thetajn_array - em_estimate) / em_sigma)**2)

def apply_nmma_em_constraints_thetajn(thetajn_array):
  return np.exp(-0.5 * ((thetajn_array - em_estimate) / em_sigma)**2)"""


# True injected values
true_dl = 150      # Mpc
true_thetajn = 0.0  # rad

# Decide how precise fake EM instrument is
em_sigma_dl = 10        # Mpc -- placeholder, refine later
em_sigma_thetajn = 0.1  # rad -- placeholder, refine later

em_estimate_dl = true_dl + np.random.normal(0, em_sigma_dl)
em_estimate_thetajn = true_thetajn + np.random.normal(0, em_sigma_thetajn)


print(em_estimate_dl)
print(em_estimate_thetajn)