"""
This file conducts the postprocessing pipeline for the 4 cases of .json outputs from bilby_cbc_param_estimation.py.
By applying Bayesian re-weighting on the posteriors for each case, it constrains the parameter probabilities
even further. There will be 3 main postprocessing lines: sGRBs, host galaxy redshift, and both. 
"""
import bilby
import numpy as np 
from scipy.stats import entropy
import matplotlib.pyplot as plt

def em_likelihood_dl(dl_array, em_estimate_dl, em_sigma_dl):
  weights = np.exp(-0.5 * ((dl_array - em_estimate_dl) / em_sigma_dl)**2)
  weights /= np.sum(weights)
  return weights 


def em_host_galaxy_postprocessing(case_no, randomseed, true_dl_param, true_thetajn_param):
  np.random.seed(randomseed)

  # Load the same GW-only posterior file as before
  gw_results = bilby.result.read_in_result(filename=f"outdir/{case_no}_result.json")
  dl_array = gw_results.posterior["luminosity_distance"].values

  # Host-galaxy redshift constrains ONLY distance -- theta_jn is untouched here,
  # so true_thetajn_param is accepted but unused (kept for consistent function signatures
  # across your 3 pipelines, and for logging/future joint-combination use)

  true_dl = true_dl_param

  # Sigma as a PERCENTAGE of true distance, based on GW170817's host galaxy (NGC 4993)
  # direct distance measurement: 40.7 +/- 2.36 Mpc (surface brightness fluctuation method)
  # 2.36 / 40.7 = 0.058 -> ~5.8% precision
  em_sigma_dl = 0.058 * true_dl

  em_estimate_dl = true_dl + np.random.normal(0, em_sigma_dl)

  em_dl_array = em_likelihood_dl(dl_array=dl_array, em_estimate_dl=em_estimate_dl, em_sigma_dl=em_sigma_dl)

  uniform_weights_dl = np.ones_like(dl_array) / len(dl_array)

  dl_entropy = entropy(em_dl_array, uniform_weights_dl)
  ess_dl = 1 / np.sum(em_dl_array**2)

  print("\n\n\n")
  print("------------- Host Galaxy KL Divergence -------------")
  print(f"Luminosity Distance: {dl_entropy}")
  print(f"ESS luminosity distance: {ess_dl:.1f} / {len(dl_array)}")
  print("\n\n\n")

  return dl_entropy