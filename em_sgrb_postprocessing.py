import bilby
import numpy as np
from scipy.stats import entropy

def em_likelihood_thetajn(thetajn_array, em_estimate_thetajn, em_sigma_thetajn):
  weights = np.exp(-0.5 * ((thetajn_array - em_estimate_thetajn) / em_sigma_thetajn)**2)
  weights /= np.sum(weights)
  return weights

def em_sgrb_postprocessing(case_no, randomseed, true_thetajn_param, verbose=False):
  np.random.seed(randomseed)
  gw_results = bilby.result.read_in_result(filename=f"outdir/{case_no}_result.json")
  thetajn_array = gw_results.posterior["theta_jn"].values

  true_thetajn = true_thetajn_param
  em_sigma_thetajn = 0.1  # rad -- VLBI-based, GW170817 (avg of 0.05-0.12 rad range)
  em_estimate_thetajn = true_thetajn + np.random.normal(0, em_sigma_thetajn)

  em_thetajn_array = em_likelihood_thetajn(thetajn_array, em_estimate_thetajn, em_sigma_thetajn)
  uniform_weights_thetajn = np.ones_like(thetajn_array) / len(thetajn_array)

  thetajn_entropy = entropy(em_thetajn_array, uniform_weights_thetajn)
  ess_thetajn = 1 / np.sum(em_thetajn_array**2)

  window = 2 * em_sigma_thetajn
  local_density = np.mean(np.abs(thetajn_array - true_thetajn) < window)

  if verbose:
    print(f"[sGRB] seed {randomseed}: KL={thetajn_entropy:.4f}, ESS={ess_thetajn:.1f}/{len(thetajn_array)}, local_density={local_density:.4f}")

  return [thetajn_array, em_thetajn_array, thetajn_entropy, ess_thetajn, local_density]