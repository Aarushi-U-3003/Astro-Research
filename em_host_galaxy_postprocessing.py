import bilby
import numpy as np

def em_likelihood_dl(dl_array, em_estimate_dl, em_sigma_dl):
  weights = np.exp(-0.5 * ((dl_array - em_estimate_dl) / em_sigma_dl)**2)
  weights /= np.sum(weights)
  return weights

def em_host_galaxy_postprocessing(case_no, randomseed, true_dl_param, verbose=False):
  np.random.seed(randomseed)
  gw_results = bilby.result.read_in_result(filename=f"outdir/{case_no}_result.json")
  dl_array = gw_results.posterior["luminosity_distance"].values

  true_dl = true_dl_param
  em_sigma_dl = 0.058 * true_dl  # ~5.8% precision, from NGC 4993 SBF distance (Cantiello et al. 2018)
  em_estimate_dl = true_dl + np.random.normal(0, em_sigma_dl)

  em_dl_array = em_likelihood_dl(dl_array, em_estimate_dl, em_sigma_dl)
  uniform_weights_dl = np.ones_like(dl_array) / len(dl_array)

  from scipy.stats import entropy
  dl_entropy = entropy(em_dl_array, uniform_weights_dl)
  ess_dl = 1 / np.sum(em_dl_array**2)

  if verbose:
    print(f"[Host Galaxy] seed {randomseed}: KL={dl_entropy:.4f}, ESS={ess_dl:.1f}/{len(dl_array)}")

  return [dl_array, em_dl_array, dl_entropy, ess_dl]