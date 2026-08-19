"""
This file conducts the postprocessing pipeline for the 4 cases of .json outputs from bilby_cbc_param_estimation.py.
By applying Bayesian re-weighting on the posteriors for each case, it constrains the parameter probabilities
even further. There will be 3 main postprocessing lines: sGRBs, host galaxy redshift, and both. 
"""
import bilby
import numpy as np 
from scipy.stats import entropy
import matplotlib.pyplot as plt


np.random.seed(42)

# Load the result object using bilby -- can also be done with the json library
gw_results = bilby.result.read_in_result(filename="outdir/Case1_result.json")

# Extract luminosity_distance and theta_jn as a numpy arrays
dl_array = gw_results.posterior["luminosity_distance"].values
thetajn_array = gw_results.posterior["theta_jn"].values
# The above will be the GW-only probability distribution. 

# True injected values
true_dl = 150      # Mpc
true_thetajn = 0.3  # rad

# Decide how precise fake EM instrument is
em_sigma_dl = 8        # Mpc -- placeholder, refine later
em_sigma_thetajn = 0.15  # rad -- placeholder, refine later

em_estimate_dl = true_dl + np.random.normal(0, em_sigma_dl)
em_estimate_thetajn = true_thetajn + np.random.normal(0, em_sigma_thetajn)

def em_likelihood_thetajn(thetajn_array, em_estimate_thetajn, em_sigma_thetajn):
  weights = np.exp(-0.5 * ((thetajn_array - em_estimate_thetajn) / em_sigma_thetajn)**2)
  weights/= np.sum(weights)
  return weights

def em_likelihood_dl(dl_array, em_estimate_dl, em_sigma_dl):
  weights = np.exp(-0.5 * ((dl_array - em_estimate_dl) / em_sigma_dl)**2)
  weights /= np.sum(weights)
  return weights 

em_thetajn_array = em_likelihood_thetajn(thetajn_array=thetajn_array, em_estimate_thetajn=em_estimate_thetajn, em_sigma_thetajn=em_sigma_thetajn)
em_dl_array = em_likelihood_dl(dl_array=dl_array, em_estimate_dl=em_estimate_dl, em_sigma_dl=em_sigma_dl)

uniform_weights_dl = np.ones_like(dl_array) / len(dl_array)
uniform_weights_thetajn = np.ones_like(thetajn_array) / len(thetajn_array)

dl_entropy = entropy(em_dl_array, uniform_weights_dl)
thetajn_entropy = entropy(em_thetajn_array, uniform_weights_thetajn)

ess_dl = 1 / np.sum(em_dl_array**2)
ess_thetajn = 1 / np.sum(em_thetajn_array**2)


"""# matplotlib to make comparison histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distance panel
axes[0].hist(dl_array, bins=50, alpha=0.5, density=True, label="GW-only")
axes[0].hist(dl_array, bins=50, weights=em_dl_array, alpha=0.5, density=True, label="GW + EM")
axes[0].axvline(true_dl, color="black", linestyle="--", label="true value")
axes[0].set_xlabel("Luminosity Distance (Mpc)")
axes[0].legend()

# Inclination panel
axes[1].hist(thetajn_array, bins=50, alpha=0.5, density=True, label="GW-only")
axes[1].hist(thetajn_array, bins=50, weights=em_thetajn_array, alpha=0.5, density=True, label="GW + EM")
axes[1].axvline(true_thetajn, color="black", linestyle="--", label="true value")
axes[1].set_xlabel("theta_jn (rad)")
axes[1].legend()

plt.tight_layout()
plt.savefig("outdir/reweighting_comparison.png")
plt.show()
"""


print("\n\n\n")
print("------------- KL Divergence -------------")
print(f"Luminosity Distance: {dl_entropy}")
print(f"Inclination Angle: {thetajn_entropy}")
print(f"ESS luminosity distance: {ess_dl:.1f} / {len(dl_array)}")
print(f"ESS inclination angle: {ess_thetajn:.1f} / {len(thetajn_array)}")
print("\n\n\n")