"""
This file conducts the postprocessing pipeline for the 4 cases of .json outputs from bilby_cbc_param_estimation.py.
By applying Bayesian re-weighting on the posteriors for each case, it constrains the parameter probabilities
even further. There will be 3 main postprocessing lines: sGRBs, host galaxy redshift, and both. 
"""
import bilby
import numpy as np 
from scipy.stats import entropy
import matplotlib.pyplot as plt

def em_likelihood_thetajn(thetajn_array, em_estimate_thetajn, em_sigma_thetajn):
  weights = np.exp(-0.5 * ((thetajn_array - em_estimate_thetajn) / em_sigma_thetajn)**2)
  weights/= np.sum(weights)
  return weights

def em_likelihood_dl(dl_array, em_estimate_dl, em_sigma_dl):
  weights = np.exp(-0.5 * ((dl_array - em_estimate_dl) / em_sigma_dl)**2)
  weights /= np.sum(weights)
  return weights 

def em_postprocessing(case_no, randomseed, true_dl_param, true_thetajn_param):
  np.random.seed(randomseed)

  # Load the result object using bilby -- can also be done with the json library
  gw_results = bilby.result.read_in_result(filename=f"outdir/{case_no}_result.json")

  # Extract luminosity_distance and theta_jn as a numpy arrays
  dl_array = gw_results.posterior["luminosity_distance"].values
  thetajn_array = gw_results.posterior["theta_jn"].values
  # The above will be the GW-only probability distribution. 

  # True injected values
  true_dl = true_dl_param # Mpc
  true_thetajn = true_thetajn_param  # rad

  # Decide how precise fake EM instrument is
  em_sigma_dl = 0.058 * true_dl   # Mpc -- GW170817 host galaxy precision was 2.36/40.7=5.8%
  em_sigma_thetajn = 0.09  # rad -- based on VLBI that constrained GW170817 to 
                          # 14-28 deg --> (28-14)/2 = 7 deg = 0.12 rad
                          # then 19-25 deg --> (25-19)/2 = 3 deg = 0.05 rad
                          # avg of 0.05 and 0.12 is approx. 0.09

  em_estimate_dl = true_dl + np.random.normal(0, em_sigma_dl)
  em_estimate_thetajn = true_thetajn + np.random.normal(0, em_sigma_thetajn)

  em_thetajn_array = em_likelihood_thetajn(thetajn_array=thetajn_array, em_estimate_thetajn=em_estimate_thetajn, em_sigma_thetajn=em_sigma_thetajn)
  em_dl_array = em_likelihood_dl(dl_array=dl_array, em_estimate_dl=em_estimate_dl, em_sigma_dl=em_sigma_dl)
  joint_weights = em_dl_array * em_thetajn_array
  joint_weights /= np.sum(joint_weights)

  # array of ones divided by length
  # used to find entropy later 
  uniform_weights_dl = np.ones_like(dl_array) / len(dl_array) 
  uniform_weights_thetajn = np.ones_like(thetajn_array) / len(thetajn_array)
  joint_uniform_weights = np.ones_like(joint_weights) / len(joint_weights)

  # when passing in two params to entropy(), the metric switches from Shannon entropy
  # to KL Divergence 
  dl_entropy = entropy(em_dl_array, uniform_weights_dl)
  thetajn_entropy = entropy(em_thetajn_array, uniform_weights_thetajn)
  joint_entropy = entropy (joint_weights, joint_uniform_weights)

  ess_dl = 1 / np.sum(em_dl_array**2)
  ess_thetajn = 1 / np.sum(em_thetajn_array**2)
  ess_joint = 1 / np.sum(joint_weights**2)


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
  print(f"Joint Weighting: {joint_entropy}\n")
  print(f"ESS luminosity distance: {ess_dl:.1f} / {len(dl_array)}")
  print(f"ESS inclination angle: {ess_thetajn:.1f} / {len(thetajn_array)}")
  print(f"ESS joint: {ess_joint:.1f} / {len(joint_weights)}")
  print("\n\n\n")

  return [dl_entropy, thetajn_entropy, joint_entropy]


em_three_vals = [] # stores dl info gain, thetajn info gain, and joint info gain for each iteration -- 2d array

for seed in range (10):
  em_three_vals.extend([em_postprocessing("Case1", seed, 150, 0.0)])

dl_mean_arr = []
thetajn_mean_arr = []
joint_mean_arr = []

for iter in range(len(em_three_vals)):
  dl_mean_arr.append(em_three_vals[iter][0])
  thetajn_mean_arr.append(em_three_vals[iter][1])
  joint_mean_arr.append(em_three_vals[iter][2])

dl_mean = np.mean(dl_mean_arr)
thetajn_mean = np.mean(thetajn_mean_arr)
joint_mean = np.mean(joint_mean_arr)

dl_std = np.std(dl_mean_arr)
thetajn_std = np.std(thetajn_mean_arr)
joint_std = np.std(joint_mean_arr)

dl_relspread = dl_std/dl_mean
thetajn_relspread = thetajn_std/thetajn_mean
joint_relspread = joint_std/joint_mean

print(f"Distance mean, std, relative spread: {dl_mean}, {dl_std}, {dl_relspread}")
print(f"Inclination mean, std, relative spread: {thetajn_mean}, {thetajn_std},{thetajn_relspread}")
print(f"Joint mean, std, relative spread: {joint_mean}, {joint_std},{joint_relspread}")
print("\n\n")