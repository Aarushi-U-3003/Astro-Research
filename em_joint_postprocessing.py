import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import bilby
from em_sgrb_postprocessing import em_sgrb_postprocessing
from em_host_galaxy_postprocessing import em_host_galaxy_postprocessing
from sklearn.feature_selection import mutual_info_regression


def compute_joint(em_dl_array, em_thetajn_array):
  joint_weights = em_dl_array * em_thetajn_array
  joint_weights /= np.sum(joint_weights)
  joint_uniform_weights = np.ones_like(joint_weights) / len(joint_weights)
  joint_entropy = entropy(joint_weights, joint_uniform_weights)
  ess_joint = 1 / np.sum(joint_weights**2)
  return joint_weights, joint_entropy, ess_joint


def plot_averaged_results(dl_array, thetajn_array, avg_em_dl, avg_em_thetajn, avg_joint, true_dl, true_thetajn, case_no):
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  axes[0].hist(thetajn_array, bins=50, alpha=0.5, density=True, label="GW-only")
  axes[0].hist(thetajn_array, bins=50, weights=avg_em_thetajn, alpha=0.5, density=True, label="GW + sGRB")
  axes[0].axvline(true_thetajn, color="black", linestyle="--", label="true value")
  axes[0].set_xlabel("theta_jn (rad)")
  axes[0].legend()

  axes[1].hist(dl_array, bins=50, alpha=0.5, density=True, label="GW-only")
  axes[1].hist(dl_array, bins=50, weights=avg_em_dl, alpha=0.5, density=True, label="GW + Host Galaxy")
  axes[1].axvline(true_dl, color="black", linestyle="--", label="true value")
  axes[1].set_xlabel("d_L (Mpc)")
  axes[1].legend()

  # Joint is inherently 2D (dl AND theta_jn constrained together) -- a 1D histogram
  # of the weight values themselves isn't meaningful, so this shows the joint-weighted
  # density directly in the (theta_jn, dl) plane instead.
  hb = axes[2].hexbin(thetajn_array, dl_array, C=avg_joint, gridsize=30, cmap="viridis", reduce_C_function=np.sum)
  axes[2].plot(true_thetajn, true_dl, marker="*", color="red", markersize=15, label="true value")
  axes[2].set_xlabel("theta_jn (rad)")
  axes[2].set_ylabel("d_L (Mpc)")
  axes[2].legend()
  fig.colorbar(hb, ax=axes[2], label="joint weight density")

  plt.tight_layout()
  plt.savefig(f"outdir/{case_no}_reweighting_comparison.png")
  plt.show()


def run_pipeline(case_no, true_dl, true_thetajn, n_seeds=10, ess_threshold=30):
  dl_entropies, thetajn_entropies, joint_entropies = [], [], []
  ess_dl_list, ess_thetajn_list, ess_joint_list = [], [], []
  local_densities = []
  em_dl_arrays, em_thetajn_arrays, joint_arrays = [], [], []

  gw_results = bilby.result.read_in_result(filename=f"outdir/{case_no}_result.json")
  num_samples = len(gw_results.posterior)

  dl_array = thetajn_array = None  # same across seeds -- captured once below

  for seed in range(n_seeds):
    dl_array, em_dl_array, dl_entropy, ess_dl = em_host_galaxy_postprocessing(case_no, seed, true_dl)
    raw_thetajn_array, thetajn_array, em_thetajn_array, thetajn_entropy, ess_thetajn, local_density = em_sgrb_postprocessing(case_no, seed, true_thetajn)
    joint_weights, joint_entropy, ess_joint = compute_joint(em_dl_array, em_thetajn_array)

    dl_entropies.append(dl_entropy)
    thetajn_entropies.append(thetajn_entropy)
    joint_entropies.append(joint_entropy)
    ess_dl_list.append(ess_dl)
    ess_thetajn_list.append(ess_thetajn)
    ess_joint_list.append(ess_joint)
    local_densities.append(local_density)

    em_dl_arrays.append(em_dl_array)
    em_thetajn_arrays.append(em_thetajn_array)
    joint_arrays.append(joint_weights)

  def summarize(name, values, ess_values):
    values = np.array(values)
    ess_values = np.array(ess_values)
    invalid = np.sum(ess_values < ess_threshold)
    print(f"{name}: KL mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
          f"rel_spread={np.std(values)/np.mean(values):.4f} | "
          f"ESS mean={np.mean(ess_values):.1f}/{num_samples}, invalid runs (ESS<{ess_threshold}): {invalid}/{n_seeds}")

  mi_score = mutual_info_regression(em_dl_array, em_thetajn_array)[0]
  
  print(f"Mutual Information: {mi_score:.3f} nats")
  print(f"\n===== {case_no} summary (true_dl={true_dl}, true_thetajn={true_thetajn}, n_seeds={n_seeds}) =====")
  summarize("Distance", dl_entropies, ess_dl_list)
  summarize("Inclination", thetajn_entropies, ess_thetajn_list)
  summarize("Joint", joint_entropies, ess_joint_list)
  print(f"Mean local density (sGRB, within 2*sigma of truth): {np.mean(local_densities):.4f}")

  print("=" * 60 + "\n")

  avg_em_dl = np.mean(em_dl_arrays, axis=0)
  avg_em_dl /= np.sum(avg_em_dl)
  avg_em_thetajn = np.mean(em_thetajn_arrays, axis=0)
  avg_em_thetajn /= np.sum(avg_em_thetajn)
  avg_joint = np.mean(joint_arrays, axis=0)
  avg_joint /= np.sum(avg_joint)

  plot_averaged_results(dl_array, thetajn_array, avg_em_dl, avg_em_thetajn, avg_joint, true_dl, true_thetajn, case_no)

  return {
    "dl_entropies": dl_entropies, "thetajn_entropies": thetajn_entropies, "joint_entropies": joint_entropies,
    "ess_dl": ess_dl_list, "ess_thetajn": ess_thetajn_list, "ess_joint": ess_joint_list,
  }


if __name__ == "__main__":
  run_pipeline("Case4", true_dl=120, true_thetajn=1.57)