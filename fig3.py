import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import bilby
from em_sgrb_postprocessing import em_sgrb_postprocessing
from em_host_galaxy_postprocessing import em_host_galaxy_postprocessing

cases = ["Case1", "Case2", "Case3", "Case4"]
true_dls = [150.0, 75.0, 100.0, 120.0]
true_thetas = [0.0, 1.0, 0.5, 1.57]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for i, case in enumerate(cases):
    # Load raw GW samples
    res = bilby.result.read_in_result(f"outdir/{case}_result.json")
    dl_gw = res.posterior["luminosity_distance"].values
    raw_theta = res.posterior["theta_jn"].values
    theta_gw = np.minimum(raw_theta, np.pi - raw_theta)
    
    # Get weights for seed 0 (or average across seeds)
    _, _, w_s, *_ = em_sgrb_postprocessing(case, 0, true_thetas[i])
    _, w_h, *_ = em_host_galaxy_postprocessing(case, 0, true_dls[i])
    
    # Top Row: d_L Leakage (GW-only vs sGRB-reweighted)
    sns.kdeplot(dl_gw, ax=axes[0, i], color="blue", fill=True, alpha=0.3, label="GW-only")
    sns.kdeplot(x=dl_gw, weights=w_s, ax=axes[0, i], color="darkorange", fill=True, alpha=0.5, label="sGRB-only")
    axes[0, i].axvline(true_dls[i], color="black", linestyle="--", label="True Value")
    axes[0, i].set_title(f"{case}: $d_L$ Leakage")
    axes[0, i].set_xlabel("Luminosity Distance $d_L$ (Mpc)")
    
    # Bottom Row: Theta Leakage (GW-only vs HG-reweighted)
    sns.kdeplot(theta_gw, ax=axes[1, i], color="blue", fill=True, alpha=0.3, label="GW-only")
    sns.kdeplot(x=theta_gw, weights=w_h, ax=axes[1, i], color="green", fill=True, alpha=0.5, label="HG-only")
    true_folded_theta = min(true_thetas[i], np.pi - true_thetas[i])
    axes[1, i].axvline(true_folded_theta, color="black", linestyle="--", label="True Value")
    axes[1, i].set_title(f"{case}: $\Theta$ Leakage")
    axes[1, i].set_xlabel("Folded Viewing Angle $\Theta$ (rad)")

axes[0, 0].legend()
axes[1, 0].legend()
plt.tight_layout()
plt.savefig("figure3_leakage.png", dpi=300)
plt.show()