import numpy as np, bilby
from em_sgrb_postprocessing import em_sgrb_postprocessing
from em_host_galaxy_postprocessing import em_host_galaxy_postprocessing
from em_joint_postprocessing import compute_joint

CASE, TRUE_DL, TRUE_THETA, NSEED = "Case4", 120.0, 1.57, 10

res    = bilby.result.read_in_result(f"outdir/{CASE}_result.json")
dl0    = res.posterior["luminosity_distance"].values
theta0 = res.posterior["theta_jn"].values

seed_diff = 0

for seed in range(NSEED):
    theta, th_fold, w_s, th_entropy, *_ = em_sgrb_postprocessing(CASE, seed, TRUE_THETA)
    dl, w_h, dl_entropy, *_ = em_host_galaxy_postprocessing(CASE, seed, TRUE_DL)
    joint, joint_entropy, *_ = compute_joint(w_h, w_s)
    seed_diff += joint_entropy - th_entropy - dl_entropy

seed_diff /= NSEED 

print(f"AVG PAIRED SEED DIFFERENCES {CASE}: {seed_diff}")