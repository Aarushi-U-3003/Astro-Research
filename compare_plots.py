import bilby
import numpy as np

# 1. Load the two results you generated
try:
    result_A = bilby.core.result.read_in_result(filename="outdir/NS_CaseA_face_on_result.json")
    result_B = bilby.core.result.read_in_result(filename="outdir/NS_CaseB_edge_on_result.json")
except Exception as e:
    print("Could not find the files. Check your outdir folder filenames!")
    print(f"Error: {e}")
    exit()

print("==================================================")
print("   DISTANCE-INCLINATION DEGENERACY REPORT         ")
print("==================================================")

# 2. Extract and print data for Case A
dist_A = np.array(result_A.posterior["luminosity_distance"])
theta_A = np.array(result_A.posterior["theta_jn"])

print("\n[CASE A: Injected as 150 Mpc, Face-On (0.0 rad)]")
print(f"  Calculated Distance Median: {np.median(dist_A):.2f} Mpc")
print(f"  Distance Uncertainty (Std Dev): {np.std(dist_A):.2f} Mpc")
print(f"  Calculated Inclination Median: {np.median(theta_A):.2f} rad")

# 3. Extract and print data for Case B
dist_B = np.array(result_B.posterior["luminosity_distance"])
theta_B = np.array(result_B.posterior["theta_jn"])

print("\n[CASE B: Injected as 75 Mpc, Tilted (1.0 rad)]")
print(f"  Calculated Distance Median: {np.median(dist_B):.2f} Mpc")
print(f"  Distance Uncertainty (Std Dev): {np.std(dist_B):.2f} Mpc")
print(f"  Calculated Inclination Median: {np.median(theta_B):.2f} rad")

print("\n==================================================")
# 4. Check for Overlap
# Let's see if the minimum guess of Case A is lower than the maximum guess of Case B
if np.min(dist_A) < np.max(dist_B):
    print("DEGENERACY CONFIRMED: The distance estimations overlap!")
    print("LIGO alone cannot completely separate these two events.")
else:
    print("The distributions are fully separated.")
print("==================================================")