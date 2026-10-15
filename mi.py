import numpy as np
import bilby
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score

gw_results = bilby.result.read_in_result(filename=f"outdir/Case4_result.json")
# Reshape because scikit-learn expects 2D array for features
inclination_array = gw_results.posterior["theta_jn"].values
inclination_2d = inclination_array.reshape(-1, 1)

distance_array = gw_results.posterior["luminosity_distance"].values

# Calculate mutual information
mi_score = mutual_info_regression(inclination_2d, distance_array)[0]
print(f"Mutual Information: {mi_score:.3f} nats")

distance_binned = np.digitize(distance_array, np.linspace(distance_array.min(), distance_array.max(), 50))
inclination_binned = np.digitize(inclination_array, np.linspace(inclination_array.min(), inclination_array.max(), 50))

nmi = normalized_mutual_info_score(distance_binned, inclination_binned, average_method='geometric')
print(f"Normalized Mutual Information: {nmi:.3f}")