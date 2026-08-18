import bilby
import numpy as np
import scipy.stats as stats

# 1. Load the results file that Bilby saved
# (Double check your 'outdir' folder to make sure the filename matches exactly!)
result_path = "outdir/linear_regression_result.json"
result = bilby.core.result.read_in_result(filename=result_path)

# 2. Extract the raw posterior samples using NumPy
# This gives you an array of all the valid guesses the sampler made
m_samples = np.array(result.posterior["m"])
c_samples = np.array(result.posterior["c"])

print("--- RAW DATA STATS ---")
print(f"Total number of posterior samples collected: {len(m_samples)}")

# 3. Use NumPy to calculate the width (uncertainty) of the humps
m_median = np.median(m_samples)
m_std = np.std(m_samples) # Standard deviation measures the width/uncertainty

c_median = np.median(c_samples)
c_std = np.std(c_samples)

print(f"\nParameter m (Slope):")
print(f"  Calculated Median: {m_median:.4f}")
print(f"  Uncertainty (Std Dev): {m_std:.4f}")

print(f"\nParameter c (Y-Intercept):")
print(f"  Calculated Median: {c_median:.4f}")
print(f"  Uncertainty (Std Dev): {c_std:.4f}")

# 4. Use SciPy to calculate Information Gain (KL Divergence)
# For your upcoming paper, you'll need to show how much an error bar shrinks.
# Let's mock up a "Prior" (before observing data) vs "Posterior" (after observing data)
print("\n--- INFORMATION THEORY MATH ---")

# Let's approximate our data as a smooth Gaussian/Normal distribution using SciPy
posterior_distribution = stats.norm(loc=m_median, scale=m_std)

# Imagine before running the code, our uncertainty (prior) was much wider (scale=1.0)
prior_distribution = stats.norm(loc=m_median, scale=1.0)

# We generate a grid of points to evaluate the math across
x_axis = np.linspace(m_median - 3, m_median + 3, 1000)

# Get the probability densities
p_posterior = posterior_distribution.pdf(x_axis)
q_prior = prior_distribution.pdf(x_axis)

# Calculate Kullback-Leibler Divergence using SciPy
# This calculates exactly how many 'bits' of information were gained!
kl_divergence = stats.entropy(p_posterior, q_prior)
print(f"Information Gained (KL Divergence): {kl_divergence:.4f} nats")