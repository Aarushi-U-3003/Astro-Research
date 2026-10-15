import matplotlib.pyplot as plt
import numpy as np

cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4']
# Using the delta values from your Table 5
delta_means = [0.158, -1.226, -0.017, -1.501]
# Approximate std dev from your seed data (replace with actual std of deltas if you have them)
delta_stds = [0.2, 0.15, 0.1, 0.2] 

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(cases, delta_means, yerr=delta_stds, fmt='o', color='purple', capsize=5, markersize=8)
ax.axhline(0, color='black', linestyle='--', label='Additive Threshold ($\Delta = 0$)')

ax.set_ylabel('Paired Difference $\Delta$ (nats)')
ax.set_title('Joint Constraint Redundancy vs. Complementarity')
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.savefig('figure2_delta.png', dpi=300, bbox_inches='tight')
plt.show()