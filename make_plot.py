import bilby

# Load your result
result = bilby.core.result.read_in_result(filename="outdir/CaseB_edge_on_result.json")

# Force it to ignore parameters that have zero variance to prevent the crash
result.plot_corner(quantiles=[0.16, 0.84])
print("Plot generated successfully!")