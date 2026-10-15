import bilby

result = bilby.result.read_in_result(filename=f"outdir/Case4_result.json")
# Assuming 'result' is your Bilby Result object
result.plot_corner(filename="Annotated_Plots4/bilby_plot1", parameters=["luminosity_distance", "theta_jn"]) 