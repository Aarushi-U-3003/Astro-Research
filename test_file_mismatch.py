import bilby
import numpy as np

gw_results = bilby.read_in_result("Phase2/outdir_dl_50_th_0.00/dl50_th0.00_result.json")

injection_parameters = dict(
    mass_1=1.4,
    mass_2=1.3,
    a_1=0.05,
    a_2=0.03,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=50,
    theta_jn=0.0,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    lambda_1=300.0,
    lambda_2=550.0
)

def make_waveform_generator(source_model):
    # fresh dict per generator, so the likelihood's "fiducial" flag
    # can't leak into the injection generator
    wf_args = dict(
        waveform_approximant="IMRPhenomXP_NRTidalv2",
        reference_frequency=100.0,
        minimum_frequency=100.0,
    )
    return bilby.gw.WaveformGenerator(
        duration=4.0,
        sampling_frequency=1024,
        frequency_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=wf_args,
    )

injection_wfg = make_waveform_generator(bilby.gw.source.lal_binary_neutron_star)
likelihood_wfg = make_waveform_generator(bilby.gw.source.lal_binary_neutron_star_relative_binning) 

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for ifo in ifos:
    ifo.minimum_frequency = 100.0
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=1024,
    duration=4.0,
    start_time=injection_parameters["geocent_time"] - 2,
)


time_delay = ifos[0].time_delay_from_geocenter(
    injection_parameters["ra"],
    injection_parameters["dec"],
    injection_parameters["geocent_time"],
)

injection_parameters["H1_time"] = injection_parameters["geocent_time"] + time_delay

priors = bilby.gw.prior.BNSPriorDict()
for key in ["geocent_time", "lambda_1", "lambda_2"]:
    priors.pop(key, None)

priors.minimum_frequency = 100.0

priors["H1_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["H1_time"] - 0.1,
    maximum=injection_parameters["H1_time"] + 0.1,
    name="H1_time",
    latex_label="$t_H$",
    unit="$s$",
)

priors["lambda_1"] = 300.0
priors["lambda_2"] = 550.0

for key in ["chi_1", "chi_2"]:
    priors.pop(key, None)

priors["a_1"] = bilby.core.prior.Uniform(0, 0.3, name="a_1", latex_label="$a_1$")
priors["a_2"] = bilby.core.prior.Uniform(0, 0.3, name="a_2", latex_label="$a_2$")
priors["tilt_1"] = bilby.core.prior.Sine(name="tilt_1", latex_label="$\\theta_1$")
priors["tilt_2"] = bilby.core.prior.Sine(name="tilt_2", latex_label="$\\theta_2$")
priors["phi_12"] = bilby.core.prior.Uniform(0, 2*np.pi, name="phi_12", boundary="periodic", latex_label="$\\phi_{12}$")
priors["phi_jl"] = bilby.core.prior.Uniform(0, 2*np.pi, name="phi_jl", boundary="periodic", latex_label="$\\phi_{JL}$")

# Force the sampler to look in a realistic neighborhood instead of the whole universe
# Root distance prior mins and maxes in literature -- can influence results
priors['luminosity_distance'] = bilby.core.prior.PowerLaw(alpha=2, minimum=10, maximum=500, name='luminosity_distance', latex_label='$d_L$')
# Check Baral paper for literature support -- standard range for low-level unbiased NS-NS detection 
priors['theta_jn'] = bilby.core.prior.Sine(minimum=0, maximum=np.pi, name='theta_jn', latex_label='$\\theta_{JN}$') 


fiducial_params = injection_parameters.copy()

likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=likelihood_wfg,
    priors=priors,
    distance_marginalization=False, # perhaps can be turned to True
    phase_marginalization=False, # only works well with dominant mode (2, 2) -- unsure for IMRPhenomXPHM
    time_marginalization=False,
    reference_frame="sky",
    time_reference="H1",
    fiducial_parameters=fiducial_params
)

full_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=injection_wfg,
    priors=priors,
    distance_marginalization=False,
    phase_marginalization=False,
    time_marginalization=False,
    reference_frame="sky",
    time_reference="H1",
)

# load the actual saved result from the (50, 0.0) run
best_idx = gw_results.posterior["log_likelihood"].idxmax()
print(gw_results.posterior.loc[best_idx, ["theta_jn", "log_likelihood"]])
print(gw_results.posterior["log_likelihood"].describe())

real_best_idx = gw_results.posterior["log_likelihood"].idxmax()
best_point = gw_results.posterior.loc[real_best_idx].to_dict()

# pull out just the sampled parameters the likelihood needs
sampled_keys = [k for k in injection_parameters if k in best_point] + ["theta_jn", "luminosity_distance", "psi", "phase", "H1_time", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"]
pt_best = {k: best_point[k] for k in set(sampled_keys) if k in best_point}
# fill in anything missing (masses, sky location, lambdas) from injection_parameters
full_pt = {**injection_parameters, **pt_best}

likelihood.parameters.update(full_pt)
full_likelihood.parameters.update(full_pt)
print("sampler's best point -- relative-binned:", likelihood.log_likelihood_ratio(),
      "full:", full_likelihood.log_likelihood_ratio())
print("theta_jn at this point:", full_pt["theta_jn"])
print("luminosity_distance at this point:", full_pt.get("luminosity_distance"))
print("psi at this point:", full_pt.get("psi"))