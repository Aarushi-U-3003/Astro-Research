#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a full 15 parameter
space for an injected cbc signal. This is the standard injection analysis script
one can modify for the study of injected CBC events.

This will take many hours to run.
"""
import bilby
import logging
import numpy as np
import warnings
import argparse
from bilby.core.utils import random

bilby.core.utils.logger.setLevel(logging.ERROR)
if __name__ == '__main__':
    # Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
    random.seed(123)

    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    duration = 4.0
    sampling_frequency = 1024.0

    parser = argparse.ArgumentParser()
    parser.add_argument("--dl", type=float, required=True, help="Injected Luminosity Distance")
    parser.add_argument("--theta", type=float, required=True, help="Injected Theta JN")
    parser.add_argument("--npool", type=int, default=1, help="Dynesty pool workers per case")
    args = parser.parse_args()

    true_dl = args.dl
    true_theta = args.theta

    # Specify the output directory and the name of the simulation.
    outdir = f"Phase2/outdir_dl_{int(true_dl)}_th_{true_theta:.2f}"
    case_no = f"dl{int(true_dl)}_th{true_theta:.2f}"
    label = case_no
    bilby.core.utils.setup_logger(outdir=outdir, label=label)
    bilby.core.utils.logger.setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", message="Cannot convert from zenith/azimuth to ra/dec falling back to provided ra/dec")
    warnings.filterwarnings("ignore", category=UserWarning)


    # We are going to inject a binary black hole waveform.  We first establish a
    # dictionary of parameters that includes all of the different waveform
    # parameters, including masses of the two black holes (mass_1, mass_2),
    # spins of both black holes (a, tilt, phi), etc.
    injection_parameters = dict(
        mass_1=1.4,
        mass_2=1.3,
        a_1=0.05,
        a_2=0.03,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=true_dl,
        theta_jn=true_theta,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
        lambda_1=300.0,
        lambda_2=550.0
    )
    res = bilby.gw.conversion.generate_tidal_parameters(injection_parameters)
    injection_parameters = res[0] if isinstance(res, tuple) else res

    # Fixed arguments passed into the source model
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP_NRTidalv2", 
        # Phase 2 now handles tidal deformabilities
        reference_frequency=100.0,
        minimum_frequency=100.0,
    )

    # Create the waveform_generator using a LAL BinaryNeutronStar source function
    # the generator will convert all the parameters
    def make_waveform_generator(source_model):
        # fresh dict per generator, so the likelihood's "fiducial" flag
        # can't leak into the injection generator
        wf_args = dict(
            waveform_approximant="IMRPhenomXP_NRTidalv2",
            reference_frequency=100.0,
            minimum_frequency=100.0,
        )
        return bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=source_model,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=wf_args,
        )

    injection_wfg = make_waveform_generator(bilby.gw.source.lal_binary_neutron_star)
    #likelihood_wfg = make_waveform_generator(bilby.gw.source.lal_binary_neutron_star_relative_binning) 


    # Set up interferometers.  In this case we'll use two interferometers
    # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
    # sensitivity
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    for ifo in ifos:
        ifo.minimum_frequency = 100.0
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )

    ifos.inject_signal(
        waveform_generator=injection_wfg, parameters=injection_parameters
    )

    time_delay = ifos[0].time_delay_from_geocenter(
        injection_parameters["ra"],
        injection_parameters["dec"],
        injection_parameters["geocent_time"],
    )

    injection_parameters["H1_time"] = injection_parameters["geocent_time"] + time_delay
    fiducial_params = injection_parameters.copy()

    # For this analysis, we implement the standard BBH priors defined, except for
    # the definition of the time prior, which is defined as uniform about the
    # injected value.
    # We change the mass boundaries to be more targeted for the source we
    # injected.
    # We define a prior in the time at the Hanford interferometer
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
    # Between 0 and pi because that's the range of inverse cosine
    # Need to explain how cosine of inclination angle results in uniform probability
    # across plane, avoiding "squeezing" and overfitting near the poles
    # Plus, we're not restricting to (0, pi/2) because the bimodality at the poles is
    # required so that the degeneracy exists to break w/ EM data later
    # Degeneracy is a dominant-mode effect, and while higher modes in 
    # IMRPhenomXPHM slightly break it, it's not enough to be significant. 

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveoform generator, as well the priors.
    # The explicit distance marginalization is turned on to improve
    # convergence, and the posterior is recovered by the conversion function.
    '''likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=likelihood_wfg,
        priors=priors,
        distance_marginalization=False, # perhaps can be turned to True
        phase_marginalization=False, # only works well with dominant mode (2, 2) -- unsure for IMRPhenomXPHM
        time_marginalization=False,
        reference_frame="sky",
        time_reference="H1",
        fiducial_parameters=fiducial_params
    )'''

    # Run sampler. In this case we're going to use the `dynesty` sampler
    # Note that the `nlive`, `naccept`, and `sample` parameters are specified
    # to ensure sufficient convergence of the analysis.
    # We set `npool=10` to parallelize the analysis over 10 cores -- 6 performance and 4 efficiency.
    # The conversion function will determine the distance posterior in post processing

    #for key, p in priors.items():
        #print(key, type(p).__name__)
    #print("sampled:", sorted(priors.non_fixed_keys))
    '''likelihood.parameters.update(injection_parameters)
    print(likelihood.log_likelihood_ratio())
    for ifo in ifos:
        print(ifo.name, ifo.meta_data['optimal_SNR'] if 'optimal_SNR' in ifo.meta_data else ifo.optimal_snr_squared(injection_parameters)**0.5)
    ''' 

    full_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=injection_wfg,
        priors=priors,
        distance_marginalization=True,
        phase_marginalization=False,
        time_marginalization=False,
        reference_frame="sky",
        time_reference="H1",
    )

    '''for theta_jn_test in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]:
        pt = injection_parameters.copy()
        pt["theta_jn"] = theta_jn_test
        likelihood.parameters.update(pt)
        print(theta_jn_test, likelihood.log_likelihood_ratio())'''
        
    result = bilby.run_sampler(
        likelihood=full_likelihood,
        priors=priors,
        #clean=True,
        sampler='dynesty',
        sample='acceptance-walk',
        naccept=60,
        nlive=500,                  
        check_point_delta_t=900,
        resume=True,
        npool=args.npool,          # One worker per vCPU; set with --npool

        check_point_plot=False,    # Turn off the checkpoint plots that are crashing
        label=label,
        outdir=f"Phase2/outdir_dl_{int(true_dl)}_th_{true_theta:.2f}"
    )
    
    result.plot_corner()