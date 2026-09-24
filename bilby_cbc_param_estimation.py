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
from bilby.core.utils import random

bilby.core.utils.logger.setLevel(logging.ERROR)
if __name__ == '__main__':
    # Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
    random.seed(123)

    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    duration = 4.0
    sampling_frequency = 1024.0

    # Specify the output directory and the name of the simulation.
    outdir = "outdir"
    label = "Case4"
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
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=120.0,
        theta_jn=1.57,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )

    # Fixed arguments passed into the source model
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXPHM",
        reference_frequency=100.0,
        minimum_frequency=100.0,
    )

    # Create the waveform_generator using a LAL BinaryNeutronStar source function
    # the generator will convert all the parameters
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments,
    )

    # Set up interferometers.  In this case we'll use two interferometers
    # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
    # sensitivity
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency = 100.0
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )

    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    time_delay = ifos[0].time_delay_from_geocenter(
        injection_parameters["ra"],
        injection_parameters["dec"],
        injection_parameters["geocent_time"],
    )

    injection_parameters["H1_time"] = injection_parameters["geocent_time"] + time_delay

    # For this analysis, we implement the standard BBH priors defined, except for
    # the definition of the time prior, which is defined as uniform about the
    # injected value.
    # We change the mass boundaries to be more targeted for the source we
    # injected.
    # We define a prior in the time at the Hanford interferometer
    priors = bilby.gw.prior.BNSPriorDict()

    priors.minimum_frequency = 100.0

    priors["H1_time"] = bilby.core.prior.Uniform(
        minimum=injection_parameters["H1_time"] - 0.1,
        maximum=injection_parameters["H1_time"] + 0.1,
        name="H1_time",
        latex_label="$t_H$",
        unit="$s$",
    )
    priors['lambda_1'] = 0.0
    priors['lambda_2'] = 0.0   

    # Force the sampler to look in a realistic neighborhood instead of the whole universe
    # Root distance prior mins and maxs in literature -- can influence results
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
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        distance_marginalization=False,
        phase_marginalization=False,
        time_marginalization=False,
        reference_frame="H1L1",
        time_reference="H1",
    )

    # Run sampler. In this case we're going to use the `dynesty` sampler
    # Note that the `nlive`, `naccept`, and `sample` parameters are specified
    # to ensure sufficient convergence of the analysis.
    # We set `npool=10` to parallelize the analysis over 10 cores -- 6 performance and 4 efficiency.
    # The conversion function will determine the distance posterior in post processing

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        clean=True,
        sampler='dynesty',
        sample='rwalk',
        nlive=1000,                  # Keep this tiny -- make larger later for formal runs
        npool=10,                   # Decided to use all 10 cores to parallelize evaluations
        walks=10,                 # Shorter chains per point
        check_point_plot=False,    # Turn off the checkpoint plots that are crashing
        label=label,
        outdir='outdir'
    )
    
    result.plot_corner()