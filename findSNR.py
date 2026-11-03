import bilby
import numpy as np

injection_parameters = dict(
    mass_1=1.4, mass_2=1.3, a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=150.0, theta_jn=0.0,
    psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
)

duration = 4.0
sampling_frequency = 1024.0
fmin = 100.0

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                            reference_frequency=100.0, minimum_frequency=fmin),
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
for ifo in ifos:
    ifo.minimum_frequency = fmin
    ifo.maximum_frequency = sampling_frequency / 2

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)

ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

snrs = {}
for ifo in ifos:
    # Optimal SNR: the injected signal's matched-filter SNR against the PSD (no noise)
    snrs[ifo.name] = ifo.meta_data["optimal_SNR"]
    print(f"{ifo.name} optimal SNR: {snrs[ifo.name]:.2f}")

network_snr = np.sqrt(sum(s**2 for s in snrs.values()))
print(f"--> Network optimal SNR: {network_snr:.2f}")