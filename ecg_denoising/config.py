MITBIH_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124,
    200, 201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
    228, 230, 231, 232, 233, 234
]

CONFIG = {
    # Data paths
    'data_dir':           './ECG_data/',
    'output_dir':         './plots/',
    'model_dir':          './models/',        # where trained dictionary is saved

    # ECG signal parameters
    'channel':            'MLII',
    'fs':                 360,               # sampling frequency (Hz)
    'gain':               200.0,             # ADC units per millivolt
    'baseline_adc':       1024.0,            # ADC value at 0 mV (11-bit midpoint)
    'n_samples_per_file': 10000,
    'test_record':        100,
    'clip_mv':            5.0,               # clip signal beyond ±5 mV
    'min_std_threshold':  0.005,             # reject flat/corrupt signals below this std

    # Dictionary Learning parameters
    'window_size':        64,                # patch size for dictionary learning
    'stride':             32,                # stride for standard OLA (display only)
    'n_atoms':            128,               # number of dictionary atoms
    'sparsity':           10,                # OMP non-zero coefficients
    'max_train_windows':  15000,             # cap on training patches
    'n_extra_per_record': 1000,              # patches sampled per training record
    'n_train_records':    10,                # how many training records to use

    # Noise synthesis
    'noise_snr_db':       10,                # target noise SNR (dB)

    # Post-processing
    'savgol_window':      11,
    'savgol_polyorder':   3,

    # Baseline wander removal
    'hp_cutoff':          0.5,               # high-pass cutoff (Hz)
    'hp_order':           4,

    # Butterworth bandpass baseline
    'bp_low':             0.5,
    'bp_high':            40.0,
    'bp_order':           4,

    # Moving average baseline
    'ma_window':          15,

    # Wavelet baseline
    'wavelet':            'db6',
    'wavelet_level':      6,
}
