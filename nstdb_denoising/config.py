CONFIG = {
    # ── Paths ──
    'data_dir':          './ECG_clean_noise_data/',
    'output_dir':        './nstdb_plots/',
    'model_dir':         './models/',

    # ── Signal parameters ──
    'fs':                360,           # sampling frequency (Hz)
    'n_samples':         10000,         # samples to extract per signal
    'signal_offset':     100000,        # skip noise-free preamble
    'channel':           0,             # first ECG lead

    # ── NSTDB structure ──
    'snr_levels':        [24, 18, 12, 6, 0, -6],
    'clean_snr':         24,            # ground-truth reference level
    'test_patient':      '16265',       # held-out test patient

    # ── Pre-processing ──
    'hp_cutoff':         0.67,          # HP filter cutoff (Hz) — removes EM BW
    'hp_order':          4,             # Butterworth HP filter order

    # ── Dictionary Learning ──
    'window_size':       64,            # patch size
    'n_atoms':           128,           # dictionary atoms
    'sparsity':          3,             # OMP non-zero coeffs (aggressive)
    'max_train_patches': 15000,         # max patches for training
    'n_extra_per_patient': 1000,        # patches sampled per training patient
    'batch_size':        256,           # MiniBatchDictionaryLearning batch

    # ── Post-processing ──
    'savgol_window':     11,            # Savitzky-Golay window
    'savgol_polyorder':  3,             # Savitzky-Golay polynomial order

    # ── Baselines ──
    'bp_low':            0.5,           # Butterworth BPF low cutoff
    'bp_high':           40.0,          # Butterworth BPF high cutoff
    'bp_order':          4,             # Butterworth BPF order
    'wavelet':           'db6',         # Wavelet family
    'wavelet_level':     6,             # Wavelet decomposition level
}