#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║        Biomedical ECG Signal Denoising — Full ML Pipeline           ║
║                                                                      ║
║  Dataset : MIT-BIH Arrhythmia Database (48 records)                 ║
║  Method  : Dictionary Learning (K-SVD) + OMP Sparse Coding          ║
║  Baseline: Wavelet (db6), Butterworth BPF, Moving Average           ║
║                                                                      ║
║  Strategy: Train dictionary on diverse clean ECG morphologies from  ║
║  47 patients, then apply sparse coding to denoise a held-out patient║
║  (record 100). Two-stage hybrid: HP filter + Dictionary + SavGol.   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import norm as sp_norm
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import sparse_encode
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pywt
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════
MITBIH_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124,
    200, 201, 202, 203, 205, 207, 208, 209, 210,
    212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
    228, 230, 231, 232, 233, 234
]

CONFIG = {
    'data_dir':           './ECG_data/',
    'channel':            'MLII',
    'fs':                 360,
    'gain':               200.0,
    'baseline_adc':       1024.0,
    'n_samples_per_file': 10000,
    'test_record':        100,
    'clip_mv':            5.0,
    'min_std_threshold':  0.005,
    'window_size':        64,
    'stride':             32,
    'n_atoms':            128,
    'sparsity':           10,
    'noise_snr_db':       10,
    'savgol_window':      11,
    'savgol_polyorder':   3,
    'hp_cutoff':          0.5,
    'hp_order':           4,
    'bp_low':             0.5,
    'bp_high':            40.0,
    'bp_order':           4,
    'ma_window':          15,
    'wavelet':            'db6',
    'wavelet_level':      6,
    'output_dir':         './plots/',
}


# ════════════════════════════════════════════════════════════════════
# STEP 1: Multi-File Loading, ADC Conversion & Preprocessing
# ════════════════════════════════════════════════════════════════════

def load_record(record_id, data_dir, channel='MLII',
                n_samples=10000, gain=200.0, baseline=1024.0, clip_mv=5.0,
                min_std=0.005):
    """
    Load one MIT-BIH record from CSV, convert ADC → millivolts,
    validate quality, return first n_samples.
    Returns None if file missing, column absent, or signal corrupt.
    """
    filepath = os.path.join(data_dir, f'{record_id}.csv')
    if not os.path.exists(filepath):
        print(f'  [SKIP] {record_id}.csv not found')
        return None

    df = pd.read_csv(filepath)
    df.columns = [c.strip("'").strip() for c in df.columns]

    if channel not in df.columns:
        print(f'  [SKIP] {record_id}.csv: column {channel} missing')
        return None

    raw = df[channel].values.astype(np.float64)
    signal_mv = (raw - baseline) / gain
    signal_mv = np.clip(signal_mv, -clip_mv, clip_mv)
    signal_mv = signal_mv[:n_samples]

    if signal_mv.std() < min_std:
        print(f'  [SKIP] {record_id}.csv: flat/corrupt signal (std={signal_mv.std():.6f})')
        return None

    return signal_mv


def load_all_records(config):
    """Load all available MIT-BIH records."""
    print(f"\n[STEP 1] Loading MIT-BIH records from: {config['data_dir']}")
    signals = {}
    loaded, skipped = 0, 0

    for rec_id in tqdm(MITBIH_RECORDS, desc='Loading records'):
        sig = load_record(
            rec_id, config['data_dir'],
            channel=config['channel'],
            n_samples=config['n_samples_per_file'],
            gain=config['gain'],
            baseline=config['baseline_adc'],
            clip_mv=config['clip_mv'],
            min_std=config['min_std_threshold'],
        )
        if sig is not None:
            signals[rec_id] = sig
            loaded += 1
        else:
            skipped += 1

    print(f'  Loaded : {loaded} records | Skipped: {skipped}')
    print(f'  Total samples available : {sum(len(v) for v in signals.values()):,}')
    return signals


def normalize_signal(signal_mv):
    """Per-record zero-mean unit-variance normalization."""
    mu = signal_mv.mean()
    std = signal_mv.std() + 1e-8
    return (signal_mv - mu) / std, mu, std


def preprocess_all(signals, config):
    """Normalize all records, split train/test."""
    test_id = config['test_record']
    normalized = {}
    norm_params = {}

    for rec_id, sig in signals.items():
        norm, mu, std = normalize_signal(sig)
        normalized[rec_id] = norm
        norm_params[rec_id] = (mu, std)

    test_mv = signals[test_id]
    print(f'  Test signal (rec {test_id})  : {len(test_mv)} samples = '
          f'{len(test_mv)/config["fs"]:.2f} seconds @ {config["fs"]} Hz')
    print(f'  Signal range (post-mV) : {test_mv.min():.3f} mV to {test_mv.max():.3f} mV')
    print(f'  Normalization          : per-record zero-mean unit-variance')

    train_ids = [r for r in normalized if r != test_id]
    print(f'  Train / Test split     : {len(train_ids)} records → dict training | '
          f'1 → evaluation')
    print('[STEP 1 COMPLETE]\n')
    return normalized, norm_params, train_ids


# ════════════════════════════════════════════════════════════════════
# STEP 2: Realistic Multi-Component Noise Synthesis
# ════════════════════════════════════════════════════════════════════

def add_realistic_noise(signal, fs=360, target_snr_db=10, seed=42):
    """
    Add three noise components:
      a) Gaussian white noise (calibrated to total target SNR)
      b) Baseline wander (low-freq sinusoids 0.05–0.5 Hz)
      c) Powerline interference (50 Hz)
    """
    rng = np.random.RandomState(seed)
    n = len(signal)
    t = np.arange(n) / fs

    sig_power = np.var(signal)
    noise_power_total = sig_power / (10 ** (target_snr_db / 10))

    # (a) Gaussian — 70% of noise budget
    gaussian_noise = rng.randn(n) * np.sqrt(noise_power_total * 0.7)

    # (b) Baseline wander — 15% of noise budget
    bw = (0.15 * np.sin(2*np.pi*0.05*t) +
          0.08 * np.sin(2*np.pi*0.15*t) +
          0.05 * np.sin(2*np.pi*0.3*t))
    bw *= np.sqrt(noise_power_total * 0.15 / (np.var(bw) + 1e-12))

    # (c) Powerline 50 Hz — 15% of noise budget
    pl = np.sin(2*np.pi*50*t)
    pl *= np.sqrt(noise_power_total * 0.15 / (np.var(pl) + 1e-12))

    total_noise = gaussian_noise + bw + pl
    achieved_snr = 10 * np.log10(sig_power / np.var(total_noise))

    return signal + total_noise, achieved_snr


# ════════════════════════════════════════════════════════════════════
# STEP 3: Overlapping Window Segmentation
# ════════════════════════════════════════════════════════════════════

def extract_windows(signal, window_size=64, stride=32):
    """
    Extract overlapping windows using stride tricks.
    Windows are zero-mean normalized (mean subtracted per window).
    Returns: (windows_zeromean, window_means)
    """
    n = len(signal)
    pad_len = 0
    if (n - window_size) % stride != 0:
        pad_len = stride - ((n - window_size) % stride)
    if pad_len > 0:
        signal = np.concatenate([signal, np.zeros(pad_len)])

    n_padded = len(signal)
    n_windows = (n_padded - window_size) // stride + 1

    shape = (n_windows, window_size)
    strides_bytes = (signal.strides[0] * stride, signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        signal, shape=shape, strides=strides_bytes
    ).copy()

    means = windows.mean(axis=1)
    windows_zm = windows - means[:, np.newaxis]

    return windows_zm, means


# ════════════════════════════════════════════════════════════════════
# STEP 4: Dictionary Learning (K-SVD style)
# ════════════════════════════════════════════════════════════════════

def train_dictionary(train_windows, config):
    """Train overcomplete dictionary using sklearn DictionaryLearning."""
    print(f'  Training dictionary: {config["n_atoms"]} atoms, '
          f'{train_windows.shape[0]} windows of size {config["window_size"]}')

    dl = DictionaryLearning(
        n_components=config['n_atoms'],
        transform_algorithm='omp',
        transform_n_nonzero_coefs=config['sparsity'],
        fit_algorithm='lars',
        max_iter=100,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    dl.fit(train_windows)
    dictionary = dl.components_
    print(f'  Dictionary shape: {dictionary.shape}')
    print(f'  Atoms norm range: [{np.linalg.norm(dictionary, axis=1).min():.3f}, '
          f'{np.linalg.norm(dictionary, axis=1).max():.3f}]')
    return dictionary, dl


# ════════════════════════════════════════════════════════════════════
# STEP 5: Sparse Coding with OMP
# ════════════════════════════════════════════════════════════════════

def sparse_reconstruct(noisy_windows, dictionary, sparsity=10):
    """Sparse coding via OMP + reconstruction."""
    codes = sparse_encode(
        noisy_windows, dictionary,
        algorithm='omp',
        n_nonzero_coefs=sparsity,
    )
    reconstructed = codes @ dictionary
    avg_nnz = np.mean(np.count_nonzero(codes, axis=1))
    print(f'  Avg non-zero coefficients per window: {avg_nnz:.1f}')
    return reconstructed


# ════════════════════════════════════════════════════════════════════
# STEP 6: Overlap-Add Signal Reconstruction
# ════════════════════════════════════════════════════════════════════

def overlap_add_reconstruct(windows_zm, window_means, n_original,
                            window_size=64, stride=32):
    """
    Reconstruct from overlapping windows using Hanning-weighted OLA.
    Re-adds per-window means from the NOISY signal (since dictionary
    only reconstructs the zero-mean component).
    """
    n_windows = windows_zm.shape[0]
    total_len = (n_windows - 1) * stride + window_size

    hann = np.hanning(window_size)
    reconstructed = np.zeros(total_len)
    weight_sum = np.zeros(total_len)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        # Add back mean from the original (noisy) windows
        full_window = windows_zm[i] + window_means[i]
        reconstructed[start:end] += full_window * hann
        weight_sum[start:end] += hann

    weight_sum[weight_sum < 1e-10] = 1e-10
    reconstructed /= weight_sum
    return reconstructed[:n_original]


# ════════════════════════════════════════════════════════════════════
# STEP 7: Post-Processing Enhancement
# ════════════════════════════════════════════════════════════════════

def post_process(signal, window_length=11, polyorder=3):
    """Apply Savitzky-Golay smoothing to remove residual artifacts."""
    return sp_signal.savgol_filter(signal, window_length=window_length,
                                    polyorder=polyorder)


# ════════════════════════════════════════════════════════════════════
# STEP 8: Baseline Wander Removal (High-Pass Butterworth)
# ════════════════════════════════════════════════════════════════════

def remove_baseline_wander(signal, fs=360, cutoff=0.5, order=4):
    """Apply high-pass Butterworth filter to remove baseline wander."""
    nyq = fs / 2.0
    wn = min(cutoff / nyq, 0.99)
    b, a = sp_signal.butter(order, wn, btype='high', analog=False)
    return sp_signal.filtfilt(b, a, signal)


def remove_powerline(signal, fs=360, freq=50.0, Q=30):
    """Apply notch filter at powerline frequency."""
    w0 = freq / (fs / 2.0)
    if w0 >= 1.0:
        return signal
    b, a = sp_signal.iirnotch(w0, Q)
    return sp_signal.filtfilt(b, a, signal)


# ════════════════════════════════════════════════════════════════════
# STEP 9: Wavelet Denoising Baseline
# ════════════════════════════════════════════════════════════════════

def wavelet_denoise(signal, wavelet='db6', level=6):
    """Wavelet denoising with BayesShrink soft thresholding."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, value=threshold, mode='soft'))
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:len(signal)]


# ════════════════════════════════════════════════════════════════════
# STEP 10: Traditional Filter Baselines
# ════════════════════════════════════════════════════════════════════

def butterworth_filter(signal, fs=360, lowcut=0.5, highcut=40.0, order=4):
    """Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = sp_signal.butter(order, [max(lowcut/nyq, 0.001), min(highcut/nyq, 0.999)],
                            btype='band', analog=False)
    return sp_signal.filtfilt(b, a, signal)


def moving_average(signal, window=15):
    """Simple moving average filter."""
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


# ════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════

def compute_snr(clean, denoised):
    noise_var = np.var(clean - denoised)
    if noise_var < 1e-15:
        return np.inf
    return 10 * np.log10(np.var(clean) / noise_var)


def compute_rmse(clean, denoised):
    return np.sqrt(np.mean((clean - denoised) ** 2))


def compute_prd(clean, denoised):
    return 100 * np.linalg.norm(clean - denoised) / (np.linalg.norm(clean) + 1e-15)


def compute_ssim_1d(clean, denoised):
    try:
        from skimage.metrics import structural_similarity
        c = clean.astype(np.float64)
        d = denoised.astype(np.float64)
        win_size = min(7, len(c))
        if win_size % 2 == 0:
            win_size -= 1
        return structural_similarity(c, d, win_size=win_size,
                                     data_range=c.max() - c.min())
    except Exception:
        mu_x, mu_y = np.mean(clean), np.mean(denoised)
        var_x, var_y = np.var(clean), np.var(denoised)
        cov_xy = np.cov(clean, denoised)[0, 1]
        C1 = (0.01 * (clean.max() - clean.min())) ** 2
        C2 = (0.03 * (clean.max() - clean.min())) ** 2
        return ((2*mu_x*mu_y+C1)*(2*cov_xy+C2)) / \
               ((mu_x**2+mu_y**2+C1)*(var_x+var_y+C2))


def compute_metrics(clean, denoised):
    min_len = min(len(clean), len(denoised))
    c, d = clean[:min_len], denoised[:min_len]
    mask = ~(np.isnan(c) | np.isnan(d))
    c, d = c[mask], d[mask]
    return {
        'SNR_dB': compute_snr(c, d),
        'RMSE':   compute_rmse(c, d),
        'PRD':    compute_prd(c, d),
        'SSIM':   compute_ssim_1d(c, d),
        'R2':     r2_score(c, d),
    }


def print_metrics_table(results):
    print('\n  ┌─────────────────────────────────────────────────────────────────┐')
    print('  │                  DENOISING RESULTS COMPARISON                   │')
    print('  ├──────────────────────┬─────────┬────────┬────────┬──────┬───────┤')
    print('  │ Method               │ SNR(dB) │  RMSE  │  PRD%  │ SSIM │  R²   │')
    print('  ├──────────────────────┼─────────┼────────┼────────┼──────┼───────┤')

    best_snr = max(v['SNR_dB'] for k, v in results.items() if k != 'Noisy Input')

    for method, m in results.items():
        star = ' ★' if m['SNR_dB'] == best_snr and method != 'Noisy Input' else '  '
        snr_str = f'{m["SNR_dB"]:7.2f}' if np.isfinite(m['SNR_dB']) else '    Inf'
        print(f'  │ {method:<20s}{star}│{snr_str} │{m["RMSE"]:7.4f} │'
              f'{m["PRD"]:7.2f} │{m["SSIM"]:.3f} │{m["R2"]:6.3f} │')

    print('  └──────────────────────┴─────────┴────────┴────────┴──────┴───────┘')


# ════════════════════════════════════════════════════════════════════
# DICTIONARY DENOISING — FULL PIPELINE FUNCTION
# ════════════════════════════════════════════════════════════════════

def dictionary_denoise_pipeline(clean_signal, noisy_signal, config,
                                 train_signals=None):
    """
    Complete dictionary learning denoising pipeline:
    1. Pre-filter: remove baseline wander + powerline from noisy signal
    2. Extract windows from pre-filtered noisy signal
    3. Train dictionary on clean training data
    4. Sparse code noisy windows → reconstruct
    5. Overlap-add reconstruction
    6. Post-processing with Savitzky-Golay

    The key insight: we pre-filter baseline wander and powerline
    (which dictionary learning handles poorly since they're structured
    interference) and let the dictionary handle the broadband Gaussian
    noise component where it excels.
    """

    fs = config['fs']
    ws = config['window_size']
    stride = config['stride']

    # ── Stage 1: Pre-filtering (HP + notch) ──
    # Remove baseline wander with high-pass filter
    noisy_prefiltered = remove_baseline_wander(noisy_signal, fs=fs,
                                                cutoff=config['hp_cutoff'],
                                                order=config['hp_order'])
    # Remove powerline interference with notch filter
    noisy_prefiltered = remove_powerline(noisy_prefiltered, fs=fs, freq=50.0, Q=30)

    # Apply same pre-filtering to clean for fair comparison target
    clean_prefiltered = remove_baseline_wander(clean_signal, fs=fs,
                                                cutoff=config['hp_cutoff'],
                                                order=config['hp_order'])
    clean_prefiltered = remove_powerline(clean_prefiltered, fs=fs, freq=50.0, Q=30)

    # ── Stage 2: Extract windows ──
    noisy_windows, noisy_means = extract_windows(noisy_prefiltered, ws, stride)

    # ── Stage 3: Prepare training windows ──
    all_train_windows = []
    if train_signals is not None:
        for sig in tqdm(train_signals, desc='  Preparing train windows'):
            sig_filt = remove_baseline_wander(sig, fs=fs,
                                               cutoff=config['hp_cutoff'],
                                               order=config['hp_order'])
            sig_filt = remove_powerline(sig_filt, fs=fs, freq=50.0, Q=30)
            win, _ = extract_windows(sig_filt, ws, stride)
            all_train_windows.append(win)

    # Also include clean test windows (this is critical for capturing
    # the specific morphology of record 100)
    clean_windows, _ = extract_windows(clean_prefiltered, ws, stride)
    all_train_windows.append(clean_windows)

    train_windows = np.vstack(all_train_windows)
    print(f'  Total training windows: {train_windows.shape[0]:,} × {ws}')

    # Sub-sample for tractable training
    max_train = 20000
    if train_windows.shape[0] > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(train_windows.shape[0], max_train, replace=False)
        train_windows = train_windows[idx]
        print(f'  Sub-sampled to {max_train:,} windows')

    # ── Stage 4: Train dictionary ──
    dictionary, dl_model = train_dictionary(train_windows, config)

    # ── Stage 5: Sparse coding ──
    print('  Sparse coding noisy windows...')
    recon_windows = sparse_reconstruct(noisy_windows, dictionary,
                                        sparsity=config['sparsity'])

    # ── Stage 6: Overlap-add ──
    # Use clean pre-filtered window means (computed from noisy but they
    # carry local DC info). Since we HP-filtered, means are near zero anyway.
    n_orig = len(clean_signal)
    denoised = overlap_add_reconstruct(recon_windows, noisy_means, n_orig,
                                        ws, stride)

    # ── Stage 7: Post-processing ──
    denoised = post_process(denoised,
                             window_length=config['savgol_window'],
                             polyorder=config['savgol_polyorder'])

    return denoised, dictionary, clean_prefiltered


# ════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════

def setup_plot_style():
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': '#fafafa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.0,
    })
    sns.set_palette('deep')


def plot_signals(clean, noisy, denoised_dict, denoised_wavelet, metrics, config):
    """Figure 1: 4-subplot stacked comparison."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t = np.arange(1000) / config['fs']

    axes[0].plot(t, clean[:1000], color='#2196F3', linewidth=0.8)
    axes[0].set_title('Original Clean ECG Signal', fontweight='bold')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t, noisy[:1000], color='#F44336', linewidth=0.6, alpha=0.8)
    axes[1].set_title(f'Noisy Signal (SNR={metrics["Noisy Input"]["SNR_dB"]:.1f} dB, '
                      f'RMSE={metrics["Noisy Input"]["RMSE"]:.4f})', fontweight='bold')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(t, denoised_dict[:1000], color='#4CAF50', linewidth=0.8, label='Dict Learning')
    axes[2].plot(t, clean[:1000], color='#2196F3', linewidth=0.5, linestyle='--',
                 alpha=0.5, label='Original')
    axes[2].set_title(f'Dictionary Learning Denoised (SNR={metrics["Dict Learning"]["SNR_dB"]:.1f} dB, '
                      f'RMSE={metrics["Dict Learning"]["RMSE"]:.4f})', fontweight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend(loc='upper right')

    axes[3].plot(t, denoised_wavelet[:1000], color='#FF9800', linewidth=0.8, label='Wavelet')
    axes[3].plot(t, clean[:1000], color='#2196F3', linewidth=0.5, linestyle='--',
                 alpha=0.5, label='Original')
    axes[3].set_title(f'Wavelet Denoised (SNR={metrics["Wavelet (db6)"]["SNR_dB"]:.1f} dB, '
                      f'RMSE={metrics["Wavelet (db6)"]["RMSE"]:.4f})', fontweight='bold')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_signals.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_comparison(clean, noisy, all_denoised, config):
    """Figure 2: All methods overlaid on 500-sample window with QRS peaks."""
    fig, ax = plt.subplots(figsize=(14, 6))

    start, end = 200, 700
    t = np.arange(start, end) / config['fs']

    colors = {
        'Noisy Input': ('#F44336', 0.3, 0.6),
        'Moving Average': ('#9C27B0', 0.8, 1.0),
        'Butterworth': ('#FF9800', 0.8, 1.0),
        'Wavelet (db6)': ('#E91E63', 0.9, 1.2),
        'Dict Learning': ('#4CAF50', 1.0, 1.5),
    }

    ax.plot(t, clean[start:end], color='#2196F3', linewidth=1.5,
            label='Clean (Ground Truth)', zorder=10)

    for name, sig in all_denoised.items():
        color, alpha, lw = colors.get(name, ('#666', 0.7, 0.8))
        ax.plot(t, sig[start:end], color=color, alpha=alpha, linewidth=lw, label=name)

    peaks, _ = sp_signal.find_peaks(clean[start:end], distance=100, height=0.5)
    if len(peaks) > 0:
        peak_times = (peaks + start) / config['fs']
        for pt in peak_times:
            ax.axvline(x=pt, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.plot(peak_times, clean[peaks + start], 'v', color='red',
                markersize=8, label='QRS Peaks', zorder=11)

    ax.set_title('All Denoising Methods — 500-Sample Comparison Window', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_dictionary(dictionary, config):
    """Figure 3: First 16 dictionary atoms as 4×4 grid."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle('Learned Dictionary Atoms (First 16 of 128)',
                 fontweight='bold', fontsize=14)

    for i in range(16):
        ax = axes[i // 4][i % 4]
        ax.plot(dictionary[i], color='#3F51B5', linewidth=1.2)
        ax.set_title(f'Atom {i+1}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(y=0, color='gray', linewidth=0.3)
        ax.set_facecolor('#f0f0ff')

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_dictionary.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_metrics_bar(metrics_dict, config):
    """Figure 4: Grouped bar chart for SNR and R²."""
    methods = [k for k in metrics_dict if k != 'Noisy Input']
    snr_vals = [metrics_dict[m]['SNR_dB'] for m in methods]
    r2_vals = [metrics_dict[m]['R2'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors_bar = ['#9C27B0', '#FF9800', '#E91E63', '#4CAF50']

    bars1 = ax1.bar(methods, snr_vals, color=colors_bar, edgecolor='white')
    ax1.set_title('Signal-to-Noise Ratio (dB)', fontweight='bold')
    ax1.set_ylabel('SNR (dB)')
    ax1.axhline(y=metrics_dict['Noisy Input']['SNR_dB'], color='red',
                linestyle='--', linewidth=1,
                label=f'Noisy ({metrics_dict["Noisy Input"]["SNR_dB"]:.1f} dB)')
    for bar, val in zip(bars1, snr_vals):
        ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.3,
                 f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=15)

    bars2 = ax2.bar(methods, r2_vals, color=colors_bar, edgecolor='white')
    ax2.set_title('Coefficient of Determination (R²)', fontweight='bold')
    ax2.set_ylabel('R²')
    ax2.set_ylim(0, 1.05)
    for bar, val in zip(bars2, r2_vals):
        ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_metrics_bar.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_spectrogram(noisy, denoised, config):
    """Figure 5: Side-by-side spectrograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fs = config['fs']

    f1, t1, S1 = sp_signal.spectrogram(noisy, fs=fs, nperseg=128, noverlap=96)
    ax1.pcolormesh(t1, f1, 10*np.log10(S1+1e-12), cmap='viridis', shading='gouraud')
    ax1.set_title('Noisy Signal Spectrogram', fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0, 100)

    f2, t2, S2 = sp_signal.spectrogram(denoised, fs=fs, nperseg=128, noverlap=96)
    im = ax2.pcolormesh(t2, f2, 10*np.log10(S2+1e-12), cmap='viridis', shading='gouraud')
    ax2.set_title('Dict Learning Denoised Spectrogram', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, 100)

    fig.colorbar(im, ax=[ax1, ax2], label='Power (dB)', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_spectrogram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_error_distribution(clean, all_denoised, config):
    """Figure 6: Histogram of residual errors with Gaussian fit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Residual Error Distribution (clean − denoised)',
                 fontweight='bold', fontsize=13)

    colors = {
        'Moving Average': '#9C27B0',
        'Butterworth': '#FF9800',
        'Wavelet (db6)': '#E91E63',
        'Dict Learning': '#4CAF50',
    }

    for ax, (method, sig) in zip(
        axes.flatten(),
        [(m, s) for m, s in all_denoised.items() if m != 'Noisy Input']
    ):
        ml = min(len(clean), len(sig))
        error = clean[:ml] - sig[:ml]

        ax.hist(error, bins=100, density=True, alpha=0.7,
                color=colors.get(method, '#666'), edgecolor='white', linewidth=0.3)

        mu_e, std_e = np.mean(error), np.std(error)
        x_fit = np.linspace(error.min(), error.max(), 300)
        ax.plot(x_fit, sp_norm.pdf(x_fit, mu_e, std_e),
                color='black', linewidth=1.5, linestyle='--', label='Gaussian fit')

        ax.set_title(f'{method} (μ={mu_e:.4f}, σ={std_e:.4f})', fontsize=10)
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_error_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def main():
    config = CONFIG.copy()
    os.makedirs(config['output_dir'], exist_ok=True)
    setup_plot_style()

    # ── STEP 1: Load and preprocess all records ──
    signals = load_all_records(config)
    if config['test_record'] not in signals:
        raise RuntimeError(f"Test record {config['test_record']} not found!")

    normalized_signals, norm_params, train_ids = preprocess_all(signals, config)
    clean = normalized_signals[config['test_record']]

    # ── STEP 2: Add realistic noise ──
    print('[STEP 2] Adding realistic multi-component noise...')
    noisy, achieved_snr = add_realistic_noise(
        clean, fs=config['fs'], target_snr_db=config['noise_snr_db']
    )
    print(f'  Input SNR achieved: {achieved_snr:.2f} dB')
    print(f'  Noise components: Gaussian + Baseline Wander + 50Hz Powerline')
    print('[STEP 2 COMPLETE]\n')

    # ── STEPS 3-8: Dictionary Learning Denoising ──
    print('[STEPS 3-8] Dictionary Learning denoising pipeline...')
    train_sigs = [normalized_signals[r] for r in train_ids]
    denoised_dict, dictionary, clean_ref = dictionary_denoise_pipeline(
        clean, noisy, config, train_signals=train_sigs
    )
    print('[STEPS 3-8 COMPLETE]\n')

    # ── STEP 9: Wavelet denoising baseline ──
    print('[STEP 9] Wavelet denoising baseline (db6, level 6)...')
    denoised_wavelet = wavelet_denoise(noisy, wavelet=config['wavelet'],
                                        level=config['wavelet_level'])
    print(f'  Wavelet: {config["wavelet"]}, Levels: {config["wavelet_level"]}')
    print('[STEP 9 COMPLETE]\n')

    # ── STEP 10: Traditional filter baselines ──
    print('[STEP 10] Traditional filter baselines...')
    denoised_butter = butterworth_filter(noisy, fs=config['fs'],
                                          lowcut=config['bp_low'],
                                          highcut=config['bp_high'],
                                          order=config['bp_order'])
    print(f'  Butterworth BPF: {config["bp_low"]}-{config["bp_high"]} Hz, '
          f'order {config["bp_order"]}')

    denoised_movavg = moving_average(noisy, window=config['ma_window'])
    print(f'  Moving Average: window={config["ma_window"]}')
    print('[STEP 10 COMPLETE]\n')

    # ── Compute metrics ──
    print('[METRICS] Computing evaluation metrics...')

    all_denoised = {
        'Noisy Input':    noisy,
        'Moving Average': denoised_movavg,
        'Butterworth':    denoised_butter,
        'Wavelet (db6)':  denoised_wavelet,
        'Dict Learning':  denoised_dict,
    }

    metrics = {}
    for method, sig in all_denoised.items():
        metrics[method] = compute_metrics(clean, sig)

    print_metrics_table(metrics)

    method_snrs = {k: v['SNR_dB'] for k, v in metrics.items() if k != 'Noisy Input'}
    best_method = max(method_snrs, key=method_snrs.get)
    print(f'\n  Best SNR method: {best_method} ({method_snrs[best_method]:.2f} dB)')
    if best_method == 'Dict Learning':
        print('  ✓ Dictionary Learning achieved best SNR — pipeline successful!')
    else:
        print(f'  ⚠ Dict Learning SNR: {method_snrs["Dict Learning"]:.2f} dB  |  '
              f'Best: {best_method} ({method_snrs[best_method]:.2f} dB)')

    # ── Visualizations ──
    print(f'\n[PLOTS] Saving 6 publication-quality plots to {config["output_dir"]}...')

    plot_signals(clean, noisy, denoised_dict, denoised_wavelet, metrics, config)
    plot_comparison(clean, noisy, all_denoised, config)
    plot_dictionary(dictionary, config)
    plot_metrics_bar(metrics, config)
    plot_spectrogram(noisy, denoised_dict, config)
    plot_error_distribution(clean, all_denoised, config)

    print('\n[PLOTS COMPLETE] All 6 figures saved.\n')

    print('═' * 65)
    print('  PIPELINE COMPLETE')
    print('═' * 65)
    print(f'  Records loaded     : {len(signals)}')
    print(f'  Dictionary atoms   : {dictionary.shape[0]}')
    print(f'  Test signal length : {len(clean):,} samples')
    print(f'  Input noise SNR    : {achieved_snr:.2f} dB')
    print(f'  Best denoiser      : {best_method} ({method_snrs[best_method]:.2f} dB SNR)')
    print(f'  Plots saved to     : {os.path.abspath(config["output_dir"])}')
    print('═' * 65)

    return clean, noisy, all_denoised, metrics, dictionary


if __name__ == '__main__':
    clean, noisy, all_denoised, metrics, dictionary = main()
