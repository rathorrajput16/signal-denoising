import os
import re
import glob
import numpy as np
import scipy.io as sio
from scipy import signal as sp_signal
from tqdm import tqdm

def parse_filename(filepath):
    """
    Parse NSTDB .mat filename → (patient_id, snr_level).

    Examples
    --------
    >>> parse_filename('nsrdb_16265e24.mat')
    ('16265', 24)
    >>> parse_filename('nsrdb_16265e-6.mat')
    ('16265', -6)
    """
    basename = os.path.basename(filepath)
    match = re.match(r'nsrdb_(\d+)e(-?\d+)\.mat', basename)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))

def load_mat_signal(filepath, channel=0, n_samples=10000, offset=100000):
    """
    Load a single NSTDB .mat file using scipy.io.loadmat.

    Parameters
    ----------
    filepath : str
        Path to .mat file.
    channel : int
        ECG channel index (0 = first lead).
    n_samples : int
        Number of samples to extract.
    offset : int
        Sample offset — skips the noise-free preamble.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (clean_signal, noisy_signal, labeled_snr)
    """
    mat = sio.loadmat(filepath)
    data = mat['data'][0, 0]

    clean = data['clean_ecg'][:, channel].astype(np.float64)
    noisy = data['noisy_ecg'][:, channel].astype(np.float64)

    end = min(offset + n_samples, len(clean))
    clean = clean[offset:end]
    noisy = noisy[offset:end]

    labeled_snr = int(data['snr'].flatten()[0])
    return clean, noisy, labeled_snr

def highpass_filter(signal, fs=360, cutoff=0.67, order=4):
    """
    4th-order Butterworth high-pass filter.

    Removes EM baseline wander — critical for the zero-means
    reconstruction strategy used by dictionary learning.
    """
    nyq = fs / 2.0
    wn = min(cutoff / nyq, 0.99)
    b, a = sp_signal.butter(order, wn, btype='high', analog=False)
    return sp_signal.filtfilt(b, a, signal)

def load_nstdb_dataset(data_dir, config):
    """
    Scan directory and load all NSTDB .mat files into a nested dict.

    Returns
    -------
    dict
        dataset[patient_id][snr_level] = {
            'clean':     np.ndarray  — normalised clean signal,
            'noisy':     np.ndarray  — normalised noisy (clean's μ/σ),
            'clean_hp':  np.ndarray  — HP-filtered clean,
            'noisy_hp':  np.ndarray  — HP-filtered noisy,
            'clean_mu':  float,
            'clean_std': float,
        }
    """
    print(f'\n[STEP 1] Loading NSTDB dataset from: {data_dir}')

    files = sorted(glob.glob(os.path.join(data_dir, 'nsrdb_*e*.mat')))
    if not files:
        raise FileNotFoundError(f"No NSTDB .mat files found in {data_dir}")

    dataset = {}
    loaded = 0

    for filepath in tqdm(files, desc='Loading .mat files'):
        patient_id, snr = parse_filename(filepath)
        if patient_id is None:
            continue

        try:
            clean, noisy, _ = load_mat_signal(
                filepath,
                channel=config['channel'],
                n_samples=config['n_samples'],
                offset=config['signal_offset'],
            )
        except Exception as e:
            print(f'  [SKIP] {os.path.basename(filepath)}: {e}')
            continue
        if patient_id not in dataset:
            dataset[patient_id] = {}

        clean_mu = clean.mean()
        clean_std = clean.std() + 1e-8
        clean_norm = (clean - clean_mu) / clean_std
        noisy_norm = (noisy - clean_mu) / clean_std

        clean_hp = highpass_filter(clean_norm, config['fs'],
                                    config['hp_cutoff'], config['hp_order'])
        noisy_hp = highpass_filter(noisy_norm, config['fs'],
                                    config['hp_cutoff'], config['hp_order'])

        dataset[patient_id][snr] = {
            'clean':     clean_norm,
            'noisy':     noisy_norm,
            'clean_hp':  clean_hp,
            'noisy_hp':  noisy_hp,
            'clean_mu':  clean_mu,
            'clean_std': clean_std,
        }
        loaded += 1

    patients = sorted(dataset.keys())
    snr_set = set()
    for p in dataset:
        snr_set.update(dataset[p].keys())

    print(f'  Loaded  : {loaded} files ({len(patients)} patients)')
    print(f'  Patients: {patients}')
    print(f'  SNR levels: {sorted(snr_set)}')
    print(f'  Samples per signal: {config["n_samples"]:,}')
    print(f'  HP pre-filter: {config["hp_cutoff"]} Hz (EM baseline removal)')
    print('[STEP 1 COMPLETE]\n')

    return dataset