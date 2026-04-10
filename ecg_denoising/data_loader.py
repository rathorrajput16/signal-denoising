import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import MITBIH_RECORDS

def load_record(record_id, data_dir, channel='MLII',
                n_samples=10000, gain=200.0, baseline=1024.0,
                clip_mv=5.0, min_std=0.005):
    """
    Load one MIT-BIH record from CSV, convert ADC → millivolts,
    validate quality, return first n_samples.

    Parameters
    ----------
    record_id : int
        Record number (e.g., 100, 101, ...)
    data_dir : str
        Path to directory containing CSV files
    channel : str
        ECG lead column name (default 'MLII')
    n_samples : int
        Number of samples to extract (from start of record)
    gain : float
        ADC units per millivolt (MIT-BIH standard: 200)
    baseline : float
        ADC value at 0 mV (11-bit midpoint: 1024)
    clip_mv : float
        Clip signal beyond ±clip_mv (artifact rejection)
    min_std : float
        Reject signals with std below this (flat/corrupt)

    Returns
    -------
    np.ndarray or None
        Signal in millivolts, or None if file missing/corrupt
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
        print(f'  [SKIP] {record_id}.csv: flat/corrupt (std={signal_mv.std():.6f})')
        return None

    return signal_mv


def load_all_records(config, verbose=True):
    """
    Load all available MIT-BIH records.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping record_id → signal_mv array
    """
    if verbose:
        print(f"\n[DATA] Loading MIT-BIH records from: {config['data_dir']}")

    signals = {}
    loaded, skipped = 0, 0
    iterator = tqdm(MITBIH_RECORDS, desc='Loading records') if verbose else MITBIH_RECORDS
    for rec_id in iterator:
        sig = load_record(
            rec_id, config['data_dir'],
            channel=config['channel'],
            n_samples=config['n_samples_per_file'],
            gain=config['gain'], baseline=config['baseline_adc'],
            clip_mv=config['clip_mv'], min_std=config['min_std_threshold'],
        )
        if sig is not None:
            signals[rec_id] = sig
            loaded += 1
        else:
            skipped += 1

    if verbose:
        print(f'  Loaded : {loaded} records | Skipped: {skipped}')
        print(f'  Total samples : {sum(len(v) for v in signals.values()):,}')
    return signals


def normalize_signal(signal_mv):
    """
    Per-record zero-mean unit-variance normalization.

    Parameters
    ----------
    signal_mv : np.ndarray
        Signal in millivolts

    Returns
    -------
    tuple[np.ndarray, float, float]
        (normalized_signal, mean, std) — store mean/std for denormalization
    """
    mu = signal_mv.mean()
    std = signal_mv.std() + 1e-8
    return (signal_mv - mu) / std, mu, std


def denormalize_signal(normalized, mu, std):
    """
    Reverse normalization to recover millivolt-scale signal.

    Parameters
    ----------
    normalized : np.ndarray
        Normalized signal
    mu : float
        Original mean
    std : float
        Original standard deviation

    Returns
    -------
    np.ndarray
        Signal in millivolts
    """
    return normalized * std + mu


def load_single_csv(filepath, channel='MLII', n_samples=None,
                    gain=200.0, baseline=1024.0):
    """
    Load a single ECG CSV file (for inference on arbitrary files).

    Parameters
    ----------
    filepath : str
        Path to CSV file
    channel : str
        Column name for ECG lead
    n_samples : int or None
        If set, take only first n_samples
    gain : float
        ADC gain for conversion
    baseline : float
        ADC baseline for conversion

    Returns
    -------
    tuple[np.ndarray, float, float]
        (normalized_signal, mean, std)
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip("'").strip() for c in df.columns]

    if channel not in df.columns:
        raise ValueError(f"Column '{channel}' not found. "
                         f"Available: {list(df.columns)}")

    raw = df[channel].values.astype(np.float64)
    signal_mv = (raw - baseline) / gain

    if n_samples is not None:
        signal_mv = signal_mv[:n_samples]

    normalized, mu, std = normalize_signal(signal_mv)
    return normalized, mu, std