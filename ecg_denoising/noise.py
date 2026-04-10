import numpy as np

def add_realistic_noise(signal, fs=360, target_snr_db=10, seed=42):
    """
    Add three noise components to simulate real ECG artifacts.

    Parameters
    ----------
    signal : np.ndarray
        Clean ECG signal (normalized)
    fs : int
        Sampling frequency (Hz)
    target_snr_db : float
        Target signal-to-noise ratio (dB) for combined noise
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[np.ndarray, float]
        (noisy_signal, achieved_snr_db)

    Notes
    -----
    Noise power budget allocation:
      - 70% → Gaussian white noise
      - 15% → Baseline wander (sinusoids at 0.05, 0.15, 0.3 Hz)
      - 15% → Powerline interference (50 Hz sinusoid)
    """
    rng = np.random.RandomState(seed)
    n = len(signal)
    t = np.arange(n) / fs
    sig_power = np.var(signal)
    noise_power = sig_power / (10 ** (target_snr_db / 10))

    # (a) Gaussian white noise — 70% of budget
    gaussian = rng.randn(n) * np.sqrt(noise_power * 0.7)

    # (b) Baseline wander — 15% of budget
    bw = (0.15 * np.sin(2 * np.pi * 0.05 * t) +
          0.08 * np.sin(2 * np.pi * 0.15 * t) +
          0.05 * np.sin(2 * np.pi * 0.3 * t))
    bw *= np.sqrt(noise_power * 0.15 / (np.var(bw) + 1e-12))

    # (c) Powerline interference — 15% of budget
    pl = np.sin(2 * np.pi * 50 * t)
    pl *= np.sqrt(noise_power * 0.15 / (np.var(pl) + 1e-12))

    total_noise = gaussian + bw + pl
    achieved_snr = 10 * np.log10(sig_power / np.var(total_noise))

    return signal + total_noise, achieved_snr