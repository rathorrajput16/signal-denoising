import numpy as np
from scipy import signal as sp_signal
import pywt

def wavelet_denoise(signal, wavelet='db6', level=6):
    """
    Wavelet denoising with BayesShrink soft thresholding.

    Parameters
    ----------
    signal : np.ndarray
        Noisy signal
    wavelet : str
        Wavelet family (default 'db6')
    level : int
        Decomposition levels (default 6)

    Returns
    -------
    np.ndarray
        Denoised signal
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, value=threshold, mode='soft'))

    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[:len(signal)]


def butterworth_filter(signal, fs=360, lowcut=0.5, highcut=40.0, order=4):
    """
    Butterworth bandpass filter.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : int
        Sampling frequency (Hz)
    lowcut : float
        Lower cutoff frequency (Hz)
    highcut : float
        Upper cutoff frequency (Hz)
    order : int
        Filter order

    Returns
    -------
    np.ndarray
        Filtered signal
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(
        order, [max(lowcut/nyq, 0.001), min(highcut/nyq, 0.999)],
        btype='band', analog=False
    )
    return sp_signal.filtfilt(b, a, signal)


def moving_average(signal, window=15):
    """
    Simple moving average filter.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    window : int
        Window size

    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def remove_baseline_wander(signal, fs=360, cutoff=0.5, order=4):
    """
    Remove baseline wander using high-pass Butterworth filter.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : int
        Sampling frequency (Hz)
    cutoff : float
        High-pass cutoff frequency (Hz)
    order : int
        Filter order

    Returns
    -------
    np.ndarray
        Filtered signal (baseline removed)
    """
    nyq = fs / 2.0
    wn = min(cutoff / nyq, 0.99)
    b, a = sp_signal.butter(order, wn, btype='high', analog=False)
    return sp_signal.filtfilt(b, a, signal)