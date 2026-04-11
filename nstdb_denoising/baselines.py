import numpy as np
from scipy import signal as sp_signal
import pywt

def wavelet_denoise(signal, wavelet='db6', level=6):
    """
    Wavelet denoising with BayesShrink soft thresholding.
    Parameters
    ----------
    signal : np.ndarray
        Input 1-D signal.
    wavelet : str
        Wavelet family (default 'db6').
    level : int
        Decomposition level.

    Returns
    -------
    np.ndarray
        Denoised signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised = [coeffs[0]]
    for c in coeffs[1:]:
        denoised.append(pywt.threshold(c, value=threshold, mode='soft'))
    return pywt.waverec(denoised, wavelet)[:len(signal)]


def butterworth_filter(signal, fs=360, lowcut=0.5, highcut=40.0, order=4):
    """
    4th-order Butterworth bandpass filter.
    Parameters
    ----------
    signal : np.ndarray
        Input 1-D signal.
    fs : float
        Sampling frequency (Hz).
    lowcut : float
        Lower cutoff frequency (Hz).
    highcut : float
        Upper cutoff frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(
        order,
        [max(lowcut / nyq, 0.001), min(highcut / nyq, 0.999)],
        btype='band', analog=False,
    )
    return sp_signal.filtfilt(b, a, signal)