import os
import numpy as np
import joblib
from scipy import signal as sp_signal
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import sparse_encode

def extract_windows(signal, window_size=64, stride=32):
    """
    Extract overlapping windows with configurable stride.
    Each window is zero-mean normalized.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (windows_zeromean, window_means, n_windows)
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
    strides_b = (signal.strides[0] * stride, signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        signal, shape=shape, strides=strides_b
    ).copy()

    means = windows.mean(axis=1)
    windows_zm = windows - means[:, np.newaxis]

    return windows_zm, means, n_windows


def extract_dense_patches(signal, window_size=64):
    """
    Extract ALL overlapping patches with stride=1.
    This is the key function for dense patch averaging.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D)
    window_size : int
        Patch size (default 64)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (patches_zeromean, patch_means, n_patches)
        - patches_zeromean: (n_patches, window_size) — zero-mean patches
        - patch_means: (n_patches,) — per-patch means for reconstruction
        - n_patches: number of patches
    """
    n = len(signal)
    n_patches = n - window_size + 1

    patches = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_patches, window_size),
        strides=(signal.strides[0], signal.strides[0])
    ).copy()

    means = patches.mean(axis=1)
    patches_zm = patches - means[:, np.newaxis]

    return patches_zm, means, n_patches

def train_dictionary(train_windows, n_atoms=128, sparsity=10,
                     max_iter=100, max_train_windows=15000):
    """
    Train an overcomplete dictionary using sklearn DictionaryLearning.

    Parameters
    ----------
    train_windows : np.ndarray
        Training patches, shape (n_samples, window_size)
    n_atoms : int
        Number of dictionary atoms
    sparsity : int
        OMP non-zero coefficients
    max_iter : int
        Maximum training iterations
    max_train_windows : int
        Cap on training patches (random subsample if exceeded)

    Returns
    -------
    tuple[np.ndarray, DictionaryLearning]
        (dictionary, fitted_model)
        - dictionary: (n_atoms, window_size)
    """
    if train_windows.shape[0] > max_train_windows:
        rng = np.random.RandomState(42)
        idx = rng.choice(train_windows.shape[0], max_train_windows, replace=False)
        train_windows = train_windows[idx]
        print(f'  Sub-sampled to {max_train_windows:,} training windows')

    print(f'  Training dictionary: {n_atoms} atoms, '
          f'{train_windows.shape[0]:,} windows of size {train_windows.shape[1]}')

    dl = DictionaryLearning(
        n_components=n_atoms,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=sparsity,
        fit_algorithm='lars',
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    dl.fit(train_windows)

    dictionary = dl.components_
    print(f'  Dictionary shape: {dictionary.shape}')
    norms = np.linalg.norm(dictionary, axis=1)
    print(f'  Atoms norm range: [{norms.min():.3f}, {norms.max():.3f}]')

    return dictionary, dl

def save_dictionary(dictionary, config, filepath):
    """
    Save trained dictionary + config to disk using joblib.

    Parameters
    ----------
    dictionary : np.ndarray
        Learned dictionary, shape (n_atoms, window_size)
    config : dict
        Training config (stored alongside for reproducibility)
    filepath : str
        Output path (e.g., 'models/ecg_dictionary.pkl')
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    artifact = {
        'dictionary': dictionary,
        'n_atoms': dictionary.shape[0],
        'window_size': dictionary.shape[1],
        'sparsity': config.get('sparsity', 10),
        'savgol_window': config.get('savgol_window', 11),
        'savgol_polyorder': config.get('savgol_polyorder', 3),
        'fs': config.get('fs', 360),
        'config': config,
    }

    joblib.dump(artifact, filepath, compress=3)
    print(f'  Dictionary saved to: {filepath}')
    print(f'  Dictionary shape: {dictionary.shape}')


def load_dictionary(filepath):
    """
    Load a trained dictionary from disk.

    Parameters
    ----------
    filepath : str
        Path to saved dictionary (e.g., 'models/ecg_dictionary.pkl')

    Returns
    -------
    dict
        Contains: 'dictionary', 'n_atoms', 'window_size', 'sparsity',
                  'savgol_window', 'savgol_polyorder', 'fs', 'config'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dictionary not found: {filepath}. "
                                f"Run train.py first.")

    artifact = joblib.load(filepath)
    print(f'  Loaded dictionary from: {filepath}')
    print(f'  Shape: {artifact["dictionary"].shape} '
          f'({artifact["n_atoms"]} atoms × {artifact["window_size"]} samples)')

    return artifact

def sparse_reconstruct(noisy_windows, dictionary, sparsity=10):
    """
    Sparse coding via OMP: encode noisy windows and reconstruct.

    Parameters
    ----------
    noisy_windows : np.ndarray
        Noisy patches, shape (n_patches, window_size)
    dictionary : np.ndarray
        Learned dictionary, shape (n_atoms, window_size)
    sparsity : int
        Number of non-zero coefficients per patch

    Returns
    -------
    np.ndarray
        Reconstructed patches, shape (n_patches, window_size)
    """
    codes = sparse_encode(
        noisy_windows, dictionary,
        algorithm='omp', n_nonzero_coefs=sparsity,
    )
    reconstructed = codes @ dictionary
    avg_nnz = np.mean(np.count_nonzero(codes, axis=1))
    print(f'  Avg non-zero coefficients per patch: {avg_nnz:.1f}')
    return reconstructed

def dense_patch_reconstruct(reconstructed_patches, patch_means,
                             n_original, window_size=64):
    """
    Reconstruct signal by averaging ALL patches at each sample position.

    This is the key innovation: each sample is the average of up to
    `window_size` independent sparse estimates. Noise (random) cancels
    while signal (deterministic) reinforces.

    Parameters
    ----------
    reconstructed_patches : np.ndarray
        Sparse-coded patches, shape (n_patches, window_size)
    patch_means : np.ndarray
        Per-patch means from extraction, shape (n_patches,)
    n_original : int
        Original signal length
    window_size : int
        Patch size

    Returns
    -------
    np.ndarray
        Reconstructed signal, length n_original
    """
    n = n_original
    n_patches = reconstructed_patches.shape[0]

    rec = np.zeros(n)
    cnt = np.zeros(n)

    for p in range(n_patches):
        end = min(p + window_size, n)
        actual_ws = end - p
        rec[p:end] += (reconstructed_patches[p, :actual_ws] + patch_means[p])
        cnt[p:end] += 1

    cnt[cnt == 0] = 1
    rec /= cnt
    return rec


def overlap_add_reconstruct(windows, n_original, window_size=64, stride=32):
    """
    Standard Hanning-weighted overlap-add reconstruction.
    (Used for comparison / fallback, not the primary method.)
    """
    n_windows = windows.shape[0]
    total_len = (n_windows - 1) * stride + window_size
    hann = np.hanning(window_size)
    rec = np.zeros(total_len)
    wt = np.zeros(total_len)
    for i in range(n_windows):
        s = i * stride
        rec[s:s+window_size] += windows[i] * hann
        wt[s:s+window_size] += hann
    wt[wt < 1e-10] = 1e-10
    rec /= wt
    return rec[:n_original]

def post_process(signal, window_length=11, polyorder=3):
    """Apply Savitzky-Golay smoothing to remove residual artifacts."""
    return sp_signal.savgol_filter(signal, window_length=window_length,
                                    polyorder=polyorder)

def denoise_signal(noisy_signal, dictionary, sparsity=10,
                   window_size=64, savgol_window=11, savgol_polyorder=3,
                   verbose=True):
    """
    Denoise an ECG signal using a pre-trained dictionary.

    This is the main inference function. Given a noisy signal and a
    trained dictionary, it:
      1. Extracts dense patches (stride=1)
      2. Sparse codes each patch via OMP
      3. Reconstructs via per-sample averaging
      4. Applies Savitzky-Golay post-processing

    Parameters
    ----------
    noisy_signal : np.ndarray
        Noisy ECG signal (1D, normalized)
    dictionary : np.ndarray
        Trained dictionary, shape (n_atoms, window_size)
    sparsity : int
        OMP non-zero coefficients
    window_size : int
        Patch size (must match dictionary)
    savgol_window : int
        Savitzky-Golay window length
    savgol_polyorder : int
        Savitzky-Golay polynomial order
    verbose : bool
        Print progress info

    Returns
    -------
    np.ndarray
        Denoised signal, same length as input
    """
    n_original = len(noisy_signal)

    if verbose:
        print(f'  Signal length: {n_original:,} samples')

    patches_zm, patch_means, n_patches = extract_dense_patches(
        noisy_signal, window_size
    )
    if verbose:
        print(f'  Extracted {n_patches:,} patches (stride=1, size={window_size})')

    recon_patches = sparse_reconstruct(patches_zm, dictionary, sparsity)

    denoised = dense_patch_reconstruct(
        recon_patches, patch_means, n_original, window_size
    )

    denoised = post_process(denoised, savgol_window, savgol_polyorder)

    if verbose:
        print(f'  Denoised signal length: {len(denoised):,}')

    return denoised