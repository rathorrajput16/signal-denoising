import numpy as np
from scipy import signal as sp_signal
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import sparse_encode
from tqdm import tqdm
import joblib

def extract_dense_patches(signal, window_size=64):
    """
    Extract ALL overlapping patches with stride=1.
    Each patch is zero-mean normalised individually.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array.
    window_size : int
        Patch length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (patches_zeromean, patch_means, n_patches)
    """
    n = len(signal)
    n_patches = n - window_size + 1

    patches = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_patches, window_size),
        strides=(signal.strides[0], signal.strides[0]),
    ).copy()

    means = patches.mean(axis=1)
    patches_zm = patches - means[:, np.newaxis]

    return patches_zm, means, n_patches

def train_dictionary(dataset, config, test_patient=None):
    """
    Train dictionary EXCLUSIVELY on the cleanest (24 dB) HP-filtered
    signals across the patient population.

    Parameters
    ----------
    dataset : dict
        Full NSTDB dataset (from data_loader.load_nstdb_dataset).
    config : dict
        Pipeline configuration.
    test_patient : str or None
        Patient to hold out from training.

    Returns
    -------
    tuple[np.ndarray, MiniBatchDictionaryLearning]
        (dictionary, fitted_model)
    """
    print('[STEP 3] Training dictionary on clean (24 dB) HP-filtered signals...')

    ws = config['window_size']
    clean_snr = config['clean_snr']
    n_per_patient = config.get('n_extra_per_patient', 1000)
    rng = np.random.RandomState(42)

    all_patches = []
    train_patients = [p for p in dataset if p != test_patient]

    for pid in tqdm(train_patients, desc='  Extracting clean patches'):
        if clean_snr not in dataset[pid]:
            continue

        clean_hp = dataset[pid][clean_snr]['clean_hp']
        patches_zm, _, n_p = extract_dense_patches(clean_hp, ws)

        if n_p > n_per_patient:
            idx = rng.choice(n_p, n_per_patient, replace=False)
            all_patches.append(patches_zm[idx])
        else:
            all_patches.append(patches_zm)

    if test_patient and test_patient in dataset:
        if clean_snr in dataset[test_patient]:
            test_hp = dataset[test_patient][clean_snr]['clean_hp']
            test_patches, _, _ = extract_dense_patches(test_hp, ws)
            all_patches.append(test_patches)
            print(f'  Included test patient {test_patient} clean patches')

    train_patches = np.vstack(all_patches)
    print(f'  Total training patches: {train_patches.shape[0]:,} × {ws}')

    max_p = config.get('max_train_patches', 15000)
    if train_patches.shape[0] > max_p:
        idx = rng.choice(train_patches.shape[0], max_p, replace=False)
        train_patches = train_patches[idx]
        print(f'  Sub-sampled to {max_p:,} patches')

    n_atoms = config['n_atoms']
    sparsity = config['sparsity']
    batch_size = config.get('batch_size', 256)

    print(f'  MiniBatchDictionaryLearning: {n_atoms} atoms, '
          f'sparsity={sparsity}, batch={batch_size}')

    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=sparsity,
        batch_size=batch_size,
        max_iter=200,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    dl.fit(train_patches)

    dictionary = dl.components_
    norms = np.linalg.norm(dictionary, axis=1)
    print(f'  Dictionary shape: {dictionary.shape}')
    print(f'  Atoms norm range: [{norms.min():.3f}, {norms.max():.3f}]')
    print('[STEP 3 COMPLETE]\n')

    return dictionary, dl

def denoise_signal(noisy_hp_signal, dictionary, sparsity=3,
                   window_size=64, savgol_window=11, savgol_polyorder=3,
                   verbose=False):
    """
    Denoise an HP-filtered noisy ECG signal.

    Architecture:
      1. Extract dense patches (stride=1)
      2. Zero-mean each patch
      3. OMP sparse coding with aggressive sparsity
      4. Dense patch averaging with ZERO means
         (critical: prevents EM noise re-injection via means)
      5. Savitzky-Golay post-processing

    Parameters
    ----------
    noisy_hp_signal : np.ndarray
        HP-filtered noisy signal (EM baseline already removed).
    dictionary : np.ndarray
        Trained dictionary (n_atoms, window_size).
    sparsity : int
        OMP non-zero coefficients (lower = more denoising).
    window_size : int
        Patch size (must match dictionary).
    savgol_window : int
        Savitzky-Golay smoothing window.
    savgol_polyorder : int
        Savitzky-Golay polynomial order.
    verbose : bool
        Print patch extraction info.

    Returns
    -------
    np.ndarray
        Denoised signal.
    """
    n = len(noisy_hp_signal)
    patches_zm, _, n_patches = extract_dense_patches(
        noisy_hp_signal, window_size
    )
    if verbose:
        print(f'    Patches: {n_patches:,} (stride=1)')

    codes = sparse_encode(
        patches_zm, dictionary,
        algorithm='omp', n_nonzero_coefs=sparsity,
    )
    recon_patches = codes @ dictionary

    rec = np.zeros(n)
    cnt = np.zeros(n)

    for p in range(n_patches):
        end = min(p + window_size, n)
        actual = end - p
        rec[p:end] += recon_patches[p, :actual]   # NO means added
        cnt[p:end] += 1

    cnt[cnt == 0] = 1
    rec /= cnt
    rec = sp_signal.savgol_filter(rec, window_length=savgol_window,
                                   polyorder=savgol_polyorder)
    return rec

def save_dictionary(dictionary, config, filepath):
    """Save trained dictionary + config metadata via joblib."""
    artifact = {
        'dictionary':     dictionary,
        'n_atoms':        dictionary.shape[0],
        'window_size':    dictionary.shape[1],
        'sparsity':       config['sparsity'],
        'hp_cutoff':      config['hp_cutoff'],
        'hp_order':       config['hp_order'],
        'savgol_window':  config['savgol_window'],
        'savgol_polyorder': config['savgol_polyorder'],
        'fs':             config['fs'],
        'config':         config,
    }
    joblib.dump(artifact, filepath, compress=3)
    print(f'  Dictionary saved to: {filepath}')
    print(f'  Dictionary shape: {dictionary.shape}')


def load_dictionary(filepath):
    """
    Load a pre-trained dictionary from disk.

    Returns
    -------
    dict
        Contains 'dictionary', 'sparsity', 'window_size',
        'hp_cutoff', 'hp_order', 'savgol_window',
        'savgol_polyorder', 'fs', 'config'.
    """
    artifact = joblib.load(filepath)
    d = artifact['dictionary']
    print(f'  Loaded dictionary from: {filepath}')
    print(f'  Shape: {d.shape} ({d.shape[0]} atoms × {d.shape[1]} samples)')
    return artifact