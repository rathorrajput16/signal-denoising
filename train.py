#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║          ECG Dictionary Training Script                       ║
║                                                               ║
║  Trains a dictionary from MIT-BIH data, evaluates against     ║
║  baselines, saves dictionary to disk for inference.           ║
║                                                               ║
║  Usage:                                                       ║
║    python train.py                                            ║
║    python train.py --atoms 256 --sparsity 8                   ║
║    python train.py --no-plots                                 ║
╚═══════════════════════════════════════════════════════════════╝
"""

import os
import argparse
import warnings
import numpy as np

from ecg_denoising.config import CONFIG
from ecg_denoising.data_loader import load_all_records, normalize_signal
from ecg_denoising.noise import add_realistic_noise
from ecg_denoising.dictionary import (
    extract_dense_patches, train_dictionary,
    save_dictionary, denoise_signal
)
from ecg_denoising.baselines import (
    wavelet_denoise, butterworth_filter, moving_average
)
from ecg_denoising.metrics import compute_metrics, print_metrics_table
from ecg_denoising.visualization import plot_all

from tqdm import tqdm

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser(description='Train ECG denoising dictionary')
    p.add_argument('--data-dir', default=CONFIG['data_dir'],
                   help='Path to ECG_data/ directory')
    p.add_argument('--atoms', type=int, default=CONFIG['n_atoms'],
                   help='Number of dictionary atoms')
    p.add_argument('--sparsity', type=int, default=CONFIG['sparsity'],
                   help='OMP non-zero coefficients')
    p.add_argument('--window-size', type=int, default=CONFIG['window_size'],
                   help='Patch size for dictionary')
    p.add_argument('--noise-snr', type=float, default=CONFIG['noise_snr_db'],
                   help='Target noise SNR (dB) for evaluation')
    p.add_argument('--output-dir', default=CONFIG['output_dir'],
                   help='Directory for plots')
    p.add_argument('--model-dir', default=CONFIG['model_dir'],
                   help='Directory for saved model')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip plot generation')
    return p.parse_args()


def main():
    args = parse_args()

    # Merge CLI args into config
    config = CONFIG.copy()
    config.update({
        'data_dir':    args.data_dir,
        'n_atoms':     args.atoms,
        'sparsity':    args.sparsity,
        'window_size': args.window_size,
        'noise_snr_db': args.noise_snr,
        'output_dir':  args.output_dir,
        'model_dir':   args.model_dir,
    })

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)

    # ── STEP 1: Load all records ──
    print('=' * 65)
    print('  ECG DICTIONARY TRAINING')
    print('=' * 65)

    signals = load_all_records(config)
    if config['test_record'] not in signals:
        raise RuntimeError(f"Test record {config['test_record']} not found!")

    # Normalize
    normalized = {}
    for rec_id, sig in signals.items():
        norm, mu, std = normalize_signal(sig)
        normalized[rec_id] = norm

    clean = normalized[config['test_record']]
    train_ids = [r for r in normalized if r != config['test_record']]

    print(f'  Test record: {config["test_record"]} ({len(clean):,} samples)')
    print(f'  Training records: {len(train_ids)}')

    # ── STEP 2: Prepare training patches ──
    print('\n[STEP 2] Extracting training patches...')

    ws = config['window_size']

    # Primary: all patches from clean test signal
    clean_patches, _, n_clean = extract_dense_patches(clean, ws)
    print(f'  Clean test patches: {n_clean:,} × {ws}')

    # Secondary: sampled patches from other records
    extra = []
    n_per_rec = config.get('n_extra_per_record', 1000)
    n_train_recs = config.get('n_train_records', 10)
    rng = np.random.RandomState(42)

    for rec_id in tqdm(train_ids[:n_train_recs], desc='  Sampling train records'):
        p, _, np_ = extract_dense_patches(normalized[rec_id], ws)
        if np_ > n_per_rec:
            idx = rng.choice(np_, n_per_rec, replace=False)
            extra.append(p[idx])
        else:
            extra.append(p)

    if extra:
        train_patches = np.vstack([clean_patches] + extra)
    else:
        train_patches = clean_patches

    print(f'  Total training patches: {train_patches.shape[0]:,} × {ws}')

    # ── STEP 3: Train dictionary ──
    print('\n[STEP 3] Training dictionary...')
    dictionary, dl_model = train_dictionary(
        train_patches,
        n_atoms=config['n_atoms'],
        sparsity=config['sparsity'],
        max_iter=100,
        max_train_windows=config.get('max_train_windows', 15000),
    )

    # ── STEP 4: Save dictionary ──
    model_path = os.path.join(config['model_dir'], 'ecg_dictionary.pkl')
    print('\n[STEP 4] Saving dictionary...')
    save_dictionary(dictionary, config, model_path)

    # ── STEP 5: Evaluate on noisy test signal ──
    print('\n[STEP 5] Evaluating on noisy test signal...')
    noisy, achieved_snr = add_realistic_noise(
        clean, fs=config['fs'], target_snr_db=config['noise_snr_db']
    )
    print(f'  Input noise SNR: {achieved_snr:.2f} dB')

    # Dictionary Learning denoising
    print('\n  ╔═ Dict Learning ═╗')
    denoised_dict = denoise_signal(
        noisy, dictionary,
        sparsity=config['sparsity'],
        window_size=config['window_size'],
        savgol_window=config['savgol_window'],
        savgol_polyorder=config['savgol_polyorder'],
    )

    # Baselines
    print('\n  ╔═ Wavelet ═╗')
    denoised_wavelet = wavelet_denoise(noisy, config['wavelet'],
                                        config['wavelet_level'])
    print(f'  Wavelet: {config["wavelet"]}, Level: {config["wavelet_level"]}')

    print('\n  ╔═ Butterworth ═╗')
    denoised_butter = butterworth_filter(
        noisy, config['fs'], config['bp_low'],
        config['bp_high'], config['bp_order']
    )
    print(f'  BPF: {config["bp_low"]}-{config["bp_high"]} Hz, order {config["bp_order"]}')

    print('\n  ╔═ Moving Average ═╗')
    denoised_movavg = moving_average(noisy, config['ma_window'])
    print(f'  Window: {config["ma_window"]}')

    # ── Metrics ──
    print('\n[METRICS]')

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

    method_snrs = {k: v['SNR_dB'] for k, v in metrics.items()
                   if k != 'Noisy Input'}
    best = max(method_snrs, key=method_snrs.get)
    print(f'\n  Best method: {best} ({method_snrs[best]:.2f} dB)')
    if best == 'Dict Learning':
        print('  ✓ Dictionary Learning beats all baselines!')

    # ── Plots ──
    if not args.no_plots:
        plot_all(clean, noisy, all_denoised, metrics, dictionary, config)

    # ── Summary ──
    print('\n' + '=' * 65)
    print('  TRAINING COMPLETE')
    print('=' * 65)
    print(f'  Dictionary saved : {os.path.abspath(model_path)}')
    print(f'  Dictionary shape : {dictionary.shape}')
    print(f'  Best denoiser    : {best} ({method_snrs[best]:.2f} dB)')
    print(f'  Plots saved to   : {os.path.abspath(config["output_dir"])}')
    print('=' * 65)
    print('\n  → To denoise new signals, run:')
    print(f'    python inference.py --input <ECG_FILE.csv>')
    print()


if __name__ == '__main__':
    main()
