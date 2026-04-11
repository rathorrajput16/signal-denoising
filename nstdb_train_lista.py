import os
import sys
import argparse
import warnings
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nstdb_denoising import (
    CONFIG,
    load_nstdb_dataset, highpass_filter,
    extract_dense_patches, denoise_signal,
    butterworth_filter, wavelet_denoise,
    compute_all_metrics, print_results_table,
    setup_plot_style,
)
from nstdb_denoising.lista_model import (
    learn_initial_dictionary, prepare_patches_for_lista,
    build_lista_model, denoise_signal_lista,
    save_lista_model,
)
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description='CNN-LISTA Training & Evaluation for NSTDB ECG Denoising',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir', type=str, default=CONFIG['data_dir'],
                   help='Path to NSTDB .mat directory')
    p.add_argument('--model_dir', type=str, default=CONFIG['model_dir'],
                   help='Directory to save LISTA model')
    p.add_argument('--output_dir', type=str, default='./lista_plots/',
                   help='Directory for output plots')
    p.add_argument('--test_patient', type=str,
                   default=CONFIG['test_patient'],
                   help='Patient ID held out for testing')
    p.add_argument('--num_atoms', type=int,
                   default=CONFIG['lista_num_atoms'],
                   help='LISTA sparse code dimension')
    p.add_argument('--iterations', type=int,
                   default=CONFIG['lista_iterations'],
                   help='Unrolled ISTA iterations')
    p.add_argument('--epochs', type=int,
                   default=CONFIG['lista_epochs'],
                   help='Training epochs')
    p.add_argument('--batch_size', type=int,
                   default=CONFIG['lista_batch_size'],
                   help='Training batch size')
    p.add_argument('--lr', type=float,
                   default=CONFIG['lista_lr'],
                   help='Adam learning rate')
    p.add_argument('--sparsity_penalty', type=float,
                   default=CONFIG['lista_sparsity_penalty'],
                   help='L1 sparsity weight on alpha')
    p.add_argument('--patches_per_signal', type=int,
                   default=CONFIG['lista_patches_per_signal'],
                   help='Max patches sub-sampled per noise-level signal')
    p.add_argument('--dict_max_iter', type=int,
                   default=CONFIG['lista_dict_max_iter'],
                   help='sklearn dictionary learning iterations')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip plot generation')
    return p.parse_args()

def build_config(args):
    config = CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['model_dir'] = args.model_dir
    config['output_dir'] = args.output_dir
    config['test_patient'] = args.test_patient
    config['lista_num_atoms'] = args.num_atoms
    config['lista_iterations'] = args.iterations
    config['lista_epochs'] = args.epochs
    config['lista_batch_size'] = args.batch_size
    config['lista_lr'] = args.lr
    config['lista_sparsity_penalty'] = args.sparsity_penalty
    config['lista_patches_per_signal'] = args.patches_per_signal
    config['lista_dict_max_iter'] = args.dict_max_iter
    return config

def extract_lista_training_data(dataset, config, test_patient=None):
    """
    Extract paired (noisy, clean) zero-mean patches for supervised
    LISTA training. Patches are aligned position-by-position.

    Training data comes from ALL noise levels (excluding 24 dB ground
    truth) across all patients EXCEPT the held-out test patient.
    """
    ws = config['window_size']
    clean_snr = config['clean_snr']
    patches_per_signal = config.get('lista_patches_per_signal', 2000)
    rng = np.random.RandomState(42)

    train_patients = [p for p in sorted(dataset.keys())
                      if p != test_patient]
    noise_levels = sorted(
        [s for s in dataset[train_patients[0]].keys() if s != clean_snr],
    )

    print(f'  Training patients : {len(train_patients)}')
    print(f'  Noise levels      : {noise_levels}')
    print(f'  Patches per signal: {patches_per_signal}')

    all_noisy, all_clean = [], []

    for pid in tqdm(train_patients, desc='  Extracting patches'):
        for snr in noise_levels:
            if snr not in dataset[pid]:
                continue

            entry = dataset[pid][snr]
            noisy_zm, _, n_p = extract_dense_patches(
                entry['noisy_hp'], ws)
            clean_zm, _, _ = extract_dense_patches(
                entry['clean_hp'], ws)

            if n_p > patches_per_signal:
                idx = rng.choice(n_p, patches_per_signal, replace=False)
                noisy_zm = noisy_zm[idx]
                clean_zm = clean_zm[idx]

            all_noisy.append(noisy_zm)
            all_clean.append(clean_zm)

    X = np.vstack(all_noisy)
    Y = np.vstack(all_clean)

    perm = rng.permutation(X.shape[0])
    X, Y = X[perm], Y[perm]

    print(f'  Total patches     : {X.shape[0]:,} × {ws}')
    return X, Y

def plot_training_history(history, config):
    """Training loss / MSE / L1 curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('loss', 'val_loss', 'Total Loss (MSE + L1)'),
        ('mse', 'val_mse', 'MSE Reconstruction'),
        ('l1_sparsity', 'val_l1_sparsity', 'L1 Sparsity Penalty'),
    ]

    for ax, (tk, vk, title) in zip(axes, metrics):
        if tk in history.history:
            ax.plot(history.history[tk], label='Train',
                    linewidth=2, color='#2196F3')
        if vk in history.history:
            ax.plot(history.history[vk], label='Val',
                    linewidth=2, linestyle='--', color='#F44336')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title.split('(')[0].strip())
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('CNN-LISTA Training Curves', fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'lista_training_curves.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'    Saved: {path}')

def plot_lista_stress_test(ground_truth_hp, noisy_hp, lista_out, omp_out,
                           butter_out, wavelet_out, snr_level, config):
    """5-panel comparison at a given SNR level."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    n_show = min(1500, len(ground_truth_hp))
    t = np.arange(n_show) / config['fs']

    panels = [
        (ground_truth_hp, 'Ground Truth (24 dB HP-filtered)', '#2196F3'),
        (noisy_hp, f'Noisy Input ({snr_level} dB EM Artifact)', '#F44336'),
        (butter_out, 'Butterworth BPF', '#FF9800'),
        (omp_out, 'OMP Sparse Coding (64 atoms)', '#9C27B0'),
        (lista_out, 'CNN-LISTA (Deep Unrolled)', '#4CAF50'),
    ]

    for ax, (sig, title, color) in zip(axes, panels):
        ax.plot(t, sig[:n_show], color=color, linewidth=0.8)
        if sig is not ground_truth_hp:
            ax.plot(t, ground_truth_hp[:n_show], color='#2196F3',
                    linewidth=0.3, linestyle='--', alpha=0.3)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Amplitude')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'LISTA vs Classical Methods — {snr_level} dB EM Noise '
                 f'(Patient {config["test_patient"]})',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'lista_stress_test.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'    Saved: {path}')

def plot_lista_degradation(results_by_snr, config):
    """Output SNR / RMSE vs Input SNR with LISTA included."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods_style = {
        'CNN-LISTA':      ('#4CAF50', 'D', 2.5),
        'OMP (64 atoms)': ('#9C27B0', 'o', 2.0),
        'Wavelet (db6)':  ('#E91E63', 's', 2.0),
        'Butterworth':    ('#FF9800', '^', 2.0),
    }
    snr_levels = sorted(results_by_snr.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for method, (color, marker, lw) in methods_style.items():
        snr_vals = [results_by_snr[s].get(method, {}).get(
            'SNR_dB', np.nan) for s in snr_levels]
        rmse_vals = [results_by_snr[s].get(method, {}).get(
            'RMSE', np.nan) for s in snr_levels]
        ax1.plot(snr_levels, snr_vals, color=color, marker=marker,
                 linewidth=lw, markersize=9, label=method, zorder=5)
        ax2.plot(snr_levels, rmse_vals, color=color, marker=marker,
                 linewidth=lw, markersize=9, label=method)

    inp_snr = [results_by_snr[s]['Noisy Input']['SNR_dB']
               for s in snr_levels]
    ax1.plot(snr_levels, inp_snr, 'k--', linewidth=1.5, marker='x',
             markersize=6, label='Noisy Input', alpha=0.5)

    ax1.set_xlabel('Input SNR (dB)', fontweight='bold')
    ax1.set_ylabel('Output SNR (dB)', fontweight='bold')
    ax1.set_title('Denoising Performance: LISTA vs Classical',
                  fontweight='bold', fontsize=13)
    ax1.legend(framealpha=0.9)
    ax1.set_xticks(snr_levels)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Input SNR (dB)', fontweight='bold')
    ax2.set_ylabel('RMSE', fontweight='bold')
    ax2.set_title('Reconstruction Error', fontweight='bold', fontsize=13)
    ax2.legend(framealpha=0.9)
    ax2.set_xticks(snr_levels)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'lista_degradation_curve.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'    Saved: {path}')

def plot_lista_metrics_bar(results_by_snr, stress_snr, config):
    """Bar chart comparing all metrics at a specific SNR level."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = ['Butterworth', 'Wavelet (db6)', 'OMP (64 atoms)', 'CNN-LISTA']
    colors = ['#FF9800', '#E91E63', '#9C27B0', '#4CAF50']
    metric_keys = ['SNR_dB', 'RMSE', 'PRD', 'SSIM', 'R2']
    metric_labels = ['SNR (dB)', 'RMSE', 'PRD (%)', 'SSIM', 'R²']

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(20, 5))
    mr = results_by_snr[stress_snr]

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        vals = [mr.get(m, {}).get(mk, 0) for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor='black',
                      linewidth=0.5)
        ax.set_title(ml, fontweight='bold')
        ax.set_ylabel(ml)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle(f'Method Comparison at {stress_snr} dB EM Noise',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'lista_metrics_bar.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'    Saved: {path}')

def main():
    args = parse_args()
    config = build_config(args)

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)

    print('=' * 70)
    print('  CNN-LISTA TRAINING — NSTDB ECG Denoising')
    print('  Architecture: HP Pre-filter + Deep Unrolled ISTA')
    print('  Decoder: Frozen sklearn Dictionary')
    print('=' * 70)

    # ── STEP 1: Load dataset ──
    dataset = load_nstdb_dataset(config['data_dir'], config)

    test_patient = config['test_patient']
    if test_patient not in dataset:
        test_patient = sorted(dataset.keys())[0]
        config['test_patient'] = test_patient
        print(f'  ⚠ Fallback test patient: {test_patient}')

    clean_snr = config['clean_snr']
    ground_truth_hp = dataset[test_patient][clean_snr]['clean_hp']
    print(f'  Test patient : {test_patient}')
    print(f'  Ground truth : {clean_snr} dB ({len(ground_truth_hp):,} samples)')

    # ── STEP 2: Extract supervised training patches ──
    print('\n[STEP 2] Extracting paired (noisy→clean) patches...')
    X_patches, Y_patches = extract_lista_training_data(
        dataset, config, test_patient=test_patient,
    )

    # ── STEP 3: Learn sklearn dictionary for decoder init ──
    print('\n[STEP 3] Learning sklearn dictionary for decoder init...')
    num_atoms = config['lista_num_atoms']
    sklearn_dict = learn_initial_dictionary(
        Y_patches, n_components=num_atoms,
        max_iter=config['lista_dict_max_iter'],
    )

    # ── STEP 4: Build CNN-LISTA model ──
    print('\n[STEP 4] Building CNN-LISTA model...')
    ws = config['window_size']

    # Reshape for TF Conv1D: (N, 64) → (N, 64, 1)
    X_train, Y_train = prepare_patches_for_lista(X_patches, Y_patches)
    print(f'  X_train: {X_train.shape}  (Conv1D format)')
    print(f'  Y_train: {Y_train.shape}')

    # Compute total training steps for cosine LR schedule
    n_train = int(X_train.shape[0] * 0.9)  # 90% train after val_split
    steps_per_epoch = n_train // config['lista_batch_size']
    total_steps = steps_per_epoch * config['lista_epochs']

    model = build_lista_model(
        dictionary_weights=sklearn_dict,
        input_dim=ws,
        num_atoms=num_atoms,
        iterations=config['lista_iterations'],
        sparsity_penalty=config['lista_sparsity_penalty'],
        lr=config['lista_lr'],
        total_steps=total_steps,
    )

    # ── STEP 5: Train ──
    print('\n[STEP 5] Training CNN-LISTA...')

    import tensorflow as tf
    best_weights_path = os.path.join(
        config['model_dir'], 'lista_best.weights.h5',
    )
    callbacks = [
        # Save the best model weights (by val loss)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        # Stop early if val_loss stalls for 8 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, Y_train,
        batch_size=config['lista_batch_size'],
        epochs=config['lista_epochs'],
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Restore best weights (EarlyStopping does this, but be explicit)
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
        print(f'  Loaded best weights from: {best_weights_path}')

    print('[STEP 5 COMPLETE]\n')

    # ── Save model ──
    print('[SAVE] Persisting LISTA model...')
    save_lista_model(model, sklearn_dict, config, config['model_dir'])

    # ── STEP 6: Multi-level stress testing ──
    print('\n[STEP 6] Multi-level stress testing...')
    print(f'  Test patient: {test_patient}')
    print(f'  Reference   : {clean_snr} dB HP-filtered signal\n')

    results_by_snr = {}
    lista_outputs = {}
    omp_outputs = {}
    noisy_hp_by_snr = {}
    denoised_signals_by_snr = {}

    available_snrs = sorted(
        [s for s in dataset[test_patient] if s != clean_snr], reverse=True,
    )
    print(f'  Testing SNR levels: {available_snrs}\n')

    for snr_level in tqdm(available_snrs, desc='  Stress testing'):
        entry = dataset[test_patient][snr_level]
        noisy_hp = entry['noisy_hp']
        noisy_raw = entry['noisy']
        noisy_hp_by_snr[snr_level] = noisy_hp

        input_metrics = compute_all_metrics(ground_truth_hp, noisy_hp)

        # CNN-LISTA
        lista_denoised = denoise_signal_lista(
            noisy_hp, model,
            window_size=ws,
            savgol_window=config['savgol_window'],
            savgol_polyorder=config['savgol_polyorder'],
        )
        lista_metrics = compute_all_metrics(ground_truth_hp, lista_denoised)
        lista_outputs[snr_level] = lista_denoised

        # OMP (64 atoms, same sklearn dictionary)
        omp_denoised = denoise_signal(
            noisy_hp, sklearn_dict,
            sparsity=3,
            window_size=ws,
            savgol_window=config['savgol_window'],
            savgol_polyorder=config['savgol_polyorder'],
        )
        omp_metrics = compute_all_metrics(ground_truth_hp, omp_denoised)
        omp_outputs[snr_level] = omp_denoised

        # Butterworth
        butter_out = butterworth_filter(
            noisy_raw, config['fs'],
            config['bp_low'], config['bp_high'], config['bp_order'],
        )
        butter_hp = highpass_filter(butter_out, config['fs'],
                                     config['hp_cutoff'], config['hp_order'])
        butter_metrics = compute_all_metrics(ground_truth_hp, butter_hp)

        # Wavelet
        wav_out = wavelet_denoise(
            noisy_raw, config['wavelet'], config['wavelet_level'],
        )
        wav_hp = highpass_filter(wav_out, config['fs'],
                                  config['hp_cutoff'], config['hp_order'])
        wav_metrics = compute_all_metrics(ground_truth_hp, wav_hp)

        results_by_snr[snr_level] = {
            'Noisy Input':    input_metrics,
            'Butterworth':    butter_metrics,
            'Wavelet (db6)':  wav_metrics,
            'OMP (64 atoms)': omp_metrics,
            'CNN-LISTA':      lista_metrics,
        }

        denoised_signals_by_snr[snr_level] = {
            'Butterworth':    butter_hp,
            'Wavelet (db6)':  wav_hp,
            'OMP (64 atoms)': omp_denoised,
            'CNN-LISTA':      lista_denoised,
        }

        tqdm.write(
            f'    {snr_level:3d} dB │ '
            f'LISTA: {lista_metrics["SNR_dB"]:6.2f} │ '
            f'OMP: {omp_metrics["SNR_dB"]:6.2f} │ '
            f'Wav: {wav_metrics["SNR_dB"]:6.2f} │ '
            f'BPF: {butter_metrics["SNR_dB"]:6.2f}'
        )

    print('\n[STEP 6 COMPLETE]\n')

    # ── Results table ──
    methods = ['Noisy Input', 'Butterworth', 'Wavelet (db6)',
               'OMP (64 atoms)', 'CNN-LISTA']
    print_results_table(results_by_snr, methods)

    # ── Per-level winners ──
    dm = ['Butterworth', 'Wavelet (db6)', 'OMP (64 atoms)', 'CNN-LISTA']
    print('\n  Per-level winners:')
    lista_wins = 0
    for snr in sorted(results_by_snr.keys(), reverse=True):
        best = max(dm, key=lambda m: results_by_snr[snr].get(
            m, {}).get('SNR_dB', float('-inf')))
        best_snr = results_by_snr[snr][best]['SNR_dB']
        marker = '★' if best == 'CNN-LISTA' else ' '
        if best == 'CNN-LISTA':
            lista_wins += 1
        print(f'    {snr:3d} dB → {best} ({best_snr:.2f} dB) {marker}')

    # ── Plots ──
    if not args.no_plots:
        print('\n[PLOTS] Generating visualizations...')
        setup_plot_style()
        plot_training_history(history, config)
        plot_lista_degradation(results_by_snr, config)

        stress_snr = 0 if 0 in available_snrs else available_snrs[-1]
        plot_lista_stress_test(
            ground_truth_hp,
            noisy_hp_by_snr[stress_snr],
            lista_outputs[stress_snr],
            omp_outputs[stress_snr],
            denoised_signals_by_snr[stress_snr]['Butterworth'],
            denoised_signals_by_snr[stress_snr]['Wavelet (db6)'],
            stress_snr, config,
        )
        plot_lista_metrics_bar(results_by_snr, stress_snr, config)
        print('[PLOTS COMPLETE]')

    # ── Summary ──
    print('\n' + '=' * 70)
    print('  CNN-LISTA TRAINING COMPLETE')
    print('=' * 70)
    print(f'  Dataset           : NSTDB ({len(dataset)} patients)')
    print(f'  Test patient      : {test_patient}')
    print(f'  LISTA architecture: {ws}→{num_atoms} atoms, '
          f'{config["lista_iterations"]} iterations')
    print(f'  Epochs            : {config["lista_epochs"]}')
    print(f'  Training patches  : {X_train.shape[0]:,}')
    print(f'  LISTA wins        : {lista_wins}/{len(available_snrs)} levels')
    print(f'  Model saved to    : {os.path.abspath(config["model_dir"])}')
    print(f'  Plots saved to    : {os.path.abspath(config["output_dir"])}')
    print('=' * 70)

if __name__ == '__main__':
    main()
