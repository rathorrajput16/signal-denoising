import os
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import norm as sp_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def setup_plot_style():
    """Configure publication-quality aesthetics."""
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 150,
        'font.size': 10, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'legend.fontsize': 9,
        'figure.facecolor': 'white', 'axes.facecolor': '#fafafa',
        'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.2,
    })
    sns.set_palette('deep')

def plot_degradation_curve(results_by_snr, config):
    """Output SNR & RMSE vs calibrated Input SNR for all methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    methods = ['Dict Learning', 'Wavelet (db6)', 'Butterworth']
    colors = {'Dict Learning': '#4CAF50', 'Wavelet (db6)': '#E91E63',
              'Butterworth': '#FF9800'}
    markers = {'Dict Learning': 'o', 'Wavelet (db6)': 's',
               'Butterworth': '^'}

    snr_levels = sorted(results_by_snr.keys())

    for method in methods:
        vals = [results_by_snr[s].get(method, {}).get('SNR_dB', np.nan)
                for s in snr_levels]
        ax1.plot(snr_levels, vals, color=colors[method],
                 marker=markers[method], linewidth=2.5, markersize=9,
                 label=method, zorder=5)

    inp = [results_by_snr[s]['Noisy Input']['SNR_dB'] for s in snr_levels]
    ax1.plot(snr_levels, inp, color='gray', linestyle='--', linewidth=1.5,
             marker='x', markersize=6, label='Noisy Input', alpha=0.7)

    ax1.set_xlabel('Calibrated Input SNR (dB)', fontweight='bold')
    ax1.set_ylabel('Output SNR (dB)', fontweight='bold')
    ax1.set_title('Denoising Performance vs EM Noise Level',
                  fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_xticks(snr_levels)

    cross_idx = None
    for i, s in enumerate(snr_levels):
        d_snr = results_by_snr[s].get('Dict Learning', {}).get(
            'SNR_dB', float('-inf'))
        b_snr = results_by_snr[s].get('Butterworth', {}).get(
            'SNR_dB', float('-inf'))
        if d_snr > b_snr and cross_idx is None:
            cross_idx = i
    if cross_idx is not None:
        ax1.axvspan(snr_levels[0] - 1, snr_levels[cross_idx],
                    alpha=0.08, color='green',
                    label='Dict Learning Dominance')

    for method in methods:
        vals = [results_by_snr[s].get(method, {}).get('RMSE', np.nan)
                for s in snr_levels]
        ax2.plot(snr_levels, vals, color=colors[method],
                 marker=markers[method], linewidth=2.5, markersize=9,
                 label=method)

    inp_rmse = [results_by_snr[s]['Noisy Input']['RMSE'] for s in snr_levels]
    ax2.plot(snr_levels, inp_rmse, color='gray', linestyle='--',
             linewidth=1.5, marker='x', markersize=6,
             label='Noisy Input', alpha=0.7)

    ax2.set_xlabel('Calibrated Input SNR (dB)', fontweight='bold')
    ax2.set_ylabel('RMSE', fontweight='bold')
    ax2.set_title('Reconstruction Error vs EM Noise Level',
                  fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_xticks(snr_levels)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_degradation_curve.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_extreme_stress_test(clean_hp, noisy, butter_out, dict_out,
                              snr_level, config):
    """4-subplot comparison at extreme noise level."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    n_show = min(1000, len(clean_hp))
    t = np.arange(n_show) / config['fs']

    axes[0].plot(t, clean_hp[:n_show], color='#2196F3', linewidth=0.9)
    axes[0].set_title('Ground Truth (24 dB — Cleanest)', fontweight='bold')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t, noisy[:n_show], color='#F44336', linewidth=0.6,
                 alpha=0.8)
    axes[1].set_title(f'Noisy Input ({snr_level} dB Real EM Artifact)',
                      fontweight='bold')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(t, butter_out[:n_show], color='#FF9800', linewidth=0.9,
                 label='Butterworth')
    axes[2].plot(t, clean_hp[:n_show], color='#2196F3', linewidth=0.4,
                 linestyle='--', alpha=0.4, label='Ground Truth')
    axes[2].set_title('Butterworth Bandpass (0.5–40 Hz)', fontweight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend(loc='upper right', fontsize=8)

    axes[3].plot(t, dict_out[:n_show], color='#4CAF50', linewidth=0.9,
                 label='Dict Learning')
    axes[3].plot(t, clean_hp[:n_show], color='#2196F3', linewidth=0.4,
                 linestyle='--', alpha=0.4, label='Ground Truth')
    axes[3].set_title('Dictionary Learning (128 atoms, OMP, sparsity=3)',
                      fontweight='bold')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Extreme Stress Test — {snr_level} dB Real EM Noise '
                 f'(Patient {config["test_patient"]})',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_extreme_stress.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_all_snr_levels(clean_hp, noisy_by_snr, dict_by_snr, config):
    """Multi-panel showing noisy vs dict-denoised at ALL SNR levels."""
    snr_levels = sorted(noisy_by_snr.keys(), reverse=True)
    n_levels = len(snr_levels)

    fig, axes = plt.subplots(n_levels + 1, 1,
                              figsize=(14, 3 * (n_levels + 1)),
                              sharex=True)

    n_show = min(1500, len(clean_hp))
    t = np.arange(n_show) / config['fs']

    axes[0].plot(t, clean_hp[:n_show], color='#2196F3', linewidth=0.8)
    axes[0].set_title('Ground Truth (24 dB HP-filtered)', fontweight='bold')
    axes[0].set_ylabel('Amplitude')

    for i, snr in enumerate(snr_levels):
        ax = axes[i + 1]
        noisy = noisy_by_snr[snr]
        denoised = dict_by_snr.get(snr)

        ax.plot(t, noisy[:n_show], color='#F44336', linewidth=0.4,
                alpha=0.5, label=f'Noisy ({snr} dB)')
        if denoised is not None:
            ax.plot(t, denoised[:n_show], color='#4CAF50', linewidth=0.8,
                    label='Dict Denoised')
        ax.set_title(f'{snr} dB — EM Artifact', fontweight='bold')
        ax.set_ylabel('Amp')
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Multi-Level Stress Test — Patient {config["test_patient"]}',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_all_snr_levels.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_dictionary_atoms(dictionary, config):
    """Plot first 16 learned dictionary atoms."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle(
        f'Learned Dictionary Atoms (First 16 of {dictionary.shape[0]}) '
        f'— Trained on HP-filtered Clean ECG',
        fontweight='bold', fontsize=13)
    for i in range(min(16, dictionary.shape[0])):
        ax = axes[i // 4][i % 4]
        ax.plot(dictionary[i], color='#3F51B5', linewidth=1.2)
        ax.set_title(f'Atom {i + 1}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(y=0, color='gray', linewidth=0.3)
        ax.set_facecolor('#f0f0ff')
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_dictionary_atoms.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_snr_improvement_heatmap(results_by_snr, config):
    """Heatmap of SNR improvement over noisy input."""
    methods = ['Dict Learning', 'Wavelet (db6)', 'Butterworth']
    snr_levels = sorted(results_by_snr.keys(), reverse=True)

    improvement = np.zeros((len(methods), len(snr_levels)))
    for j, snr in enumerate(snr_levels):
        input_snr = results_by_snr[snr]['Noisy Input']['SNR_dB']
        for i, m in enumerate(methods):
            out_snr = results_by_snr[snr].get(m, {}).get('SNR_dB', np.nan)
            improvement[i, j] = out_snr - input_snr

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(improvement, cmap='RdYlGn', aspect='auto',
                   vmin=min(-2, np.nanmin(improvement)),
                   vmax=np.nanmax(improvement) + 1)
    ax.set_xticks(range(len(snr_levels)))
    ax.set_xticklabels([f'{s} dB' for s in snr_levels])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel('Calibrated Input SNR')
    ax.set_title('SNR Improvement Over Noisy Input (dB)',
                 fontweight='bold', fontsize=13)

    for i in range(len(methods)):
        for j in range(len(snr_levels)):
            val = improvement[i, j]
            color = 'white' if abs(val) > 3 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, label='SNR Improvement (dB)', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_snr_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_spectrograms(clean_hp, noisy_hp, denoised, snr_level, config):
    """Triple spectrogram — clean, noisy, denoised."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    fs = config['fs']

    for ax, sig, title, cmap in [
        (a1, clean_hp, 'Ground Truth (24 dB)', 'viridis'),
        (a2, noisy_hp, f'Noisy ({snr_level} dB)', 'inferno'),
        (a3, denoised, 'Dict Denoised', 'viridis'),
    ]:
        f, t, S = sp_signal.spectrogram(sig, fs=fs, nperseg=128, noverlap=96)
        ax.pcolormesh(t, f, 10 * np.log10(S + 1e-12), cmap=cmap,
                      shading='gouraud')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0, 100)

    fig.suptitle(f'Spectrogram — {snr_level} dB EM Noise',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_spectrograms.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_error_distributions(clean_hp, denoised_signals, snr_level, config):
    """Residual error distributions for all methods."""
    methods_colors = {
        'Butterworth':   '#FF9800',
        'Wavelet (db6)': '#E91E63',
        'Dict Learning': '#4CAF50',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'Residual Error Distribution at {snr_level} dB EM Noise',
                 fontweight='bold', fontsize=13)

    for ax, (method, color) in zip(axes, methods_colors.items()):
        sig = denoised_signals.get(method)
        if sig is None:
            continue

        ml = min(len(clean_hp), len(sig))
        err = clean_hp[:ml] - sig[:ml]
        ax.hist(err, bins=100, density=True, alpha=0.7, color=color,
                edgecolor='white', linewidth=0.3)
        mu_e, std_e = err.mean(), err.std()
        x_fit = np.linspace(err.min(), err.max(), 300)
        ax.plot(x_fit, sp_norm.pdf(x_fit, mu_e, std_e), 'k--',
                linewidth=1.5, label='Gaussian fit')
        ax.set_title(f'{method}\n(μ={mu_e:.4f}, σ={std_e:.4f})', fontsize=10)
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_error_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_metrics_bar(results_by_snr, stress_snr, config):
    """Bar chart comparing all 5 metrics at a given SNR level."""
    methods = ['Butterworth', 'Wavelet (db6)', 'Dict Learning']
    colors = ['#FF9800', '#E91E63', '#4CAF50']

    mr = results_by_snr[stress_snr]

    metrics = ['SNR_dB', 'RMSE', 'PRD', 'SSIM', 'R2']
    titles = ['SNR (dB)', 'RMSE', 'PRD (%)', 'SSIM', 'R²']

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'Metrics Comparison at {stress_snr} dB EM Noise',
                 fontweight='bold', fontsize=13, y=1.02)

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [mr.get(m, {}).get(metric, 0) for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=25)

        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.,
                    b.get_height() + 0.01 * abs(b.get_height()),
                    f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_metrics_bar.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')

def plot_all(ground_truth_hp, noisy_hp_by_snr, dict_outputs,
             denoised_signals_by_snr, results_by_snr, dictionary, config):
    """
    Generate all 8 plots.

    Parameters
    ----------
    ground_truth_hp : np.ndarray
        HP-filtered ground truth (24 dB) signal.
    noisy_hp_by_snr : dict
        {snr: noisy_hp_signal}
    dict_outputs : dict
        {snr: dict_denoised_signal}
    denoised_signals_by_snr : dict
        {snr: {'Butterworth': ..., 'Wavelet (db6)': ..., 'Dict Learning': ...}}
    results_by_snr : dict
        {snr: {method: metrics_dict}}
    dictionary : np.ndarray
        Trained dictionary.
    config : dict
        Pipeline configuration.
    """
    setup_plot_style()
    print(f'[PLOTS] Generating 8 visualisations to {config["output_dir"]}...')

    stress_snr = 0 if 0 in noisy_hp_by_snr else min(noisy_hp_by_snr.keys())

    plot_degradation_curve(results_by_snr, config)
    plot_extreme_stress_test(
        ground_truth_hp,
        noisy_hp_by_snr[stress_snr],
        denoised_signals_by_snr[stress_snr]['Butterworth'],
        dict_outputs[stress_snr],
        stress_snr, config,
    )
    plot_all_snr_levels(ground_truth_hp, noisy_hp_by_snr, dict_outputs, config)
    plot_dictionary_atoms(dictionary, config)
    plot_snr_improvement_heatmap(results_by_snr, config)
    plot_spectrograms(ground_truth_hp, noisy_hp_by_snr[stress_snr],
                       dict_outputs[stress_snr], stress_snr, config)
    plot_error_distributions(ground_truth_hp,
                              denoised_signals_by_snr[stress_snr],
                              stress_snr, config)
    plot_metrics_bar(results_by_snr, stress_snr, config)

    print('[PLOTS COMPLETE]\n')
