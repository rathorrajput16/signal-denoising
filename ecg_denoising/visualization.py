import os
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import norm as sp_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """Configure publication-quality plot aesthetics."""
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 150,
        'font.size': 10, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'legend.fontsize': 8,
        'figure.facecolor': 'white', 'axes.facecolor': '#fafafa',
        'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.0,
    })
    sns.set_palette('deep')


def plot_signals(clean, noisy, denoised_dict, denoised_wavelet,
                 metrics, config):
    """Figure 1: 4-subplot stacked comparison."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t = np.arange(1000) / config['fs']

    axes[0].plot(t, clean[:1000], color='#2196F3', linewidth=0.8)
    axes[0].set_title('Original Clean ECG Signal', fontweight='bold')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t, noisy[:1000], color='#F44336', linewidth=0.6, alpha=0.8)
    axes[1].set_title(
        f'Noisy Signal (SNR={metrics["Noisy Input"]["SNR_dB"]:.1f} dB, '
        f'RMSE={metrics["Noisy Input"]["RMSE"]:.4f})', fontweight='bold')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(t, denoised_dict[:1000], color='#4CAF50', linewidth=0.8,
                 label='Dict Learning')
    axes[2].plot(t, clean[:1000], color='#2196F3', linewidth=0.5,
                 linestyle='--', alpha=0.5, label='Original')
    axes[2].set_title(
        f'Dictionary Learning Denoised '
        f'(SNR={metrics["Dict Learning"]["SNR_dB"]:.1f} dB, '
        f'RMSE={metrics["Dict Learning"]["RMSE"]:.4f})', fontweight='bold')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend(loc='upper right')

    axes[3].plot(t, denoised_wavelet[:1000], color='#FF9800', linewidth=0.8,
                 label='Wavelet')
    axes[3].plot(t, clean[:1000], color='#2196F3', linewidth=0.5,
                 linestyle='--', alpha=0.5, label='Original')
    axes[3].set_title(
        f'Wavelet Denoised '
        f'(SNR={metrics["Wavelet (db6)"]["SNR_dB"]:.1f} dB, '
        f'RMSE={metrics["Wavelet (db6)"]["RMSE"]:.4f})', fontweight='bold')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_signals.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_comparison(clean, noisy, all_denoised, config):
    """Figure 2: All methods overlaid on 500-sample window."""
    fig, ax = plt.subplots(figsize=(14, 6))
    start, end = 200, 700
    t = np.arange(start, end) / config['fs']

    colors = {
        'Noisy Input':    ('#F44336', 0.3, 0.6),
        'Moving Average': ('#9C27B0', 0.8, 1.0),
        'Butterworth':    ('#FF9800', 0.8, 1.0),
        'Wavelet (db6)':  ('#E91E63', 0.9, 1.2),
        'Dict Learning':  ('#4CAF50', 1.0, 1.5),
    }

    ax.plot(t, clean[start:end], color='#2196F3', linewidth=1.5,
            label='Clean (Ground Truth)', zorder=10)

    for name, sig in all_denoised.items():
        clr, alph, lw = colors.get(name, ('#666', 0.7, 0.8))
        ax.plot(t, sig[start:end], color=clr, alpha=alph,
                linewidth=lw, label=name)

    peaks, _ = sp_signal.find_peaks(clean[start:end], distance=100, height=0.5)
    if len(peaks) > 0:
        pt = (peaks + start) / config['fs']
        for p in pt:
            ax.axvline(x=p, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.plot(pt, clean[peaks + start], 'v', color='red',
                markersize=8, label='QRS Peaks', zorder=11)

    ax.set_title('All Denoising Methods — 500-Sample Comparison',
                 fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_dictionary(dictionary, config):
    """Figure 3: First 16 dictionary atoms as 4×4 grid."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.suptitle('Learned Dictionary Atoms (First 16 of 128)',
                 fontweight='bold', fontsize=14)
    for i in range(16):
        ax = axes[i // 4][i % 4]
        ax.plot(dictionary[i], color='#3F51B5', linewidth=1.2)
        ax.set_title(f'Atom {i + 1}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(y=0, color='gray', linewidth=0.3)
        ax.set_facecolor('#f0f0ff')
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_dictionary.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_metrics_bar(metrics_dict, config):
    """Figure 4: Grouped bar chart for SNR and R²."""
    methods = [k for k in metrics_dict if k != 'Noisy Input']
    snr = [metrics_dict[m]['SNR_dB'] for m in methods]
    r2 = [metrics_dict[m]['R2'] for m in methods]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    clrs = ['#9C27B0', '#FF9800', '#E91E63', '#4CAF50']

    bars1 = a1.bar(methods, snr, color=clrs, edgecolor='white')
    a1.set_title('Signal-to-Noise Ratio (dB)', fontweight='bold')
    a1.set_ylabel('SNR (dB)')
    a1.axhline(y=metrics_dict['Noisy Input']['SNR_dB'], color='red',
               linestyle='--', linewidth=1,
               label=f'Noisy ({metrics_dict["Noisy Input"]["SNR_dB"]:.1f} dB)')
    for b, v in zip(bars1, snr):
        a1.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.3,
                f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    a1.legend()
    a1.tick_params(axis='x', rotation=15)

    bars2 = a2.bar(methods, r2, color=clrs, edgecolor='white')
    a2.set_title('Coefficient of Determination (R²)', fontweight='bold')
    a2.set_ylabel('R²')
    a2.set_ylim(0, 1.05)
    for b, v in zip(bars2, r2):
        a2.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    a2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_metrics_bar.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_spectrogram(noisy, denoised, config):
    """Figure 5: Side-by-side spectrograms."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    fs = config['fs']

    f1, t1, S1 = sp_signal.spectrogram(noisy, fs=fs, nperseg=128, noverlap=96)
    a1.pcolormesh(t1, f1, 10 * np.log10(S1 + 1e-12), cmap='viridis',
                  shading='gouraud')
    a1.set_title('Noisy Signal Spectrogram', fontweight='bold')
    a1.set_ylabel('Frequency (Hz)')
    a1.set_xlabel('Time (s)')
    a1.set_ylim(0, 100)

    f2, t2, S2 = sp_signal.spectrogram(denoised, fs=fs, nperseg=128, noverlap=96)
    im = a2.pcolormesh(t2, f2, 10 * np.log10(S2 + 1e-12), cmap='viridis',
                       shading='gouraud')
    a2.set_title('Dict Learning Denoised Spectrogram', fontweight='bold')
    a2.set_ylabel('Frequency (Hz)')
    a2.set_xlabel('Time (s)')
    a2.set_ylim(0, 100)

    fig.colorbar(im, ax=[a1, a2], label='Power (dB)', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_spectrogram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_error_distribution(clean, all_denoised, config):
    """Figure 6: Histogram of residual errors with Gaussian fit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Residual Error Distribution (clean − denoised)',
                 fontweight='bold', fontsize=13)
    colors = {
        'Moving Average': '#9C27B0', 'Butterworth': '#FF9800',
        'Wavelet (db6)': '#E91E63', 'Dict Learning': '#4CAF50',
    }
    for ax, (method, sig) in zip(
        axes.flatten(),
        [(m, s) for m, s in all_denoised.items() if m != 'Noisy Input']
    ):
        ml = min(len(clean), len(sig))
        err = clean[:ml] - sig[:ml]
        ax.hist(err, bins=100, density=True, alpha=0.7,
                color=colors.get(method, '#666'), edgecolor='white', linewidth=0.3)
        mu_e, std_e = np.mean(err), np.std(err)
        x_fit = np.linspace(err.min(), err.max(), 300)
        ax.plot(x_fit, sp_norm.pdf(x_fit, mu_e, std_e),
                color='black', linewidth=1.5, linestyle='--', label='Gaussian fit')
        ax.set_title(f'{method} (μ={mu_e:.4f}, σ={std_e:.4f})', fontsize=10)
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(config['output_dir'], 'plot_error_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def plot_all(clean, noisy, all_denoised, metrics, dictionary, config):
    """Generate all 6 publication-quality plots."""
    os.makedirs(config['output_dir'], exist_ok=True)
    setup_plot_style()

    print(f'\n[PLOTS] Saving 6 plots to {config["output_dir"]}...')
    plot_signals(clean, noisy, all_denoised['Dict Learning'],
                 all_denoised['Wavelet (db6)'], metrics, config)
    plot_comparison(clean, noisy, all_denoised, config)
    plot_dictionary(dictionary, config)
    plot_metrics_bar(metrics, config)
    plot_spectrogram(noisy, all_denoised['Dict Learning'], config)
    plot_error_distribution(clean, all_denoised, config)
    print('[PLOTS COMPLETE] All 6 figures saved.\n')
