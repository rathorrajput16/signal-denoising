import os
import sys
import argparse
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from nstdb_denoising import (
    CONFIG,
    load_mat_signal, highpass_filter, parse_filename,
    load_dictionary, denoise_signal,
    butterworth_filter, wavelet_denoise,
    compute_all_metrics,
)

def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description='Denoise an ECG signal using a pre-trained dictionary',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', type=str, required=True,
                   help='Input signal: NSTDB .mat file or .npy array')
    p.add_argument('--model', type=str,
                   default='./models/nstdb_dictionary.pkl',
                   help='Path to pre-trained dictionary (.pkl)')
    p.add_argument('--output_dir', type=str,
                   default='./inference_output/',
                   help='Directory for output files and plots')
    p.add_argument('--channel', type=int, default=0,
                   help='ECG channel index (for .mat files)')
    p.add_argument('--n_samples', type=int, default=10000,
                   help='Number of samples to process')
    p.add_argument('--offset', type=int, default=100000,
                   help='Sample offset (for .mat files, skips preamble)')
    p.add_argument('--compare', action='store_true',
                   help='Compare against Butterworth and Wavelet baselines')
    p.add_argument('--save_npy', action='store_true',
                   help='Save denoised signal as .npy file')
    return p.parse_args()

def load_input_signal(args):
    """
    Load the input signal based on file extension.

    Returns
    -------
    tuple
        (noisy_signal, clean_signal_or_None, signal_info_dict)
    """
    filepath = args.input
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.mat':
        clean, noisy, labeled_snr = load_mat_signal(
            filepath,
            channel=args.channel,
            n_samples=args.n_samples,
            offset=args.offset,
        )
        patient_id, snr_level = parse_filename(filepath)
        info = {
            'source': 'NSTDB .mat',
            'patient_id': patient_id,
            'snr_level': snr_level,
            'labeled_snr': labeled_snr,
            'filename': os.path.basename(filepath),
        }
        return noisy, clean, info

    elif ext == '.npy':
        noisy = np.load(filepath).astype(np.float64)
        if noisy.ndim > 1:
            noisy = noisy[:, args.channel]
        if args.n_samples and len(noisy) > args.n_samples:
            noisy = noisy[:args.n_samples]
        info = {
            'source': 'NumPy .npy',
            'filename': os.path.basename(filepath),
            'patient_id': 'unknown',
            'snr_level': None,
        }
        return noisy, None, info

    else:
        raise ValueError(f'Unsupported file format: {ext}. '
                         f'Use .mat (NSTDB) or .npy')

def plot_inference_result(clean_hp, noisy_hp, denoised, info, config,
                          baselines=None, metrics=None, output_dir='.'):
    """
    Generate a visual comparison plot for inference results.

    Parameters
    ----------
    clean_hp : np.ndarray or None
        HP-filtered clean ground truth (if available).
    noisy_hp : np.ndarray
        HP-filtered noisy input.
    denoised : np.ndarray
        Dictionary-denoised output.
    info : dict
        Signal metadata.
    config : dict
        Pipeline config (for fs).
    baselines : dict or None
        {'Butterworth': signal, 'Wavelet': signal}
    metrics : dict or None
        {method_name: metrics_dict}
    output_dir : str
        Where to save the plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fs = config['fs']
    has_clean = clean_hp is not None
    n_panels = 2 + (1 if has_clean else 0)
    if baselines:
        n_panels += len(baselines)

    fig, axes = plt.subplots(n_panels, 1,
                              figsize=(14, 3 * n_panels), sharex=True)

    n_show = min(2000, len(noisy_hp))
    t = np.arange(n_show) / fs
    panel = 0
    if has_clean:
        axes[panel].plot(t, clean_hp[:n_show], color='#2196F3',
                         linewidth=0.9)
        title = 'Ground Truth (24 dB)'
        if metrics and 'Noisy Input' in metrics:
            title += f'  |  Input SNR: {metrics["Noisy Input"]["SNR_dB"]:.2f} dB'
        axes[panel].set_title(title, fontweight='bold')
        axes[panel].set_ylabel('Amplitude')
        panel += 1

    axes[panel].plot(t, noisy_hp[:n_show], color='#F44336',
                     linewidth=0.6, alpha=0.8)
    snr_label = f'{info.get("snr_level", "?")} dB ' if info.get('snr_level') else ''
    axes[panel].set_title(f'Noisy Input ({snr_label}EM Artifact)',
                          fontweight='bold')
    axes[panel].set_ylabel('Amplitude')
    panel += 1

    if baselines:
        baseline_colors = {'Butterworth': '#FF9800', 'Wavelet (db6)': '#E91E63'}
        for method, sig in baselines.items():
            ax = axes[panel]
            ax.plot(t, sig[:n_show], color=baseline_colors.get(method, 'gray'),
                    linewidth=0.9, label=method)
            if has_clean:
                ax.plot(t, clean_hp[:n_show], color='#2196F3',
                        linewidth=0.3, linestyle='--', alpha=0.3,
                        label='Ground Truth')
            title = method
            if metrics and method in metrics:
                title += f'  |  SNR: {metrics[method]["SNR_dB"]:.2f} dB'
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right', fontsize=7)
            panel += 1

    ax = axes[panel]
    ax.plot(t, denoised[:n_show], color='#4CAF50', linewidth=0.9,
            label='Dict Learning')
    if has_clean:
        ax.plot(t, clean_hp[:n_show], color='#2196F3',
                linewidth=0.3, linestyle='--', alpha=0.3,
                label='Ground Truth')
    title = 'Dictionary Learning (denoised)'
    if metrics and 'Dict Learning' in metrics:
        title += f'  |  SNR: {metrics["Dict Learning"]["SNR_dB"]:.2f} dB'
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=7)

    fig.suptitle(f'ECG Denoising Inference — {info["filename"]}',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'inference_result.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Plot saved: {path}')

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print('=' * 70)
    print('  ECG DENOISING — INFERENCE')
    print('=' * 70)
    print('\n[1] Loading pre-trained dictionary...')
    artifact = load_dictionary(args.model)
    dictionary = artifact['dictionary']
    config = artifact.get('config', CONFIG.copy())

    sparsity = artifact.get('sparsity', config['sparsity'])
    window_size = artifact.get('window_size', config['window_size'])
    hp_cutoff = artifact.get('hp_cutoff', config['hp_cutoff'])
    hp_order = artifact.get('hp_order', config['hp_order'])
    savgol_window = artifact.get('savgol_window', config['savgol_window'])
    savgol_polyorder = artifact.get('savgol_polyorder', config['savgol_polyorder'])
    fs = artifact.get('fs', config['fs'])
    config['fs'] = fs
    config['output_dir'] = args.output_dir

    print(f'  Atoms    : {dictionary.shape[0]}')
    print(f'  Sparsity : {sparsity}')
    print(f'  HP cutoff: {hp_cutoff} Hz')
    print('\n[2] Loading input signal...')
    noisy_raw, clean_raw, info = load_input_signal(args)
    print(f'  Source   : {info["source"]}')
    print(f'  File     : {info["filename"]}')
    print(f'  Samples  : {len(noisy_raw):,}')

    if info.get('snr_level') is not None:
        print(f'  SNR level: {info["snr_level"]} dB')

    if clean_raw is not None:
        clean_mu = clean_raw.mean()
        clean_std = clean_raw.std() + 1e-8
    else:
        clean_mu = noisy_raw.mean()
        clean_std = noisy_raw.std() + 1e-8

    noisy_norm = (noisy_raw - clean_mu) / clean_std
    clean_norm = (clean_raw - clean_mu) / clean_std if clean_raw is not None else None

    noisy_hp = highpass_filter(noisy_norm, fs, hp_cutoff, hp_order)
    clean_hp = highpass_filter(clean_norm, fs, hp_cutoff, hp_order) if clean_norm is not None else None
    print('\n[3] Running Dictionary Learning denoising...')
    denoised = denoise_signal(
        noisy_hp, dictionary,
        sparsity=sparsity,
        window_size=window_size,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        verbose=True,
    )
    print('  Denoising complete.')

    metrics = {}
    if clean_hp is not None:
        print('\n[4] Computing evaluation metrics...')
        metrics['Noisy Input'] = compute_all_metrics(clean_hp, noisy_hp)
        metrics['Dict Learning'] = compute_all_metrics(clean_hp, denoised)

        print(f'  Input SNR : {metrics["Noisy Input"]["SNR_dB"]:7.2f} dB')
        print(f'  Output SNR: {metrics["Dict Learning"]["SNR_dB"]:7.2f} dB  '
              f'(Δ = {metrics["Dict Learning"]["SNR_dB"] - metrics["Noisy Input"]["SNR_dB"]:+.2f} dB)')
        print(f'  RMSE      : {metrics["Dict Learning"]["RMSE"]:.4f}')
        print(f'  PRD       : {metrics["Dict Learning"]["PRD"]:.2f}%')
        print(f'  SSIM      : {metrics["Dict Learning"]["SSIM"]:.4f}')
        print(f'  R²        : {metrics["Dict Learning"]["R2"]:.4f}')

    baselines = {}
    if args.compare:
        print('\n[5] Running baseline comparisons...')
        butter_out = butterworth_filter(
            noisy_norm, fs,
            config.get('bp_low', 0.5),
            config.get('bp_high', 40.0),
            config.get('bp_order', 4),
        )
        butter_hp = highpass_filter(butter_out, fs, hp_cutoff, hp_order)
        baselines['Butterworth'] = butter_hp
        wav_out = wavelet_denoise(
            noisy_norm,
            config.get('wavelet', 'db6'),
            config.get('wavelet_level', 6),
        )
        wav_hp = highpass_filter(wav_out, fs, hp_cutoff, hp_order)
        baselines['Wavelet (db6)'] = wav_hp

        if clean_hp is not None:
            for method, sig in baselines.items():
                m = compute_all_metrics(clean_hp, sig)
                metrics[method] = m
                print(f'  {method:15s}: SNR={m["SNR_dB"]:6.2f} dB, '
                      f'RMSE={m["RMSE"]:.4f}')

    print(f'\n[6] Saving results to {args.output_dir}/')
    if args.save_npy:
        npy_path = os.path.join(args.output_dir, 'denoised_signal.npy')
        np.save(npy_path, denoised)
        print(f'  Denoised signal: {npy_path}')

    plot_inference_result(
        clean_hp, noisy_hp, denoised, info, config,
        baselines=baselines if args.compare else None,
        metrics=metrics if metrics else None,
        output_dir=args.output_dir,
    )

    print('\n' + '=' * 70)
    print('  INFERENCE COMPLETE')
    print('=' * 70)
    print(f'  Input            : {args.input}')
    print(f'  Dictionary       : {args.model}')
    if metrics and 'Dict Learning' in metrics:
        snr_imp = metrics['Dict Learning']['SNR_dB'] - metrics['Noisy Input']['SNR_dB']
        print(f'  SNR improvement  : {snr_imp:+.2f} dB')
    print(f'  Output directory : {os.path.abspath(args.output_dir)}')
    print('=' * 70)

if __name__ == '__main__':
    main()