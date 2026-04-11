import os
import sys
import argparse
import warnings
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nstdb_denoising import (
    CONFIG,
    load_mat_signal, highpass_filter, parse_filename,
    butterworth_filter, wavelet_denoise,
    compute_all_metrics,
)
from nstdb_denoising.lista_model import (
    load_lista_model, denoise_signal_lista,
)

def parse_args():
    p = argparse.ArgumentParser(
        description='Denoise an ECG signal using a pre-trained CNN-LISTA model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input', type=str, required=True,
                   help='Input: NSTDB .mat file or raw .npy array')
    p.add_argument('--model_dir', type=str, default='./models/',
                   help='Directory with lista_weights.weights.h5 + lista_config.pkl')
    p.add_argument('--output_dir', type=str,
                   default='./lista_inference_output/',
                   help='Directory for output files and plots')
    p.add_argument('--channel', type=int, default=0,
                   help='ECG channel index (for .mat files)')
    p.add_argument('--n_samples', type=int, default=10000,
                   help='Number of samples to process')
    p.add_argument('--offset', type=int, default=100000,
                   help='Sample offset (skip preamble in .mat files)')
    p.add_argument('--compare', action='store_true',
                   help='Compare against Butterworth, Wavelet, and OMP')
    p.add_argument('--save_npy', action='store_true',
                   help='Save denoised signal as .npy file')
    return p.parse_args()

def load_input_signal(args):
    """Load .mat or .npy input signal with metadata."""
    filepath = args.input
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.mat':
        clean, noisy, labeled_snr = load_mat_signal(
            filepath, channel=args.channel,
            n_samples=args.n_samples, offset=args.offset,
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
        raise ValueError(f'Unsupported file format: {ext}. Use .mat or .npy')

def plot_inference_result(clean_hp, noisy_hp, denoised, info, config,
                          baselines=None, metrics=None, output_dir='.'):
    """Generate CNN-LISTA inference comparison plot."""
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
    if n_panels == 1:
        axes = [axes]

    n_show = min(2000, len(noisy_hp))
    t = np.arange(n_show) / fs
    panel = 0

    if has_clean:
        axes[panel].plot(t, clean_hp[:n_show], color='#2196F3',
                         linewidth=0.9)
        title = 'Ground Truth (24 dB)'
        if metrics and 'Noisy Input' in metrics:
            title += (f'  |  Input SNR: '
                      f'{metrics["Noisy Input"]["SNR_dB"]:.2f} dB')
        axes[panel].set_title(title, fontweight='bold')
        axes[panel].set_ylabel('Amplitude')
        panel += 1

    axes[panel].plot(t, noisy_hp[:n_show], color='#F44336',
                     linewidth=0.6, alpha=0.8)
    snr_label = (f'{info.get("snr_level", "?")} dB '
                 if info.get('snr_level') else '')
    axes[panel].set_title(f'Noisy Input ({snr_label}EM Artifact)',
                          fontweight='bold')
    axes[panel].set_ylabel('Amplitude')
    panel += 1

    if baselines:
        baseline_colors = {
            'Butterworth': '#FF9800',
            'Wavelet (db6)': '#E91E63',
            'OMP': '#9C27B0',
        }
        for method, sig in baselines.items():
            ax = axes[panel]
            ax.plot(t, sig[:n_show],
                    color=baseline_colors.get(method, 'gray'),
                    linewidth=0.9, label=method)
            if has_clean:
                ax.plot(t, clean_hp[:n_show], color='#2196F3',
                        linewidth=0.3, linestyle='--', alpha=0.3)
            title = method
            if metrics and method in metrics:
                title += f'  |  SNR: {metrics[method]["SNR_dB"]:.2f} dB'
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right', fontsize=7)
            panel += 1

    ax = axes[panel]
    ax.plot(t, denoised[:n_show], color='#4CAF50', linewidth=0.9,
            label='CNN-LISTA')
    if has_clean:
        ax.plot(t, clean_hp[:n_show], color='#2196F3',
                linewidth=0.3, linestyle='--', alpha=0.3)
    title = 'CNN-LISTA (denoised)'
    if metrics and 'CNN-LISTA' in metrics:
        title += f'  |  SNR: {metrics["CNN-LISTA"]["SNR_dB"]:.2f} dB'
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=7)

    fig.suptitle(f'CNN-LISTA Inference — {info["filename"]}',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'lista_inference_result.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Plot saved: {path}')

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('  CNN-LISTA INFERENCE')
    print('=' * 70)

    # ── Load model ──
    print('\n[1] Loading CNN-LISTA model...')
    model, artifact = load_lista_model(args.model_dir)
    config = artifact.get('config', CONFIG.copy())

    ws = artifact.get('window_size', config['window_size'])
    hp_cutoff = artifact.get('hp_cutoff', config['hp_cutoff'])
    hp_order = artifact.get('hp_order', config['hp_order'])
    savgol_window = artifact.get('savgol_window', config['savgol_window'])
    savgol_polyorder = artifact.get('savgol_polyorder',
                                    config['savgol_polyorder'])
    fs = artifact.get('fs', config['fs'])
    config['fs'] = fs
    config['output_dir'] = args.output_dir

    # ── Load input ──
    print('\n[2] Loading input signal...')
    noisy_raw, clean_raw, info = load_input_signal(args)
    print(f'  Source   : {info["source"]}')
    print(f'  File     : {info["filename"]}')
    print(f'  Samples  : {len(noisy_raw):,}')
    if info.get('snr_level') is not None:
        print(f'  SNR level: {info["snr_level"]} dB')

    # ── Normalize ──
    if clean_raw is not None:
        clean_mu = clean_raw.mean()
        clean_std = clean_raw.std() + 1e-8
    else:
        clean_mu = noisy_raw.mean()
        clean_std = noisy_raw.std() + 1e-8

    noisy_norm = (noisy_raw - clean_mu) / clean_std
    clean_norm = ((clean_raw - clean_mu) / clean_std
                  if clean_raw is not None else None)

    # ── HP filter ──
    noisy_hp = highpass_filter(noisy_norm, fs, hp_cutoff, hp_order)
    clean_hp = (highpass_filter(clean_norm, fs, hp_cutoff, hp_order)
                if clean_norm is not None else None)

    # ── Denoise ──
    print('\n[3] Running CNN-LISTA denoising...')
    denoised = denoise_signal_lista(
        noisy_hp, model,
        window_size=ws,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        verbose=True,
    )
    print('  Denoising complete.')

    # ── Metrics ──
    metrics = {}
    if clean_hp is not None:
        print('\n[4] Computing evaluation metrics...')
        metrics['Noisy Input'] = compute_all_metrics(clean_hp, noisy_hp)
        metrics['CNN-LISTA'] = compute_all_metrics(clean_hp, denoised)

        snr_imp = (metrics['CNN-LISTA']['SNR_dB']
                   - metrics['Noisy Input']['SNR_dB'])
        print(f'  Input SNR : {metrics["Noisy Input"]["SNR_dB"]:7.2f} dB')
        print(f'  Output SNR: {metrics["CNN-LISTA"]["SNR_dB"]:7.2f} dB  '
              f'(Δ = {snr_imp:+.2f} dB)')
        print(f'  RMSE      : {metrics["CNN-LISTA"]["RMSE"]:.4f}')
        print(f'  PRD       : {metrics["CNN-LISTA"]["PRD"]:.2f}%')
        print(f'  SSIM      : {metrics["CNN-LISTA"]["SSIM"]:.4f}')
        print(f'  R²        : {metrics["CNN-LISTA"]["R2"]:.4f}')

    # ── Baselines comparison ──
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

        # OMP with the saved 128-atom dictionary (if available)
        try:
            from nstdb_denoising import denoise_signal, load_dictionary
            omp_dict_path = os.path.join(args.model_dir,
                                         'nstdb_dictionary.pkl')
            if os.path.exists(omp_dict_path):
                omp_artifact = load_dictionary(omp_dict_path)
                omp_out = denoise_signal(
                    noisy_hp, omp_artifact['dictionary'],
                    sparsity=omp_artifact.get('sparsity', 3),
                    window_size=omp_artifact.get('window_size', 64),
                    savgol_window=savgol_window,
                    savgol_polyorder=savgol_polyorder,
                )
                baselines['OMP'] = omp_out
        except Exception:
            pass

        if clean_hp is not None:
            for method, sig in baselines.items():
                m = compute_all_metrics(clean_hp, sig)
                metrics[method] = m
                print(f'  {method:15s}: SNR={m["SNR_dB"]:6.2f} dB, '
                      f'RMSE={m["RMSE"]:.4f}')

    # ── Save outputs ──
    print(f'\n[6] Saving results to {args.output_dir}/')
    if args.save_npy:
        npy_path = os.path.join(args.output_dir, 'lista_denoised.npy')
        np.save(npy_path, denoised)
        print(f'  Denoised signal: {npy_path}')

    plot_inference_result(
        clean_hp, noisy_hp, denoised, info, config,
        baselines=baselines if args.compare else None,
        metrics=metrics if metrics else None,
        output_dir=args.output_dir,
    )

    # ── Summary ──
    print('\n' + '=' * 70)
    print('  CNN-LISTA INFERENCE COMPLETE')
    print('=' * 70)
    print(f'  Input            : {args.input}')
    print(f'  Model            : {args.model_dir}')
    if metrics and 'CNN-LISTA' in metrics:
        snr_imp = (metrics['CNN-LISTA']['SNR_dB']
                   - metrics['Noisy Input']['SNR_dB'])
        print(f'  SNR improvement  : {snr_imp:+.2f} dB')
    print(f'  Output directory : {os.path.abspath(args.output_dir)}')
    print('=' * 70)

if __name__ == '__main__':
    main()
