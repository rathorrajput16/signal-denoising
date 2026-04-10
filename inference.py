import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ecg_denoising.config import CONFIG
from ecg_denoising.data_loader import load_single_csv, denormalize_signal
from ecg_denoising.noise import add_realistic_noise
from ecg_denoising.dictionary import load_dictionary, denoise_signal
from ecg_denoising.baselines import wavelet_denoise
from ecg_denoising.metrics import compute_metrics, print_metrics_table

warnings.filterwarnings('ignore')

def parse_args():
    p = argparse.ArgumentParser(
        description='Denoise an ECG signal using a pre-trained dictionary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Denoise a noisy ECG file:
  python inference.py --input noisy_ecg.csv

  # Add synthetic noise to a clean file, then denoise:
  python inference.py --input ECG_data/100.csv --add-noise --noise-snr 10

  # Denoise and generate comparison plot:
  python inference.py --input ECG_data/100.csv --add-noise --plot

  # Use a different channel:
  python inference.py --input data.csv --channel V5

  # Save denoised output:
  python inference.py --input noisy.csv --output denoised.csv
        """)

    p.add_argument('--input', '-i', required=True,
                   help='Path to input ECG CSV file')
    p.add_argument('--output', '-o', default=None,
                   help='Path to save denoised signal as CSV (optional)')
    p.add_argument('--model', '-m',
                   default=os.path.join(CONFIG['model_dir'], 'ecg_dictionary.pkl'),
                   help='Path to trained dictionary (default: models/ecg_dictionary.pkl)')
    p.add_argument('--channel', default=CONFIG['channel'],
                   help='ECG lead column name (default: MLII)')
    p.add_argument('--n-samples', type=int, default=None,
                   help='Number of samples to process (default: all)')

    # Noise simulation (for testing)
    p.add_argument('--add-noise', action='store_true',
                   help='Add synthetic noise (for testing with clean signals)')
    p.add_argument('--noise-snr', type=float, default=10,
                   help='Target noise SNR in dB (with --add-noise)')

    # Visualization
    p.add_argument('--plot', action='store_true',
                   help='Generate before/after plot')
    p.add_argument('--plot-dir', default='./inference_plots/',
                   help='Directory for inference plots')

    return p.parse_args()


def plot_inference_result(original, denoised, fs, output_path,
                          noisy=None, title_suffix=''):
    """Generate a clean before/after plot for inference."""
    fig, axes = plt.subplots(2 if noisy is None else 3, 1,
                              figsize=(14, 6 if noisy is None else 9),
                              sharex=True)

    n_show = min(2000, len(original))
    t = np.arange(n_show) / fs

    ax_idx = 0

    if noisy is not None:
        axes[ax_idx].plot(t, original[:n_show], color='#2196F3', linewidth=0.8)
        axes[ax_idx].set_title(f'Original Clean Signal{title_suffix}',
                               fontweight='bold')
        axes[ax_idx].set_ylabel('Amplitude')
        ax_idx += 1

        axes[ax_idx].plot(t, noisy[:n_show], color='#F44336',
                          linewidth=0.6, alpha=0.8)
        axes[ax_idx].set_title('Noisy Signal', fontweight='bold')
        axes[ax_idx].set_ylabel('Amplitude')
        ax_idx += 1
    else:
        axes[ax_idx].plot(t, original[:n_show], color='#F44336',
                          linewidth=0.7)
        axes[ax_idx].set_title(f'Input Signal (Noisy){title_suffix}',
                               fontweight='bold')
        axes[ax_idx].set_ylabel('Amplitude')
        ax_idx += 1

    axes[ax_idx].plot(t, denoised[:n_show], color='#4CAF50', linewidth=0.8,
                      label='Dict Learning')
    if noisy is not None:
        axes[ax_idx].plot(t, original[:n_show], color='#2196F3',
                          linewidth=0.4, linestyle='--', alpha=0.5,
                          label='Original')
    axes[ax_idx].set_title('Denoised Output (Dictionary Learning)',
                           fontweight='bold')
    axes[ax_idx].set_ylabel('Amplitude')
    axes[ax_idx].set_xlabel('Time (s)')
    axes[ax_idx].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Plot saved: {output_path}')


def main():
    args = parse_args()

    print('=' * 65)
    print('  ECG SIGNAL DENOISING — INFERENCE')
    print('=' * 65)

    print('\n[1] Loading pre-trained dictionary...')
    artifact = load_dictionary(args.model)
    dictionary = artifact['dictionary']
    sparsity = artifact['sparsity']
    window_size = artifact['window_size']
    savgol_window = artifact['savgol_window']
    savgol_polyorder = artifact['savgol_polyorder']
    fs = artifact['fs']

    print(f'\n[2] Loading input signal: {args.input}')
    normalized, mu, std = load_single_csv(
        args.input, channel=args.channel,
        n_samples=args.n_samples,
        gain=CONFIG['gain'], baseline=CONFIG['baseline_adc']
    )
    print(f'  Samples: {len(normalized):,}')
    print(f'  Duration: {len(normalized)/fs:.2f} seconds @ {fs} Hz')
    print(f'  Normalization: μ={mu:.4f}, σ={std:.4f}')

    clean_ref = None
    noisy_for_plot = None

    if args.add_noise:
        print(f'\n[2b] Adding synthetic noise (target SNR={args.noise_snr} dB)...')
        clean_ref = normalized.copy()
        normalized, achieved_snr = add_realistic_noise(
            normalized, fs=fs, target_snr_db=args.noise_snr
        )
        noisy_for_plot = normalized.copy()
        print(f'  Achieved SNR: {achieved_snr:.2f} dB')

    print(f'\n[3] Denoising with dictionary ({dictionary.shape[0]} atoms, '
          f'sparsity={sparsity})...')
    denoised = denoise_signal(
        normalized, dictionary,
        sparsity=sparsity,
        window_size=window_size,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )

    # ── Metrics (if clean reference available) ──
    if clean_ref is not None:
        print('\n[4] Computing metrics...')

        # Also run wavelet baseline for comparison
        wav_denoised = wavelet_denoise(normalized, CONFIG['wavelet'],
                                        CONFIG['wavelet_level'])

        results = {
            'Noisy Input':   compute_metrics(clean_ref, normalized),
            'Wavelet (db6)': compute_metrics(clean_ref, wav_denoised),
            'Dict Learning': compute_metrics(clean_ref, denoised),
        }
        print_metrics_table(results)

        method_snrs = {k: v['SNR_dB'] for k, v in results.items()
                       if k != 'Noisy Input'}
        best = max(method_snrs, key=method_snrs.get)
        print(f'\n  Best method: {best} ({method_snrs[best]:.2f} dB)')
    else:
        print('\n[4] No clean reference — skipping metrics.')
        print('    (Use --add-noise to evaluate with synthetic noise)')

    # ── Save output ──
    if args.output:
        print(f'\n[5] Saving denoised signal to: {args.output}')
        import pandas as pd
        # Denormalize to millivolts
        denoised_mv = denormalize_signal(denoised, mu, std)
        df_out = pd.DataFrame({'denoised_mV': denoised_mv})
        if clean_ref is not None:
            df_out['clean_mV'] = denormalize_signal(clean_ref, mu, std)
        if noisy_for_plot is not None:
            df_out['noisy_mV'] = denormalize_signal(noisy_for_plot, mu, std)
        df_out.to_csv(args.output, index=False)
        print(f'  Saved {len(df_out)} samples')

    # ── Plot ──
    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.input))[0]
        plot_path = os.path.join(args.plot_dir, f'denoised_{basename}.png')
        print(f'\n[6] Generating plot...')

        if clean_ref is not None:
            plot_inference_result(
                clean_ref, denoised, fs, plot_path,
                noisy=noisy_for_plot,
                title_suffix=f' (Record {basename})'
            )
        else:
            plot_inference_result(
                normalized, denoised, fs, plot_path,
                title_suffix=f' (Record {basename})'
            )

    # ── Summary ──
    print('\n' + '=' * 65)
    print('  INFERENCE COMPLETE')
    print('=' * 65)
    print(f'  Input file     : {args.input}')
    print(f'  Signal length  : {len(denoised):,} samples '
          f'({len(denoised)/fs:.2f}s)')
    print(f'  Dictionary     : {dictionary.shape[0]} atoms × '
          f'{dictionary.shape[1]} window')
    if clean_ref is not None and 'Dict Learning' in results:
        print(f'  Dict SNR       : {results["Dict Learning"]["SNR_dB"]:.2f} dB')
    if args.output:
        print(f'  Output saved   : {args.output}')
    if args.plot:
        print(f'  Plot saved     : {plot_path}')
    print('=' * 65)
    print()


if __name__ == '__main__':
    main()