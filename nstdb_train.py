import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
from nstdb_denoising import (
    CONFIG,
    load_nstdb_dataset,
    train_dictionary, denoise_signal, save_dictionary,
    butterworth_filter, wavelet_denoise, highpass_filter,
    compute_all_metrics, print_results_table,
    plot_all, setup_plot_style,
)
from tqdm import tqdm

def parse_args():
    """Parse CLI arguments to override CONFIG defaults."""
    p = argparse.ArgumentParser(
        description='NSTDB ECG Dictionary Training & Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir', type=str, default=CONFIG['data_dir'],
                   help='Path to NSTDB .mat directory')
    p.add_argument('--model_dir', type=str, default=CONFIG['model_dir'],
                   help='Directory to save the trained dictionary')
    p.add_argument('--output_dir', type=str, default=CONFIG['output_dir'],
                   help='Directory for output plots')
    p.add_argument('--test_patient', type=str,
                   default=CONFIG['test_patient'],
                   help='Patient ID to hold out for testing')
    p.add_argument('--n_atoms', type=int, default=CONFIG['n_atoms'],
                   help='Number of dictionary atoms')
    p.add_argument('--sparsity', type=int, default=CONFIG['sparsity'],
                   help='OMP non-zero coefficients')
    p.add_argument('--window_size', type=int,
                   default=CONFIG['window_size'],
                   help='Patch/window size')
    p.add_argument('--max_train_patches', type=int,
                   default=CONFIG['max_train_patches'],
                   help='Max patches for dictionary training')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip plot generation')
    return p.parse_args()

def build_config(args):
    """Merge CLI args into a copy of the default CONFIG."""
    config = CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['model_dir'] = args.model_dir
    config['output_dir'] = args.output_dir
    config['test_patient'] = args.test_patient
    config['n_atoms'] = args.n_atoms
    config['sparsity'] = args.sparsity
    config['window_size'] = args.window_size
    config['max_train_patches'] = args.max_train_patches
    return config

def main():
    args = parse_args()
    config = build_config(args)

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)

    print('=' * 70)
    print('  ECG DENOISING — NSTDB DICTIONARY TRAINING')
    print('  Architecture: HP Pre-filter + Dict Learning (Zero-Means)')
    print('=' * 70)

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

    dictionary, dl_model = train_dictionary(
        dataset, config, test_patient=test_patient,
    )

    model_path = os.path.join(config['model_dir'], 'nstdb_dictionary.pkl')
    save_dictionary(dictionary, config, model_path)

    print('\n[STEP 4] Multi-level stress testing on real EM artifacts...')
    print(f'  Test patient: {test_patient}')
    print(f'  Reference   : {clean_snr} dB HP-filtered signal\n')

    results_by_snr = {}
    dict_outputs = {}
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
        dict_denoised = denoise_signal(
            noisy_hp, dictionary,
            sparsity=config['sparsity'],
            window_size=config['window_size'],
            savgol_window=config['savgol_window'],
            savgol_polyorder=config['savgol_polyorder'],
        )
        dict_metrics = compute_all_metrics(ground_truth_hp, dict_denoised)
        dict_outputs[snr_level] = dict_denoised

        butter_denoised = butterworth_filter(
            noisy_raw, config['fs'],
            config['bp_low'], config['bp_high'], config['bp_order'],
        )
        butter_hp = highpass_filter(butter_denoised, config['fs'],
                                     config['hp_cutoff'], config['hp_order'])
        butter_metrics = compute_all_metrics(ground_truth_hp, butter_hp)

        wav_denoised = wavelet_denoise(
            noisy_raw, config['wavelet'], config['wavelet_level'],
        )
        wav_hp = highpass_filter(wav_denoised, config['fs'],
                                  config['hp_cutoff'], config['hp_order'])
        wav_metrics = compute_all_metrics(ground_truth_hp, wav_hp)

        results_by_snr[snr_level] = {
            'Noisy Input':   input_metrics,
            'Butterworth':   butter_metrics,
            'Wavelet (db6)': wav_metrics,
            'Dict Learning': dict_metrics,
        }

        denoised_signals_by_snr[snr_level] = {
            'Butterworth':   butter_hp,
            'Wavelet (db6)': wav_hp,
            'Dict Learning': dict_denoised,
        }

        tqdm.write(
            f'    {snr_level:3d} dB │ Input: {input_metrics["SNR_dB"]:6.2f} │ '
            f'Dict: {dict_metrics["SNR_dB"]:6.2f} │ '
            f'Wav: {wav_metrics["SNR_dB"]:6.2f} │ '
            f'BPF: {butter_metrics["SNR_dB"]:6.2f}'
        )

    print('\n[STEP 4 COMPLETE]\n')

    print('[RESULTS] Robustness evaluation...')
    methods = ['Noisy Input', 'Butterworth', 'Wavelet (db6)', 'Dict Learning']
    print_results_table(results_by_snr, methods)
    denoising_methods = ['Butterworth', 'Wavelet (db6)', 'Dict Learning']
    print('\n  Per-level winners:')
    dl_wins = 0
    for snr in sorted(results_by_snr.keys(), reverse=True):
        best_m = max(denoising_methods,
                     key=lambda m: results_by_snr[snr].get(m, {}).get(
                         'SNR_dB', float('-inf')))
        best_snr = results_by_snr[snr][best_m]['SNR_dB']
        marker = '★' if best_m == 'Dict Learning' else ' '
        if best_m == 'Dict Learning':
            dl_wins += 1
        print(f'    {snr:3d} dB → {best_m} ({best_snr:.2f} dB) {marker}')

    if not args.no_plots:
        plot_all(
            ground_truth_hp, noisy_hp_by_snr, dict_outputs,
            denoised_signals_by_snr, results_by_snr, dictionary, config,
        )

    print('=' * 70)
    print('  TRAINING COMPLETE')
    print('=' * 70)
    print(f'  Dataset           : NSTDB ({len(dataset)} patients)')
    print(f'  Test patient      : {test_patient}')
    print(f'  Dictionary        : {dictionary.shape}')
    print(f'  HP pre-filter     : {config["hp_cutoff"]} Hz')
    print(f'  Sparsity          : {config["sparsity"]}')
    print(f'  Dict Learning wins: {dl_wins}/{len(available_snrs)} levels')
    print(f'  Dictionary saved  : {os.path.abspath(model_path)}')
    print(f'  Plots saved to    : {os.path.abspath(config["output_dir"])}')
    print('=' * 70)

if __name__ == '__main__':
    main()