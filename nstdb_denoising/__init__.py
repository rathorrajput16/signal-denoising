"""
NSTDB ECG Denoising Package
============================

Real-world Electrode Motion (EM) artifact denoising using
Dictionary Learning + OMP Sparse Coding with dense-overlap
patch averaging reconstruction.

Architecture:
  1. HP pre-filter (0.67 Hz) removes EM baseline wander
  2. Dictionary trained on HP-filtered clean (24 dB) signals
  3. OMP sparse coding with aggressive sparsity (3 coeffs)
  4. Dense patch averaging with ZERO means reconstruction
  5. Savitzky-Golay post-processing

Modules:
  config        — Centralized pipeline parameters
  data_loader   — NSTDB .mat file loading and preprocessing
  dictionary    — Dictionary training, sparse coding, reconstruction
  baselines     — Wavelet and Butterworth baseline methods
  metrics       — SNR, RMSE, PRD, SSIM, R² evaluation
  visualization — Publication-quality plot generation
"""

__version__ = '2.0.0'
__author__ = 'ECG Pipeline'

from .config import CONFIG
from .data_loader import (
    parse_filename, load_mat_signal, highpass_filter,
    load_nstdb_dataset,
)
from .dictionary import (
    extract_dense_patches, train_dictionary, denoise_signal,
    save_dictionary, load_dictionary,
)
from .baselines import wavelet_denoise, butterworth_filter
from .metrics import (
    compute_snr, compute_rmse, compute_prd, compute_ssim_1d,
    compute_all_metrics, print_results_table,
)
from .visualization import (
    setup_plot_style, plot_degradation_curve, plot_extreme_stress_test,
    plot_all_snr_levels, plot_dictionary_atoms, plot_snr_improvement_heatmap,
    plot_spectrograms, plot_error_distributions, plot_metrics_bar, plot_all,
)
