import numpy as np
from sklearn.metrics import r2_score

def compute_snr(clean, denoised):
    """SNR (dB) = 10 * log10(var(clean) / var(clean - denoised))"""
    noise_var = np.var(clean - denoised)
    if noise_var < 1e-15:
        return np.inf
    return 10 * np.log10(np.var(clean) / noise_var)


def compute_rmse(clean, denoised):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((clean - denoised) ** 2))


def compute_prd(clean, denoised):
    """Percentage Root-mean-square Difference."""
    return 100 * np.linalg.norm(clean - denoised) / (np.linalg.norm(clean) + 1e-15)


def compute_ssim_1d(clean, denoised):
    """1D Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity
        c = clean.astype(np.float64)
        d = denoised.astype(np.float64)
        win = min(7, len(c))
        if win % 2 == 0:
            win -= 1
        return structural_similarity(c, d, win_size=win,
                                     data_range=c.max() - c.min())
    except Exception:
        mu_x, mu_y = np.mean(clean), np.mean(denoised)
        var_x, var_y = np.var(clean), np.var(denoised)
        cov_xy = np.cov(clean, denoised)[0, 1]
        dr = clean.max() - clean.min()
        C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
        return ((2*mu_x*mu_y + C1) * (2*cov_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2))


def compute_metrics(clean, denoised):
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    clean : np.ndarray
        Ground-truth clean signal
    denoised : np.ndarray
        Denoised signal

    Returns
    -------
    dict
        Keys: 'SNR_dB', 'RMSE', 'PRD', 'SSIM', 'R2'
    """
    ml = min(len(clean), len(denoised))
    c, d = clean[:ml], denoised[:ml]
    mask = ~(np.isnan(c) | np.isnan(d))
    c, d = c[mask], d[mask]
    return {
        'SNR_dB': compute_snr(c, d),
        'RMSE':   compute_rmse(c, d),
        'PRD':    compute_prd(c, d),
        'SSIM':   compute_ssim_1d(c, d),
        'R2':     r2_score(c, d),
    }


def print_metrics_table(results):
    """Print a formatted comparison table to console."""
    print('\n  ┌─────────────────────────────────────────────────────────────────┐')
    print('  │                  DENOISING RESULTS COMPARISON                   │')
    print('  ├──────────────────────┬─────────┬────────┬────────┬──────┬───────┤')
    print('  │ Method               │ SNR(dB) │  RMSE  │  PRD%  │ SSIM │  R²   │')
    print('  ├──────────────────────┼─────────┼────────┼────────┼──────┼───────┤')

    best_snr = max(v['SNR_dB'] for k, v in results.items() if k != 'Noisy Input')

    for method, m in results.items():
        star = ' ★' if m['SNR_dB'] == best_snr and method != 'Noisy Input' else '  '
        snr_s = f'{m["SNR_dB"]:7.2f}' if np.isfinite(m['SNR_dB']) else '    Inf'
        print(f'  │ {method:<20s}{star}│{snr_s} │{m["RMSE"]:7.4f} │'
              f'{m["PRD"]:7.2f} │{m["SSIM"]:.3f} │{m["R2"]:6.3f} │')

    print('  └──────────────────────┴─────────┴────────┴────────┴──────┴───────┘')