import numpy as np
from sklearn.metrics import r2_score

def compute_snr(clean, denoised):
    """Output SNR in dB."""
    noise_var = np.var(clean - denoised)
    if noise_var < 1e-15:
        return np.inf
    return 10 * np.log10(np.var(clean) / noise_var)


def compute_rmse(clean, denoised):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((clean - denoised) ** 2))


def compute_prd(clean, denoised):
    """Percentage Root-mean-square Difference."""
    return 100 * np.linalg.norm(clean - denoised) / \
        (np.linalg.norm(clean) + 1e-15)


def compute_ssim_1d(clean, denoised):
    """
    1-D Structural Similarity Index.

    Falls back to a manual implementation if scikit-image is
    unavailable.
    """
    try:
        from skimage.metrics import structural_similarity
        win = min(7, len(clean))
        if win % 2 == 0:
            win -= 1
        return structural_similarity(
            clean.astype(np.float64),
            denoised.astype(np.float64),
            win_size=win,
            data_range=clean.max() - clean.min(),
        )
    except Exception:
        mu_x, mu_y = clean.mean(), denoised.mean()
        var_x, var_y = clean.var(), denoised.var()
        cov_xy = np.cov(clean, denoised)[0, 1]
        dr = clean.max() - clean.min()
        c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
        return ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))


def compute_all_metrics(clean, denoised):
    """
    Compute all evaluation metrics.

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


def print_results_table(results_by_snr, methods):
    """
    Print a formatted multi-SNR results comparison table.

    Parameters
    ----------
    results_by_snr : dict
        {snr_level: {method_name: metrics_dict}}
    methods : list[str]
        Ordered list of method names for columns.
    """
    n_m = len(methods)
    line = '─' * 10 + '┼' + ('─' * 17 + '┼') * (n_m - 1) + '─' * 17
    top = '─' * 10 + '┬' + ('─' * 17 + '┬') * (n_m - 1) + '─' * 17
    bot = '─' * 10 + '┴' + ('─' * 17 + '┴') * (n_m - 1) + '─' * 17

    header = f'  │{"Input SNR":^10s}│'
    for m in methods:
        header += f'{m:^17s}│'

    print(f'\n  ┌{top}┐')
    print(header)
    print(f'  ├{line}┤')

    for snr_level in sorted(results_by_snr.keys(), reverse=True):
        mr = results_by_snr[snr_level]
        row = f'  │{snr_level:^10d}│'

        snr_vals = {}
        for m in methods:
            if m != 'Noisy Input':
                snr_vals[m] = mr.get(m, {}).get('SNR_dB', float('-inf'))
        best_m = max(snr_vals, key=snr_vals.get) if snr_vals else None

        for m in methods:
            val = mr.get(m, {}).get('SNR_dB', float('nan'))
            star = '★' if m == best_m and m != 'Noisy Input' else ' '
            if np.isfinite(val):
                row += f' {val:6.2f} dB {star}     │'
            else:
                row += f'      Inf {star}     │'
        print(row)

    print(f'  └{bot}┘')