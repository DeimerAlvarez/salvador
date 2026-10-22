"""Final abundances of a star: mean, sigma-clipping and errors (extract_abu*), linear fits."""
import numpy as np
from scipy import stats
from astropy.stats import mad_std
from astropy.stats import sigma_clip

from .config import cfg
from .elements import abundance_link, get_element_symbol, strip_ion, symbol_map

def _clip_and_average(nelem, abu, abu_mc, nel, flag_abu, param_errors=None):
    """Common core of extract_abu_fe, extract_abu and extract_abu_isor.

    Selects the valid lines of the species nel (abundance != 99.99 and flag 0), sigma-clips them
    (median and MAD, SCLIP sigma; only with 3 or more lines and a non-zero dispersion) and sets
    flag 1, in place, for the lines used. Returns None if there are no valid lines, otherwise a dict:
      n       number of lines used
      mean    mean of the (absolute) values of the lines used
      stat    statistical error: the MC error for one line, std/sqrt(n) for more lines
      params  (sigma_T, sigma_g, sigma_v): mean parameter errors of the lines used, or None if
              param_errors (the three arrays abu_pT, abu_pg, abu_pv) is not given
    """
    mask = (nelem == nel) & (abu != 99.99) & (flag_abu == 0)
    nlin = np.count_nonzero(mask)
    if nlin == 0:
        return None
    original_indices = np.where(mask)[0]
    abu_elem = abu[mask]

    if nlin == 1:
        flag_abu[original_indices[0]] = 1
        mean = abu_elem[0]
        stat = abu_mc[mask][0]
        params = tuple(err[mask][0] for err in param_errors) if param_errors is not None else None
        return {'n': nlin, 'mean': mean, 'stat': stat, 'params': params}    # nlin == 1 (same integer type as before)

    clip_mask = np.ones(nlin, dtype=bool)
    if nlin >= 3:
        disp = mad_std(abu_elem)
        if np.isfinite(disp) and disp > 0.0:
            clipped_abu = sigma_clip(abu_elem, sigma=cfg.sclip, cenfunc='median', stdfunc=mad_std)
            if clipped_abu.count() > 0:
                clip_mask = ~clipped_abu.mask
                nlin = clipped_abu.count()

    flag_abu[original_indices[clip_mask]] = 1       # lines used
    mean = np.mean(abu_elem[clip_mask])
    stat = np.std(abu_elem[clip_mask], ddof=1) / np.sqrt(nlin)
    params = tuple(np.mean(err[mask][clip_mask]) for err in param_errors) if param_errors is not None else None
    return {'n': nlin, 'mean': mean, 'stat': stat, 'params': params}


def extract_abu_fe(nelem, abu, abu_mc, nel, flag_abu, abu_pT=None, abu_pg=None, abu_pv=None):
    """[Fe/H] from the Fe I (or Fe II) lines: mean, errors and the absolute abundance (params_XH).
    With the three parameter-error arrays: (nlin, [Fe/H], sigma_tot, sigma_stat, sigma_T, sigma_g,
    sigma_v, flags, sigma_params, params_XH); without: (nlin, [Fe/H], sigma_stat, flags, 99.99, params_XH)."""
    full_error_mode = all(err is not None for err in [abu_pT, abu_pg, abu_pv])
    res = _clip_and_average(nelem, abu, abu_mc, nel, flag_abu, (abu_pT, abu_pg, abu_pv) if full_error_mode else None)

    abu_sol = abundance_link.get(strip_ion(get_element_symbol(nel, symbol_map))) if res is not None else None
    if res is None or np.isnan(res['mean'] - abu_sol):
        params_XH = {'abu_abs': 99.99, 's_tot': 99.99, 's_int': 99.99, 's_params': 99.99}
        return (0, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, flag_abu, 99.99, params_XH) if full_error_mode else (0, 99.99, 99.99, flag_abu, 99.99, params_XH)

    nlin, s1 = res['n'], res['stat']
    mean = res['mean'] - abu_sol                    # [Fe/H]
    mean_abs = np.floor((mean + abu_sol) * 100 + 0.5) / 100

    if full_error_mode:
        s2, s3, s4 = res['params']
        sigma_total = np.sqrt(s1**2 + s2**2 + s3**2 + s4**2)
        sigma_params = np.sqrt(s2**2 + s3**2 + s4**2)
        params_XH = {'abu_abs': mean_abs, 's_tot': sigma_total, 's_int': s1, 's_params': sigma_params}
        return nlin, mean, sigma_total, s1, s2, s3, s4, flag_abu, sigma_params, params_XH
    else:
        params_XH = {'abu_abs': mean_abs, 's_tot': s1, 's_int': s1, 's_params': 99.99}
        return nlin, mean, s1, flag_abu, 99.99, params_XH


def extract_abu(nelem, abu, abu_mc, nel, feh, flag_abu, s_stat_Fe, param_errors=None, fe_errors=None):
    """[X/Fe] of the species nel and its errors, combined with those of Fe (s_stat_Fe, fe_errors);
    params_XH has the absolute abundance and the [X/H] errors.
    Returns (nlin, [X/Fe], sigma_tot, flags, sigma_params, sigma_int, params_XH), rounded to 0.01."""
    full_error_mode = param_errors is not None and fe_errors is not None
    res = _clip_and_average(nelem, abu, abu_mc, nel, flag_abu, param_errors if full_error_mode else None)

    abu_sol = abundance_link.get(strip_ion(get_element_symbol(nel, symbol_map))) if res is not None else None
    if res is None or np.isnan(res['mean'] - abu_sol - feh):
        params_XH = {'abu_abs': 99.99, 's_tot': 99.99, 's_int': 99.99, 's_params': 99.99}
        return 0, 99.99, 99.99, flag_abu, 99.99, 99.99, params_XH

    nlin, sigma_stat_X = res['n'], res['stat']
    mean = res['mean'] - abu_sol - feh          # [X/Fe]

    if full_error_mode:
        sigma_T_X, sigma_g_X, sigma_v_X = res['params']
        s_T_Fe, s_g_Fe, s_v_Fe = fe_errors
        # [X/Fe]: the parameter errors of X and Fe are correlated, hence the differences
        variance = (s_stat_Fe**2 + sigma_stat_X**2 + (sigma_T_X - s_T_Fe)**2 + (sigma_g_X - s_g_Fe)**2 + (sigma_v_X - s_v_Fe)**2)
        final_sigma = np.sqrt(variance)
        sigma_params = np.sqrt((sigma_T_X - s_T_Fe)**2 + (sigma_g_X - s_g_Fe)**2 + (sigma_v_X - s_v_Fe)**2)
        sigma_int = np.sqrt(s_stat_Fe**2 + sigma_stat_X**2)
        # [X/H]
        final_sigma_XH = np.sqrt(sigma_stat_X**2 + sigma_T_X**2 + sigma_g_X**2 + sigma_v_X**2)
        sigma_params_XH = np.sqrt(sigma_T_X**2 + sigma_g_X**2 + sigma_v_X**2)
    else:
        if s_stat_Fe != 99.99:
            final_sigma = np.sqrt(s_stat_Fe**2 + sigma_stat_X**2)
            sigma_int = np.sqrt(s_stat_Fe**2 + sigma_stat_X**2)
        else:
            final_sigma = sigma_stat_X
            sigma_int = sigma_stat_X
        sigma_params = 99.99
        # [X/H]
        final_sigma_XH = sigma_stat_X
        sigma_params_XH = 99.99
    sigma_int_XH = sigma_stat_X

    mean_abs = np.floor((mean + abu_sol + feh) * 100 + 0.5) / 100
    params_XH = {'abu_abs': mean_abs, 's_tot': final_sigma_XH, 's_int': sigma_int_XH, 's_params': sigma_params_XH}
    return nlin, np.round(mean, 2), np.round(final_sigma, 2), flag_abu, np.round(sigma_params, 2), np.round(sigma_int, 2), params_XH


def extract_abu_isor(nelem, abu, abu_mc, nel, flag_abu, param_errors=None):
    """Isotopic ratio of the species nel: mean and errors, rounded to integers.
    Returns (nlin, ratio, sigma_tot, flags, sigma_params, sigma_int); 999 marks missing errors."""
    full_error_mode = param_errors is not None
    res = _clip_and_average(nelem, abu, abu_mc, nel, flag_abu, param_errors if full_error_mode else None)

    if res is None or np.isnan(res['mean']):
        return 0, 99.99, 999, flag_abu, 999, 999

    nlin, mean, sigma_stat_X = res['n'], res['mean'], res['stat']
    if full_error_mode:
        sigma_T_X, sigma_g_X, sigma_v_X = res['params']
        final_sigma = np.sqrt(sigma_stat_X**2 + sigma_T_X**2 + sigma_g_X**2 + sigma_v_X**2)
        sigma_params = np.sqrt(sigma_T_X**2 + sigma_g_X**2 + sigma_v_X**2)
    else:
        final_sigma = sigma_stat_X
        sigma_params = 999
    sigma_int = sigma_stat_X

    mean = int(np.floor(mean + 0.5))
    final_sigma = int(np.floor(final_sigma + 0.5))
    sigma_params = int(np.floor(sigma_params + 0.5))
    sigma_int = int(np.floor(sigma_int + 0.5))

    return nlin, mean, final_sigma, flag_abu, sigma_params, sigma_int


def get_coeffs(xx, yy):
    """Linear fit y = m x + c. Returns (fitted values, m, c, error of m, error of c)."""
    result = stats.linregress(xx, yy)

    m = result.slope
    c = result.intercept
    em = result.stderr
    ec = result.intercept_stderr

    y_fit = m * xx + c
    return y_fit, m, c, em, ec
