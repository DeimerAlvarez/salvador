"""Final abundances of a star: mean, sigma-clipping and errors (extract_abu*), the rows of
abundance_final.txt (star_abundance_rows, also used by salvador-merge), linear fits."""
import numpy as np
from scipy import stats
from astropy.stats import mad_std
from astropy.stats import sigma_clip

from .config import cfg
from .elements import abu_absolute, abundance_link, get_element_symbol, strip_ion, symbol_map

MISSING = 99.99     # value not available (in the .fit files and in the results)


def _mean_valid(values: np.ndarray) -> float:
    """Mean of the values that are available (not 99.99 and not NaN); 99.99 if there are none.

    FIX: a parameter error of 99.99 (the fit with the changed parameter could not be measured) or
    missing (NaN) entered the mean of the species, e.g. 25 dex from one line in four."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values) & (values != MISSING)
    return float(np.mean(values[ok])) if np.any(ok) else MISSING


def _quad(*terms: float) -> float:
    """Sum in quadrature; 99.99 if a term is not available."""
    if any(t == MISSING for t in terms):
        return MISSING
    return np.sqrt(sum(t**2 for t in terms))


def _diff(a: float, b: float) -> float:
    """a - b; 99.99 if a or b is not available."""
    return MISSING if MISSING in (a, b) else a - b


def _clip_and_average(nelem, abu, abu_mc, nel, flag_abu, param_errors=None, select_flag=0, clip=True):
    """Common core of extract_abu_fe, extract_abu and extract_abu_isor.

    Selects the valid lines of the species nel (abundance != 99.99 and flag select_flag), sigma-clips
    them if clip (median and MAD, SCLIP sigma; only with 3 or more lines and a non-zero dispersion)
    and sets flag 1, in place, for the lines used. Returns None if there are no valid lines, otherwise a dict:
      n       number of lines used
      mean    mean of the (absolute) values of the lines used
      stat    statistical error: the MC error for one line, std/sqrt(n) for more lines
      params  (sigma_T, sigma_g, sigma_v): mean parameter errors (signed) of the lines used, 99.99 if
              not available, or None if param_errors (the three arrays abu_pT, abu_pg, abu_pv) is not given
    select_flag = 1 and clip = False use the lines already selected (salvador-merge, flags kept).
    """
    mask = (nelem == nel) & (abu != 99.99) & (flag_abu == select_flag)
    nlin = np.count_nonzero(mask)
    if nlin == 0:
        return None
    original_indices = np.where(mask)[0]
    abu_elem = abu[mask]

    if nlin == 1:
        flag_abu[original_indices[0]] = 1
        mean = abu_elem[0]
        stat = abu_mc[mask][0]
        params = tuple(_mean_valid(err[mask]) for err in param_errors) if param_errors is not None else None
        return {'n': nlin, 'mean': mean, 'stat': stat, 'params': params}    # nlin == 1 (same integer type as before)

    clip_mask = np.ones(nlin, dtype=bool)
    if clip and nlin >= 3:
        disp = mad_std(abu_elem)
        if np.isfinite(disp) and disp > 0.0:
            clipped_abu = sigma_clip(abu_elem, sigma=cfg.sclip, cenfunc='median', stdfunc=mad_std)
            if clipped_abu.count() > 0:
                clip_mask = ~clipped_abu.mask
                nlin = clipped_abu.count()

    flag_abu[original_indices[clip_mask]] = 1       # lines used
    mean = np.mean(abu_elem[clip_mask])
    stat = np.std(abu_elem[clip_mask], ddof=1) / np.sqrt(nlin)
    params = tuple(_mean_valid(err[mask][clip_mask]) for err in param_errors) if param_errors is not None else None
    return {'n': nlin, 'mean': mean, 'stat': stat, 'params': params}


def extract_abu_fe(nelem, abu, abu_mc, nel, flag_abu, abu_pT=None, abu_pg=None, abu_pv=None, select_flag=0, clip=True):
    """[Fe/H] from the Fe I (or Fe II) lines: mean, errors and the absolute abundance (params_XH).
    With the three parameter-error arrays: (nlin, [Fe/H], sigma_tot, sigma_stat, sigma_T, sigma_g,
    sigma_v, flags, sigma_params, params_XH); without: (nlin, [Fe/H], sigma_stat, flags, 99.99, params_XH)."""
    full_error_mode = all(err is not None for err in [abu_pT, abu_pg, abu_pv])
    res = _clip_and_average(nelem, abu, abu_mc, nel, flag_abu, (abu_pT, abu_pg, abu_pv) if full_error_mode else None,
                            select_flag, clip)

    abu_sol = abundance_link.get(strip_ion(get_element_symbol(nel, symbol_map))) if res is not None else None
    if res is None or np.isnan(res['mean'] - abu_sol):
        params_XH = {'abu_abs': 99.99, 's_tot': 99.99, 's_int': 99.99, 's_params': 99.99}
        return (0, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, flag_abu, 99.99, params_XH) if full_error_mode else (0, 99.99, 99.99, flag_abu, 99.99, params_XH)

    nlin, s1 = res['n'], res['stat']
    mean = res['mean'] - abu_sol                    # [Fe/H]
    mean_abs = np.floor((mean + abu_sol) * 100 + 0.5) / 100

    if full_error_mode:
        s2, s3, s4 = res['params']
        sigma_total = _quad(s1, s2, s3, s4)
        sigma_params = _quad(s2, s3, s4)
        params_XH = {'abu_abs': mean_abs, 's_tot': sigma_total, 's_int': s1, 's_params': sigma_params}
        return nlin, mean, sigma_total, s1, s2, s3, s4, flag_abu, sigma_params, params_XH
    else:
        params_XH = {'abu_abs': mean_abs, 's_tot': s1, 's_int': s1, 's_params': 99.99}
        return nlin, mean, s1, flag_abu, 99.99, params_XH


def extract_abu(nelem, abu, abu_mc, nel, feh, flag_abu, s_stat_Fe, param_errors=None, fe_errors=None, select_flag=0, clip=True):
    """[X/Fe] of the species nel and its errors, combined with those of Fe (s_stat_Fe, fe_errors);
    params_XH has the absolute abundance and the [X/H] errors. feh None ([Fe/H] not known, only in
    salvador-merge): [X/Fe] and its errors are 99.99, the absolute abundance and [X/H] errors are given.
    Returns (nlin, [X/Fe], sigma_tot, flags, sigma_params, sigma_int, params_XH), rounded to 0.01."""
    full_error_mode = param_errors is not None and fe_errors is not None
    res = _clip_and_average(nelem, abu, abu_mc, nel, flag_abu, param_errors if full_error_mode else None,
                            select_flag, clip)

    abu_sol = abundance_link.get(strip_ion(get_element_symbol(nel, symbol_map))) if res is not None else None
    if res is None or np.isnan(res['mean'] - abu_sol - (0.0 if feh is None else feh)):
        params_XH = {'abu_abs': 99.99, 's_tot': 99.99, 's_int': 99.99, 's_params': 99.99}
        return 0, 99.99, 99.99, flag_abu, 99.99, 99.99, params_XH

    nlin, sigma_stat_X = res['n'], res['stat']

    if full_error_mode:
        sigma_T_X, sigma_g_X, sigma_v_X = res['params']
        s_T_Fe, s_g_Fe, s_v_Fe = fe_errors
        # [X/Fe]: the parameter errors of X and Fe are correlated, hence the differences (with their
        # sign: species that change in the opposite way to Fe have a larger error)
        final_sigma = _quad(s_stat_Fe, sigma_stat_X, _diff(sigma_T_X, s_T_Fe), _diff(sigma_g_X, s_g_Fe), _diff(sigma_v_X, s_v_Fe))
        sigma_params = _quad(_diff(sigma_T_X, s_T_Fe), _diff(sigma_g_X, s_g_Fe), _diff(sigma_v_X, s_v_Fe))
        sigma_int = _quad(s_stat_Fe, sigma_stat_X)
        # [X/H]
        final_sigma_XH = _quad(sigma_stat_X, sigma_T_X, sigma_g_X, sigma_v_X)
        sigma_params_XH = _quad(sigma_T_X, sigma_g_X, sigma_v_X)
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

    if feh is None:         # [Fe/H] not known: no [X/Fe]
        mean = final_sigma = sigma_params = sigma_int = 99.99
        mean_abs = np.floor(res['mean'] * 100 + 0.5) / 100
    else:
        mean = res['mean'] - abu_sol - feh          # [X/Fe]
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
        final_sigma = _quad(sigma_stat_X, sigma_T_X, sigma_g_X, sigma_v_X)
        sigma_params = _quad(sigma_T_X, sigma_g_X, sigma_v_X)
    else:
        final_sigma = sigma_stat_X
        sigma_params = 999
    sigma_int = sigma_stat_X

    def to_int(value: float) -> int:
        """Rounded half up; 999 if not available."""
        return 999 if value == MISSING else int(np.floor(value + 0.5))

    return nlin, to_int(mean), to_int(final_sigma), flag_abu, to_int(sigma_params), to_int(sigma_int)


def _row(ID, code, nlin, abu, err, e_int, e_param, a_abs, e_xh, e_xh_int, e_xh_param) -> str:
    """One row of abundance_final.txt (and abundance_final_merged.txt)."""
    return (f"{ID:<30}  {code:>5.2f}  {nlin:>5}  {abu:>5.2f}  {err:>5.2f}  {e_int:>5.2f}  {e_param:>5.2f}       "
            f"{a_abs:>5.2f}  {e_xh:>5.2f}   {e_xh_int:>5.2f}   {e_xh_param:>5.2f}")


def star_abundance_rows(ID, nelem, abu, abu_mc, flag_line, param_errors=None, feh_model=None, external_feh=None,
                        select_flag=0, clip=True):
    """Rows of abundance_final.txt of one star: [Fe/H] first, then [X/Fe] of the other species
    (Fe II: [Fe II/H]). The same function is used by salvador and by salvador-merge.

    Args:
        ID: Name of the star in the rows.
        nelem, abu, abu_mc: Species code, abundance and MC error of each line.
        flag_line: Flag of each line (0 = usable): updated (1 = used) and returned.
        param_errors: (d_Teff, d_logg, d_vt) of each line, or None.
        feh_model: [Fe/H] used without Fe I lines: the [M/H] of the model (salvador), or None
            (salvador-merge: [X/Fe] not available).
        external_feh: (value, error) of an external [Fe/H] (salvador-merge), or None.
        select_flag, clip: lines to use and sigma-clipping (see _clip_and_average).

    Returns:
        (list of rows, flags).
    """
    rows = []
    errors = param_errors is not None
    fe_errors = None
    has_fe = 26.00 in nelem
    if has_fe:
        if errors:
            nlin, feh, sigma, s1, s2, s3, s4, flag_line, s_param, params_XH = extract_abu_fe(
                nelem, abu, abu_mc, 26.00, flag_line, *param_errors, select_flag=select_flag, clip=clip)
            fe_errors = (s2, s3, s4)
        else:
            nlin, feh, sigma, flag_line, s_param, params_XH = extract_abu_fe(nelem, abu, abu_mc, 26.00, flag_line,
                                                                             select_flag=select_flag, clip=clip)
            s1 = sigma
        fe_row = (nlin, feh, sigma, s1, s_param, params_XH['abu_abs'], params_XH['s_tot'], params_XH['s_int'], params_XH['s_params'])
    elif feh_model is not None:     # salvador without Fe I lines: the [M/H] of the model
        feh = feh_model
        if errors:
            s1, fe_errors = 0.0, (0.0, 0.0, 0.0)
            fe_row = (0, feh, 0.0, 0.0, 0.0, abu_absolute('Fe', feh, feh), 99.99, 99.99, 99.99)
        else:
            s1 = 99.99
            fe_row = (0, feh, 99.99, 99.99, 99.99, abu_absolute('Fe', feh, feh), 99.99, 99.99, 99.99)
    else:                           # salvador-merge without Fe I lines: [Fe/H] not known
        feh, s1 = None, 99.99
        fe_errors = (0.0, 0.0, 0.0) if errors else None
        fe_row = (0, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99)

    if external_feh is not None:    # salvador-merge: external [Fe/H] and its error
        feh, s1 = external_feh
        if errors and fe_errors is None:
            fe_errors = (0.0, 0.0, 0.0)
        fe_row = (0, feh, s1, s1, 99.99, abu_absolute('Fe', feh, feh), s1, s1, 99.99)
    rows.append(_row(ID, 26.00, *fe_row))

    for nel in np.unique(nelem):
        if nel == 26.00:
            continue
        if nel == 26.01:
            if errors:
                nlin, mean, sigma, s_int, *_, flag_line, s_param, params_XH = extract_abu_fe(
                    nelem, abu, abu_mc, nel, flag_line, *param_errors, select_flag=select_flag, clip=clip)
            else:
                nlin, mean, sigma, flag_line, s_param, params_XH = extract_abu_fe(nelem, abu, abu_mc, nel, flag_line,
                                                                                  select_flag=select_flag, clip=clip)
                s_int = sigma
        else:
            nlin, mean, sigma, flag_line, s_param, s_int, params_XH = extract_abu(
                nelem, abu, abu_mc, nel, feh, flag_line, s1, param_errors=param_errors if errors else None,
                fe_errors=fe_errors, select_flag=select_flag, clip=clip)
        rows.append(_row(ID, nel, nlin, mean, sigma, s_int, s_param, params_XH['abu_abs'], params_XH['s_tot'],
                         params_XH['s_int'], params_XH['s_params']))
    return rows, flag_line


def get_coeffs(xx, yy):
    """Linear fit y = m x + c. Returns (fitted values, m, c, error of m, error of c)."""
    result = stats.linregress(xx, yy)

    m = result.slope
    c = result.intercept
    em = result.stderr
    ec = result.intercept_stderr

    y_fit = m * xx + c
    return y_fit, m, c, em, ec
