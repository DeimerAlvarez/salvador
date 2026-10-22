"""The salvador command: abundance analysis of the stars of input.salvador.

Run it in the folder with input.salvador (and the files it names): for every star the lines of
its linelist are analysed (in parallel with NPROC > 1), the results are written to <star>.fit,
<star>_isor.fit, <star>_rot.fit and abundance_final*.txt, and the plots to <star>.pdf.
This is the former salvador.py: same steps, same order, same output files.
"""
import argparse
import importlib.metadata
import multiprocessing
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfWriter

from . import __version__
from .config import WORK_DIR, c, cfg, configure_warnings, path_tmp, patterns_to_delete, plt, select_backend
from .elements import abu_absolute, define_abu0, get_element_symbol, solar_table_path, strip_ion, symbol_map
from .fit_files import (add_errors_to_file, append_flag_column, check_if_has_data, count_header_cols,
                        get_header_names, read_fit_file)
from .k2_interface import autok, cleanup_linelist_cut
from .line_analysis import ABU_KEYS, measure_line
from .normalization import find_clipped_regions, load_cw_regions
from .parallel import cleanup_parallel_work, ensure_models, init_parallel_worker, parallel_available
from .plotting import apply_plot_style
from .spectra_io import (cleaning_tmp_files, grid_name_fields, grid_name_matches, is_line_in_grid,
                         process_names_grid, read_spec)
from .summary import extract_abu, extract_abu_fe, extract_abu_isor, get_coeffs

###
### ANALYSIS OF ONE LINE
###
### analyse_line computes everything for one line (nominal fit and parameter-error fits) and
### returns the results without writing files; write_line_result writes them, always in the
### order of the linelist. With NPROC > 1 analyse_line runs in parallel worker processes.
STAR = {}       # star-level data, filled for each star in the main loop
curr_path = Path.cwd()  # working directory of the run (set by run())
start_time = time.time()    # start of the run (set by run()): only newer temporary files are deleted

def analyse_line(task):
    """Analysis of one line of the linelist: nominal fit (measure_line) and, if the star has the
    parameter errors, the fits with Teff + dTeff, logg + dlogg (ionized species) and vt + dvt
    (EW > 50 mA). Returns a dictionary with the results; writes nothing (see write_line_result),
    so it can run in parallel workers."""
    tana, line, nelem, res, wv_l, wv_r, delta, ct_elem = task
    S = STAR
    wv, fl = S['wv'], S['fl']
    teff, logg, met, alpha, vt = S['teff'], S['logg'], S['met'], S['alpha'], S['vt']
    abu_var, name_star = S['abu_var'], S['name_star']
    T_err, g_err, v_err = S['T_err'], S['g_err'], S['v_err']
    elem = get_element_symbol(nelem, symbol_map)
    rec = {'line': line, 'nelem': nelem, 'tana': tana, 'elem': elem, 'skipped': False,
           'flag': 0, 'errors': None, 'Abu': 99.99}

    # CHECK: lines that cannot be analysed are skipped with a message (before: a crash later on)
    valid_keys = ABU_KEYS + ('ROTAT',)
    if tana not in valid_keys:
        reason = f"unknown keyword '{tana}' (valid: {', '.join(valid_keys)})"
    elif elem is None:
        reason = f'element code {nelem:.2f} not in solar_abu.txt'
    elif tana == 'ROTAT' and cfg.grid == 'y':
        reason = 'ROTAT needs synthetics from K2: not available with PATH_SYN (grid mode)'
    elif tana == 'ABU2' and not any(float(entry[1]) == line for entry in S['edges_abu2']):
        reason = 'ABU2 needs 9 columns (the second pair of edges)'
    else:
        reason = None
    if reason:
        print(f'---\nLine {line:.3f} skipped: {reason}')
        rec.update(skipped=True, reason=reason)
        return rec

    if cfg.type_norm == 2:
        mask = (S['w1_star'] >= line - 50) & (S['w1_star'] <= line + 50)
        w1, w2 = S['w1_star'][mask], S['w2_star'][mask]

        mask = (wv >= line - 50) & (wv <= line + 50)
        wave, flux = wv, fl

    elif cfg.type_norm == 1:
        w1, w2 = 0, 0
        mask = (wv >= line - 50) & (wv <= line + 50)
        wave, flux = wv[mask], fl[mask]

    elif cfg.type_norm == 0:
        w1, w2 = 0, 0
        wave, flux = wv, fl

    elif cfg.type_norm == 3:    #normalization for specific regions. Work on it
        w1, w2 = 0, 0
        mask = (wv >= line - 50) & (wv <= line + 50)
        wave, flux = wv[mask], fl[mask]

    # CALCULATION OF EDGES
    N_sig = 3
    fwhm = line / res
    sigma = fwhm / 2.35
    wind_left = line - N_sig*sigma
    wind_right = line + N_sig*sigma

    wind_l, wind_r = (wv_l, wv_r) if wv_l and wv_r else (wind_left, wind_right)

    ###
    arg_extra = {}

    if tana == 'ABU2':
        wind_l, wind_r = wv_l, wv_r

        wv_l1, wv_r1 = None, None
        edges = [entry for entry in S['edges_abu2'] if float(entry[1]) == line]
        for entry in edges:
            wv_r1 = float(entry[-1])
            wv_l1 = float(entry[-2])

        arg_extra['wind_l1'] = wv_l1
        arg_extra['wind_r1'] = wv_r1

    elif tana == 'ROTAT':
        # optional columns of the linelist: initial absolute abundance, +- range of the abundance,
        # minimum and maximum rotation. Without them the input.salvador values (or defaults) are used.
        for entry in [r for r in S['rows_rotat'] if float(r[1]) == line]:
            arg_extra['abu_init'] = float(entry[-4])
            arg_extra['abu_span'] = float(entry[-3])
            arg_extra['rot_min'] = float(entry[-2])
            arg_extra['rot_max'] = float(entry[-1])

    # SYNTHETIC
    if cfg.grid == 'y':
        names_syn = np.atleast_1d(S['names_syn_all'])

        elem_tmp = strip_ion(elem)
        # FIX: exact match of the fields of the names (see grid_name_matches)
        mask = np.array([grid_name_matches(name, S['teff1'], S['logg1'], S['vt1'], elem_tmp) for name in names_syn], dtype=bool)

        names_syn_tmp = names_syn[mask]
        mets = {grid_name_fields(name)['met'] for name in names_syn_tmp}
        if len(mets) > 1:
            print(f"WARNING ({cfg.path_syn}): synthetics of line {line} with different [M/H] in the names ({', '.join(sorted(mets))}): all are used")
        names_syn_tmp1, abu_syn = process_names_grid(names_syn_tmp)
        in_grid = np.array([is_line_in_grid(filename, line) for filename in names_syn_tmp1], dtype=bool)
        names_syn = [filename for filename, ok in zip(names_syn_tmp1, in_grid) if ok]
        abu_syn = np.asarray(abu_syn)[in_grid] if in_grid.size else np.asarray(abu_syn)
        if not names_syn:
            print('---')
            print(f"No synthetic spectra for line {line}")
            rec.update(skipped=True, reason='no synthetic spectra in the grid')
            return rec

    elif cfg.grid == 'n':
        names_syn = np.array(['aaa', 'bbb'])
        abu_syn = np.array([0.0, 1.0])

    print('')
    print('------------------------------------------')
    print('LINE = %.3f [A] --- ' % (line), 'ELEM: ', elem)
    print('')

    ###
    param_kur3 = {'teff': teff, 'logg': logg, 'met': met, 'alpha': alpha, 'vt': vt, 'res': res,
    'line': line, 'nelem': nelem, 'abu_var': abu_var, 'delta': delta, 'tana': tana}

    wave_cen = np.round((np.min(wave) + np.max(wave)) / 2, 3)
    abu0_fix = define_abu0(26, alpha, met, 0.0, 'Fe')
    fixed_params = {'teff': teff, 'logg': logg, 'met': met, 'alpha': alpha, 'vt': vt, 'res': res,
    'line': wave_cen, 'nelem': '26', 'abu_var': abu_var, 'delta': 0.0, 'tana': 'ABU', 'abu0': abu0_fix}

    ####################   ABUNDANCE DETERMINATION   ####################
    results = measure_line(wave, flux, names_syn, wind_l, wind_r, elem, param_kur3, name_star, abu_syn, w1, w2, ct_elem, fixed_params, **arg_extra)
    #####################################################################

    ans = results['answer']
    rec['ans'] = ans
    if ans in ['y', 'r'] and not tana.startswith('ROTAT'):
        Abu = results['abu_fin']
        ew =  results['ew_k2']
        emc = results['abu_err_mc']
        abu0 = results['abu0']
        EP = results['EP']
        loggf = results['loggf']
        rec.update(Abu=Abu, ew=ew, emc=emc, EP=EP, loggf=loggf)

        ewr = np.log10((max(ew, 1e-9) / 1000) / line)
        if Abu != 99.99 and not (cfg.ewr_min <= ewr <= cfg.ewr_max) and not tana.startswith(('ROTAT', 'MOLEC', 'ABU2', 'ISOR')):
            rec['flag'] = -1

        if Abu != 99.99 and not (cfg.ep_min <= EP <= cfg.ep_max) and EP != 99.999 and not tana.startswith(('ROTAT', 'MOLEC', 'ABU2', 'ISOR')):
            rec['flag'] = -2

        if Abu == 99.99 and not tana.startswith(('ISOR')):
            rec['flag'] = -3

    elif ans in ['y', 'r'] and tana == 'ROTAT':
        rec.update(rotat=results['rot_fin'], err_rot=results['abu_err_mc'],
                   abu_rot=results['abu_rot'], err_abu_rot=results['err_abu_rot'])
        return rec      # no parameter errors for ROTAT (before, this case stopped with an error)

    elif ans == 'e':
        return rec

    ### errors due to the parameters: each parameter is varied by its error (vt only for EW > 50 mA, logg only for ionized species)
    if T_err is not None and ans != 'r':
        if tana != 'ISOR':
            delta = Abu - abu0
        elif tana == 'ISOR':
            delta = Abu

        ### +T_eff
        T_tmp = int(teff + T_err)
        param_kur3 = {'teff': T_tmp, 'logg': logg, 'met': met, 'alpha': alpha, 'vt': vt, 'res': res,
        'line': line, 'nelem': nelem, 'abu_var': abu_var, 'delta': delta, 'tana': tana}
        Abu_pT = Abu
        if Abu != 99.99 and T_err != 0 and rec['flag'] == 0:
            print(f"Variation of Teff: \u0394T = {T_err:.0f} [K] \n")
            results=measure_line(wave, flux, names_syn, wind_l, wind_r, elem, param_kur3, name_star, abu_syn, w1, w2, ct_elem, fixed_params, **arg_extra, save_plot=False)
            Abu_pT = results['abu_fin']

        ### +logg
        Abu_pg = Abu
        logg_tmp = float(logg + g_err)
        param_kur3 = {'teff': teff, 'logg': logg_tmp, 'met': met, 'alpha': alpha, 'vt': vt, 'res': res,
                'line': line, 'nelem': nelem, 'abu_var': abu_var, 'delta': delta, 'tana': tana}
        if "{:.2f}".format(nelem).endswith("01") and Abu!=99.99 and g_err != 0 and rec['flag'] == 0 and tana != 'ISOR':
            print('---')
            print('Variation of logg: \u0394logg = %.2f [dex]' % (g_err), '\n')
            results=measure_line(wave, flux, names_syn, wind_l, wind_r, elem, param_kur3, name_star, abu_syn, w1, w2, ct_elem, fixed_params, **arg_extra, save_plot=False)
            Abu_pg = results['abu_fin']

        elif Abu!=99.99 and g_err != 0 and tana == 'ISOR':
            print('---')
            print('Variation of logg: \u0394logg = %.2f [dex]' % (g_err), '\n')
            results=measure_line(wave, flux, names_syn, wind_l, wind_r, elem, param_kur3, name_star, abu_syn, w1, w2, ct_elem, fixed_params, **arg_extra, save_plot=False)
            Abu_pg = results['abu_fin']

        ### +vt
        Abu_pv = Abu
        if ew > 50 and Abu!=99.99 and v_err != 0 and rec['flag'] == 0:
            print('---')
            print('Variation of v_t: \u0394vt = %.2f [km/s]' % (v_err), '\n')
            vt_tmp = float(vt + v_err)
            param_kur3 = {'teff': teff, 'logg': logg, 'met': met, 'alpha': alpha, 'vt': vt_tmp, 'res': res,
            'line': line, 'nelem': nelem, 'abu_var': abu_var, 'delta': delta, 'tana': tana}
            results=measure_line(wave, flux, names_syn, wind_l, wind_r, elem, param_kur3, name_star, abu_syn, w1, w2, ct_elem, fixed_params, **arg_extra, save_plot=False)
            Abu_pv = results['abu_fin']

        ###
        delta_pT = np.abs(Abu - Abu_pT) if Abu_pT != 99.99 else 99.99
        delta_pg = np.abs(Abu - Abu_pg) if Abu_pg != 99.99 else 99.99
        delta_pv = np.abs(Abu - Abu_pv) if Abu_pv != 99.99 else 99.99
        rec['errors'] = (delta_pT, delta_pg, delta_pv)

    return rec


def write_line_result(rec, abu_file, abu_file_isor, flags, kept):
    """Write the results of one line, exactly as the former loop did."""
    kept.append(not rec['skipped'])
    if rec['skipped']:
        return
    flags.append(rec['flag'])
    ans, tana, line, nelem = rec['ans'], rec['tana'], rec['line'], rec['nelem']
    name_star = STAR['name_star']

    if ans in ['y', 'r'] and not tana.startswith('ROTAT'):
        Abu, ew, EP, loggf, emc = rec['Abu'], rec['ew'], rec['EP'], rec['loggf'], rec['emc']
        if tana != 'ISOR':
            with open(abu_file, 'a') as f:
                f.write(f"{name_star:<30}  {line:>8.3f}  {nelem:>5.2f}  {Abu:>5.2f}  {ew:>6.2f}  {EP:>6.3f}  {loggf:>6.3f}  {emc:>5.2f}\n")
        elif tana == 'ISOR':
            with open(abu_file_isor, 'a') as f:
                f.write(f"{name_star:<30}  {line:>8.3f}  {nelem:>5.2f}  {Abu:>5.0f}  {emc:>3.0f}\n")

    elif ans in ['y', 'r'] and tana == 'ROTAT':
        # NEW: rotation results in their own file star_rot.fit (before, in star.fit, where the summary
        # read the rotation value as an abundance)
        abu_file_rot = name_star + '_rot.fit'
        if not os.path.exists(abu_file_rot):
            with open(abu_file_rot, 'w') as f:
                f.write('#SPECTRUM                           LINE   ELEM    ROT   eROT    ABU   eABU\n')
        # rejected line (r): 99.99, as in the abundance files (with 0 decimals it would read 100)
        rot_txt = '99.99' if rec['rotat'] == 99.99 else f"{rec['rotat']:>5.0f}"
        # error of the rotation as an integer (km/s), rounded half up
        err_txt = '99.99' if rec['err_rot'] == 99.99 else f"{int(np.floor(rec['err_rot'] + 0.5)):>5d}"
        # absolute abundance of the rotation fit and its Monte Carlo error (0.00 if the abundance is fixed)
        abu_txt = f"{rec['abu_rot']:>5.2f}"
        err_abu_txt = f"{rec['err_abu_rot']:>5.2f}"
        with open(abu_file_rot, 'a') as f:
            f.write(f"{name_star:<30}  {line:>8.3f}  {nelem:>5.2f}  {rot_txt:>5}  {err_txt:>5}  {abu_txt:>5}  {err_abu_txt:>5}\n")

    elif ans == 'e':
        cleaning_tmp_files(patterns_to_delete, directory=curr_path, newer_than=start_time)
        cleanup_linelist_cut()
        if os.path.isdir(path_tmp):
            shutil.rmtree(path_tmp)
        sys.exit()

    if rec['errors'] is not None:
        delta_pT, delta_pg, delta_pv = rec['errors']
        new_headers = ["d_Teff", "d_logg", " d_vt"]
        # FIX: only in the file where this line was written (before: also the last row of the other file)
        if tana != 'ISOR':
            add_errors_to_file(abu_file, new_headers, 8, f"{delta_pT:>5.2f}", f"{delta_pg:>5.2f}", f"{delta_pv:>5.2f}")
        else:
            add_errors_to_file(abu_file_isor, new_headers, 5, f"{delta_pT:>6.0f}", f"{delta_pg:>5.0f}", f"{delta_pv:>4.0f}")


def prepare_parallel_star(tasks):
    """Before starting the workers: compute the shared resources that workers would otherwise
    build at the same time: the model atmospheres and the NORM = 2 reference synthetic."""
    S = STAR
    models = [(S['teff'], S['logg'], S['met'], S['alpha'], S['vt'])]
    if S['T_err'] is not None:
        models += [(int(S['teff'] + S['T_err']), S['logg'], S['met'], S['alpha'], S['vt']),
                   (S['teff'], float(S['logg'] + S['g_err']), S['met'], S['alpha'], S['vt']),
                   (S['teff'], S['logg'], S['met'], S['alpha'], float(S['vt'] + S['v_err']))]
    if cfg.grid == 'n':     # FIX: with a grid (PATH_SYN) K2 is not needed (before: NPROC > 1 required K2)
        print('\nPreparing model atmospheres for the parallel analysis')
        ensure_models(models)

    if cfg.type_norm == 2 and cfg.grid == 'n':
        wv = S['wv']
        res0 = tasks[0][3]
        wave_cen = np.round((np.min(wv) + np.max(wv)) / 2, 3)
        abu0_fix = define_abu0(26, S['alpha'], S['met'], 0.0, 'Fe')
        fixed_params = {'teff': S['teff'], 'logg': S['logg'], 'met': S['met'], 'alpha': S['alpha'], 'vt': S['vt'], 'res': res0,
        'line': wave_cen, 'nelem': '26', 'abu_var': S['abu_var'], 'delta': 0.0, 'tana': 'ABU', 'abu0': abu0_fix}
        autok(fixed_params, 0.0, np.round(wv[-1] - wv[0], 3))


def run() -> None:
    """Analysis of all the stars of the input file (cfg.input_stars): the former main block of
    salvador.py, unchanged. The settings (cfg) must be loaded before (see main)."""
    global curr_path, start_time
    start_time = time.time()

    # CHECK: K2 is needed unless a grid of synthetics is used (before: 'k2.py: command not found'
    # and a confusing error at the first line)
    if cfg.grid == 'n' and shutil.which('k2.py') is None:
        sys.exit('Error: k2.py not found in PATH. salvador computes the synthetic spectra with K2: '
                 'install it and make sure k2.py is executable and in your PATH.')

    ### main block of analysis
    print("#########################################################")
    print("")
    print("   ####    #    #      #   #    #    ####    ###   ####  ")
    print("  #       # #   #      #   #   # #   #   #  #   #  #   # ")
    print("   ###   #   #  #      #   #  #   #  #   #  #   #  ####  ")
    print("      #  # # #  #       # #   # # #  #   #  #   #  #   # ")
    print("  ####   #   #  #####    #    #   #  ####    ###   #   # ")
    print("")
    print("#########################################################")

    print('')

    ###
    with open('abundance_final.txt','w') as f:
        f.write('#STAR\t\t= name of the spectrum \n')
        f.write('#CODE\t\t= element code in kurucz format \n')
        f.write('#N_LIN\t\t= number of lines used to compute the mean \n')
        f.write('#ABU\t\t= abundance in the format [Fe/H] or [X/Fe] \n')
        f.write('#ERROR\t\t= total error of ABU. Sum in quadrature of of E_INT + E_PARAM \n')
        f.write('#E_INT\t\t= internal error. Sigma / sqrt(N_LIN) or MC error \n')
        f.write('#E_PARAM\t= atmospheric params error. Sum in quadrature of errors due to Teff, logg, and vt \n')
        f.write('#A_ABS\t\t= absolute abundance \n')
        f.write('#E_X/H\t\t= total error of abundance in [X/H] format. Sum in quadrature of of E_X/H_INT + E_X/H_PARAM \n')
        f.write('#E_X/H_INT\t= internal error. Sigma / sqrt(N_LIN) or MC error \n')
        f.write('#E_X/H_PARAM\t= atmospheric params error. Sum in quadrature of error due to Teff, logg, and vt \n')
        f.write('#' + '-'*110 + '\n')
        f.write('#STAR                            CODE  N_LIN    ABU  ERROR  E_INT  E_PARAM     A_ABS   E_XH E_XH_INT E_XH_PARAM\n')

    ###
    with open('abundance_final_isor.txt','w') as f:
        f.write('#STAR\t\t= name of the spectrum \n')
        f.write('#CODE\t\t= element code in kurucz format \n')
        f.write('#N_LIN\t\t= number of lines used to compute the mean \n')
        f.write('#ISOR\t\t= isotopic ratio \n')
        f.write('#ERROR\t\t= total error of ISOR. Sum in quadrature of of E_INT + E_PARAM \n')
        f.write('#E_INT\t\t= internal error. Sigma / sqrt(N_LIN) or MC error \n')
        f.write('#E_PARAM\t= atmospheric params error. Sum in quadrature of error due to Teff, logg, and vt \n')
        f.write('#' + '-'*110 + '\n')
        f.write('#STAR                            CODE  N_LIN   ISOR  ERROR  E_INT  E_PARAM\n')


    ### logs of the parallel workers of a previous run
    cleanup_parallel_work(keep_logs=False)

    ### READING THE LISTA STARS AND THE CORRESPONDING PARAMETERS FOR EACH STAR PRESENT IN THE LIST
    ID, ABU_FILES, LINELIST, CONT_WIND = np.genfromtxt(cfg.input_stars, comments='#', usecols=(0,7,8,9), dtype='str', unpack=True)
    TEFF, LOGG, MET, ALPHA, VT, RV = np.genfromtxt(cfg.input_stars, comments='#', usecols=(1,2,3,4,5,6), dtype='float', unpack=True)

    ### CHECK THAT ALL VARIABLES ARE ARRAYS (EVEN IF THE FILE HAS ONLY ONE ROW)
    arrays = [ID, ABU_FILES, LINELIST, CONT_WIND, TEFF, LOGG, MET, ALPHA, VT, RV]
    ID, ABU_FILES, LINELIST, CONT_WIND, TEFF, LOGG, MET, ALPHA, VT, RV = [np.atleast_1d(a) for a in arrays]

    curr_path = Path.cwd()

    ### SPEED: files that were re-read for every line are now read once
    with open(cfg.input_stars, "r") as f:
        stars_rows = [row.strip().split() for row in f if not row.lstrip().startswith("#") and row.strip()]

    # CHECK: a row with 11 or 12 columns silently had no parameter errors (they need 3 values)
    for _cols in stars_rows:
        if len(_cols) in (11, 12):
            print(f"WARNING ({cfg.input_stars}): star {_cols[0]} has {len(_cols)} columns: the errors on the parameters "
                  f"need 3 values (dTeff, dlogg, dvt, 13 columns in total), so they are NOT computed for this star")
        elif len(_cols) not in (10, 13):
            print(f"WARNING ({cfg.input_stars}): star {_cols[0]} has {len(_cols)} columns (expected 10, or 13 with the errors)")

    if cfg.grid == 'y':
        names_syn_all = np.genfromtxt(cfg.path_syn, comments='#', usecols=(0), dtype=None, encoding='utf-8', unpack=True)

    ###
    for ids, rv, cw, teff, logg, met, alpha, vt, linelist_file, abu_var in zip(ID, RV, CONT_WIND, TEFF, LOGG, MET, ALPHA, VT, LINELIST, ABU_FILES):
        # CHECK: alpha is the K2 code of the alpha-enhancement (before: a crash in write_name_syn)
        if int(alpha) not in (1, 2, 3, 4, 5):
            print(f'Star {ids} skipped: alpha = {alpha} in {cfg.input_stars} must be 1-5 (K2 code of the alpha-enhancement)')
            continue
        if os.path.exists(path_tmp):
            shutil.rmtree(path_tmp)

        os.makedirs(path_tmp)


        plot_file = 'nomi_plot.txt'
        path_plot_file = path_tmp / plot_file
        with open(path_plot_file, "w") as file:
            pass

        ### reads the spectrum
        wv, fl = read_spec(ids)

        ### check if the spectrum is okay or not
        if wv is None or fl is None:
            print(f"Spectrum {ids} not read")
            continue

        ### correction for the radial velocity
        wv = wv / (1 + (rv / c))

        ### print of the info of the star under analysis
        print('########################################')
        print('OBSERVED SPECTRUM = ', ids)
        print('TEFF  = %.0f    [K]' % (teff))
        print('LOGG  = %.2f    [dex]' % (logg))
        print('VT    = %.2f    [km/s]' % (vt))
        print('START = %.2f [A]' % (wv[0]))
        print('END   = %.2f [A]' % (wv[-1]))
        print('------------------------------------------')

        teff1 = str(teff.astype(int))
        logg1 = f"{logg:.2f}".replace('.', '')
        vt1 = f"{vt:.2f}".replace('.', '')
        ### READING LINELIST
        try:
            LINES, NELEM, WV_L, WV_R, DELTA = np.genfromtxt(linelist_file, comments='#', usecols=(1,2,4,5,6), dtype='float', unpack=True)
            RES = np.genfromtxt(linelist_file, comments='#', usecols=(3), dtype='int', unpack=True)
            T_ABU = np.genfromtxt(linelist_file, comments='#', usecols=(0), dtype='str', unpack=True)

            LINES = np.atleast_1d(LINES)
            NELEM = np.atleast_1d(NELEM)
            WV_L = np.atleast_1d(WV_L)
            WV_R = np.atleast_1d(WV_R)
            DELTA = np.atleast_1d(DELTA)
            RES = np.atleast_1d(RES)
            T_ABU = np.atleast_1d(T_ABU)

            _, unique_indices = np.unique(LINES, return_index=True)
            sorted_indices = np.sort(unique_indices)
            # CHECK: rows with the same wavelength as an earlier row are ignored (before: silently)
            for k in np.setdiff1d(np.arange(LINES.size), sorted_indices):
                print(f"WARNING ({linelist_file}): {T_ABU[k]} line {LINES[k]:.3f} has the same wavelength as an earlier row: ignored")

            LINES = LINES[sorted_indices]
            NELEM = NELEM[sorted_indices]
            WV_L = WV_L[sorted_indices]
            WV_R = WV_R[sorted_indices]
            DELTA = DELTA[sorted_indices]
            RES = RES[sorted_indices]
            T_ABU = T_ABU[sorted_indices]

        except FileNotFoundError:
            print(f"Linelist {linelist_file} does not exist")
            continue

        ### exclusion of all the lines outside the spectrum
        index = (LINES >= min(wv)) & (LINES <= max(wv))
        lines_out = LINES[~index]
        T_ABU, LINES, NELEM, RES, WV_L, WV_R, DELTA = T_ABU[index], LINES[index], NELEM[index], RES[index], WV_L[index], WV_R[index], DELTA[index]
        if lines_out.size > 0:
            print("--- Lines excluded for being outside the spectral range ---")
            for line_val in lines_out:
                print(f"Line {line_val:.3f} outside the spectrum.")

        ### counting how many lines of each element are present. Needed for the MC. If for a given element there are less than 10 lines then the MC is computed
        counts = Counter(NELEM)
        COUNTS_ELEM = np.array([counts[c] for c in NELEM])

        ### CREATION OF FILE ABUNDANCES
        name_star = ids.split('.')[0]

        abu_file = name_star+'.fit'
        with open(abu_file,'w') as f:
            f.write('#   FLAG =  1  (USED LINE)\n')
            f.write('#   FLAG =  0  (REJ. SIGMA-CLIPPING)\n')
            f.write('#   FLAG = -1  (REJ. EW)\n')
            f.write('#   FLAG = -2  (REJ. E.P.)\n')
            f.write('#   FLAG = -3  (REJ. ABU)\n')
            f.write('#' + '-'*39 + '\n')
            f.write('#SPECTRUM                           LINE   ELEM    ABU      EW      EP  log_gf    eMC\n')

        # star_rot.fit is created at the first ROTAT line; remove the one of a previous run
        if os.path.exists(name_star + '_rot.fit'):
            os.remove(name_star + '_rot.fit')

        abu_file_isor = name_star+'_isor.fit'
        with open(abu_file_isor,'w') as f:
            f.write('#   FLAG =  1  (USED LINE)\n')
            f.write('#   FLAG =  0  (REJ. SIGMA-CLIPPING)\n')
            f.write('#' + '-'*39 + '\n')
            f.write('#SPECTRUM                           LINE   ELEM   ISOR  eMC\n')


        ### SPEED: per-star quantities, computed once instead of for every line ---------------
        # errors on the atmospheric parameters (rows with exactly 13 columns)
        T_err, g_err, v_err = None, None, None
        for entry in [cols for cols in stars_rows if cols[0] == ids and len(cols) == 13]:
            T_err = int(float(entry[-3]))   # FIX: '50.0' was a crash
            g_err = float(entry[-2])
            v_err = float(entry[-1])

        # ABU2 edges: linelist rows with exactly 9 columns
        with open(linelist_file, "r") as f:
            linelist_rows = [line_row.strip().split() for line_row in f if not line_row.lstrip().startswith("#") and line_row.strip()]
        # CHECK: columns of each row (7; 9 for ABU2 with the second pair of edges; 11 for ROTAT with options)
        for _cols in linelist_rows:
            _n, _key = len(_cols), _cols[0]
            if _key == 'ABU2' and _n != 9:
                print(f"WARNING ({linelist_file}): ABU2 line {_cols[1]} has {_n} columns: it needs 9 (the second pair of edges)")
            elif _n not in (7, 9, 11) or (_n == 9 and _key != 'ABU2') or (_n == 11 and _key != 'ROTAT'):
                print(f"WARNING ({linelist_file}): {_key} line {_cols[1]} has {_n} columns: the extra columns are ignored")
        edges_abu2 = [cols for cols in linelist_rows if len(cols) == 9]
        # ROTAT options: rows with 11 columns (abu_init, abu_span, rot_min, rot_max)
        rows_rotat = [cols for cols in linelist_rows if len(cols) == 11 and cols[0] == 'ROTAT']

        # continuum regions (NORM = 2): same file and same spectrum for every line, so the
        # sigma-clipping of all the regions is done once here; each line only selects +-50 A.
        if cfg.type_norm == 2:
            w1_star, w2_star = load_cw_regions(cw)
            w1_star, w2_star = find_clipped_regions(w1_star, w2_star, wv, fl)
        # ----------------------------------------------------------------------------------------

        col_names_err = ['abu_pT', 'abu_pg', 'abu_pv']
        col_names_base_isor = ['ID', 'line', 'nelem', 'abu', 'abu_mc']
        col_names_base = ['ID', 'line', 'nelem', 'abu', 'ew', 'EP', 'loggf', 'abu_mc']

        ### star-level data used by analyse_line (inherited by the parallel workers)
        STAR.update(wv=wv, fl=fl, teff=teff, logg=logg, met=met, alpha=alpha, vt=vt,
                    abu_var=os.path.abspath(abu_var), name_star=name_star, T_err=T_err, g_err=g_err, v_err=v_err,
                    edges_abu2=edges_abu2, rows_rotat=rows_rotat, teff1=teff1, logg1=logg1, vt1=vt1)
        if cfg.type_norm == 2:
            STAR.update(w1_star=w1_star, w2_star=w2_star)
        if cfg.grid == 'y':
            STAR.update(names_syn_all=names_syn_all)

        tasks = list(zip(T_ABU, LINES, NELEM, RES, WV_L, WV_R, DELTA, COUNTS_ELEM))
        n_workers = parallel_available(len(tasks))

        ### analysis of each line present in the linelist file
        flags, kept = [], []
        if n_workers == 1:
            for task in tasks:
                rec = analyse_line(task)
                write_line_result(rec, abu_file, abu_file_isor, flags, kept)
        else:
            prepare_parallel_star(tasks)
            print(f'Analysing {len(tasks)} lines with {n_workers} parallel processes (logs in {WORK_DIR.name}/)')
            pool = multiprocessing.get_context('fork').Pool(n_workers, initializer=init_parallel_worker)
            try:
                for k, rec in enumerate(pool.imap(analyse_line, tasks), 1):
                    write_line_result(rec, abu_file, abu_file_isor, flags, kept)
                    if rec['skipped']:
                        print(f'[{k:>3}/{len(tasks)}] {rec["line"]:9.3f} A  skipped: {rec["reason"]}')
                    elif rec['tana'] == 'ROTAT':
                        print(f'[{k:>3}/{len(tasks)}] {rec["line"]:9.3f} A  {rec["elem"]:<5} {rec["tana"]:<6} vsini  = {rec["rotat"]:.0f} +- {np.floor(rec["err_rot"] + 0.5):.0f} km/s, A = {rec["abu_rot"]:.2f} +- {rec["err_abu_rot"]:.2f}')
                    else:
                        print(f'[{k:>3}/{len(tasks)}] {rec["line"]:9.3f} A  {rec["elem"]:<5} {rec["tana"]:<6} result = {rec["Abu"]:.2f}')
                print('')
                pool.close()
            except BaseException:
                pool.terminate()
                print(f'Parallel analysis stopped: see the worker logs in {WORK_DIR}/')
                raise
            pool.join()
            cleanup_parallel_work()

        # lines without synthetic spectra (grid mode) are not in the .fit file
        T_ABU = T_ABU[np.array(kept, dtype=bool)] if kept else T_ABU[:0]
        flag_line_all = np.array(flags, dtype=int)

        # END OF THE ANALYSIS OF A STAR

        # SINGLE PDF WITH ALL THE PLOTS
        plot_list_file = path_tmp / plot_file
        try:
            if not plot_list_file.exists():
                print("Warning: Plot list file not found. No PDF will be created.")

            pdf_filenames = np.unique(np.genfromtxt(plot_list_file, dtype=str, defaultfmt=" "))
            pdf_filenames = np.atleast_1d(pdf_filenames)

            destination_path = curr_path / f"{name_star}.pdf"

            if pdf_filenames.size == 0:
                print("No plot files to merge.")

            elif pdf_filenames.size == 1:
                source_path = path_tmp / str(pdf_filenames[0])
                if source_path.exists():
                    shutil.move(source_path, destination_path)

            else:
                pdf_merger = PdfWriter()
                for filename in pdf_filenames:
                    source_path = path_tmp / filename
                    if source_path.exists():
                        pdf_merger.append(source_path)

                pdf_merger.write(destination_path)
                pdf_merger.close()

        finally:
            if path_tmp.is_dir():
                shutil.rmtree(path_tmp)

        ### SPLIT BETWEEN ISOR AND OTHER KEYWORDS
        mask_isor = (T_ABU == 'ISOR')
        mask_others = (T_ABU != 'ISOR') & (T_ABU != 'ROTAT')    # ROTAT lines are in star_rot.fit

        flag_line_isor = flag_line_all[mask_isor]
        flag_line = flag_line_all[mask_others]


        # CREATION OF ABUNDANCE_FINAL.TXT FROM ABU_FILE
        if check_if_has_data(abu_file, '#SPECTRUM'):
            ID, nelem, abu, abu_mc, other_nelems, df = read_fit_file(count_header_cols(abu_file), col_names_base, col_names_err, abu_file)  # FIX: column count from the file header

            # This list will hold all the lines we want to write to the file.
            lines_to_write = []

            # Handle the two cases (with/without errors) by calling the correct functions.
            # --- Case with parameter errors ---
            if 'abu_pT' in df.columns:
                abu_pT = df['abu_pT'].values
                abu_pg = df['abu_pg'].values
                abu_pv = df['abu_pv'].values

                if 26.00 in nelem:
                    nlin, feh, sigma, s1, s2, s3, s4, flag_line, s_param, params_XH = extract_abu_fe(nelem, abu, abu_mc, 26.00, flag_line, abu_pT, abu_pg, abu_pv)
                    abu_abs_fe = params_XH['abu_abs']
                    s_tot_feh = params_XH['s_tot']
                    s_int_feh = params_XH['s_int']
                    s_params_feh = params_XH['s_params']
                else:
                    abu_abs_fe = abu_absolute('Fe', met, met)
                    nlin, feh, sigma, s1, s2, s3, s4, s_param, s_tot_feh, s_int_feh, s_params_feh = 0, met, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.99, 99.99, 99.99

                lines_to_write.append(f"{ID:<30}  {26.00:5.2f}  {nlin:>5}  {feh:>5.2f}  {sigma:>5.2f}  {s1:>5.2f}  {s_param:>5.2f}       {abu_abs_fe:>5.2f}  {s_tot_feh:>5.2f}   {s_int_feh:>5.2f}   {s_params_feh:>5.2f}")

                for nel in other_nelems:
                    if nel == 26.01:
                        nlin, mean, sigma, s_int, *_, flag_line, s_param, params_XH = extract_abu_fe(nelem, abu, abu_mc, nel, flag_line, abu_pT, abu_pg, abu_pv)
                    else:
                        nlin, mean, sigma, flag_line, s_param, s_int, params_XH = extract_abu(nelem, abu, abu_mc, nel, feh, flag_line, s1, param_errors=(abu_pT,abu_pg,abu_pv), fe_errors=(s2,s3,s4))

                    abu_abs_xh = params_XH['abu_abs']
                    s_tot_xh = params_XH['s_tot']
                    s_int_xh = params_XH['s_int']
                    s_params_xh = params_XH['s_params']

                    lines_to_write.append(f"{ID:<30}  {nel:>5.2f}  {nlin:>5}  {mean:>5.2f}  {sigma:>5.2f}  {s_int:>5.2f}  {s_param:>5.2f}       {abu_abs_xh:>5.2f}  {s_tot_xh:>5.2f}   {s_int_xh:>5.2f}   {s_params_xh:>5.2f}")

            # --- Case without parameter errors ---
            else:
                if 26.00 in nelem:
                    nlin, feh, sigma, flag_line, s_param, params_XH = extract_abu_fe(nelem, abu, abu_mc, 26.00, flag_line)
                    abu_abs_fe = params_XH['abu_abs']
                    s_tot_feh = params_XH['s_tot']
                    s_int_feh = params_XH['s_int']
                    s_params_feh = params_XH['s_params']
                else:
                    abu_abs_fe = abu_absolute('Fe', met, met)
                    nlin, feh, sigma, s_param, s_tot_feh, s_int_feh, s_params_feh = 0, met, 99.99, 99.99, 99.99, 99.99, 99.99

                s1 = sigma
                lines_to_write.append(f"{ID:<30}  {26.00:5.2f}  {nlin:>5}  {feh:5.2f}  {sigma:5.2f}  {sigma:>5.2f}  {s_param:>5.2f}       {abu_abs_fe:>5.2f}  {s_tot_feh:>5.2f}   {s_int_feh:>5.2f}   {s_params_feh:>5.2f}")

                for nel in other_nelems:
                    if nel == 26.01:
                        nlin, mean, sigma, flag_line, s_param, params_XH = extract_abu_fe(nelem, abu, abu_mc, nel, flag_line)
                        s_int = sigma
                    else:
                        nlin, mean, sigma, flag_line, s_param, s_int, params_XH = extract_abu(nelem, abu, abu_mc, nel, feh, flag_line, s1)

                    abu_abs_xh = params_XH['abu_abs']
                    s_tot_xh = params_XH['s_tot']
                    s_int_xh = params_XH['s_int']
                    s_params_xh = params_XH['s_params']

                    lines_to_write.append(f"{ID:<30}  {nel:>5.2f}  {nlin:>5}  {mean:>5.2f}  {sigma:>5.2f}  {s_int:>5.2f}  {s_param:>5.2f}       {abu_abs_xh:>5.2f}  {s_tot_xh:>5.2f}   {s_int_xh:>5.2f}   {s_params_xh:>5.2f}")

            # Write all the generated lines to the final file in ONE operation.
            with open('abundance_final.txt', 'a+') as f:
                for line in lines_to_write:
                    f.write(line + '\n')
                f.write('#' + '-'*39 + '\n')


            # update the abu_file with the flag for the used lines
            append_flag_column(abu_file, flag_line)


            #----------------------------------------------------------- Plot Gala-like
            column_names = get_header_names(abu_file)
            if column_names:
                try:
                    data = pd.read_csv(abu_file, sep=r"\s+", comment='#', header=None, names=column_names)
                except FileNotFoundError:
                    print(f"Error: File not found '{abu_file}'")
                    data = pd.DataFrame()

            else:
                print("No column names.")
                data = pd.DataFrame()

            # --- 1. Data Selection ---
            fe_data = data[(data['ELEM'] == 26.00) & (data['ABU'] != 99.99)].copy()
            feii_data = data[(data['ELEM'] == 26.01) & (data['ABU'] != 99.99)].copy()

            # MAIN CHECK: without Fe I nothing can be done.
            if not fe_data.empty and cfg.grid == 'n':
                # --- Prepare Data for Plotting ---

                # 1. Fe I (it exists)
                fe_data['reduced_ew'] = np.log10((fe_data['EW'].clip(lower=1e-9)/1000) / fe_data['LINE'])
                used_data = fe_data[fe_data['flag_line'] == 1]
                discarded_data = fe_data[fe_data['flag_line'] != 1]

                # 2. Fe II (ONLY IF IT EXISTS)
                has_feii = not feii_data.empty  # Convenience flag

                if has_feii:
                    feii_data['reduced_ew'] = np.log10((feii_data['EW'].clip(lower=1e-9)/1000) / feii_data['LINE'])
                    used_feii = feii_data[feii_data['flag_line'] == 1]
                    discarded_feii = feii_data[feii_data['flag_line'] == 0]

                # --- Common limits (Fe I + Fe II) ---

                # 1. Fe I and Fe II together, only to compute the min/max
                if not feii_data.empty:
                    all_data_limits = pd.concat([fe_data, feii_data])
                else:
                    all_data_limits = fe_data

                # 2. Limits from the joined data (all_data_limits)
                # Filter on ABU and LINE (excluding 99.99)
                valid_rows_abu = all_data_limits[all_data_limits['ABU'] != 99.99]
                valid_abu = valid_rows_abu['ABU']
                valid_wv = valid_rows_abu['LINE']

                miny = valid_abu.min()
                maxy = valid_abu.max()
                minx1 = valid_wv.min()
                maxx1 = valid_wv.max()

                # Filter on EP (excluding 99.999)
                valid_ep = all_data_limits[all_data_limits['EP'] != 99.999]['EP']
                minx2 = valid_ep.min()
                maxx2 = valid_ep.max()

                # --- Create the 3-Panel Plot ---
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6.75), sharey=True)

                # --- Panel 1: Wave ---
                # Fit on Fe I only
                if len(used_data['LINE']) >= 3:
                    y_fit1, m1, c1, em1, ec1 = get_coeffs(used_data['LINE'], used_data['ABU'])
                    ax1.plot(used_data['LINE'].to_numpy(), y_fit1.to_numpy(), c='red', label=rf'$y = ({m1:.3f} \pm {em1:.3f})x + ({c1:.2f} \pm {ec1:.2f})$')

                # Plot Fe I
                ax1.scatter(discarded_data['LINE'].to_numpy(), discarded_data['ABU'].to_numpy(), color='gray', alpha=0.5, label='Discarded', lw=0.35)
                ax1.scatter(used_data['LINE'].to_numpy(), used_data['ABU'].to_numpy(), color='blue', ec='k', label='Used', lw=0.35)

                # Fe II (ONLY IF IT EXISTS)
                if has_feii:
                    ax1.scatter(discarded_feii['LINE'].to_numpy(), discarded_feii['ABU'].to_numpy(), color='lightcoral', marker='^', alpha=0.5, lw=0.35)
                    ax1.scatter(used_feii['LINE'].to_numpy(), used_feii['ABU'].to_numpy(), color='red', ec='k', marker='^', label='Fe II', lw=0.35)

                # Limiti e stile
                if np.isfinite(miny) and np.isfinite(maxy):
                    ax1.set_ylim(miny - 0.2, maxy + 0.2)
                    ax1.set_xlim(minx1 - 50, maxx1 + 50)
                else:
                    ax1.set_ylim(0, 1)
                apply_plot_style(ax1, r'Wavelength [$\AA$]', 'A(Fe)')
                ax1.legend(fontsize=7)

                # --- Panel 2: EP ---
                # Fit FeI
                if len(used_data['LINE']) >= 3:
                    y_fit2, m2, c2, em2, ec2 = get_coeffs(used_data['EP'], used_data['ABU'])
                    ax2.plot(used_data['EP'].to_numpy(), y_fit2.to_numpy(), c='red', label=rf'$y = ({m2:.2f} \pm {em2:.2f})x + ({c2:.2f} \pm {ec2:.2f})$')

                # Plot Fe I
                ax2.scatter(discarded_data['EP'].to_numpy(), discarded_data['ABU'].to_numpy(), color='gray', alpha=0.5, lw=0.35)
                ax2.scatter(used_data['EP'].to_numpy(), used_data['ABU'].to_numpy(), color='blue', ec='k', lw=0.35)

                # Fe II (ONLY IF IT EXISTS)
                if has_feii:
                    ax2.scatter(discarded_feii['EP'].to_numpy(), discarded_feii['ABU'].to_numpy(), color='lightcoral', marker='^', alpha=0.5, lw=0.35)
                    ax2.scatter(used_feii['EP'].to_numpy(), used_feii['ABU'].to_numpy(), color='red', ec='k', marker='^', lw=0.35)

                ax2.set_xlim(minx2 - 0.2, maxx2 + 0.2)
                apply_plot_style(ax2, 'Excitation Potential [eV]', '')
                if ax2.get_legend_handles_labels()[0]:  # legend only if there is a fit (>= 3 lines)
                    ax2.legend(fontsize=7)

                # --- Panel 3: Reduced EW ---
                # Fit FeI
                if len(used_data['LINE']) >= 3:
                    y_fit3, m3, c3, em3, ec3 = get_coeffs(used_data['reduced_ew'], used_data['ABU'])
                    ax3.plot(used_data['reduced_ew'].to_numpy(), y_fit3.to_numpy(), c='red', label=rf'$y = ({m3:.2f} \pm {em3:.2f})x + ({c3:.2f} \pm {ec3:.2f})$')

                # Plot Fe I
                ax3.scatter(discarded_data['reduced_ew'].to_numpy(), discarded_data['ABU'].to_numpy(), color='gray', alpha=0.5, lw=0.35)
                ax3.scatter(used_data['reduced_ew'].to_numpy(), used_data['ABU'].to_numpy(), color='blue', ec='k', lw=0.35)

                # Fe II (ONLY IF IT EXISTS)
                if has_feii:
                    ax3.scatter(discarded_feii['reduced_ew'].to_numpy(), discarded_feii['ABU'].to_numpy(), color='lightcoral', marker='^', alpha=0.5, lw=0.35)
                    ax3.scatter(used_feii['reduced_ew'].to_numpy(), used_feii['ABU'].to_numpy(), color='red', ec='k', marker='^', lw=0.35)

                ax3.set_xlim(cfg.ewr_min - 0.1, cfg.ewr_max + 0.1)
                apply_plot_style(ax3, r'log(EW / $\lambda$)', '')
                if ax3.get_legend_handles_labels()[0]:  # legend only if there is a fit (>= 3 lines)
                    ax3.legend(fontsize=7)

                # --- Finalization ---
                plt.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.75, wspace=0.1)
                plt.savefig(name_star+'_tmp.pdf')
                plt.close(fig)  # FIX: one figure was left open for every star

            # --- 1. File names ---
            # The file just saved
            tmp_pdf_path = Path(name_star+'_tmp.pdf')
            # The main file the temporary one is added to
            main_pdf_path = Path(name_star+'.pdf')
            # A temporary name for the output, to avoid conflicts
            output_pdf_path = Path('__merged_temp.pdf')

            # --- 2. Check that both files exist ---
            if tmp_pdf_path.exists() and main_pdf_path.exists():
                # --- 3. Merge ---
                try:
                    merger = PdfWriter()

                    merger.append(main_pdf_path)
                    merger.append(tmp_pdf_path)

                    merger.write(output_pdf_path)
                    merger.close()

                    # --- 4. Replace the main file and clean up ---
                    # Delete the original 'name_star.pdf'
                    main_pdf_path.unlink()

                    # Rename the merged file ('__merged_temp.pdf') as the main file
                    output_pdf_path.rename(main_pdf_path)

                    # Delete the temporary plot file ('name_star_tmp.pdf')
                    tmp_pdf_path.unlink()

                except Exception as e:
                    print(f"Error during merging process: {e}")
                    if output_pdf_path.exists():
                        output_pdf_path.unlink()


        else:
            os.remove(abu_file)

        #--------------------------------------------------------- abu_file_isor
        if check_if_has_data(abu_file_isor, '#SPECTRUM'):
            ID, nelem, abu, abu_mc, other_nelems, df = read_fit_file(count_header_cols(abu_file_isor), col_names_base_isor, col_names_err, abu_file_isor)   # FIX: column count from the file header

            lines_to_write = []
            # --- Case with parameter errors ---
            if 'abu_pT' in df.columns:
                abu_pT = df['abu_pT'].values
                abu_pg = df['abu_pg'].values
                abu_pv = df['abu_pv'].values

                for nel in other_nelems:
                    nlin, mean, sigma, flag_line_isor, s_param, s_int = extract_abu_isor(nelem, abu, abu_mc, nel, flag_line_isor, param_errors=(abu_pT,abu_pg,abu_pv))
                    # FIX: append moved inside the loop (before, only the last species was written)
                    lines_to_write.append(f"{ID:<30}  {nel:>5.2f}  {nlin:>5}  {mean:>5.0f}  {sigma:>5.0f}  {s_int:>5.0f}  {s_param:>5.0f}")

            # --- Case without parameter errors ---
            else:
                for nel in other_nelems:
                    nlin, mean, sigma, flag_line_isor, s_param, s_int = extract_abu_isor(nelem, abu, abu_mc, nel, flag_line_isor)
                    # FIX: append moved inside the loop (before, only the last species was written)
                    lines_to_write.append(f"{ID:<30}  {nel:>5.2f}  {nlin:>5}  {mean:>5.0f}  {sigma:>5.0f}  {s_int:>5.0f}  {s_param:>5.0f}")

            # Write all the generated lines to the final file in ONE operation.
            with open('abundance_final_isor.txt', 'a+') as f:
                for line in lines_to_write:
                    f.write(line + '\n')
                f.write('#' + '-'*39 + '\n')


            # update the abu_file with the flag for the used lines
            append_flag_column(abu_file_isor, flag_line_isor)

        else:
            os.remove(abu_file_isor)


    if not check_if_has_data('abundance_final_isor.txt', '#STAR'):
        os.remove('abundance_final_isor.txt')

    if not check_if_has_data('abundance_final.txt', '#STAR'):
        os.remove('abundance_final.txt')

    ### cleaning the unnecessary files from k2
    cleaning_tmp_files(patterns_to_delete, directory=curr_path, newer_than=start_time)
    cleanup_linelist_cut()


    ### calculation of the execution time
    end_time = time.time()
    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"\nExecution time: {hours}h {minutes}m {seconds:.1f}s \n")

    print("Deep in the human unconscious is a")
    print("pervasive need for a logical universe")
    print("that makes sense. But the real universe")
    print("is always one step beyond logic.")


def check_installation() -> int:
    """Print the versions of Python and of the packages salvador needs, where K2 and the table of
    solar abundances are (the former 'make check'). Returns 1 if a package is missing."""
    print(f'salvador {__version__}')
    print(f'Python {sys.version.split()[0]} ({sys.executable})')
    ok = True
    for dist in ('numpy', 'scipy', 'pandas', 'matplotlib', 'astropy', 'specutils', 'PyAstronomy', 'pypdf'):
        try:
            print(f'  [OK]   {dist} {importlib.metadata.version(dist)}')
        except importlib.metadata.PackageNotFoundError:
            print(f'  [FAIL] {dist} is missing (pip install {dist})')
            ok = False
    k2 = shutil.which('k2.py')
    if k2:
        print(f'  [OK]   K2: {k2}')
    else:
        print('  [WARN] k2.py is not in the PATH: K2 is needed unless you only use a grid of synthetics (PATH_SYN)')
    print(f'  solar abundances: {solar_table_path()}')
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point of the salvador command (also: python -m salvador)."""
    parser = argparse.ArgumentParser(
        prog='salvador',
        description='Chemical abundances by spectral synthesis with the Kurucz codes (through K2). '
                    'Run it in the folder of input.salvador.')
    parser.add_argument('input', nargs='?', default='input.salvador', help='input file (default: input.salvador)')
    parser.add_argument('--version', action='version', version=f'salvador {__version__}')
    parser.add_argument('--check', action='store_true', help='check the installation (packages, K2) and exit')
    args = parser.parse_args(argv)
    if args.check:
        return check_installation()

    cfg.load(args.input)
    configure_warnings()
    select_backend(cfg.inter)
    run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
