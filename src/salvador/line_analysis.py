"""Analysis of one line: measure_line (first fit, interactive menu, automatic iterations) and helpers."""
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import astropy.units as u
from specutils import Spectrum

from .config import cfg, init_ans, path_plot_file, path_tmp
from .elements import abu_absolute, define_abu0
from .k2_interface import autok, eqw_k2
from .spectra_io import readsyn
from .normalization import find_clipped_regions, norm1, norm2, norm3
from .fitting import abu_determination, abu_mc_line, convolve_gauss, convolve_gauss2, mc_seed, new_res_est
from .rotation import analysis_rotat
from .plotting import NullPlotter, PlotManager

### keywords of the lines analysed as abundances (ROTAT is the rotation)
ABU_KEYS = ('ABU', 'ABU2', 'MOLEC', 'ABU_GF', 'ABUF', 'ISOR')

def _rejected_result():
    """Result of a line that could not be measured (out of limit, or no usable synthetics)."""
    return {"abu_fin": 99.99, "answer": 'y', "ew_k2": 99.99, "snr": 0, "abu_err_mc": 99.99, "abu0": 99.99, "EP": 99.999, "loggf": 99.999}

### SPEED: cache of the NORM = 2 reference synthetic (whole spectral range), see measure_line
_norm2_syn_cache = {}


def _calculate_line_area(wave, flux, fit1, fit2):
    """Calculates the area of a spectral line within a given window."""
    mask = (wave.value > fit1) & (wave.value < fit2)
    w_masked, f_masked = wave[mask], flux[mask]

    if w_masked.size == 0:
        return 0.0

    absorption_depth = 1 - f_masked.value
    return trapezoid(absorption_depth, w_masked.value)


def _update_grid_step(direction, abu, delta_arr, fl_syn_br, param1, window, resol, tana, fl_syn_nb=None):
    """Move the 3 synthetics of the fit by one step (0.2 dex; 10 for ISOR, not below 1) towards
    higher ('up') or lower ('down') abundances: one new synthetic is computed and the one at the
    other end is dropped. Also the unbroadened synthetics, if given (interactive mode)."""
    if tana != 'ISOR':
        step = 0.20
    elif tana == 'ISOR':
        step = 10

    if direction == 'up':
        new_abu = abu[-1] + step
        new_delta = delta_arr[-1] + step

        wl_syn, new_raw_spectrum = autok(param1, new_delta, window)
        new_fl_syn_br = convolve_gauss2(wl_syn, new_raw_spectrum, resol)

        # Roll arrays to the left and update the last element
        abu = np.roll(abu, -1)
        abu[-1] = new_abu
        delta_arr = np.roll(delta_arr, -1)
        delta_arr[-1] = new_delta
        fl_syn_br = tuple(fl_syn_br[1:]) + (new_fl_syn_br,)
        if fl_syn_nb is not None:
            fl_syn_nb = tuple(fl_syn_nb[1:]) + (new_raw_spectrum * u.count,)

    elif direction == 'down':
        if tana != 'ISOR':
            new_abu = abu[0] - step
            new_delta = delta_arr[0] - step
        elif tana == 'ISOR':
            new_abu = max(abu[0] - step, 1)
            new_delta = max(delta_arr[0] - step, 1)

        wl_syn, new_raw_spectrum = autok(param1, new_delta, window)
        new_fl_syn_br = convolve_gauss2(wl_syn, new_raw_spectrum, resol)

        # Roll arrays to the right and update the first element
        abu = np.roll(abu, 1)
        abu[0] = new_abu
        delta_arr = np.roll(delta_arr, 1)
        delta_arr[0] = new_delta
        fl_syn_br = (new_fl_syn_br,) + tuple(fl_syn_br[:-1])
        if fl_syn_nb is not None:
            fl_syn_nb = (new_raw_spectrum * u.count,) + tuple(fl_syn_nb[:-1])

    if fl_syn_nb is not None:
        return abu, delta_arr, fl_syn_br, wl_syn, fl_syn_nb
    return abu, delta_arr, fl_syn_br, wl_syn

def get_type_input(prompt, target_type):
    """Ask for a value until it can be converted to target_type (e.g. float)."""
    while True:
        try:
            value = input(prompt)
            return target_type(value)
        except ValueError:
            print('Insert a valid value.')

def confirm_resolution_change(old_resol, new_resol):
    # Whether to apply the new value
    """New resolution, asking for a confirmation if it differs from the old one by more than a
    factor 1.5 (or less than 0.5). Returns the old resolution if the change is cancelled."""
    apply_change = True

    if old_resol != 0:
        ratio = new_resol / old_resol

        # Check whether the ratio is OUTSIDE the safe range
        if not (0.5 <= ratio <= 1.5):
            # Chiedi conferma
            prompt = (f"\nWarning: The new resolution ({new_resol}) is a drastic change.\n"
                f"Apply anyway? (y/n): ")

            confirmation = input(prompt).lower()

            if confirmation != 'y':
                apply_change = False # Cancel the change

    if apply_change:
        print(f"Resolution updated: {new_resol}")
        return new_resol # Apply the change
    else:
        print("Change canceled. The old resolution remains.")
        return old_resol # Cancel the change: return the old value


def measure_line(wvl, flux, sss, fit1, fit2, elem, param1, name_star, abu, wv_left, wv_right, ct_elem, fixed_params, **kwargs):
    # window of THIS line: starts from WINDOW of input.salvador and may be widened below (it was a
    # global variable modified here and read silently by eqw_k2 and abu_determination)
    """Measure one line: abundance (ABU, ABU2, MOLEC, ABU_GF, ABUF), isotopic ratio (ISOR) or
    rotation (ROTAT).
    Steps: normalization of the spectrum (NORM), local RV correction, synthetic spectra around the
    observed line (from K2, or from the grid), first fit and plot. Then either the interactive menu
    (INTER = y) or, in automatic mode, new synthetics until the chi2 minimum is inside the grid.
    Returns a dictionary with the result (see abundance_result, rotation_result).
    save_plot=False for the parameter-error reruns: no plot, no EW and no Monte Carlo."""
    window = cfg.window

    # edges for ABU2 inside kwargs
    save_plot = kwargs.pop('save_plot', True)
    arg_extra = kwargs.copy()

    # param k2
    teff = int(param1['teff'])
    logg = float(param1['logg'])
    met = float(param1['met'])
    alpha = int(param1['alpha'])
    vt = float(param1['vt'])
    resol = int(param1['res'])
    line = float(param1['line'])
    nelem = float(param1['nelem'])
    abu_var = str(param1['abu_var'])
    delta = float(param1['delta'])
    tana = str(param1['tana'])

    save_flag = 'y' if save_plot else 'n'

    # re-defining window
    if tana == 'ABU2':
        wind_tmp = int(np.ceil((kwargs.get('wind_r1')+1) - (fit1 - 1)))
        if wind_tmp > window:
            window = wind_tmp
    else:
        wind_tmp = int(np.ceil((fit2 + 1) - (fit1 - 1)))
        if wind_tmp > window:
            window = wind_tmp

    # defining abu0
    abu0 = define_abu0(int(nelem), alpha, met, delta, elem)
    param1['abu0'] = abu0

    # adding name_star to param1
    param1['name_star'] = name_star

    # Copies of original arrays
    wvl_orig, flux_orig = np.copy(wvl), np.copy(flux)
    cshift_orig, wv_shift_orig = 1.0, 0.0
    fit1_orig, fit2_orig = np.copy(fit1), np.copy(fit2)
    wv_left_orig, wv_right_orig = np.copy(wv_left), np.copy(wv_right)

    def perform_analysis(wvl, flux, fit1, fit2, wv_left, wv_right, answer, wl_syn, fl_syn_br, abu, param1, **new_args):
        """Abundance fit (abu_determination) of the current spectrum, edges and synthetics."""
        spec = Spectrum(spectral_axis=wvl * u.AA, flux=flux * u.count)
        return abu_determination(spec, fit1, fit2, wl_syn, fl_syn_br, abu, line, elem, answer, wv_left, wv_right, param1, save=save_flag, window=window, **new_args)

    ### Normalization
    if cfg.type_norm == 1 and cfg.grid == 'n':
        wl_syn_tmp, fl_syn_tmp = autok(param1, delta, window)
        fl_syn_tmp = convolve_gauss2(wl_syn_tmp, fl_syn_tmp, resol)

        flux = norm1(wvl, flux, wl_syn_tmp.value, fl_syn_tmp.value)

    elif cfg.type_norm == 2:
        fl_syn_norm2 = None

        if cfg.grid == 'n':
            wind_fixed = np.round(wvl[-1] - wvl[0], 3)
            # SPEED: this reference synthetic covers the whole spectrum and depends only on the star,
            # the resolution and the observed wavelength grid, so it is computed once and reused for
            # every line and every parameter-error rerun (before: ~150 ms of reading/convolving each time).
            norm2_key = (tuple(sorted(fixed_params.items())), wind_fixed, resol, wvl.size, hash(np.ascontiguousarray(wvl).tobytes()))
            fl_syn_norm2 = _norm2_syn_cache.get(norm2_key)
            if fl_syn_norm2 is None:
                wl_syn_tmp, fl_syn_tmp = autok(fixed_params, float(fixed_params['delta']), wind_fixed)
                fl_syn_tmp = convolve_gauss2(wl_syn_tmp, fl_syn_tmp, resol)

                interpolator = interp1d(wl_syn_tmp, fl_syn_tmp, kind='cubic', bounds_error=False, fill_value=1.0)
                fl_syn_norm2 = interpolator(wvl)
                fl_syn_norm2.setflags(write=False)
                if len(_norm2_syn_cache) >= 8:
                    _norm2_syn_cache.clear()
                _norm2_syn_cache[norm2_key] = fl_syn_norm2

        flux = norm2(wv_left, wv_right, wvl, flux, fl_syn_norm2)

    elif cfg.type_norm == 3:
        param_tmp = (teff, logg, met, alpha, vt, resol, line, nelem, abu_var)
        wl_syn_tmp, fl_syn_tmp = autok(param_tmp, 0.0, window)
        fl_syn_tmp = convolve_gauss2(wl_syn_tmp, fl_syn_tmp, resol)
        flux = norm3(wvl, flux, wl_syn_tmp.value, fl_syn_tmp.value)


    ### local correction for RV according to the wave of minimum flux on the spectrum
    newr, fwhm_obs, pos_line, area_line = new_res_est(wvl, flux, line, resol, fit1, fit2, tana)

    # 1.5 FWHM left and 1.5 FWHM right = 3 FWHM total
    wvl_start = line - (1.5 * fwhm_obs)
    wvl_end = line + (1.5 * fwhm_obs)

    # Let's create a mask to take only the pixels inside this region
    mask_3fwhm = (wvl >= wvl_start) & (wvl <= wvl_end)
    wvl_region = wvl[mask_3fwhm]

    # We calculate the pixel step if there are enough pixels in the region
    if len(wvl_region) > 1:
        pixel_step = np.mean(np.diff(wvl_region))
        pixel_per_fwhm = fwhm_obs / pixel_step

        if cfg.verbose == 1:
            print(f"\n--- Local RV correction for line {line} A ---")
            print(f"Average pixel step in region 3 FWHM: {pixel_step: .4f} A/pixel")

        if pixel_per_fwhm < 2.0:
            print("WARNING: The line is not correctly sampled.")

    else:
        print("Error: No pixels found in the specified region. Please check your data or FWHM value.")
        pixel_step = 0.0

    # tolerance from number of pixels to range in Ångstrom
    tol_wv = cfg.n_pix_rv * pixel_step

    # Let's calculate the shift (distance between the measured centroid and the laboratory value)
    delta_wv = pos_line - line
    if abs(delta_wv) < tol_wv:
        wvl = wvl - delta_wv
        if cfg.verbose == 1:
            print(f"Spectrum realigned! Shifted by {-delta_wv:.4f} A ({abs(delta_wv)/pixel_step:.2f} pixels)\n")
    else:
        if cfg.verbose == 1:
            print(f"Shift too large ({abs(delta_wv):.4f} A = {abs(delta_wv)/pixel_step:.1f} pixels). "
            f"Exceeds threshold of {cfg.n_pix_rv} pixels. Spectrum has NOT been modified.\n")


    ###  syn
    if cfg.grid == 'y':
        abu = abu_absolute(elem, abu, met)
        wl_syn, fl_syn_br, used = readsyn(sss, resol, line)
        # FIX: keep names and abundances aligned with the spectra actually read
        sss, abu = [sss[k] for k in used], np.asarray(abu)[used]
        if len(used) < 2:
            print('---')
            print(f'Fewer than 2 usable synthetic spectra for line {line}: line skipped\n')
            return _rejected_result()
        if cfg.type_norm == 1:
            flux = norm1(wvl, flux, wl_syn.value, fl_syn_br[0].value)

    elif cfg.grid == 'n' and tana != 'ROTAT':
        # 1. Calculate the area of the first synthetic spectrum.
        wl_syn1, fl_syn1 = autok(param1, delta, window)
        area_syn1 = _calculate_line_area(wl_syn1, fl_syn1, fit1, fit2)

        # 2. Determine the direction of the abundance change needed.
        #    direction = 1  --> need higher abundance
        #    direction = -1 --> need lower abundance
        direction = 1 if area_line > area_syn1 else -1

        # 3. Calculate the second spectrum at a 0.4 dex step in the determined direction.
        if tana != 'ISOR':
            delta2 = delta + 0.4 * direction
            wl_syn2, fl_syn2 = autok(param1, delta2, window)
        elif tana == 'ISOR':
            delta2 = delta - 20 * direction
            delta2 = max(delta2, 5)
            wl_syn2, fl_syn2 = autok(param1, delta2, window)

        area_syn2 = _calculate_line_area(wl_syn2, fl_syn2, fit1, fit2)

        # 4. Check for convergence. The condition is now generalized.
        #    If direction=1, this is `area_line < area_syn2`.
        #    If direction=-1, this is `area_line > area_syn2`.
        has_converged = (area_line - area_syn2) * direction < 0
        if has_converged:
            # CONVERGED: The true value is between delta and delta2.
            # Calculate the midpoint spectrum.
            if tana != 'ISOR':
                delta3 = delta + 0.2 * direction
                wl_syn3, fl_syn3 = autok(param1, delta3, window)
            elif tana == 'ISOR':
                delta3 = delta - 10 * direction
                delta3 = max(delta3, 7)
                if delta2 == 5:
                    delta3 = 7
                wl_syn3, fl_syn3 = autok(param1, delta3, window)

            # Prepare the final arrays.
            deltas = [delta, delta3, delta2]
            spectra = [fl_syn1, fl_syn3, fl_syn2]

        else:
            # NOT CONVERGED: The true value is beyond delta2.
            # Calculate two more spectra at larger steps.
            if tana != 'ISOR':
                delta3 = delta + 0.6 * direction
                delta4 = delta + 0.8 * direction
                wl_syn3, fl_syn3 = autok(param1, delta4, window) # Note: 0.8 is calculated first
                wl_syn4, fl_syn4 = autok(param1, delta3, window)
            elif tana == 'ISOR':
                delta3 = delta - 30 * direction
                delta4 = delta - 40 * direction
                delta3 = max(delta3, 3)
                delta4 = max(delta4, 1)
                if delta2 == 5:
                    delta3 = 3
                    delta4 = 1
                wl_syn3, fl_syn3 = autok(param1, delta4, window) # Note: 0.8 is calculated first
                wl_syn4, fl_syn4 = autok(param1, delta3, window)


            # Prepare the final arrays.
            deltas = [delta2, delta3, delta4]
            spectra = [fl_syn2, fl_syn4, fl_syn3]

        # 5. Final assignment of results, sorted by abundance.
        sort_indices = np.argsort(deltas)
        delta_arr = np.array(deltas)[sort_indices]

        if tana != 'ISOR':
            abu = abu0 + (delta_arr - delta) # Calculate final abundances based on the initial one
        elif tana == 'ISOR':
            abu = delta_arr

        # The base wavelength grid is the same for all spectra
        wl_syn = wl_syn1
        # Sort the spectra to match the sorted deltas
        fl_syn_tmp = tuple(np.array(spectra)[sort_indices])
        fl_syn_nb = [flux_array * u.count for flux_array in fl_syn_tmp]

        ### broadening the original spectra to the given resolution
        fl_syn_br, res_obs = convolve_gauss(wl_syn, fl_syn_nb, resol, fwhm_obs, line, fit1, fit2, tana, init_ans)

        resol = res_obs
        param1['res'] = resol

        # copies of the unbroadened (nb) and broadened (br) synthetics
        resol_orig = np.copy(resol)
        fl_syn_br_orig = fl_syn_br[:]
        fl_syn_nb_orig = fl_syn_nb[:]
        wl_syn_orig = wl_syn.copy()
        abu_orig = np.copy(abu)
        delta_arr_orig = np.copy(delta_arr)


    # FIX: initial values for the 'restore' command (i) also for ROTAT and grid mode: they were
    # saved only for abundance lines without grid, so 'i' stopped with an error in these cases
    if not (cfg.grid == 'n' and tana != 'ROTAT'):
        resol_orig = np.copy(resol)
        abu_orig = np.copy(abu)
        delta_arr_orig = None

    ### first calculation of the abundance or rotation and plot
    name_title = rf'{name_star} -- $T_{{\rm eff}}={teff}$ K -- $\log g={logg}$ -- $\xi={vt}$ km/s'

    def rotation_fit():
        """Rotation (and abundance) fit with the current settings, with the keys the plot needs."""
        fit_data = analysis_rotat(nelem, alpha, met, delta, elem, wvl, flux, line, window, param1, resol, fit1, fit2, **arg_extra)
        fit_data.update({"wv_left": wv_left, "wv_right": wv_right, "type_norm": cfg.type_norm})
        return fit_data

    def refit(answer):
        """Fit again with the current settings (edges, continuum, shift, windows, resolution,
        synthetics) and update the plot. Used after every menu command and in the automatic mode."""
        nonlocal abu_fin, ew, snr, chi, param_mc, plot_data, rot_data
        if tana.startswith(ABU_KEYS):
            abu_fin, ew, snr, chi, param_mc, plot_data = perform_analysis(wvl, flux, fit1, fit2, wv_left, wv_right, answer, wl_syn, fl_syn_br, abu, param1, **arg_extra)
            plotter.update(plot_data)
        elif tana == 'ROTAT':
            rot_data = rotation_fit()
            plotter.update(rot_data)

    def save_line_plot():
        """Save the plot of the line (PDF), record its name for the final merged PDF, close it."""
        filename = plotter.save(path_tmp, line)
        plotter.write_log(path_plot_file, filename)
        plotter.close()

    def final_ew_and_mc():
        """EW, EP and log gf of the line (from K2, or the fitted EW in grid mode) and the Monte Carlo
        error. Only for the nominal fit (save_plot): the parameter-error reruns skip them."""
        if cfg.grid == 'y':
            ew_k2, EP, loggf = (ew if save_plot else 99.99), 99.999, 99.999
        elif tana not in ['MOLEC', 'ABU2', 'ISOR']:
            ew_k2, EP, loggf = (eqw_k2(param1, abu_fin-abu0+delta, window) if save_plot else (99.99, 99.999, 99.999))
        else:
            ew_k2, EP, loggf = 99.99, 99.999, 99.999
        abu_err_mc = abu_mc_line(param_mc, snr, abu, seed=mc_seed(param1)) if save_plot and ct_elem < 10 else 99.99
        return ew_k2, EP, loggf, abu_err_mc

    def abundance_result(answer, ew_k2, EP, loggf, abu_err_mc):
        """Result of an abundance (or isotopic ratio) line, as read by salvador.py."""
        return {"abu_fin": abu_fin, "answer": answer, "ew_k2": ew_k2, "snr": snr, "abu_err_mc": abu_err_mc, "abu0": abu0, "EP": EP, "loggf": loggf}

    def rotation_result(answer):
        """Result of a ROTAT line (last fit; 99.99 if rejected), as read by salvador.py."""
        if answer == 'r':
            return {"rot_fin": 99.99, "answer": answer, "abu_err_mc": 99.99, "abu_rot": 99.99, "err_abu_rot": 99.99}
        return {"rot_fin": rot_data['best_rot_fine'], "answer": answer, "abu_err_mc": rot_data['err_rot'],
                "abu_rot": rot_data['best_abu_fine'], "err_abu_rot": rot_data['err_abu']}

    abu_fin = ew = snr = chi = param_mc = plot_data = rot_data = None
    if tana.startswith(ABU_KEYS):
        abu_fin, ew, snr, chi, param_mc, plot_data = perform_analysis(wvl, flux, fit1, fit2, wv_left, wv_right, init_ans, wl_syn, fl_syn_br, abu, param1, **arg_extra)
        first_plot = plot_data
    elif tana == 'ROTAT':
        rot_data = rotation_fit()       # always the latest fit (updated by every menu command)
        first_plot = rot_data
    plotter = PlotManager(star_title=name_title, tana=tana) if (cfg.inter == 'y' or save_plot) else NullPlotter()   # SPEED
    plotter.update(first_plot)


    ### INTERACTIVE MODE
    if cfg.inter == 'y':
        while True:
            print('\n# ok and next line (y)' )
            if cfg.grid == 'n': print('# new fit (f)' )
            print('# rule out and next line (r)')
            print('# new continuum (c)')
            print('# wave shift in Angstroms (wa)')
            print('# new left edge (f1), new right edge (f2)')
            if tana == 'ABU2': print('# new left_new edge (f3), new right_new edge (f4)')
            print('# new resolution (res)')
            if cfg.type_norm == 2: print('# delete window (dw)')
            if cfg.type_norm == 2: print('# add window (aw)')
            print('# restore initial values (i)')
            print('# exit (e) \n')

            answer = input('insert: ').lower()

            if answer not in ['c', 'r', 'y', 'f1', 'f2', 'wa', 'res', 'i', 'e', 'dw', 'aw', 'f', 'f3', 'f4']:
                print('Wrong option.\n')
                continue

            if answer == 'c':
                cshift = get_type_input('new continuum value: ', float)
                flux = flux*cshift
                refit(answer)

            elif answer == 'f1':
                print(f'old left edge = {fit1:.2f}')
                fit1 = get_type_input('new left edge: ', float)
                while fit1 >= fit2:
                    print('left edge >= right edge', float)
                    fit1 = get_type_input('new left edge: ', float)
                refit(answer)

            elif answer == 'f2':
                print(f'old right edge = {fit2:.2f}')
                fit2 = get_type_input('new right edge: ', float)
                while fit2 <= fit1:
                    print('right edge <= left edge')
                    fit2 = get_type_input('new right edge: ', float)
                refit(answer)

            elif tana == 'ABU2' and answer == 'f3':
                wind_l1 = arg_extra['wind_l1']
                print(f'old left_new edge = {wind_l1:.2f}')
                fit3 = get_type_input('new left_new edge: ', float)
                while fit3 >= arg_extra['wind_r1']:
                    print('left edge >= right edge', float)
                    fit3 = get_type_input('new left_new edge: ', float)
                arg_extra['wind_l1'] = fit3
                refit(answer)

            elif tana == 'ABU2' and answer == 'f4':
                wind_r1 = arg_extra['wind_r1']
                print(f'old right_new edge = {wind_r1:.2f}')
                fit4 = get_type_input('new right_new edge: ', float)
                while fit4 <= arg_extra['wind_l1']:
                    print('right edge <= left edge')
                    fit4 = get_type_input('new right_new edge: ', float)
                arg_extra['wind_r1'] = fit4
                refit(answer)

            elif answer == 'wa':
                wv_shift = get_type_input('new wv shift: ', float)
                wvl = wvl - wv_shift
                refit(answer)

            elif answer == 'res' and (tana.startswith(ABU_KEYS) or tana == 'ROTAT'):
                print(f'old res = {resol}')
                resol_new = get_type_input('new res: ', int)
                resol = confirm_resolution_change(resol, resol_new)
                param1['res'] = resol
                if tana.startswith(ABU_KEYS) and cfg.grid == 'y':
                    # FIX: the confirmed resolution (before: resol_new, also when the change was cancelled)
                    wl_syn, fl_syn_br, _ = readsyn(sss, resol, line)
                elif tana.startswith(ABU_KEYS) and cfg.grid == 'n':
                    fl_syn_br, res_obs = convolve_gauss(wl_syn, fl_syn_nb, resol, fwhm_obs, line, fit1, fit2, tana, answer)
                # ROTAT: the resolution is passed to the fit directly
                refit(answer)

            elif answer == 'dw' and cfg.type_norm == 2:
                wtd = get_type_input('select window: ', int)
                wv_left = np.delete(wv_left, wtd - 1)
                wv_right = np.delete(wv_right, wtd - 1)

                flux = flux_orig
                flux = norm2(wv_left, wv_right, wvl, flux, fl_syn_norm2)
                refit(answer)

            elif answer == 'aw' and cfg.type_norm == 2:
                new_wv_left = get_type_input('new left window: ', float)
                new_wv_right = get_type_input('new right window: ', float)

                flux = flux_orig

                new_islands_left, new_islands_right = find_clipped_regions([new_wv_left], [new_wv_right], wvl, flux)

                wv_left = np.append(wv_left, new_islands_left)
                wv_right = np.append(wv_right, new_islands_right)

                # 3. Sort the windows
                ind_sort = np.argsort(wv_left)
                wv_left = wv_left[ind_sort]
                wv_right = wv_right[ind_sort]

                flux = norm2(wv_left, wv_right, wvl, flux, fl_syn_norm2)
                refit(answer)

            elif answer in ['aw', 'dw'] and cfg.type_norm != 2:
                print('Option valid only for NORM = 2 \n')

            elif answer in ['f3', 'f4'] and tana != 'ABU2':
                print('Option valid only with ABU2 \n')

            elif answer == 'f' and cfg.grid == 'y':
                print('Working on a grid. Cannot call k2!!! \n')

            elif answer == 'f' and cfg.grid == 'n' and tana != 'ROTAT':
                # new synthetic beyond the edge of the grid where the chi2 decreases, as in the automatic
                # mode (same helper: steps of 0.2 dex, or 10 for ISOR, which cannot go below 1)
                if chi[-1] <= chi[-2]:
                    direction = 'up'
                elif chi[0] <= chi[1]:
                    direction = 'down'
                else:
                    direction = None

                if direction is None:
                    # the minimum is already inside the grid: no new synthetic (before, one was computed and not used)
                    print('The chi2 minimum is already inside the grid: no new synthetic spectrum needed.\n')
                else:
                    abu, delta_arr, fl_syn_br, wl_syn, fl_syn_nb = _update_grid_step(direction, abu, delta_arr, fl_syn_br, param1, window, resol, tana, fl_syn_nb=fl_syn_nb)
                refit(answer)

            elif answer == 'i':
                resol, wv_shift, cshift, wvl, flux = resol_orig, wv_shift_orig, cshift_orig, wvl_orig, flux_orig
                fit1, fit2, wv_left, wv_right, abu = fit1_orig, fit2_orig, wv_left_orig, wv_right_orig, abu_orig
                delta_arr = delta_arr_orig
                arg_extra = kwargs.copy()

                if cfg.type_norm == 2:
                    flux = norm2(wv_left, wv_right, wvl, flux, fl_syn_norm2)

                if cfg.type_norm == 1 and cfg.grid == 'n':
                    wl_syn_tmp, fl_syn_tmp = autok(param1, delta, window)
                    fl_syn_tmp = convolve_gauss2(wl_syn_tmp, fl_syn_tmp, resol)
                    flux = norm1(wvl, flux, wl_syn_tmp.value, fl_syn_tmp.value)

                if tana.startswith(ABU_KEYS) and cfg.grid == 'y':
                    wl_syn, fl_syn_br, _ = readsyn(sss, resol, line)
                    if cfg.type_norm == 1:
                        flux = norm1(wvl, flux, wl_syn.value, fl_syn_br[0].value)   # FIX: wl_syn_tmp does not exist in grid mode
                elif tana.startswith(ABU_KEYS) and cfg.grid == 'n':
                    fl_syn_nb, fl_syn_br, wl_syn = fl_syn_nb_orig, fl_syn_br_orig, wl_syn_orig
                refit(answer)

            elif answer == 'y':
                refit(answer)
                plotter.close()
                if tana.startswith(ABU_KEYS):
                    ew_k2, EP, loggf, abu_err_mc = final_ew_and_mc()
                break

            elif answer in ['r', 'e']:
                plotter.close()
                abu_fin, abu0, abu_err_mc, EP, loggf, ew_k2 = 99.99, 99.99, 99.99, 99.999, 99.999, 99.99
                break

        if answer in ['y', 'r'] and save_flag == 'y':
            save_line_plot()

        if tana.startswith(ABU_KEYS):
            return abundance_result(answer, ew_k2, EP, loggf, abu_err_mc)
        elif tana == 'ROTAT':
            # result of the LAST fit, error from the Monte Carlo; 99.99 if rejected (r)
            return rotation_result(answer)


    ### AUTOMATIC MODE
    elif cfg.inter == 'n' and cfg.grid == 'n' and tana == 'ROTAT':
        if save_flag == 'y':
            save_line_plot()
        return rotation_result('y')

    elif cfg.inter == 'n' and cfg.grid == 'y' and tana.startswith(ABU_KEYS):    # FIX: ISOR was missing (function returned None)
        if save_flag == 'y':
            save_line_plot()
        ew_k2, EP, loggf, abu_err_mc = final_ew_and_mc()
        return abundance_result('y', ew_k2, EP, loggf, abu_err_mc)

    elif cfg.inter =='n' and cfg.grid == 'n' and tana.startswith(ABU_KEYS):
        iterations = 0
        answer = 'rrr'

        ### check for ISOR = 1
        skip_loop = False

        if tana == 'ISOR' and (abu_fin == 1 or len(chi) < 2):
            skip_loop = True


        while iterations < cfg.limit_iter and not skip_loop:
            direction_to_shift = None
            if chi[-1] <= chi[-2]:
                direction_to_shift = 'up'
            elif chi[0] <= chi[1]:
                direction_to_shift = 'down'

            # If the minimum is in the middle, we have converged.
            if direction_to_shift is None:
                break

            abu, delta_arr, fl_syn_br, wl_syn = _update_grid_step(direction_to_shift, abu, delta_arr, fl_syn_br, param1, window, resol, tana)

            # Recalculate chi-squared with the new grid
            refit(answer)

            iterations += 1


        # --- Final Return Block (handles both convergence and limit reached) ---
        # FIX: decide on convergence, not on the iteration counter. Before, a line whose chi2
        # minimum became bracketed exactly at the last allowed step was flagged as out of limit.
        converged = skip_loop or not (chi[-1] <= chi[-2] or chi[0] <= chi[1])
        if not converged:
            plot_data.update({"abu_fin": 99.99})
            plotter.update(plot_data)
            if save_flag == 'y':
                save_line_plot()

            print('---')
            print('Out of limit of calculation\n')
            return _rejected_result()
        else:
            # final fit (with the synthetics at +-0.2 dex from the best fit for the plot)
            refit('y')
            if save_flag == 'y':
                save_line_plot()
            ew_k2, EP, loggf, abu_err_mc = final_ew_and_mc()
            return abundance_result('y', ew_k2, EP, loggf, abu_err_mc)
