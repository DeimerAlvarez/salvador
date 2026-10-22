"""Rotational velocity (and abundance) of a line: analysis_rotat."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from PyAstronomy import pyasl

from .config import cfg
from .elements import define_abu0
from .k2_interface import autok
from .fitting import convolve_gauss2, mc_seed


def analysis_rotat(nelem, alpha, met, delta, elem, wvl, flux, line, window, param1, resol, fit1, fit2, **kwargs):
    """Rotational velocity (and abundance) of a line: coarse grid of synthetic spectra (abundance x
    rotation), interpolated fine grid, chi2 minimum, Monte Carlo error. Options in input.salvador:
    ROT_ABU_SPAN (dex, 0 = abundance fixed), ROT_MIN, ROT_MAX (km/s), ROT_STEP, ROT_NMC.
    For each line, abu_init, abu_span, rot_min and rot_max can be given in the linelist (optional
    columns 8-11 of a ROTAT row); they override the input.salvador values."""
    # Abundance of the model atmosphere: K2 receives the offsets from this value (DELTA = A - A_model)
    abu_model = define_abu0(int(nelem), alpha, met, 0.0, elem)
    # initial abundance: from the linelist (absolute, optional column), otherwise the model one + var
    abu0 = kwargs.get('abu_init')
    if abu0 is None:
        abu0 = abu_model + delta
    abu_span = kwargs.get('abu_span', cfg.rot_abu_span)
    rot_min = kwargs.get('rot_min', cfg.rot_min)
    rot_max = kwargs.get('rot_max', cfg.rot_max)
    n_mc = kwargs.get('n_mc', cfg.rot_nmc)

    # Abundances: a single value (fixed) or a range
    abundances = np.array([abu0]) if abu_span == 0 else np.round(np.arange(abu0 - abu_span, abu0 + abu_span + 0.01, 0.1), 2)
    # FIX: rot_max is always a node of the grid (before, with e.g. rot_max = 50 the last node was 40
    # and the fine grid beyond it was NaN, giving a wrong minimum without any error)
    # step of the rotation nodes: ROT_STEP (default 10 km/s; with 20 km/s and linear interpolation the
    # result was biased by a few km/s). Rotation is applied after the synthesis: no extra K2 calls.
    rotations = np.unique(np.append(np.arange(rot_min, rot_max, kwargs.get('rot_step', cfg.rot_step)), rot_max))

    # Preliminary cut (wide mask)
    mask_cut = (wvl > line - (window / 2)) & (wvl < line + (window / 2))
    wv_cut = wvl[mask_cut]
    pix = np.round(np.min(np.diff(wv_cut)), 4)
    wl_syn_grid = np.arange(wv_cut[0], wv_cut[-1] + pix, pix)

    # Observed flux interpolated on the grid
    fl_cut = np.interp(wl_syn_grid, wvl, flux)

    num_abu = len(abundances)
    num_rot = len(rotations)

    # MATRICE 3D COARSE
    synthetic_matrix = np.zeros((num_abu, num_rot, len(wl_syn_grid)))

    for i, abu_val in enumerate(abundances):
        # FIX: offset from the MODEL abundance (before: from abu_init, so the synthetics were computed
        # at the wrong abundance whenever abu_init differed from the model value, e.g. Li)
        wl_syn_tmp, fl_syn_tmp = autok(param1, np.round(abu_val - abu_model, 2), window)
        fl_syn_instr = convolve_gauss2(wl_syn_tmp, fl_syn_tmp, resol)

        fl_base_resampled = np.interp(wl_syn_grid, wl_syn_tmp.value, fl_syn_instr.value)

        for j, rot_val in enumerate(rotations):
            if rot_val <= 0:
                synthetic_matrix[i, j, :] = fl_base_resampled
            else:
                synthetic_matrix[i, j, :] = pyasl.rotBroad(wl_syn_grid, fl_base_resampled, 0.6, rot_val)

    # Narrow mask for the fit
    mask_fit = (wl_syn_grid >= fit1) & (wl_syn_grid <= fit2)

    obs_fit = fl_cut[mask_fit]
    syn_fit_matrix = synthetic_matrix[:, :, mask_fit]

    # Chi-Quadro Coarse
    diff_squared = (obs_fit[np.newaxis, np.newaxis, :] - syn_fit_matrix)**2
    chi2_grid_coarse = np.sum(diff_squared / (syn_fit_matrix + 1e-9), axis=-1)

    best_idx = np.unravel_index(np.argmin(chi2_grid_coarse), chi2_grid_coarse.shape)
    best_abu_coarse = abundances[best_idx[0]]
    best_rot_coarse = rotations[best_idx[1]]

    # --- FINE GRID ---
    # FIX: limited to the nodes of the coarse grid (no extrapolation)
    finer_rotations = np.arange(max(rotations[0], best_rot_coarse - 20), min(rotations[-1], best_rot_coarse + 20) + 0.5, 1)

    if num_abu == 1:
        print("Fixed abundance: interpolation on rotation only.")
        finer_abundances = abundances
        matrix_1d = synthetic_matrix[0, :, :]

        # FIX: cubic needs at least 4 nodes (crash with e.g. rot_max = 40)
        kind = 'cubic' if num_rot >= 4 else 'linear'
        interpolator = interp1d(rotations, matrix_1d, axis=0, kind=kind)
        fine_syn_matrix = interpolator(finer_rotations)[np.newaxis, :, :]

    else:
        # The fine grid must not go outside the coarse grid
        min_abu_allowed = abundances[0]
        max_abu_allowed = abundances[-1]

        abu_start = max(min_abu_allowed, best_abu_coarse - 0.20)
        abu_end = min(max_abu_allowed, best_abu_coarse + 0.21)

        # FIX: rounded and clipped, so floating-point steps cannot fall outside the grid (NaN)
        finer_abundances = np.clip(np.round(np.arange(abu_start, abu_end, 0.01), 2), min_abu_allowed, max_abu_allowed)

        # bounds_error=False for safety
        # cubic (more accurate in rotation) when possible: needs >= 4 nodes per axis and SciPy >= 1.10
        try:
            interpolator = RegularGridInterpolator((abundances, rotations), synthetic_matrix,
                                               method='cubic', bounds_error=False, fill_value=np.nan)
        except ValueError:
            interpolator = RegularGridInterpolator((abundances, rotations), synthetic_matrix,
                                               method='linear', bounds_error=False, fill_value=np.nan)

        A, R = np.meshgrid(finer_abundances, finer_rotations, indexing='ij')
        coords = np.stack([A.ravel(), R.ravel()], axis=-1)

        flat_fine_matrix = interpolator(coords)
        fine_syn_matrix = flat_fine_matrix.reshape(len(finer_abundances), len(finer_rotations), len(wl_syn_grid))

    # --- CHI-QUADRO FINE ---
    syn_fine_fit = fine_syn_matrix[:, :, mask_fit]
    diff_squared_fine = (obs_fit[np.newaxis, np.newaxis, :] - syn_fine_fit)**2
    min_chi2_fine = np.sum(diff_squared_fine / (syn_fine_fit + 1e-9), axis=-1)

    best_idx_fine = np.unravel_index(np.nanargmin(min_chi2_fine), min_chi2_fine.shape)
    best_abu_fine = finer_abundances[best_idx_fine[0]]
    best_rot_fine = finer_rotations[best_idx_fine[1]]

    # ====================================================================
    # MONTE CARLO SIMULATION FOR THE ERRORS
    # ====================================================================

    # 1. Best-fit model in the fit window
    best_model_fit = syn_fine_fit[best_idx_fine[0], best_idx_fine[1], :]

    # 2. Noise estimated from the residuals (observed - model)
    residuals = obs_fit - best_model_fit
    sigma_noise = np.std(residuals)

    # 3. Monte Carlo parameters
    mc_abu = np.zeros(n_mc)
    mc_rot = np.zeros(n_mc)
    rng = np.random.default_rng(mc_seed(param1))    # reproducible, as for the abundances

    # 4. Simulation loop
    for k in range(n_mc):
        # "Mock" observed spectrum: best fit + random Gaussian noise
        mock_obs = best_model_fit + rng.normal(0, sigma_noise, size=len(best_model_fit))

        # Chi-square of the mock against the existing fine grid
        diff_sq_mc = (mock_obs[np.newaxis, np.newaxis, :] - syn_fine_fit)**2
        chi2_mc = np.sum(diff_sq_mc / (syn_fine_fit + 1e-9), axis=-1)

        # Coordinates of the new minimum
        idx_mc = np.unravel_index(np.nanargmin(chi2_mc), chi2_mc.shape)

        # Results of this iteration
        mc_abu[k] = finer_abundances[idx_mc[0]]
        mc_rot[k] = finer_rotations[idx_mc[1]]

    # 5. The error is the standard deviation (1 sigma) of the distributions
    err_abu = np.std(mc_abu)
    err_rot = np.std(mc_rot)

    print(f"Final Best Fit -> Rot: {best_rot_fine} \u00B1 {np.floor(err_rot + 0.5):.0f} km/s --- Abu: {best_abu_fine:.2f} \u00B1 {err_abu:.2f}")
    # ====================================================================


    # ====================================================================
    # FINAL SPECTRUM (best fit and +-10 km/s, for the plot)
    # ====================================================================
    wv_fin, fl_fin = autok(param1, np.round(best_abu_fine - abu_model, 2), window)
    pix_fin = np.round(np.min(np.diff(wv_fin.value)), 4)
    wv_final = np.arange(wv_fin.value[0], wv_fin.value[-1] + pix_fin, pix_fin)
    fl_final = np.interp(wv_final, wv_fin.value, fl_fin.value)

    # STEP A: instrumental convolution (always)
    fl_final_instr = convolve_gauss2(wv_final, fl_final, resol)

    # Helper: rotational broadening only for a positive rotation
    def apply_safe_rotation(wave, flux, rot_vel):
        """Rotational broadening (limb darkening 0.6), only for a positive rotation."""
        w_val = getattr(wave, 'value', wave)
        f_val = getattr(flux, 'value', flux)
        if rot_vel <= 0:
            return f_val  # No rotation: the instrumental spectrum
        return pyasl.rotBroad(w_val, f_val, 0.6, rot_vel)

    # STEP B: final rotation
    fl_final_rot = apply_safe_rotation(wv_final, fl_final_instr, best_rot_fine)

    # Spectra at +-10 km/s (not below 0)
    rot_minus = max(0, best_rot_fine - 10)
    rot_plus = best_rot_fine + 10

    fl_final_rotm = apply_safe_rotation(wv_final, fl_final_instr, rot_minus)
    fl_final_rotp = apply_safe_rotation(wv_final, fl_final_instr, rot_plus)

    # Data for the plot
    plot_data = {
        "wl_syn": wv_final, "fl_syn1": fl_final_rotp, "fl_syn2": fl_final_rotm,
        "xfit": wv_final, "bestfit": fl_final_rot, "xspec": wl_syn_grid,
        "yspec": fl_cut, "line": line, "finer_rotations": finer_rotations,
        "min_chi2_fine": min_chi2_fine, "finer_abundances": finer_abundances,
        "elem": elem, "window": window, "best_rot_fine": best_rot_fine,
        "best_abu_fine": best_abu_fine, "fit1": fit1, "fit2": fit2,
        "err_abu": err_abu, "err_rot": err_rot # Monte Carlo errors
    }

    return plot_data
