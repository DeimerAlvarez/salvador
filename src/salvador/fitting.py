"""Line profile and resolution, convolution, abundance determination (chi2), Monte Carlo errors."""
import warnings
import zlib
from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.stats import sigma_clip
from PyAstronomy import pyasl
from specutils import SpectralRegion
from specutils import Spectrum
from specutils.analysis import equivalent_width
from specutils.manipulation import extract_region

from .config import SYNTH_OUT, cfg, init_ans, plt
from .k2_interface import autok

def interp_in_abundance(abu_new, abu, spectra):
    """Linear interpolation in abundance of a set of spectra, for all pixels at once.
    abu: (N,) increasing; spectra: (N, Npix); abu_new: (M,). Returns (M, Npix).
    Same formula and edge clamping as np.interp applied pixel by pixel."""
    abu = np.asarray(abu, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    abu_new = np.clip(np.asarray(abu_new, dtype=float), abu[0], abu[-1])
    j = np.clip(np.searchsorted(abu, abu_new, side='right') - 1, 0, len(abu) - 2)
    slope = (spectra[j + 1] - spectra[j]) / (abu[j + 1] - abu[j])[:, None]
    return slope * (abu_new - abu[j])[:, None] + spectra[j]


def gaussian(x, aux):
    """Gaussian profile on a continuum: aux = (height, center, width (sigma), offset)."""
    height, center, width, offset = aux
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def gaussian_cf(x, amp, center, sigma, offset):
    """Gaussian profile on a continuum, with separate parameters (form required by curve_fit)."""
    return amp * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset

def fwhm_empirica(wvl_fit, flux_fit):
    """Empirical FWHM and center of an absorption line, from the two points where a spline of the
    profile crosses half depth. Returns (None, None) if they cannot be found."""
    min_flux = np.min(flux_fit)
    continuum = np.max(flux_fit)

    depth = continuum - min_flux
    half_max = continuum - (depth / 2.0)

    spline = UnivariateSpline(wvl_fit, flux_fit - half_max, s=0)

    roots = spline.roots()

    fwhm_measured = None
    center_measured = None
    if len(roots) >= 2:
        fwhm_measured = roots[-1] - roots[0]
        center_measured = (roots[-1] + roots[0]) / 2.0

    # --- PLOT (debug) ---
    plot = False
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(wvl_fit, flux_fit, 'o-', color='darkgray', markersize=5, label='Original data (Synthetic)')

        wvl_dense = np.linspace(wvl_fit.min(), wvl_fit.max(), 500)
        flux_dense = spline(wvl_dense) + half_max
        plt.plot(wvl_dense, flux_dense, '-', color='dodgerblue', linewidth=2, label='Spline')
        plt.axhline(half_max, color='tab:red', linestyle=':', alpha=0.7, label=f'Half-Max Level ({half_max:.3f})')

        if len(roots) >= 2:
            plt.plot(roots, [half_max] * len(roots), 'x', color='crimson', markersize=10, markeredgewidth=2, zorder=5)
            plt.hlines(y=half_max, xmin=roots[0], xmax=roots[-1], color='crimson', linewidth=3, label=f'Measured FWHM = {fwhm_measured:.4f} A')

        plt.title('Empirical measure of FWHM', fontsize=14, pad=10)
        plt.xlabel('Wavelength (A)', fontsize=12)
        plt.ylabel('Normalized Flux', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    return fwhm_measured, center_measured

def new_res_est(wvl, flux, line, resol, fit1, fit2, tana):
    """Measure the observed line between fit1 and fit2: FWHM, center and area, each the mean of a
    Gaussian fit and of the empirical profile. Returns (resolution = center/FWHM, FWHM, center,
    area); the input resolution, line/resol, line and 0 if the measure fails (and for ABU2)."""
    if tana == 'ABU2':
        return resol, line/resol, line, 0.0

    mask = (wvl > fit1) & (wvl < fit2)
    wvl_fit, flux_fit = wvl[mask], flux[mask]

    new_res = resol
    fwhm = line/resol
    mean = line
    area = 0.0

    if len(wvl_fit) == 0:
        return new_res, fwhm, mean, area

    # Parameters: [height (depth), center, width, offset (continuum)]
    initial_guess = [-0.2, line, line/resol, 1.0]
    bounds = ([-2.0, line - 0.5, 0.001, 0.5],   # Lower limits (the absorption amplitude is negative)
        [ 0.0, line + 0.5, 1.0, 1.5])           # Upper limits (the offset cannot be 100)
    try:
        optim, _ = curve_fit(gaussian_cf, wvl_fit, flux_fit, p0=initial_guess, bounds=bounds)
        if optim is not None and len(optim) == 4:
            amp_opt, mean_opt, sigma_opt, offset_opt = optim

            # Gaussian fit
            fwhm_gauss = np.sqrt(8 * np.log(2)) * np.abs(sigma_opt)
            mean_gauss = mean_opt

            wvl_dense = np.linspace(wvl_fit[0], wvl_fit[-1], max(200, len(wvl_fit) * 2))
            y_fit_dense = gaussian(wvl_dense, optim)
            area_gauss = max(trapezoid(1.0 - y_fit_dense, wvl_dense), 0.0)

            # Empirical fit
            fwhm_emp, mean_emp = fwhm_empirica(wvl_fit, flux_fit)
            area_emp = max(trapezoid(1.0 - flux_fit, wvl_fit), 0.0)

            # ==========================================
            valid_fwhm = [f for f in (fwhm_emp, fwhm_gauss) if f is not None and f > 0]
            fwhm = np.mean(valid_fwhm) if valid_fwhm else line / resol

            valid_means = [m for m in (mean_emp, mean_gauss) if m is not None and m > 0]
            mean = np.mean(valid_means) if valid_means else line

            valid_areas = [a for a in (area_emp, area_gauss) if a is not None and a > 0]
            area = np.mean(valid_areas) if valid_areas else 0.0

            new_res = int(mean / fwhm)

            # Plot
            plot = False
            if plot:
                print(f'FWHM: {fwhm:.4f} | Res_est: {new_res} | In_Res: {resol}')

                plt.figure()
                plt.step(wvl_dense, y_fit_dense, c='k', alpha=0.5, label='Gaussian fit')
                plt.step(wvl, flux, c='r', label='Data')
                plt.axvline(mean, c='b', ls=':', label='Center')
                plt.axvline(fit1, c='k', ls='--')
                plt.axvline(fit2, c='k', ls='--')
                plt.xlim(fit1-2, fit2+2)
                plt.ylim(min(flux_fit)-0.1, 1.2)
                plt.legend()
                plt.show()

    except (RuntimeError, ValueError, TypeError, ZeroDivisionError):
        pass

    return new_res, fwhm, mean, area


def isolate_pure_line(wave, flux):
    """Part of a synthetic profile that belongs to the main line only: from the minimum, outwards
    until the flux starts decreasing again (beginning of a nearby line). Returns (wave, flux)."""
    core_idx = np.argmin(flux)
    tol = 1e-5

    left_idx = core_idx
    while left_idx > 0:
        if flux[left_idx - 1] < flux[left_idx] - tol:
            break
        left_idx -= 1

    right_idx = core_idx
    while right_idx < len(flux) - 1:
        if flux[right_idx + 1] < flux[right_idx] - tol:
            break
        right_idx += 1

    return wave[left_idx:right_idx+1], flux[left_idx:right_idx+1]

def convolve_gauss(wave_syn, list_syn, res, fwhm_obs, line, fit1, fit2, tana, ans):
    ## reading the spectra NOT broadened
    """Broaden the synthetic spectra to the resolution res (Gaussian instrumental profile). With
    AUTO_RES, at the first fit the resolution is re-estimated from the observed line, within the
    allowed range. Returns (tuple of broadened spectra, resolution used)."""
    fl_syn_res = ()
    for index in range(len(list_syn)):
        bbb, fwhm_syn = pyasl.instrBroadGaussFast(wave_syn.value, list_syn[index].value, res, edgeHandling='firstlast', fullout=True, maxsig=5., equid=True)
        bbb = bbb * u.count

        fl_syn_res += (bbb,)

    ### fit RES vs FWHM. This provides an automatic estimation of the resolution to apply to the line
    if cfg.auto_res_limit > 0. and ans == init_ans:
        mask_syn = (wave_syn.value > fit1) & (wave_syn.value < fit2)
        wave_syn_nb, fl_syn_nb = isolate_pure_line(wave_syn.value[mask_syn], list_syn[0].value[mask_syn])
        _, fwhm_syn_pure, _, _ = new_res_est(wave_syn_nb, fl_syn_nb, line, res, fit1, fit2, tana)

        if fwhm_obs > fwhm_syn_pure:
            fwhm_instr_needed = np.sqrt(fwhm_obs**2 - fwhm_syn_pure**2)

            r_estimated = line / fwhm_instr_needed
            r_estimated = int(round(r_estimated, -2))
        else:
            r_estimated = res


        var_perc = abs(r_estimated - res) / res
        if var_perc > 0.:
            if cfg.verbose == 1:
                print(f"Suggested res = {r_estimated:.0f} (Variation: {var_perc*100:.1f}%).\n")

        min_tol = 0.02 if cfg.auto_res_limit > 0.02 else 0.0
        if min_tol < var_perc <= cfg.auto_res_limit:
            if cfg.verbose == 0:
                print(f"for line {line:.3f} is tried res = {r_estimated}")

            temp_fl_syn_res = ()
            try:
                for index in range(len(list_syn)):
                    bbb, fwhm_syn = pyasl.instrBroadGaussFast(wave_syn.value, list_syn[index].value, r_estimated,
                    edgeHandling='firstlast', fullout=True, maxsig=5., equid=True)
                    bbb = bbb * u.count
                    temp_fl_syn_res += (bbb,)

                res = r_estimated
                fl_syn_res = temp_fl_syn_res

            except ValueError:
                if cfg.verbose == 1:
                    print(f"Interpolation error (weak or noisy line). \nKept res = {res}")
                pass

        elif var_perc > cfg.auto_res_limit:
            if cfg.verbose == 1:
                print(f"Security Block: Estimated res ({r_estimated}) differs by >{cfg.auto_res_limit*100:.0f}% from input. \nKept res = {res}")

    return fl_syn_res, res


def convolve_gauss2(wave_syn, syn, res):
    ## reading the spectrum NOT broadened
    """Broaden one spectrum to the resolution res (Gaussian instrumental profile)."""
    w = getattr(wave_syn, 'value', wave_syn)
    f = getattr(syn, 'value', syn)
    bbb, _ = pyasl.instrBroadGaussFast(w, f, res, edgeHandling='firstlast', fullout=True, maxsig=5., equid=True)
    bbb = bbb * u.count

    return bbb


def _estimate_snr(wave: np.ndarray, flux: np.ndarray, line: float,
                  wv_left: Sequence[float], wv_right: Sequence[float]) -> float:
    """SNR within +-20 A of the line.

    NORM = 2: mean/std of the pixels inside the continuum windows (as before). If no window pixel
    is within +-20 A (before: SNR = NaN, hence a Monte Carlo error of exactly 0), and for NORM != 2:
    mean/std of the 3-sigma-clipped flux within +-20 A. NaN if fewer than 2 pixels remain.

    Args:
        wave: Observed wavelengths [A].
        flux: Normalized observed flux.
        line: Wavelength of the line [A].
        wv_left: Left edges of the continuum windows (NORM = 2).
        wv_right: Right edges of the continuum windows (NORM = 2).

    Returns:
        The signal-to-noise ratio, or NaN if it cannot be estimated.
    """
    near = (wave > line - 20) & (wave < line + 20)
    if cfg.type_norm == 2 and len(wv_left) > 0:
        in_windows = np.zeros(wave.size, dtype=bool)
        for left, right in zip(wv_left, wv_right):
            in_windows |= (wave > left) & (wave < right)
        sel = flux[near & in_windows]
        if sel.size >= 2 and np.std(sel) > 0:
            return float(np.mean(sel) / np.std(sel))
        warnings.warn(f'no continuum window within +-20 A of {line:.3f}: SNR from the sigma-clipped flux', stacklevel=2)
    sel = sigma_clip(flux[near], sigma=3, maxiters=5).compressed()
    if sel.size < 2 or np.std(sel) == 0:
        return float('nan')
    return float(np.mean(sel) / np.std(sel))


def abu_determination(spec, fit1, fit2, wl_syn, fl_syn_br, abu, line, elem, ans, wv_left, wv_right, param1, save='y', window=None, **kwargs):
    """Abundance of the line from the synthetics at the abundances abu: chi2 of each synthetic in the
    fit region (fit1-fit2, plus the second region of ABU2), fine grid in steps of 0.01 dex by
    interpolation, minimum. Also the EW of the best fit and the SNR near the line. With
    ans = 'y' (final fit, plot saved) the best-fit and +-0.2 dex synthetics are computed with K2.
    Returns (abundance, EW, SNR, chi2 of the fine grid, inputs for the Monte Carlo, data for the plot)."""
    if window is None:      # window of the line (passed by measure_line)
        window = cfg.window
    fit11 = kwargs.get('wind_l1')
    fit22 = kwargs.get('wind_r1')

    abu0 = float(param1['abu0'])
    nelem = float(param1['nelem'])

    norm_dop2 = Spectrum(flux=spec.flux, spectral_axis=spec.spectral_axis)
    reg = SpectralRegion(fit1*u.AA, fit2*u.AA)
    if fit11 is not None:
        reg2 = SpectralRegion(fit11*u.AA, fit22*u.AA)
        reg = reg + reg2

    fit = extract_region(norm_dop2, reg)
    if fit11 is not None:
        fitx = np.concatenate([segment.spectral_axis.value for segment in fit])
        fity = np.concatenate([segment.flux.value for segment in fit])
    else:
        fitx = fit.spectral_axis.value
        fity = fit.flux.value

    ### first guess of the abundance from the entire grid of synthetic
    if fit11 is not None:
        mask_reg1 = (wl_syn.value > fit1) & (wl_syn.value < fit2)
        mask_reg2 = (wl_syn.value > fit11) & (wl_syn.value < fit22)
        mask = mask_reg1 | mask_reg2
    else:
        mask = (wl_syn.value > fit1) & (wl_syn.value < fit2)

    wl_syn2 = np.array(wl_syn[mask])
    syn_br2 = {f'syn_{i}_br2': xtmp[mask] for i, xtmp in enumerate(fl_syn_br)}
    ss = np.array([syn_br2[f'syn_{k}_br2'] for k in range(len(abu))])
    chis = np.array([np.sum(((fity - (fake2 := np.interp(fitx, wl_syn2, ss[k]))) ** 2) / fake2) for k in range(len(abu))])
    minind = np.argmin(chis)
    abu_fin = np.round(abu[minind], 2)

    ddif = np.abs(abu - abu_fin)
    mindif = np.argmin(ddif)
    last_ind = len(abu) - 1
    ind1 = min(mindif + 1, last_ind)
    ind2 = max(mindif - 1, 0)

    type_abu = str(param1['tana'])
    if type_abu != 'ISOR':
        abu3 = np.arange(abu[ind2], abu[ind1], 0.01)
    elif type_abu == 'ISOR':
        abu3 = np.arange(abu[ind2], abu[ind1]+1, 1)

    ### determination of the abundance from the 3 synthetic spectra
    # SPEED: vectorized. Same result as the former nested np.interp loop (per pixel x per abundance):
    # the synthetics are first resampled on the observed pixels, then interpolated in abundance.
    ss = np.array([syn_br2[f'syn_{k}_br2'] for k in range(len(abu))])
    ss_fitx = np.array([np.interp(fitx, wl_syn2, ss_k) for ss_k in ss])
    grid_fitx = interp_in_abundance(abu3, abu, ss_fitx)
    chis = np.sum((fity - grid_fitx) ** 2 / grid_fitx, axis=1)
    minind = np.argmin(chis)
    abu_fin = np.round(abu3[minind], 2)

    ### creation of the interpolated spectrum corresponding to the final abundance
    if fit11 is not None:
        mask = (wl_syn.value > fit1 - (window / 2)) & (wl_syn.value < fit22 + (window / 2))
    else:
        mask = (wl_syn.value > fit1 - (window / 2)) & (wl_syn.value < fit2 + (window / 2))


    wl_syn2 = np.array(wl_syn[mask])
    syn_br2 = {f'syn_{i}_br2': xtmp[mask] for i, xtmp in enumerate(fl_syn_br)}
    ss = np.array([syn_br2[f'syn_{k}_br2'] for k in range(len(abu))])
    fake = interp_in_abundance(np.array([abu_fin]), abu, ss)[0] # SPEED: vectorized

    # EW determination on the best fit (used only with a grid of synthetic spectra: with K2 the EW
    # of the line alone is computed by eqw_k2).
    # FIX: the synthetic spectra are normalized, so the continuum is 1. Before, the continuum was
    # the mean flux of the best fit at the two edges of the fit region: with the edges in the wings
    # of the line or on nearby lines (crowded spectra) the EW was underestimated, or even negative.
    spec_ew = Spectrum(spectral_axis=wl_syn2*u.AA, flux=fake*u.count)
    EW = equivalent_width(spec_ew, continuum=1.0, regions=SpectralRegion(fit1 * u.AA, fit2 * u.AA)).value * 1000


    # estimate of SNR around each line
    SNR = _estimate_snr(spec.spectral_axis.value, spec.flux.value, line, wv_left, wv_right)

    if ans != 'y':
        print('')
        if type_abu != 'ISOR':
            print(f'A({elem}) = {abu_fin:.2f}')
        elif type_abu == 'ISOR':
            print(f'Isotopic ratio ({elem}) = {abu_fin:.0f}')

        if ans == init_ans: print('SNR = %.0f' % SNR)

    ### parameters for the Monte Carlo. TO BE REVIEWED FOR ABU2
    mask_obs = (wl_syn2 > fit1) & (wl_syn2 < fit2)
    mask_syn = (wl_syn2 > fit1 - 1.5) & (wl_syn2 < fit2 + 1.5)

    syn_mc = np.array([syn_br2[f'syn_{k}_br2'][mask_syn] for k in range(len(abu))])

    param_mc = (wl_syn2[mask_syn], syn_mc, wl_syn2[mask_obs], fake[mask_obs], abu3)

    ###
    syn1, syn2 = fl_syn_br[ind2], fl_syn_br[ind1]

    ### calculating the bestfit spectrum and saving the info for the line
    # SPEED + FIX: the 'save' argument was never used, so the parameter-error reruns (save='n')
    # computed 3 extra K2 syntheses (bestfit, -0.2, +0.2) for a plot that is never saved, and
    # overwrote the nominal bestfit_*.txt file with the perturbed fit.
    do_best = (ans == 'y' and cfg.grid == 'n' and save == 'y')
    if do_best:
        name_star = param1['name_star']

        if type_abu == 'ISOR':
            bestabu = abu_fin
        else:
            bestabu = np.round(abu_fin - abu0 + float(param1['delta']), 2)

        wl_best, fl_best = autok(param1, bestabu, window)
        fl_best = convolve_gauss2(wl_best, fl_best, int(param1['res']))

        ### saving the info
        half_wind = window / 2
        min_lim = line - half_wind
        max_lim = line + half_wind

        mask_obs1 = (norm_dop2.spectral_axis.value >= min_lim) & (norm_dop2.spectral_axis.value <= max_lim)
        wave_obs_cut = norm_dop2.spectral_axis.value[mask_obs1]
        flux_obs_cut = norm_dop2.flux.value[mask_obs1]

        flux_syn_interp = np.interp(wave_obs_cut, wl_best.value, fl_best.value)

        df_final = pd.DataFrame({'Codice_Riga': nelem,
                    'Lambda': line,
                    'Wave_Obs': wave_obs_cut,
                    'Flux_Syn': flux_syn_interp,
                    'Flux_Obs': flux_obs_cut})

        df_final['Codice_Riga'] = df_final['Codice_Riga'].map('{:>8.2f}'.format)
        df_final['Lambda'] = df_final['Lambda'].map('{:>8.3f}'.format)
        df_final['Wave_Obs'] = df_final['Wave_Obs'].map('{:>10.5f}'.format)
        df_final['Flux_Syn'] = df_final['Flux_Syn'].map('{:>8.5f}'.format)
        df_final['Flux_Obs'] = df_final['Flux_Obs'].map('{:>10.5f}'.format)

        output_dir = SYNTH_OUT
        output_dir.mkdir(exist_ok=True)

        output_file_name = f"bestfit_{name_star}_line_{line:.3f}.txt"
        destination_file_path = output_dir / output_file_name

        custom_header = "#Element_code\tLambda\t\tWave\t\tFlux_Syn\t  Flux_Obs\n"
        with open(destination_file_path, 'w') as f:
            f.write(custom_header)

            df_final.to_csv(f, sep='\t', index=False, header=False)


    ### creating 2 synthetic of +-0.2 wrt the bestfit abundance
    abu3_plot, chis_plot = abu3, chis       # chi2 shown in the plot (recomputed below with CALC_BEST = y)
    if do_best and cfg.calc_best == 'y':
        if type_abu != 'ISOR':
            delta_tmp = np.round(abu_fin - abu0 - 0.2 + float(param1['delta']), 2)
            wl_syn1, syn1 = autok(param1, delta_tmp, window)
            syn1 = convolve_gauss2(wl_syn1, syn1, int(param1['res']))

            delta_tmp = np.round(abu_fin - abu0 + 0.2 + float(param1['delta']), 2)
            wl_syn_tmp, syn2 = autok(param1, delta_tmp, window)
            syn2 = convolve_gauss2(wl_syn_tmp, syn2, int(param1['res']))

            wl_syn = wl_syn_tmp.value
            # synthetics around the best fit: (abundance, wavelength, flux), for the chi2 plot below
            around_best = [(abu_fin - 0.2, wl_syn1, syn1), (abu_fin, wl_best, fl_best), (abu_fin + 0.2, wl_syn_tmp, syn2)]
            step_plot = 0.01
        elif type_abu == 'ISOR':
            delta_tmp = abu_fin + 5
            wl_syn1, syn1 = autok(param1, delta_tmp, window)
            syn1 = convolve_gauss2(wl_syn1, syn1, int(param1['res']))

            delta_tmp = max(abu_fin - 5, 1)
            wl_syn_tmp, syn2 = autok(param1, delta_tmp, window)
            syn2 = convolve_gauss2(wl_syn_tmp, syn2, int(param1['res']))

            wl_syn = wl_syn_tmp.value
            around_best = [(max(abu_fin - 5, 1), wl_syn_tmp, syn2), (abu_fin, wl_best, fl_best), (abu_fin + 5, wl_syn1, syn1)]
            step_plot = 1

        # FIX (plot only): the chi2 panel must match the synthetics shown at +-0.2 dex (ISOR: +-5).
        # The chi2 is recomputed from these three K2 synthetics, interpolated in abundance as in the fit;
        # the abundance found (abu_fin) does not change. Before, the panel kept the range of the 3
        # synthetics of the fit, so the minimum could appear at its edge.
        levels = {}
        for a, w, f in around_best:     # distinct abundances (ISOR: the lower one cannot go below 1)
            levels.setdefault(round(float(a), 6), (w, f))
        abu_levels = np.array(sorted(levels))
        if len(abu_levels) >= 2:
            ss_plot = np.array([np.interp(fitx, np.asarray(getattr(levels[a][0], 'value', levels[a][0])),
                                          np.asarray(getattr(levels[a][1], 'value', levels[a][1]))) for a in abu_levels])
            abu3_plot = np.round(np.arange(abu_levels[0], abu_levels[-1] + step_plot / 2, step_plot), 2)
            grid_plot = interp_in_abundance(abu3_plot, abu_levels, ss_plot)
            chis_plot = np.sum((fity - grid_plot) ** 2 / grid_plot, axis=1)


    ###
    wl_syn2 = wl_best.value if do_best else wl_syn2
    fake = fl_best.value if do_best else fake

    plot_data = {"wl_syn": wl_syn, "fl_syn1": syn1, "fl_syn2": syn2, "xfit": wl_syn2, "bestfit": fake,
    "xspec": norm_dop2.spectral_axis, "yspec": norm_dop2.flux,
    "fit1": fit1, "fit2": fit2, "line": line, "abund": abu3_plot, "chi": chis_plot,
    "abu_fin": abu_fin, "elem": elem, "wv_left": wv_left, "wv_right": wv_right,
    "window": window, "type_norm": cfg.type_norm, "type_abu": type_abu}

    if fit11 is not None:
        plot_data['fit1_new'] = fit11
        plot_data['fit2_new'] = fit22

    return abu_fin, EW, SNR, chis, param_mc, plot_data

def mc_seed(param1):
    """Reproducible seed for the Monte Carlo of one line: the same star, line, element and keyword
    always give the same noise, independently of the order of the lines. crc32 (not hash()) is
    stable between Python sessions."""
    key = f"{param1['name_star']}|{float(param1['line']):.3f}|{float(param1['nelem']):.2f}|{param1['tana']}"
    return zlib.crc32(key.encode())


def abu_mc_line(param_mc, snr, abu, seed=None):
    """Monte Carlo error of the abundance of a line: the best-fit profile plus noise of the measured
    SNR is fitted again many times; the error is the dispersion of the results (5% clipped on
    each side). seed makes it reproducible (see mc_seed)."""
    print('---')
    print('calculating the MC \n')
    # FIX: an undefined SNR gave NaN noise, all chi2 NaN and an error of exactly 0
    if not np.isfinite(snr) or snr <= 0:
        print(f'Warning: SNR not defined near the line (SNR = {snr}): MC error set to 99.99')
        return 99.99
    # Unpack parameters
    wl_syn2, syn_br2, fitx, fity, abu3 = param_mc

    # ss shape: (N_abu_coarse, N_wl_syn2)
    ss = syn_br2

    #
    interpolator_wave = interp1d(wl_syn2, ss, kind='linear', axis=1, fill_value="extrapolate")
    ss_interp_wave = interpolator_wave(fitx) # Shape: (N_abu_coarse, N_pixels_fitx)

    # Pre-interpolation abundance
    interpolator_abu = interp1d(abu, ss_interp_wave, kind='linear', axis=0, fill_value="extrapolate")
    synthetic_grid = interpolator_abu(abu3)
    # synthetic_grid Shape finale: (N_abu_fine, N_pixels_fitx)

    # creation noise
    n_mc = 200
    n_pixels = len(fity)

    # Reproducible noise (seed from mc_seed); seed=None gives a random one as before
    noise_matrix = np.random.default_rng(seed).normal(0, 1/snr, (n_mc, n_pixels))

    # Matrix of the simulated observed fluxes
    flux_mc_matrix = fity + noise_matrix

    # Chi-square
    observed_broad = flux_mc_matrix[:, np.newaxis, :]
    synthetic_broad = synthetic_grid[np.newaxis, :, :]

    # Weighted squared residuals (weight = model, as in the abundance fit)
    # Note: to avoid divisions by zero or instabilities, the weight is 1 where the model is <= 0
    # FIX: np.where makes a new array. Before, 'weights' was the same array as the model,
    # so setting weights <= 0 to 1 also changed the model in the numerator.
    weights = np.where(synthetic_broad <= 0, 1.0, synthetic_broad) # Safety check

    chi_squares = np.sum(((observed_broad - synthetic_broad)**2) / weights, axis=2)
    # chi_squares Shape: (N_mc, N_abu_fine)

    # Find the minimum at each iteration
    min_indices = np.argmin(chi_squares, axis=1)
    abu_snr = abu3[min_indices]

    # final error
    abu_snr = np.sort(abu_snr)[10:-10] # Remove the outliers (clipping)
    abu_err_mc = np.std(abu_snr)
    return abu_err_mc


