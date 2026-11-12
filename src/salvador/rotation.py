"""Projected rotational velocity (and abundance) of a line: analysis_rotat, and the broadening
kernels it uses (instrumental, rotational, radial-tangential macroturbulence).

The synthetic spectra of K2 (SYNTHE) already contain the thermal and microturbulent broadening
(Doppler width with the microturbulence of lista.stars) and the damping wings. The other
broadening mechanisms act on the disk-integrated spectrum and are applied here as convolutions:
instrumental profile (Gaussian, resolution of the linelist), macroturbulence (radial-tangential,
keyword ROT_VMAC) and rotation (keywords ROT_LD for the limb darkening). Since 1.3.0 they are
computed on a fine grid uniform in velocity, for every value of v sin i of the grid (no
interpolation in v sin i), and the model is compared with the observed pixels.
"""
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erfc

from .config import cfg
from .elements import define_abu0
from .fitting import interp_in_abundance, mc_seed
from .k2_interface import autok

C_KMS = 299792.458
FINE_DV = 0.25          # km/s: step of the velocity grid on which the broadening is computed
_OVERSAMPLE = 21        # sub-samples per pixel to integrate the kernels over the pixels
_FWHM_GAUSS = 2.0 * np.sqrt(2.0 * np.log(2.0))



# ------------------------------------------------------------------------------- kernels
def rotation_kernel(v: np.ndarray, vsini: float, eps: float) -> np.ndarray:
    """Rotational broadening profile with linear limb darkening (Gray, The Observation and Analysis
    of Stellar Photospheres, eq. 18.14), normalized to 1.

    Args:
        v: Velocities [km/s].
        vsini: Projected rotational velocity [km/s], > 0.
        eps: Linear limb-darkening coefficient (0-1).
    """
    x = np.asarray(v, dtype=float) / vsini
    out = np.zeros_like(x)
    inside = np.abs(x) < 1.0
    one_x2 = 1.0 - x[inside] ** 2
    out[inside] = (2.0 * (1.0 - eps) * np.sqrt(one_x2) + 0.5 * np.pi * eps * one_x2) / (np.pi * vsini * (1.0 - eps / 3.0))
    return out


def macroturbulence_kernel(v: np.ndarray, zeta: float) -> np.ndarray:
    """Radial-tangential macroturbulence profile (Gray 2005), with equal radial and tangential
    parts and no limb darkening, integrated over the disk:
    M(v) = 2 / (sqrt(pi) zeta) [exp(-u^2) - sqrt(pi) u erfc(u)], u = |v| / zeta. Normalized to 1.

    Args:
        v: Velocities [km/s].
        zeta: Macroturbulent velocity [km/s], > 0.
    """
    u = np.abs(np.asarray(v, dtype=float)) / zeta
    return 2.0 / (np.sqrt(np.pi) * zeta) * (np.exp(-u ** 2) - np.sqrt(np.pi) * u * erfc(u))


def gaussian_kernel(v: np.ndarray, fwhm: float) -> np.ndarray:
    """Gaussian instrumental profile with the given FWHM [km/s], normalized to 1."""
    sigma = fwhm / _FWHM_GAUSS
    return np.exp(-0.5 * (np.asarray(v, dtype=float) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def pixel_kernel(dv: float, half_width: float, profile) -> np.ndarray:
    """A profile averaged over pixels of width dv (km/s), from -half_width to +half_width,
    normalized so that its sum is 1. Averaging over the pixel keeps the kernel correct also when
    it is narrower than a few pixels (before, it was sampled at the pixel centres)."""
    n = int(np.ceil(half_width / dv)) + 1
    centres = np.arange(-n, n + 1) * dv
    sub = (np.arange(_OVERSAMPLE) + 0.5) / _OVERSAMPLE - 0.5
    kernel = profile(centres[:, None] + sub[None, :] * dv).mean(axis=1)
    return kernel / kernel.sum()


def convolve_rows(spectra: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve each row of spectra (n, npix) with kernel; the edges are extended with the first
    and last values of each row, so that the output has the same size."""
    spectra = np.atleast_2d(spectra)
    if kernel.size == 1:
        return spectra.copy()
    pad = kernel.size // 2
    ext = np.concatenate([np.repeat(spectra[:, :1], pad, axis=1), spectra, np.repeat(spectra[:, -1:], pad, axis=1)], axis=1)
    return fftconvolve(ext, kernel[None, :], mode='same', axes=1)[:, pad:-pad]


def velocity_grid(wave: np.ndarray, flux: np.ndarray, dv: float = FINE_DV) -> tuple[np.ndarray, np.ndarray]:
    """Resample a spectrum on a grid uniform in velocity (log wavelength) with step dv [km/s]."""
    log_w = np.log(np.asarray(wave, dtype=float))
    grid = np.arange(log_w[0], log_w[-1], np.log1p(dv / C_KMS))
    return np.exp(grid), np.interp(grid, log_w, np.asarray(flux, dtype=float))


# ------------------------------------------------------------------------------- macroturbulence
# NEW (1.3.0): relations for the radial-tangential macroturbulence, selected with ROT_VMAC. At
# R <= 40000 v sin i and vmac cannot be separated (the fit measures the total broadening), so for
# v sin i below ~10-15 km/s the result depends on the vmac assumed: use the relation calibrated on
# stars like the target. The calibration ranges are only used to print a warning.
def vmac_valenti_fischer(teff: float, logg: float = 4.44) -> float:
    """Valenti & Fischer (2005), eq. (1) with the sign of their Fig. 3 and text (vmac increases with
    Teff, 2.8 km/s at 5000 K; the equation is printed with a minus sign): 3.98 + (Teff - 5770)/650."""
    return max(3.98 + (teff - 5770.0) / 650.0, 0.0)


def vmac_doyle(teff: float, logg: float) -> float:
    """Doyle et al. (2014), eq. (8): 3.21 + 2.33e-3 dT + 2.00e-6 dT^2 - 2.00 (logg - 4.44),
    dT = Teff - 5777 K."""
    dt = teff - 5777.0
    return max(3.21 + 2.33e-3 * dt + 2.00e-6 * dt ** 2 - 2.00 * (logg - 4.44), 0.0)


def vmac_brewer(teff: float, logg: float) -> float:
    """Brewer et al. (2016), eq. (2): dwarfs (logg >= 4) 2.202 exp(0.0019 dT) + 1.30, subgiants
    (3 <= logg < 4) 1.166 exp(0.0028 dT) + 3.30, giants (logg < 3) 4.0 km/s; dT = Teff - 5777 K."""
    dt = teff - 5777.0
    if logg >= 4.0:
        return 2.202 * np.exp(0.0019 * dt) + 1.30
    if logg >= 3.0:
        return 1.166 * np.exp(0.0028 * dt) + 3.30
    return 4.0


def vmac_carney(teff: float, logg: float) -> float:
    """Carney et al. (2008), eq. (5), for red giants: -0.80 logg + 7.33 km/s."""
    return max(-0.80 * logg + 7.33, 0.0)


@dataclass(frozen=True)
class VmacRelation:
    """A macroturbulence relation vmac(Teff, logg) with its calibration range and scatter."""
    function: Callable[[float, float], float]
    teff_range: tuple[float, float]
    logg_range: tuple[float, float]
    scatter: float | None           # km/s, as quoted in the paper (None: not given)
    reference: str


VMAC_RELATIONS = {
    'VF05': VmacRelation(vmac_valenti_fischer, (4700.0, 6600.0), (3.5, 5.0), None,
                         'Valenti & Fischer 2005, dwarfs and subgiants'),
    'D14': VmacRelation(vmac_doyle, (5200.0, 6400.0), (4.0, 4.6), 0.73, 'Doyle et al. 2014, dwarfs'),
    'B16': VmacRelation(vmac_brewer, (4700.0, 6800.0), (3.0, 5.0), None,
                        'Brewer et al. 2016, dwarfs and subgiants (4 km/s for logg < 3)'),
    # C08: fit to disk giants of class II-III and III (Teff 4120-5000 K, logg 1.75-3.31, their Table 3);
    # their metal-poor RGB stars (Teff 3850-4410 K, logg 0.20-1.10, Table 2) follow the same relation.
    # Not valid for red horizontal branch stars (their Sect. 7.1).
    'C08': VmacRelation(vmac_carney, (3850.0, 5000.0), (0.2, 3.3), 0.8,
                        'Carney et al. 2008, red giants, not RHB stars'),
}


def star_macroturbulence(teff: float, logg: float, setting: str | float | None = None,
                         error: float | None = None, warn: bool = True) -> tuple[float, float]:
    """Macroturbulence [km/s] of a star and its uncertainty.

    Args:
        teff, logg: Parameters of the star.
        setting: A value in km/s or the name of a relation of VMAC_RELATIONS; None: the keywords
            ROT_VMAC and ROT_VMAC_ERR of input.salvador.
        error: Uncertainty of vmac [km/s]; None: the scatter of the relation (0 when unknown).
        warn: Print a warning outside the calibration of the relation.

    Returns:
        (vmac, error) in km/s.

    Raises:
        ValueError: ROT_VMAC is not set in input.salvador.
    """
    if setting is None:
        setting, error = cfg.rot_vmac, cfg.rot_vmac_err
        if setting is None:
            raise ValueError('ROT_VMAC is not set in input.salvador')
    if isinstance(setting, str):
        relation = VMAC_RELATIONS[setting.upper()]
        (t1, t2), (g1, g2) = relation.teff_range, relation.logg_range
        if warn and not (t1 <= teff <= t2 and g1 <= logg <= g2):
            print(f'WARNING: ROT_VMAC = {setting.upper()} ({relation.reference}) is calibrated on Teff = {t1:.0f}-{t2:.0f} K, '
                  f'logg = {g1:.1f}-{g2:.1f} (Teff = {teff:.0f} K, logg = {logg:.2f} here): check the macroturbulence, '
                  f'it sets v sin i below ~10-15 km/s')
        vmac = relation.function(teff, logg)
        if error is None:
            error = relation.scatter or 0.0
        return float(vmac), float(error)
    return float(setting), float(error or 0.0)


def check_rotat_settings(teff: float, logg: float) -> tuple[float, float]:
    """For a star with ROTAT lines: stop if ROT_VMAC is not set, print the macroturbulence used
    (and the warning outside the calibration of the relation), and a note when its error is 0
    because ROT_VMAC_ERR is not set and the relation gives no scatter.

    Returns:
        (vmac, error) in km/s.
    """
    if cfg.rot_vmac is None:
        sys.exit('Error: the linelist has ROTAT lines: set the macroturbulence with ROT_VMAC in input.salvador '
                 f"(a value in km/s, or one of {', '.join(VMAC_RELATIONS)}; optional: ROT_VMAC_ERR in km/s)")
    vmac, error = star_macroturbulence(teff, logg)
    source = f'{cfg.rot_vmac}, {VMAC_RELATIONS[cfg.rot_vmac].reference}' if isinstance(cfg.rot_vmac, str) else 'ROT_VMAC'
    print(f'ROTAT: vmac = {vmac:.1f} ± {error:.1f} km/s ({source})')
    if error == 0 and cfg.rot_vmac_err is None:
        print('Note: the macroturbulence has no error (ROT_VMAC_ERR not set and no scatter for this choice), so the '
              'error of v sin i does not include it. v sin i below ~10-15 km/s depends on vmac: set ROT_VMAC_ERR '
              '(km/s) to propagate it (column eROTv of star_rot.fit)')
    return vmac, error


# ------------------------------------------------------------------------------- analysis
def analysis_rotat(nelem, alpha, met, delta, elem, wvl, flux, line, window, param1, resol, fit1, fit2, **kwargs):
    """Projected rotational velocity and abundance of a line.

    For each abundance of a coarse grid (steps of 0.1 dex, ROT_ABU_SPAN), the synthetic spectrum
    of K2 is broadened by the instrumental profile and by the macroturbulence (ROT_VMAC), and then
    by the rotation for every v sin i from ROT_MIN to ROT_MAX in steps of ROT_STEP (1 km/s by
    default). The chi2 with the observed pixels between fit1 and fit2 is computed on a fine grid of
    abundances (0.01 dex, linear interpolation in abundance) and v sin i; its minimum gives the
    result. Errors: Monte Carlo (ROT_NMC) with the dispersion of the residuals as noise. If v sin i
    is smaller than its error or than ROT_UL_FWHM times the instrumental FWHM, the result is an upper
    limit; at ROT_MAX, a lower limit.

    Optional keyword arguments (from the linelist): abu_init, abu_span, rot_min, rot_max; and
    rot_step, n_mc, vmac, vmac_err, ld, ul_fwhm to override the settings of input.salvador.

    NEW: if the macroturbulence has an uncertainty (ROT_VMAC_ERR, or the scatter of the relation),
    the fit is repeated with vmac -/+ its error; err_rot_vmac is half the difference of the two
    v sin i (systematic error, added in quadrature to the Monte Carlo error for the limits).

    Returns:
        Dictionary with the result (best_rot_fine, err_rot, err_rot_vmac, best_abu_fine, err_abu, vmac, ld,
        limit: 'D' measurement, 'UL' upper limit, 'LL' lower limit; rot_value: the value to report)
        and the data for the plot.
    """
    abu_model = define_abu0(int(nelem), alpha, met, 0.0, elem)
    abu0 = kwargs.get('abu_init')
    if abu0 is None:
        abu0 = abu_model + delta
    abu_span = kwargs.get('abu_span', cfg.rot_abu_span)
    rot_min = max(float(kwargs.get('rot_min', cfg.rot_min)), 0.0)
    rot_max = float(kwargs.get('rot_max', cfg.rot_max))
    rot_step = float(kwargs.get('rot_step', cfg.rot_step))
    n_mc = int(kwargs.get('n_mc', cfg.rot_nmc))
    eps = float(kwargs.get('ld', cfg.rot_ld))
    if 'vmac' in kwargs:
        vmac, vmac_err = float(kwargs['vmac']), 0.0
    else:       # ROT_VMAC (the warnings were printed by check_rotat_settings)
        vmac, vmac_err = star_macroturbulence(float(param1['teff']), float(param1['logg']), warn=False)
    vmac_err = float(kwargs.get('vmac_err', vmac_err))

    abundances = np.array([abu0]) if abu_span == 0 else np.round(np.arange(abu0 - abu_span, abu0 + abu_span + 0.01, 0.1), 2)
    rotations = np.round(np.arange(rot_min, rot_max + 0.5 * rot_step, rot_step), 3)

    # observed pixels: the region of the plot (window) and the region of the fit (fit1-fit2)
    wvl, flux = np.asarray(wvl, dtype=float), np.asarray(flux, dtype=float)
    in_window = (wvl > line - window / 2) & (wvl < line + window / 2)
    w_obs, f_obs = wvl[in_window], flux[in_window]
    in_fit = (w_obs >= fit1) & (w_obs <= fit2)
    w_fit, obs_fit = w_obs[in_fit], f_obs[in_fit]

    # synthetic spectra (one K2 call per abundance) on a fine velocity grid, broadened by the
    # instrument and by the macroturbulence (independent of v sin i: done once)
    raw = []
    for abu_val in abundances:
        wl_k2, fl_k2 = autok(param1, np.round(abu_val - abu_model, 2), window)
        w_v, f_v = velocity_grid(getattr(wl_k2, 'value', wl_k2), getattr(fl_k2, 'value', fl_k2))
        raw.append(f_v)
    raw = np.array(raw)
    fwhm_instr = C_KMS / float(resol)
    raw = convolve_rows(raw, pixel_kernel(FINE_DV, 4.0 * fwhm_instr, lambda v: gaussian_kernel(v, fwhm_instr)))

    def macro(spectra: np.ndarray, zeta: float) -> np.ndarray:
        """Radial-tangential macroturbulence (nothing for zeta = 0)."""
        if zeta <= 0:
            return spectra
        return convolve_rows(spectra, pixel_kernel(FINE_DV, 6.0 * zeta, lambda v: macroturbulence_kernel(v, zeta)))

    base = macro(raw, vmac)

    def rotate(spectra: np.ndarray, vsini: float) -> np.ndarray:
        """Rotational broadening (exact for each v sin i: no interpolation between values)."""
        if vsini <= 0:
            return spectra
        return convolve_rows(spectra, pixel_kernel(FINE_DV, vsini, lambda v: rotation_kernel(v, vsini, eps)))

    def models_at_pixels(spectra: np.ndarray) -> np.ndarray:
        """Models at the observed pixels of the fit region: (abundance, v sin i, pixel)."""
        return np.array([[np.interp(w_fit, w_v, row) for row in rotate(spectra, vsini)] for vsini in rotations]).transpose(1, 0, 2)

    def chi2(observed: np.ndarray, model: np.ndarray) -> np.ndarray:
        """chi2 as for the abundances: sum of (observed - model)^2 / model over the pixels."""
        return np.sum((observed - model) ** 2 / np.clip(model, 1e-9, None), axis=-1)

    models = models_at_pixels(base)

    # fine grid of abundances around the best coarse abundance (+-0.20 dex, 0.01 dex)
    best_coarse = np.unravel_index(np.nanargmin(chi2(obs_fit, models)), (len(abundances), len(rotations)))
    if len(abundances) == 1:
        finer_abundances = abundances
        fine_models = models
    else:
        finer_abundances = np.round(np.arange(max(abundances[0], abundances[best_coarse[0]] - 0.20),
                                              min(abundances[-1], abundances[best_coarse[0]] + 0.205), 0.01), 2)
        fine_models = np.array([interp_in_abundance(finer_abundances, abundances, models[:, j, :])
                                for j in range(len(rotations))]).transpose(1, 0, 2)
    chi2_fine = chi2(obs_fit, fine_models)
    i_best, j_best = np.unravel_index(np.nanargmin(chi2_fine), chi2_fine.shape)
    best_abu_fine, best_rot_fine = float(finer_abundances[i_best]), float(rotations[j_best])

    # Monte Carlo: best fit + noise with the dispersion of the residuals
    best_model_fit = fine_models[i_best, j_best]
    sigma_noise = np.std(obs_fit - best_model_fit)
    rng = np.random.default_rng(mc_seed(param1))
    mc_abu, mc_rot = np.zeros(n_mc), np.zeros(n_mc)
    for k in range(n_mc):
        mock = best_model_fit + rng.normal(0.0, sigma_noise, size=best_model_fit.size)
        i_k, j_k = np.unravel_index(np.nanargmin(chi2(mock, fine_models)), chi2_fine.shape)
        mc_abu[k], mc_rot[k] = finer_abundances[i_k], rotations[j_k]
    err_abu, err_rot = (float(np.std(mc_abu)), float(np.std(mc_rot))) if n_mc > 1 else (0.0, 0.0)

    # NEW: systematic error due to the macroturbulence: v sin i of the best fits with vmac -/+ error
    # (abundance free on the coarse grid). v sin i and vmac are degenerate at these resolutions.
    err_rot_vmac = 0.0
    if vmac_err > 0:
        rot_vmac = []
        for zeta in (max(vmac - vmac_err, 0.0), vmac + vmac_err):
            chi2_z = chi2(obs_fit, models_at_pixels(macro(raw, zeta)))
            rot_vmac.append(float(rotations[np.unravel_index(np.nanargmin(chi2_z), chi2_z.shape)[1]]))
        err_rot_vmac = abs(rot_vmac[0] - rot_vmac[1]) / 2.0
    err_total = float(np.hypot(err_rot, err_rot_vmac))

    # NEW: limits. A lower limit when the minimum is at the largest v sin i of the grid. An upper limit
    # when v sin i is smaller than its error (as in Rain et al. 2026: v sin i + error) or than
    # ROT_UL_FWHM x the instrumental FWHM: in the simulations (R = 20000-40000, S/N = 50-100) smaller
    # values could not be distinguished from no rotation, and the Monte Carlo error underestimates the
    # uncertainty near v sin i = 0. The upper limit is the largest of v sin i + error and that value.
    resolution_limit = float(kwargs.get('ul_fwhm', cfg.rot_ul_fwhm)) * fwhm_instr
    if best_rot_fine >= rotations[-1] and rotations[-1] > rotations[0]:
        limit, rot_value = 'LL', best_rot_fine
    elif best_rot_fine < err_total or best_rot_fine < resolution_limit or (best_rot_fine <= rotations[0] and rotations[0] == 0):
        limit, rot_value = 'UL', max(best_rot_fine + err_total, resolution_limit)
    else:
        limit, rot_value = 'D', best_rot_fine
    # NEW: the fit region must contain the whole broadened line (simulations: with v sin i larger than
    # the half-width of the region the result can be wrong by tens of km/s)
    half_width = min(line - fit1, fit2 - line) / line * C_KMS
    if best_rot_fine > half_width:
        print(f'WARNING: line {line:.3f}: v sin i = {best_rot_fine:.0f} km/s is larger than the half-width of the fit '
              f'region ({half_width:.0f} km/s): the broadened line is not all inside {fit1}-{fit2}; widen the edges '
              f'(fifth and sixth columns of the linelist)')
    if best_rot_fine == 0:
        print(f'Note: line {line:.3f}: best fit at v sin i = 0. If this happens for most of the lines of a star, the '
              f'synthetic spectra may already be broader than the observed lines: check the resolution of the linelist '
              f'(R = {float(resol):.0f}) and the macroturbulence (vmac = {vmac:.1f} km/s)')
    limit_txt = {'D': '', 'UL': ' (upper limit: ', 'LL': ' (lower limit: '}[limit]
    limit_txt += '' if limit == 'D' else f'{np.floor(rot_value + 0.5):.0f} km/s)'
    print(f"Final Best Fit -> Rot: {best_rot_fine:.0f} ± {np.floor(err_rot + 0.5):.0f} (MC) ± {err_rot_vmac:.1f} (vmac) km/s"
          f"{limit_txt} (vmac = {vmac:.1f} ± {vmac_err:.1f} km/s, LD = {eps:.2f}) --- Abu: {best_abu_fine:.2f} ± {err_abu:.2f}")

    # spectra for the plot (whole window): best fit and +-10 km/s, from the same models as the fit
    best_base = interp_in_abundance(np.array([best_abu_fine]), abundances, base) if len(abundances) > 1 else base
    rot_minus, rot_plus = max(0.0, best_rot_fine - 10.0), best_rot_fine + 10.0
    fl_best, fl_minus, fl_plus = (rotate(best_base, v)[0] for v in (best_rot_fine, rot_minus, rot_plus))

    return {
        "wl_syn": w_v, "fl_syn1": fl_plus, "fl_syn2": fl_minus, "xfit": w_v, "bestfit": fl_best,
        "xspec": w_obs, "yspec": f_obs, "line": line, "finer_rotations": rotations,
        "min_chi2_fine": chi2_fine, "finer_abundances": finer_abundances, "elem": elem, "window": window,
        "best_rot_fine": best_rot_fine, "best_abu_fine": best_abu_fine, "fit1": fit1, "fit2": fit2,
        "err_abu": err_abu, "err_rot": err_rot, "err_rot_vmac": err_rot_vmac, "vmac": vmac, "vmac_err": vmac_err,
        "ld": eps, "limit": limit, "rot_value": rot_value,
    }
