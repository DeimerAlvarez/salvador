"""Selection of the lines by the sensitivity of their profile to the abundance (SIGMA_FIT_MAX, flag
-5, column SIGFIT), the check of the saturated Fe I lines, and the SNR from the continuum windows
(DER_SNR)."""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import EXPECTED_SUBSET, prepare_subset, run_salvador, set_keyword
from salvador.cli import print_saturation_check
from salvador.config import cfg, Config
from salvador.fitting import MIN_DER_PIXELS, MISSING_SIGMA, _estimate_snr, der_snr, profile_sigma
from salvador.summary import (FLAG_EW, FLAG_SIGMA, FLAG_SLOPE, MISSING, ew_selection_flag, saturation_check,
                              sigma_available)

STAR = 'M54_HR18_1500138'
ENV_MPL = {**os.environ, 'MPLBACKEND': 'Agg'}


def rows(path):
    return [r.split() for r in path.read_text().splitlines() if r.strip() and not r.startswith('#')]


# ------------------------------------------------------------------ profile_sigma
def test_profile_sigma_of_a_uniform_change():
    # 25 pixels, F(A + 0.2) - F(A - 0.2) = -0.04: dF/dA = -0.1 per dex, SNR 100
    sigma = profile_sigma(np.full(25, -0.04), 0.2, 100.0)
    assert sigma == pytest.approx(0.01 / (0.1 * np.sqrt(25)))


def test_profile_sigma_scales_with_the_noise():
    dflux = -0.05 * np.exp(-np.linspace(-2, 2, 21) ** 2)
    assert profile_sigma(dflux, 0.2, 50.0) == pytest.approx(2 * profile_sigma(dflux, 0.2, 100.0))


@pytest.mark.parametrize('dflux, snr', [(np.full(5, -0.01), np.nan), (np.full(5, -0.01), 0.0), (np.zeros(5), 100.0)])
def test_profile_sigma_not_available(dflux, snr):
    assert profile_sigma(dflux, 0.2, snr) == MISSING_SIGMA and not sigma_available(MISSING_SIGMA)


# ------------------------------------------------------------------ selection
@pytest.mark.parametrize('ew, slope, sigma, slope_min, sigma_max, flag', [
    (60.0, 0.30, 0.03, None, 0.20, 0),             # saturated but sensitive: kept (default)
    (60.0, 0.30, 0.30, None, 0.20, FLAG_SIGMA),    # profile hardly changes: rejected
    (60.0, 0.30, 0.03, 0.35, 0.20, FLAG_SLOPE),    # COG_SLOPE_MIN given: rejected by the slope
    (60.0, 0.30, 0.30, 0.35, 0.20, FLAG_SIGMA),    # both: the profile first
    (1000.0, 0.50, 0.30, None, 0.20, FLAG_EW),     # EWR > -4.4: the reduced EW first
    (60.0, 0.30, MISSING_SIGMA, None, 0.20, 0),    # no profile error: not judged by it
    (60.0, 0.30, 0.30, None, None, 0),             # SIGMA_FIT_MAX = 0: not used
    (MISSING, 99.999, 0.30, None, 0.20, FLAG_SIGMA),   # no EW: judged by the profile only
])
def test_selection_flag(ew, slope, sigma, slope_min, sigma_max, flag):
    assert ew_selection_flag(ew, 6000.0, slope, -6.0, -4.4, slope_min, sigma, sigma_max) == flag


# ------------------------------------------------------------------ saturated Fe I lines
def test_saturation_check():
    abu = np.array([6.15, 5.89, 5.96, 5.85, 5.90, 5.84, 5.92, 99.99])
    slope = np.array([0.60, 0.61, 0.46, 0.58, 0.20, 0.27, 0.27, 0.10])
    n_sat, n_other, diff, err = saturation_check(abu, slope)
    assert (n_sat, n_other) == (3, 4)                                   # 99.99: not used
    assert diff == pytest.approx(np.mean([5.90, 5.84, 5.92]) - np.mean([6.15, 5.89, 5.96, 5.85]))
    expected_err = np.sqrt(np.var([5.90, 5.84, 5.92], ddof=1) / 3 + np.var([6.15, 5.89, 5.96, 5.85], ddof=1) / 4)
    assert err == pytest.approx(expected_err)
    assert saturation_check(abu[:5], slope[:5]) is None                 # one saturated line only


def test_print_saturation_check(capsys):
    nelem = np.array([26.00] * 6 + [28.00])
    abu = np.array([6.15, 6.10, 6.12, 5.80, 5.78, 5.82, 4.50])
    slope = np.array([0.60, 0.61, 0.58, 0.20, 0.27, 0.27, 0.10])
    flags = np.array([0, 0, 0, FLAG_SLOPE, 0, -2, 0])                    # -2 (EP) is not used
    print_saturation_check('s', nelem, abu, slope, flags)
    out = capsys.readouterr().out
    assert 'Fe I lines with COG slope < 0.35 (2) minus the others (3): -0.33 +- 0.02 dex' in out
    assert 'check the microturbulence' in out


# ------------------------------------------------------------------ SNR
def test_snr_not_lowered_by_a_bad_pixel_in_the_windows():
    # FIX: one negative pixel in a continuum window gave SNR ~ 20 instead of ~ 100 (NORM = 2)
    rng = np.random.default_rng(1)
    wave = np.arange(7840.0, 7880.0, 0.05)
    flux = 1 + rng.normal(0, 0.01, wave.size)
    flux[np.argmin(np.abs(wave - 7865.0))] = -0.1
    cfg.use(Config(input_stars='', type_norm=2, window=5, inter='n', sclip=3))
    assert _estimate_snr(wave, flux, 7852.7, [7850.0, 7860.0], [7852.0, 7870.0]) == pytest.approx(100, rel=0.15)


@pytest.mark.parametrize('noise', [0.01, 0.002])
def test_der_snr_of_white_noise(noise):
    flux = 1 + np.random.default_rng(2).normal(0, noise, 4000)
    assert der_snr([flux]) == pytest.approx(1 / noise, rel=0.05)


def test_der_snr_ignores_a_linear_continuum():
    rng = np.random.default_rng(3)
    flux = np.linspace(0.95, 1.05, 2000) + rng.normal(0, 0.01, 2000)
    assert der_snr([flux]) == pytest.approx(100, rel=0.06)


def test_der_snr_does_not_mix_the_windows():
    # two windows with different continuum levels: a step between them is not noise
    rng = np.random.default_rng(4)
    a, b = 1 + rng.normal(0, 0.01, 200), 0.9 + rng.normal(0, 0.01, 200)
    assert der_snr([a, b]) == pytest.approx(95, rel=0.1)
    assert np.isnan(der_snr([a[:4], b[:4]]))                    # segments < 5 pixels: no differences
    assert np.isnan(der_snr([a[:MIN_DER_PIXELS + 3]]))          # MIN_DER_PIXELS - 1 differences


def test_snr_not_lowered_by_a_structure_in_a_window():
    # FIX (PEPSI 1r, 8793.342): the window 8779.7-8789.8 holds a smooth structure (flux 0.70-1.15)
    # and a spike, and the SNR was 9.5 instead of ~85
    rng = np.random.default_rng(5)
    wave = np.arange(8770.0, 8810.0, 0.0786)
    flux = (1 - 0.3 * np.exp(-0.5 * ((wave - 8781.5) / 1.2) ** 2) + 0.13 * np.exp(-0.5 * ((wave - 8787.0) / 1.5) ** 2)
            + rng.normal(0, 0.012, wave.size))
    flux[np.argmin(np.abs(wave - 8780.1))] = 1.42
    cfg.use(Config(input_stars='', type_norm=2, window=5, inter='n', sclip=3))
    assert _estimate_snr(wave, flux, 8793.3, [8779.69], [8789.76]) == pytest.approx(85, rel=0.3)   # 128 pixels: +-12%
