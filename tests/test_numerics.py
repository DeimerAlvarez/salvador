"""Numerical functions: interpolation in abundance, SNR, Monte Carlo, normalization, final means."""
import warnings

import numpy as np
import pytest

from salvador.fitting import _estimate_snr, abu_mc_line, gaussian, gaussian_cf, interp_in_abundance, mc_seed
from salvador.normalization import find_clipped_regions, load_cw_regions, merge_overlapping_regions, norm2
from salvador.summary import extract_abu, extract_abu_fe, extract_abu_isor, get_coeffs

RNG = np.random.default_rng(42)


# ------------------------------------------------------------------------- fitting
def test_interp_in_abundance_equals_np_interp_per_pixel():
    abu = np.array([6.8, 7.0, 7.2])
    spectra = RNG.uniform(0.5, 1.0, (3, 40))
    new = np.array([6.7, 6.8, 6.93, 7.0, 7.11, 7.2, 7.4])         # also outside: clamped as np.interp
    expected = np.array([[np.interp(a, abu, spectra[:, k]) for k in range(40)] for a in new])
    assert np.allclose(interp_in_abundance(new, abu, spectra), expected)


def test_gaussians_agree():
    x = np.linspace(-3, 3, 11)
    assert np.allclose(gaussian(x, (-0.3, 0.1, 0.5, 1.0)), gaussian_cf(x, -0.3, 0.1, 0.5, 1.0))


def test_mc_seed_is_stable():
    p = {'name_star': 's1', 'line': 7461.519, 'nelem': 26.0, 'tana': 'ABU'}
    assert mc_seed(p) == mc_seed(dict(p))
    assert mc_seed(p) != mc_seed(dict(p, line=7461.52))


@pytest.fixture
def flat_spectrum():
    wave = np.arange(7400.0, 7900.0, 0.05)
    flux = 1 + RNG.normal(0, 0.01, wave.size)                     # SNR ~ 100
    return wave, flux


def test_snr_from_continuum_windows(settings, flat_spectrum):
    wave, flux = flat_spectrum
    left, right = np.array([7450.0, 7480.0, 7700.0]), np.array([7452.0, 7483.0, 7702.0])
    near = (wave > 7450) & (wave < 7490) & (((wave > 7450) & (wave < 7452)) | ((wave > 7480) & (wave < 7483)))
    assert _estimate_snr(wave, flux, 7470.0, left, right) == pytest.approx(np.mean(flux[near]) / np.std(flux[near]))


def test_snr_without_windows_near_the_line(settings, flat_spectrum):
    # before: NaN, hence a Monte Carlo error of exactly 0
    wave, flux = flat_spectrum
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        snr = _estimate_snr(wave, flux, 7600.0, [7450.0], [7452.0])
    assert 70 < snr < 130
    assert any('no continuum window' in str(w.message) for w in caught)


def test_snr_outside_the_spectrum(settings, flat_spectrum):
    assert np.isnan(_estimate_snr(*flat_spectrum, 9000.0, [], []))


def test_mc_error_with_undefined_snr(capsys):
    assert abu_mc_line(None, float('nan'), None) == 99.99
    assert 'MC error set to 99.99' in capsys.readouterr().out


def test_mc_error_is_reproducible_and_sensible():
    # synthetic lines whose depth grows with the abundance; "observed" = the line at 7.00
    x = np.linspace(-1, 1, 60)
    abu = np.array([6.8, 7.0, 7.2])
    syn = np.array([1 - d * np.exp(-x**2 / 0.05) for d in (0.2, 0.3, 0.4)])
    abu3 = np.round(np.arange(6.8, 7.2, 0.01), 2)
    param_mc = (x, syn, x, syn[1], abu3)
    err1 = abu_mc_line(param_mc, 50.0, abu, seed=1)
    assert err1 == abu_mc_line(param_mc, 50.0, abu, seed=1)
    assert 0 < err1 < 0.1
    assert abu_mc_line(param_mc, 200.0, abu, seed=1) < err1      # higher SNR, smaller error


# ------------------------------------------------------------------------- normalization
def test_merge_overlapping_regions():
    left, right = merge_overlapping_regions([5, 1, 2, 10], [6, 3, 4, 11])
    assert left.tolist() == [1, 5, 10] and right.tolist() == [4, 6, 11]
    assert merge_overlapping_regions([], [])[0].size == 0


def test_load_cw_regions(tmp_path):
    path = tmp_path / 'cw.txt'
    path.write_text('7500 7502\n7450 7452\n7500 7502\n7451 7455\n')
    left, right = load_cw_regions(path)
    assert left.tolist() == [7450, 7500] and right.tolist() == [7455, 7502]
    assert load_cw_regions(tmp_path / 'none.txt')[0].size == 0


def test_norm2_recovers_a_smooth_continuum():
    wave = np.linspace(7450, 7550, 2001)
    continuum = 1000 + 2 * (wave - 7450)
    left = np.arange(7452, 7550, 10.0)
    norm = norm2(left, left + 1, wave, continuum.copy())
    assert np.allclose(norm, 1, atol=1e-3)


def test_find_clipped_regions_removes_an_absorption_line():
    wave = np.linspace(7450, 7460, 1001)
    flux = 1 + RNG.normal(0, 0.002, wave.size)
    flux[480:520] -= 0.3                                          # a line inside the window
    left, right = find_clipped_regions([7450.5], [7459.5], wave, flux)
    assert len(left) == 2                                          # the window is split in two
    assert right[0] < wave[480] and left[1] > wave[519]


# ------------------------------------------------------------------------- final abundances
def test_fe_mean_of_one_line(settings):
    nelem, abu, mc = np.array([26.0]), np.array([6.0]), np.array([0.05])
    flags = np.zeros(1, dtype=int)
    nlin, feh, s1, flags, s_param, params = extract_abu_fe(nelem, abu, mc, 26.0, flags)
    assert nlin == 1 and feh == pytest.approx(6.0 - 7.52) and s1 == 0.05 and flags.tolist() == [1]
    assert params['abu_abs'] == pytest.approx(6.0)


def test_sigma_clipping_rejects_an_outlier(settings):
    abu = np.array([6.00, 6.02, 5.98, 6.01, 5.99, 6.80])
    nelem, mc = np.full(abu.size, 26.0), np.full(abu.size, 99.99)
    flags = np.zeros(abu.size, dtype=int)
    nlin, feh, s1, flags, *_ = extract_abu_fe(nelem, abu, mc, 26.0, flags)
    assert nlin == 5 and flags.tolist() == [1, 1, 1, 1, 1, 0]
    assert feh == pytest.approx(np.mean(abu[:5]) - 7.52)
    assert s1 == pytest.approx(np.std(abu[:5], ddof=1) / np.sqrt(5))


def test_rejected_lines_are_not_used(settings):
    nelem = np.array([26.0, 26.0, 26.0])
    abu = np.array([6.0, 99.99, 6.2])
    flags = np.array([0, -3, -1])                                   # only the first line is valid
    nlin, *_ = extract_abu_fe(nelem, abu, np.full(3, 0.03), 26.0, flags)
    assert nlin == 1 and flags.tolist() == [1, -3, -1]


def test_x_over_fe_and_errors(settings):
    nelem = np.array([22.0, 22.0])
    abu = np.array([3.80, 3.90])
    flags = np.zeros(2, dtype=int)
    nlin, xfe, sigma, flags, s_param, s_int, params = extract_abu(nelem, abu, np.full(2, 0.02), 22.0, -1.58, flags, 0.04)
    s_x = np.std(abu, ddof=1) / np.sqrt(2)
    assert nlin == 2
    assert xfe == pytest.approx(round(np.mean(abu) - 4.93 + 1.58, 2))       # A(Ti)sun = 4.93 (Caffau & Lodders)
    assert sigma == pytest.approx(round(np.hypot(0.04, s_x), 2))
    assert params['s_int'] == pytest.approx(s_x)


def test_species_without_valid_lines(settings):
    flags = np.array([-3])
    nlin, xfe, *_ = extract_abu(np.array([22.0]), np.array([99.99]), np.array([99.99]), 22.0, -1.5, flags, 0.04)
    assert (nlin, xfe) == (0, 99.99)


def test_isotopic_ratio_is_rounded(settings):
    flags = np.zeros(2, dtype=int)
    nlin, mean, *_ = extract_abu_isor(np.array([6.0, 6.0]), np.array([10.0, 13.0]), np.array([1.0, 1.0]), 6.0, flags)
    assert (nlin, mean) == (2, 12)


def test_linear_fit():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_fit, m, c, em, ec = get_coeffs(x, 2 * x + 1)
    assert (m, c) == pytest.approx((2, 1)) and np.allclose(y_fit, 2 * x + 1) and em == pytest.approx(0)
