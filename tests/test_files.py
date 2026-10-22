"""Files: spectra, .fit result files, trimmed K2 linelists, clean-up, regression check."""
import os
import time

import numpy as np
import pytest

from conftest import SUBSET
from salvador import regression_check
from salvador.fit_files import (add_errors_to_file, append_flag_column, check_if_has_data, count_header_cols,
                                get_header_names, read_fit_file)
from salvador.k2_interface import _cut_linelist, _scan_linelist
from salvador.spectra_io import cleaning_tmp_files, read_spec, read_two_columns

HEADER = '#   FLAG =  1  (USED LINE)\n#SPECTRUM                           LINE   ELEM    ABU      EW      EP  log_gf    eMC\n'
ROWS = ('s1  7461.519  26.00   6.15   51.75   2.559  -3.585   0.01\n'
        's1  7489.578  22.00   3.85   16.92   2.249  -0.910   0.02\n')
BASE = ['ID', 'line', 'nelem', 'abu', 'ew', 'EP', 'loggf', 'abu_mc']
ERRS = ['abu_pT', 'abu_pg', 'abu_pv']


# ------------------------------------------------------------------------- spectra
def test_read_fits_and_ascii_spectra(tmp_path):
    wave, flux = read_spec(SUBSET / 'M54_HR18_1500138.fits')
    assert wave.size == flux.size == 8467
    assert wave[0] == pytest.approx(7459.486328, abs=1e-5) and flux[0] == pytest.approx(990.79474, abs=1e-3)
    assert np.allclose(np.diff(wave), wave[1] - wave[0])
    ascii_file = tmp_path / 'spec.txt'
    np.savetxt(ascii_file, np.c_[wave, flux], header='wave flux')
    wave2, flux2 = read_spec(ascii_file)
    assert np.allclose(wave2, wave) and np.allclose(flux2, flux)
    assert np.allclose(read_two_columns(ascii_file)[1], flux)


def test_missing_spectrum(tmp_path, capsys):
    assert read_spec(tmp_path / 'none.fits') == (None, None)
    assert 'File not found' in capsys.readouterr().out


# ------------------------------------------------------------------------- .fit files
def test_fit_file_errors_and_flags(tmp_path):
    fit = tmp_path / 's1.fit'
    fit.write_text(HEADER + ROWS)
    assert count_header_cols(fit) == 8 and check_if_has_data(fit, '#SPECTRUM')

    # the errors go to the LAST row, once; the header gets their names once
    assert add_errors_to_file(fit, ['d_Teff', 'd_logg', ' d_vt'], 8, ' 0.10', ' 0.02', ' 0.05') == 11
    assert add_errors_to_file(fit, ['d_Teff', 'd_logg', ' d_vt'], 8, ' 9.99', ' 9.99', ' 9.99') == 11
    rows = [r.split() for r in fit.read_text().splitlines() if not r.startswith('#')]
    assert rows[0][-1] == '0.01' and rows[1][-3:] == ['0.10', '0.02', '0.05']

    append_flag_column(fit, np.array([1, -1]))
    assert get_header_names(fit)[-1] == 'flag_line'
    assert [r.split()[-1] for r in fit.read_text().splitlines() if not r.startswith('#')] == ['1', '-1']


def test_flag_column_with_wrong_length(tmp_path, capsys):
    fit = tmp_path / 's1.fit'
    fit.write_text(HEADER + ROWS)
    append_flag_column(fit, np.array([1]))
    assert 'number of flags (1)' in capsys.readouterr().out
    assert fit.read_text() == HEADER + ROWS


def test_read_fit_file(tmp_path):
    fit = tmp_path / 's1.fit'
    fit.write_text(HEADER + ROWS)
    star, nelem, abu, mc, others, df = read_fit_file(count_header_cols(fit), BASE, ERRS, fit)
    assert star == 's1' and nelem.tolist() == [26.0, 22.0] and others.tolist() == [22.0]
    assert 'abu_pT' not in df.columns


def test_empty_fit_file(tmp_path):
    fit = tmp_path / 's1.fit'
    fit.write_text(HEADER)
    assert not check_if_has_data(fit, '#SPECTRUM')
    assert read_fit_file(8, BASE, ERRS, fit) == (None,) * 6


# ------------------------------------------------------------------------- trimmed linelists for K2
def make_linelist(path, waves_nm):
    path.write_text(''.join(f'{w:11.4f} some atomic data\n' for w in waves_nm))


def test_cut_of_a_sorted_linelist(tmp_path):
    src, dst = tmp_path / 'atoms.dat', tmp_path / 'cut.dat'
    waves = np.round(np.arange(740.0, 790.0, 0.37), 4)
    make_linelist(src, waves)
    assert _scan_linelist(src, 11) == (True, waves[0], waves[-1])
    assert _cut_linelist(src, dst, 750.0, 755.0, 11)
    kept = [float(r[:11]) for r in dst.read_text().splitlines()]
    assert kept == [w for w in waves if 750.0 <= w <= 755.0]
    assert not _cut_linelist(src, tmp_path / 'none.dat', 800.0, 810.0, 11)


def test_unsorted_linelist_is_not_cut(tmp_path):
    src = tmp_path / 'atoms.dat'
    make_linelist(src, [750.0, 749.0, 751.0])
    assert _scan_linelist(src, 11) == (False, None, None)


# ------------------------------------------------------------------------- clean-up
def test_cleaning_deletes_only_new_files(tmp_path):
    old_user_files = [tmp_path / 'guess_my_notes.txt', tmp_path / 'my_cl_data.txt']
    for f in old_user_files:
        f.write_text('mine')
        os.utime(f, (1e9, 1e9))
    start = time.time()
    (tmp_path / 'guess_model_4other.dat').write_text('k2')
    (tmp_path / 'am154_cl_ap04.nb').write_text('k2')
    cleaning_tmp_files(['guess*', '*_cl_*'], tmp_path, newer_than=start)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f.name for f in old_user_files)


# ------------------------------------------------------------------------- regression check
def test_regression_check(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 's1.fit').write_text(HEADER + ROWS)
    monkeypatch.setattr('sys.argv', ['salvador-regression', 'save'])
    assert regression_check.main() == 0
    monkeypatch.setattr('sys.argv', ['salvador-regression'])
    assert regression_check.main() == 0
    (tmp_path / 's1.fit').write_text(HEADER + ROWS.replace('6.15', '6.16'))
    assert regression_check.main() == 1
    assert 'reference 6.15, now 6.16' in capsys.readouterr().out
    monkeypatch.setattr('sys.argv', ['salvador-regression', '--tol', '0.011'])
    assert regression_check.main() == 0
