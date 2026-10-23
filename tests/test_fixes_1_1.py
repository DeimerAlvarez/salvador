"""Tests of the fixes and changes of salvador 1.1.0 (see CHANGELOG.md)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import salvador.cli as cli
import salvador.k2_interface as k2i
from conftest import EXPECTED_SUBSET, prepare_subset, run_salvador, set_keyword
from salvador.k2_interface import add_delta_entry, write_name_syn
from salvador.line_analysis import get_type_input
from salvador.spectra_io import read_spec
from salvador.summary import _mean_valid, extract_abu, extract_abu_fe, extract_abu_isor
from test_grid import build_grid

STAR = 'M54_HR18_1500138'
ENV_MPL = {**os.environ, 'MPLBACKEND': 'Agg'}


def rows(path):
    return [r.split() for r in Path(path).read_text().splitlines() if r.strip() and not r.startswith('#')]


def grid_case(path: Path, errors: str | None = None, perturbed_teff: float | None = None, **keywords):
    """The tutorial subset in grid mode (no K2), NPROC = 1. errors: the three errors added to
    lista.stars; perturbed_teff: also synthetics with Teff + this value, with lines 20% weaker."""
    env = prepare_subset(path)
    grid = path / 'grid'
    build_grid(path.parent / f'{path.name}_k2_cache', grid, path / 'linelist.txt')
    if perturbed_teff is not None:
        names = (grid / 'list.syn').read_text().split()
        for name in list(names):
            new = name.replace('t4248', f't{4248 + int(perturbed_teff)}')
            w, f = np.loadtxt(grid / name, unpack=True, usecols=(0, 1))
            np.savetxt(grid / new, np.c_[w, 1 - 0.8 * (1 - f)], fmt='%.4f')
            names.append(new)
        (grid / 'list.syn').write_text('\n'.join(names) + '\n')
    set_keyword(path, 'PATH_SYN', 'grid/list.syn')
    set_keyword(path, 'NPROC', '1')
    for key, value in keywords.items():
        set_keyword(path, key, value)
    if errors is not None:
        stars = path / 'lista.stars'
        stars.write_text('\n'.join(r if r.startswith('#') or not r.strip() else ' '.join(r.split()[:10] + errors.split())
                                   for r in stars.read_text().splitlines()) + '\n')
    env['PATH'] = '/usr/bin:/bin:' + str(Path(sys.executable).parent)
    return env


# ------------------------------------------------------------------ 1) grid: parameter errors
def test_grid_errors_without_perturbed_synthetics(tmp_path):
    path = tmp_path / 'run'
    run = run_salvador(path, grid_case(path, errors='100 0.1 0.1'))
    assert run.returncode == 0, run.output
    fit = {r[1]: r for r in rows(path / f'{STAR}.fit')}
    assert fit['7461.519'][8] == '99.99'                   # before: 0.00
    assert fit['7711.721'][9] == '99.99'                   # Fe II: logg error
    assert 'Grid (Teff + 100 K): no synthetic spectra' in run.output
    fe = [r for r in rows(path / 'abundance_final.txt') if r[1] == '26.00'][0]
    assert fe[6] == '99.99'                                # E_PARAM not available


def test_grid_errors_from_perturbed_synthetics(tmp_path):
    path = tmp_path / 'run'
    run = run_salvador(path, grid_case(path, errors='100 0 0', perturbed_teff=100))
    assert run.returncode == 0, run.output
    d_teff = [float(r[8]) for r in rows(path / f'{STAR}.fit') if r[3] != '99.99']
    assert any(0 < d < 0.5 for d in d_teff), d_teff        # weaker lines: higher abundance (signed)
    assert all(r[9] == '0.00' and r[10] == '0.00' for r in rows(path / f'{STAR}.fit') if r[3] != '99.99')


# ------------------------------------------------------------------ 2) names of the synthetics
def test_names_of_gf_isor_and_other_rows(tmp_path):
    def name(rows_text, tana='ABU', zcode=20, var=0.3):
        path = tmp_path / f'v{abs(hash((rows_text, tana, zcode, var)))}.txt'
        path.write_text(rows_text)
        add_delta_entry(tana, path, zcode, var)
        return write_name_syn(-1.5, 1, 4300, 1.0, 1.5, 6100, 6110, path, tana, zcode, var)

    delta = name('')
    assert delta == 'am150t4300g100k150_cl_ap04_Ca_p030_6100_6110'          # unchanged
    assert name('', 'ABU_GF') == 'am150t4300g100k150_cl_ap04_Ca_gfp030_6100_6110'
    assert name('ISOR 6 9\n') == 'am150t4300g100k150_cl_ap04_Ca_p030_C_isor9_6100_6110'
    assert name('ISOR 6 9\n') != name('ISOR 6 20\n')
    assert name('ISOF 6 0.5 0.5\n') != name('ISOF 6 0.6 0.4\n') != delta
    # ISOR line: its own ratio as before; another element's ratio in the file is added
    assert name('', 'ISOR', 6, 12.0) == 'am150t4300g100k150_cl_ap04_ISOR_C_12_6100_6110'
    assert name('ISOR 6 30\n', 'ISOR', 6, 12.0) == 'am150t4300g100k150_cl_ap04_ISOR_C_12_6100_6110'


def test_linelist_change_warning(tmp_path, monkeypatch, capsys):
    ll = tmp_path / 'atoms.dat'
    ll.write_text('one\n')
    xkur = tmp_path / 'linelist.xkur'
    xkur.write_text(f'#ATOMS\n{ll}\n')
    synth = tmp_path / 'synth_out'
    monkeypatch.setattr(k2i, 'K2_LINELIST', str(xkur))
    monkeypatch.setattr(k2i, 'SYNTH_OUT', synth)
    monkeypatch.setattr(k2i, 'LINELIST_RECORD', synth / 'linelists.json')
    assert k2i.check_synth_out_linelists() is False                    # first run: recorded
    (synth / 'a.nb').write_text('1 1\n')
    assert k2i.check_synth_out_linelists() is False                    # same linelists
    ll.write_text('one two\n')
    assert k2i.check_synth_out_linelists() is True
    assert 'atoms.dat' in capsys.readouterr().out
    (synth / 'a.nb').unlink()                                           # empty synth_out: new record
    assert k2i.check_synth_out_linelists() is False


# ------------------------------------------------------------------ 3), 4) errors of the abundances
def test_mean_of_parameter_errors_skips_missing_values():
    assert _mean_valid(np.array([0.05, 0.06, 99.99, 0.04])) == pytest.approx(0.05)
    assert _mean_valid(np.array([99.99, np.nan])) == 99.99


def test_x_fe_error_uses_the_sign(settings):
    nelem, flags = np.array([26.0, 26.0, 22.01, 22.01]), np.zeros(4, int)
    abu, mc = np.array([6.0, 6.0, 3.5, 3.5]), np.full(4, 0.0)
    dT, dg, dv = np.array([0.10, 0.10, -0.05, -0.05]), np.zeros(4), np.zeros(4)
    res = extract_abu_fe(nelem, abu, mc, 26.0, flags, dT, dg, dv)
    s1, s2, s3, s4 = res[3:7]
    out = extract_abu(nelem, abu, mc, 22.01, res[1], flags, s1, param_errors=(dT, dg, dv), fe_errors=(s2, s3, s4))
    assert out[4] == pytest.approx(0.15)                   # |-0.05 - 0.10|; with absolute values: 0.05


def test_missing_parameter_error_gives_missing_total(settings):
    nelem, flags = np.array([22.0]), np.zeros(1, int)
    out = extract_abu(nelem, np.array([3.5]), np.array([0.02]), 22.0, -1.5, flags, 0.03,
                      param_errors=(np.array([0.05]), np.array([99.99]), np.array([0.01])), fe_errors=(0.05, 0.0, 0.02))
    assert out[2] == 99.99 and out[4] == 99.99 and out[5] == pytest.approx(0.04, abs=0.005)


def test_x_fe_without_iron(settings):
    flags = np.zeros(2, int)
    out = extract_abu(np.array([22.0, 22.0]), np.array([3.5, 3.6]), np.full(2, 0.02), 22.0, None, flags, 99.99)
    assert out[1] == 99.99 and out[6]['abu_abs'] == pytest.approx(3.55)


def test_rejected_isor_line_is_not_a_ratio_of_100(settings):
    assert cli._isor_txt(99.99, 5) == '99.99' and cli._isor_txt(12.0, 5) == '   12'
    flags = np.array([0, -3])
    out = extract_abu_isor(np.array([6.0, 6.0]), np.array([10.0, 99.99]), np.array([1.0, 99.99]), 6.0, flags)
    assert out[0] == 1 and out[1] == 10


# ------------------------------------------------------------------ 5), 6), 9) and order of the lines
def test_nan_pixels_relative_path_and_order(tmp_path):
    path = tmp_path / 'run'
    env = grid_case(path)
    data = path / 'spectra'
    data.mkdir()
    from astropy.io import fits
    with fits.open(path / f'{STAR}.fits') as h:
        flux = h[0].data.astype(float)
        hd = h[0].header
        wave = (hd['CRVAL1'] + (np.arange(flux.size) + 1 - hd['CRPIX1']) * hd.get('CDELT1', hd.get('CD1_1'))) / (1 + 133.79 / 299792.458)
        flux[np.abs(wave - 7555.6) < 0.05] = np.nan
        fits.writeto(data / f'{STAR}.fits', flux, hd)
    (path / f'{STAR}.fits').unlink()
    stars = path / 'lista.stars'
    stars.write_text(stars.read_text().replace(f'{STAR}.fits', f'./spectra/{STAR}.fits'))
    linelist = path / 'linelist.txt'
    data_rows = [r for r in linelist.read_text().splitlines() if r.strip() and not r.startswith('#')]
    linelist.write_text('\n'.join(reversed(data_rows)) + '\n')
    run = run_salvador(path, env)
    assert run.returncode == 0, run.output                 # before: stopped by the NaN pixels
    assert 'with a NaN or infinite value are not used' in run.output
    result = rows(path / f'{STAR}.fit')                     # before: spectra/M54... (and '.fit' for ./)
    lines = [float(r[1]) for r in result]
    assert lines == sorted(lines)                           # order of wavelength
    assert '7555.597' in [r[1] for r in result]


def test_read_spec_removes_nan(tmp_path):
    path = tmp_path / 'spec.txt'
    path.write_text('1 1.0\n2 nan\n3 0.9\ninf 1\n')
    wave, flux = read_spec(path)
    assert wave.tolist() == [1, 3] and flux.tolist() == [1.0, 0.9]
    path.write_text('1 nan\n')
    assert read_spec(path) == (None, None)


# ------------------------------------------------------------------ 7) rejected lines with errors
def test_rejected_line_has_error_columns_and_merges(tmp_path):
    path = tmp_path / 'run'
    # the second line (Ti 7489.578) is rejected with 'r'
    run = run_salvador(path, grid_case(path, errors='100 0.1 0.1', INTER='y'), answers='y\nr\n' + 'y\n' * 30)
    assert run.returncode == 0, run.output
    result = rows(path / f'{STAR}.fit')
    assert all(len(r) == 12 for r in result)
    assert result[1][1] == '7489.578' and result[1][3] == '99.99' and result[1][8:11] == ['99.99'] * 3
    (path / 'list.txt').write_text(f'{STAR}.fit\nmerged.fit\n')
    proc = subprocess.run([sys.executable, '-m', 'salvador.merge_fit', 'list.txt', '--solar', '1', '--flags', 'keep'],
                          cwd=path, capture_output=True, text=True, env=ENV_MPL, timeout=300)
    assert proc.returncode == 0 and 'Error' not in proc.stdout, proc.stdout + proc.stderr
    assert (path / 'merged.pdf').exists()


# ------------------------------------------------------------------ 10) errors of one line
def test_error_of_one_line_skips_it(monkeypatch, capsys):
    def boom(task):
        raise RuntimeError('K2 did not create the synthetic spectrum')
    monkeypatch.setattr(cli, '_analyse_line', boom)
    rec = cli.analyse_line(('ABU', 7461.519, 26.0, 18000, 0, 0, 0, 3))
    assert rec['skipped'] and rec['failed'] and 'RuntimeError' in rec['reason']
    assert 'stopped with an error' in capsys.readouterr().out


def test_run_stops_after_failures_in_a_row(monkeypatch):
    def stop(message=None):
        raise SystemExit(message)
    monkeypatch.setattr(cli, 'stop_run', stop)
    counter = cli.FailureCounter()
    failed, ok, skipped = {'failed': True, 'skipped': True}, {'skipped': False}, {'skipped': True}
    for rec in (failed, failed, ok, failed, skipped, failed):
        counter.update(rec)                                  # a line analysed resets, a skipped one does not
    with pytest.raises(SystemExit, match='in a row'):
        counter.update(failed)


# ------------------------------------------------------------------ 13) NORM = 2 windows
def test_warning_for_windows_on_one_side(capsys):
    cli._norm2_window_warning(7500.0, np.array([7400.0, 7450.0]), np.array([7401.0, 7451.0]))
    out = capsys.readouterr().out
    assert 'right side' in out and 'left side' not in out
    cli._norm2_window_warning(7500.0, np.array([7400.0, 7550.0]), np.array([7401.0, 7551.0]))
    assert capsys.readouterr().out == ''


# ------------------------------------------------------------------ 21) interactive and PDF order
def test_valid_input_only(monkeypatch, capsys):
    answers = iter(['x', '0', '9', '2'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    assert get_type_input('window: ', int, lambda v: 1 <= v <= 3, 'from 1 to 3') == 2
    assert capsys.readouterr().out.count('from 1 to 3') == 2


def test_pdf_pages_in_order_of_wavelength():
    names = ['line_10012.5.pdf', 'line_7461.519.pdf', 'line_8806.756.pdf']
    assert sorted(names, key=cli._plot_wavelength) == ['line_7461.519.pdf', 'line_8806.756.pdf', 'line_10012.5.pdf']


def test_restore_keeps_the_local_rv_correction(tmp_path):
    # RV 6 km/s off, corrected line by line (NPIX_RV); the first line is changed with 'wa' and 'c',
    # then restored with 'i': the result must be the one of the first fit (before 1.1.0, 6.02
    # instead of 6.14, because 'i' lost the local RV correction)
    results = []
    for k, first in enumerate(('y\n', 'wa\n0.3\nc\n1.05\ny\n', 'wa\n0.3\nc\n1.05\ni\ny\n')):
        path = tmp_path / f'run{k}'
        env = grid_case(path, INTER='y', NPIX_RV='10')
        stars = path / 'lista.stars'
        stars.write_text(stars.read_text().replace('133.790', '127.790'))
        run = run_salvador(path, env, answers=first + 'y\n' * 30)
        assert run.returncode == 0, run.output
        results.append(rows(path / f'{STAR}.fit'))
    assert results[1][0][3] != results[0][0][3]              # the commands change the fit
    assert results[2] == results[0]                          # 'i' gives back the first fit


# ------------------------------------------------------------------ 8), 16), 17), 18) salvador-merge
def merge(path, args=(), answers=None):
    return subprocess.run([sys.executable, '-m', 'salvador.merge_fit', *args], cwd=path, input=answers,
                          capture_output=True, text=True, env=ENV_MPL, timeout=300)


def test_merge_gives_the_same_abundances_as_salvador(tmp_path):
    shutil.copy(EXPECTED_SUBSET / f'{STAR}.fit', tmp_path)
    (tmp_path / 'list.txt').write_text(f'{STAR}.fit\nmerged.fit\n')
    proc = merge(tmp_path, ['list.txt', '--solar', '1', '--flags', 'keep'])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'Which solar' not in proc.stdout                 # nothing asked
    merged = [r[1:] for r in rows(tmp_path / 'abundance_final_merged.txt')]
    expected = [r[1:] for r in rows(EXPECTED_SUBSET / 'abundance_final.txt')]
    assert merged == expected


def test_merge_external_feh_with_its_error(tmp_path):
    shutil.copy(EXPECTED_SUBSET / f'{STAR}.fit', tmp_path)
    (tmp_path / 'list.txt').write_text(f'{STAR}.fit\nmerged.fit  -1.50  0.07\n')
    assert merge(tmp_path, ['list.txt', '--solar', '1', '--flags', 'keep']).returncode == 0
    final = {r[1]: r for r in rows(tmp_path / 'abundance_final_merged.txt')}
    assert final['26.00'][3:6] == ['-1.50', '0.07', '0.07']
    assert final['22.00'][4] != '99.99'                     # before: 99.99 with the parameter errors
    # error not in the file: asked
    (tmp_path / 'list.txt').write_text(f'{STAR}.fit\nmerged.fit  -1.50\n')
    proc = merge(tmp_path, ['list.txt', '--solar', '1', '--flags', 'keep'], answers='-1\n0.07\n')
    assert proc.returncode == 0 and 'cannot be negative' in proc.stdout
    assert {r[1]: r for r in rows(tmp_path / 'abundance_final_merged.txt')}['26.00'][4] == '0.07'
    # not asked and not given: clear message
    proc = merge(tmp_path, ['list.txt', '--solar', '1', '--flags', 'keep'], answers='')
    assert proc.returncode != 0 and 'in the list file' in proc.stderr


def test_merge_refuses_wrong_choices(tmp_path):
    shutil.copy(EXPECTED_SUBSET / f'{STAR}.fit', tmp_path)
    (tmp_path / 'list.txt').write_text(f'{STAR}.fit\nmerged.fit\n')
    answers = 'list.txt\n7\n1\n5\n0\n-2\n3\n0\n10\n-6\n-7\n-4.4\n'    # wrong: 7, 5, -2, max -7 < min -6
    proc = merge(tmp_path, answers=answers)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count('Error:') == 4
    assert (tmp_path / 'merged.pdf').exists()
    proc = merge(tmp_path, ['list.txt', '--solar', '9'])
    assert proc.returncode == 2 and 'invalid choice' in proc.stderr
    proc = merge(tmp_path, ['list.txt', '--ep', '5', '1'])
    assert proc.returncode == 2 and 'MIN must be smaller' in proc.stderr
