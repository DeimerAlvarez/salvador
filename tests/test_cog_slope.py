"""salvador 1.4.0: slope of the curve of growth from K2 (COG_SLOPE_MIN, flag -4), EWR_MAX as an
alternative, the columns EWR and SLOPE of the .fit files (also read and merged from older files),
the K2 format check, the resolution in the plot, the regression check by column names, the
saturated lines of the species without other lines, the parameter errors of the saturated lines,
the local RV correction by template fit and the wavelength ticks of the wide windows."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader

from conftest import EXPECTED_SUBSET, KEYWORDS, prepare_case, prepare_subset, run_salvador, set_keyword
from salvador import regression_check
from salvador.config import EWR_MAX_DEFAULT, Config, load_config
from salvador.fit_files import read_fit_rows, read_fit_table
from salvador.fitting import template_shift
from salvador.k2_interface import K2_FORMAT_MESSAGE, K2FormatError, _parse_k2_linelist, read_info_ew
from salvador.plotting import wavelength_ticks
from salvador.summary import (FLAG_EW, FLAG_SLOPE, describe_ew_selection, ew_selection_flag, reduced_ew,
                              star_abundance_rows, use_saturated_species)

STAR = 'M54_HR18_1500138'
ENV_MPL = {**os.environ, 'MPLBACKEND': 'Agg'}


def rows(path):
    return [r.split() for r in Path(path).read_text().splitlines() if r.strip() and not r.startswith('#')]


# ------------------------------------------------------------------ EW file of K2
@pytest.mark.parametrize('text, expected', [
    ('6300.310 36.10 0.000 -9.776 0.935\n', (36.10, 0.0, -9.776, 0.935)),
    ('6300.310 0.00 0.000 0.000 nan\n', (0.0, 0.0, 0.0, 99.999)),        # no slope: 99.999
])
def test_read_info_ew(tmp_path, text, expected):
    (tmp_path / 'info_ew.txt').write_text(text)
    assert read_info_ew(tmp_path / 'info_ew.txt') == pytest.approx(expected)


def test_old_k2_format_is_refused(tmp_path):
    (tmp_path / 'info_ew.txt').write_text('6300.310 36.10 0.000 -9.776\n')
    with pytest.raises(K2FormatError, match='salvador needs EW file with the COG slope'):
        read_info_ew(tmp_path / 'info_ew.txt')
    (tmp_path / 'info_ew.txt').write_text('6300.310 36.10\n')
    with pytest.raises(ValueError):
        read_info_ew(tmp_path / 'info_ew.txt')


# ------------------------------------------------------------------ selection of the lines
@pytest.mark.parametrize('ew, slope, limits, flag', [
    (50.0, 0.70, (-6.0, -4.4, 0.35), 0),             # weak line, linear part
    (80.0, 0.30, (-6.0, -4.4, 0.35), FLAG_SLOPE),    # saturated: rejected by the slope
    (80.0, 0.30, (-6.0, -4.4, None), 0),             # EWR_MAX given: the slope is not used
    (500.0, 0.48, (-6.0, -4.4, 0.35), FLAG_EW),      # damping part: slope rises again, EWR > -4.4
    (80.0, 99.999, (-6.0, -4.4, 0.35), 0),           # no slope: only the EWR limits
    (80.0, float('nan'), (-6.0, -4.4, 0.35), 0),
    (0.0, 99.999, (-6.0, -4.4, 0.35), FLAG_EW),      # EW = 0: K2 did not find the line
    (0.01, 0.99, (-6.0, -4.4, 0.35), FLAG_EW),       # below EWR_MIN
    (99.99, 99.999, (-6.0, -4.4, 0.35), 0),          # no EW (e.g. molecules): not judged
])
def test_ew_selection_flag(ew, slope, limits, flag):
    assert ew_selection_flag(ew, 6000.0, slope, *limits) == flag


def test_reduced_ew():
    assert reduced_ew(60.0, 6000.0) == pytest.approx(-5.0)
    assert reduced_ew(0.0, 6000.0) == 99.99 and reduced_ew(99.99, 6000.0) == 99.99


def test_selection_settings(tmp_path):
    base = 'OBS = a\nNORM = 1\nWINDOW = 10\nINTER = n\nSCLIP = 3\n'
    (tmp_path / 'in').write_text(base)
    config = load_config(tmp_path / 'in')
    assert config.cog_slope_min == 0.35 and config.ewr_max is None
    assert config.ew_selection() == (-6.0, EWR_MAX_DEFAULT, 0.35)            # default: COG slope
    (tmp_path / 'in').write_text(base + 'EWR_MAX = -4.6\nCOG_SLOPE_MIN = 0.4\n')
    assert load_config(tmp_path / 'in').ew_selection() == (-6.0, -4.6, None)  # EWR_MAX replaces the slope
    grid = Config(input_stars='a', type_norm=1, window=10, inter='n', sclip=3, path_syn='list.syn')
    assert grid.ew_selection() == (-6.0, EWR_MAX_DEFAULT, None)              # grid: no slope
    for bad in ('COG_SLOPE_MIN = -0.1', 'EWR_MAX = -7'):
        (tmp_path / 'in').write_text(base + bad + '\n')
        with pytest.raises(SystemExit):
            load_config(tmp_path / 'in')
    assert 'COG slope >= 0.35' in describe_ew_selection(-6.0, -4.4, 0.35)
    assert 'COG slope not used' in describe_ew_selection(-6.0, -4.6, None)


def test_feh_without_usable_fe_lines(settings):
    # FIX 1.4.0: all the Fe I lines rejected -> [Fe/H] of the model (before: [Mg/Fe] = -100.51)
    nelem = np.array([26.00, 12.00])
    abu = np.array([6.44, 7.02])
    rows_, _ = star_abundance_rows('s', nelem, abu, np.array([0.03, 0.02]), np.array([FLAG_SLOPE, 0]), feh_model=-1.0)
    fe, mg = rows_[0].split(), rows_[1].split()
    assert fe[2] == '0' and fe[3] == '-1.00'
    assert abs(float(mg[3])) < 5


@pytest.mark.parametrize('nelem, flags, expected', [
    # Ti I: only saturated lines -> used; Fe I: a usable line -> the saturated one stays rejected
    ([26.0, 26.0, 22.0, 22.0], [0, -4, -4, -3], [0, -4, 0, -3]),
    ([28.0], [-4], [0]),                                  # single saturated line (e.g. K I 7699)
    ([20.0, 20.0], [-1, -4], [-1, 0]),                    # the other line rejected by EWR: not usable
    ([20.0, 20.0], [-2, -1], [-2, -1]),                   # no saturated line: nothing changes
    ([], [], []),
])
def test_use_saturated_species(nelem, flags, expected, capsys):
    given = np.array(flags, dtype=int)
    result = use_saturated_species(np.array(nelem), given, label='s')
    assert result.tolist() == expected
    assert given.tolist() == flags                       # the input is not changed
    out = capsys.readouterr().out
    assert ('all its lines are saturated' in out) == (expected != flags)


# ------------------------------------------------------------------ local RV correction (template fit)
def _gauss_line(wave, centre, depth=0.4, sigma=0.15):
    return 1 - depth * np.exp(-0.5 * ((wave - centre) / sigma) ** 2)


@pytest.mark.parametrize('true_pix', [0.0, 0.3, -0.45])
def test_template_shift_recovers_a_known_shift(true_pix):
    pix = 0.05
    wave_t = np.arange(6150, 6170, 0.005)
    flux_t = _gauss_line(wave_t, 6160.0) * _gauss_line(wave_t, 6160.6, depth=0.2)       # line + blend
    wave = np.arange(6155, 6165, pix)
    noise = np.random.default_rng(1).normal(0, 0.005, wave.size)
    flux = 1.02 + 0.8 * (np.interp(wave - true_pix * pix, wave_t, flux_t) - 1) + noise   # other continuum and depth
    res = template_shift(wave, flux, wave_t, flux_t, 6159.5, 6160.5, 0.5 * pix, pix)
    assert res.ok, res.reason
    assert res.shift / pix == pytest.approx(true_pix, abs=max(3 * res.error / pix, 0.03))
    assert 0 < res.error / pix < 0.2


def test_template_shift_refuses_unreliable_shifts():
    pix = 0.05
    wave_t = np.arange(6150, 6170, 0.005)
    flux_t = _gauss_line(wave_t, 6160.0)
    wave = np.arange(6155, 6165, pix)
    rng = np.random.default_rng(2)
    flat = 1 + rng.normal(0, 0.01, wave.size)
    res = template_shift(wave, flat, wave_t, flux_t, 6159.5, 6160.5, 0.5 * pix, pix)
    assert not res.ok                                    # no line: noise only
    far = np.interp(wave - 2.0 * pix, wave_t, flux_t)     # 2 pixels, NPIX_RV = 0.5
    res = template_shift(wave, far, wave_t, flux_t, 6159.5, 6160.5, 0.5 * pix, pix)
    assert not res.ok and res.reason == 'no minimum within the allowed shift'
    res = template_shift(wave, far, wave_t, flux_t, 6160.0, 6160.1, 0.5 * pix, pix)
    assert not res.ok and res.reason == 'too few pixels'


# ------------------------------------------------------------------ ticks of the main plot
@pytest.mark.parametrize('window, step, minor', [(5, 1, None), (10, 2, None), (20, 2, None), (21, 5, 1.0),
                                                 (35, 5, 1.0), (50, 5, 1.0)])
def test_wavelength_ticks(window, step, minor):
    x0, x1 = 6160.7 - window / 2, 6160.7 + window / 2
    ticks, minor_step = wavelength_ticks(x0, x1)
    assert minor_step == minor
    assert np.all(np.diff(ticks) == step) and x0 <= ticks[0] and ticks[-1] <= x1
    assert len(ticks) <= 10
    if step >= 5:
        assert np.all(ticks % step == 0)


@pytest.mark.parametrize('window', np.arange(1, 50.5, 0.5))
def test_wavelength_ticks_never_more_than_ten(window):
    for centre in (4000.25, 6160.7, 8805.0):
        assert len(wavelength_ticks(centre - window / 2, centre + window / 2)[0]) <= 10


# ------------------------------------------------------------------ .fit files of any version
OLD_FIT = ('#   FLAG =  1  (USED LINE)\n'
           '#SPECTRUM                           LINE   ELEM    ABU      EW      EP  log_gf    eMC  d_Teff d_logg  d_vt  flag_line\n'
           's  6000.000  26.00  6.00  60.00  2.000  -2.000  0.02  0.05  0.01  0.02   1\n'
           's  6100.000  26.00  99.99  99.99  99.999  99.999  99.99   1\n')        # 1.0 rejected row: no errors


def test_read_old_fit_file(tmp_path):
    (tmp_path / 'old.fit').write_text(OLD_FIT)
    rows_, has_errors = read_fit_rows(tmp_path / 'old.fit')
    assert has_errors and len(rows_) == 2
    assert rows_[0]['ewr'] == '-5.00' and rows_[0]['slope'] == '99.999' and rows_[0]['abu_pT'] == '0.05'
    assert rows_[1]['abu_pT'] == '99.99' and rows_[1]['flag'] == '1' and rows_[1]['ewr'] == '99.99'
    df = read_fit_table(tmp_path / 'old.fit')
    assert list(df['flag']) == [1, 1] and df['ew'].iat[0] == 60.0


def test_merge_old_and_new_fit_files(tmp_path):
    (tmp_path / 'old.fit').write_text(OLD_FIT)
    (tmp_path / 'new.fit').write_text((EXPECTED_SUBSET / f'{STAR}.fit').read_text())
    (tmp_path / 'list.txt').write_text('old.fit\nnew.fit\nmerged.fit\n')
    proc = subprocess.run([sys.executable, '-m', 'salvador.merge_fit', 'list.txt', '--solar', '1', '--flags', 'recalc',
                           '--sclip', '3', '--ep', '0', '10', '--ewr-min', '-6', '--slope-min', '0.75'],
                          cwd=tmp_path, capture_output=True, text=True, env=ENV_MPL, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'without the COG slope' in proc.stdout
    header = [r for r in (tmp_path / 'merged.fit').read_text().splitlines() if r.startswith('#SPECTRUM')][0]
    assert header.split()[5:7] == ['EWR', 'SLOPE'] and 'COG slope >= 0.75' in (tmp_path / 'merged.fit').read_text()
    merged = {r[1]: r for r in rows(tmp_path / 'merged.fit')}
    assert len(merged) == 7 and all(len(r) == 14 for r in merged.values())
    assert merged['6000.000'][5:7] == ['-5.00', '99.999'] and merged['6000.000'][-1] == '1'   # old: EWR only
    assert merged['7461.519'][-1] == '-4'      # Fe I, slope 0.738 < 0.75: rejected (6000.000 is usable)
    assert merged['7555.597'][-1] == '1'       # Ni I, slope 0.540: the only Ni line, used anyway
    assert 'Ni (28.00): all its lines are saturated' in proc.stdout


def test_regression_by_column_names(tmp_path, monkeypatch, capsys):
    # reference written by 1.3.0 (no EWR, SLOPE), results of 1.4.0: the common columns are compared
    monkeypatch.chdir(tmp_path)
    new = (EXPECTED_SUBSET / f'{STAR}.fit').read_text()
    old = '\n'.join(r.replace('     EWR   SLOPE', '') if r.startswith('#') else
                    ' '.join(f for i, f in enumerate(r.split()) if i not in (5, 6)) for r in new.splitlines()) + '\n'
    (tmp_path / f'{STAR}.fit').write_text(old)
    monkeypatch.setattr('sys.argv', ['salvador-regression', 'save'])
    assert regression_check.main() == 0
    (tmp_path / f'{STAR}.fit').write_text(new)
    monkeypatch.setattr('sys.argv', ['salvador-regression'])
    assert regression_check.main() == 0
    out = capsys.readouterr().out
    assert 'columns only in this run: EWR, SLOPE' in out and 'ALL RESULTS IDENTICAL' in out


# ------------------------------------------------------------------ K2 linelist file
def test_linelist_comments_and_lowercase_markers(tmp_path):
    (tmp_path / 'linelist.xkur').write_text('#atoms\n# a comment\nfe.dat\n#MOLECULES\nco.dat\n')
    assert _parse_k2_linelist(tmp_path / 'linelist.xkur') == [('#ATOMS', ['fe.dat']), ('#MOLECULES', ['co.dat'])]


# ------------------------------------------------------------------ complete runs (stand-in for K2)
def subset_run(path, nproc='2', **env_extra):
    env = prepare_subset(path)
    set_keyword(path, 'NPROC', nproc)
    return run_salvador(path, {**env, **env_extra})


@pytest.mark.parametrize('nproc', ['1', '2'])
def test_old_k2_stops_the_run(tmp_path, nproc):
    path = tmp_path / 'run'
    run = subset_run(path, nproc, FAKE_K2_EW_COLUMNS='4')
    assert run.returncode != 0
    assert run.output.strip().endswith(K2_FORMAT_MESSAGE)
    assert not (path / f'{STAR}.pdf').exists()


def test_saturated_species_are_used(tmp_path):
    # every line saturated (slope 0.20): flag -4, but each species of the subset has no other usable
    # line, so they are all used anyway: the results are those of the reference run
    path = tmp_path / 'run'
    run = subset_run(path, FAKE_K2_SLOPE='0.20')
    assert run.returncode == 0, run.output
    assert run.output.count('all its lines are saturated') == 4          # Fe, Ti, Ni, FeII
    assert 'Fe (26.00): all its lines are saturated (COG slope < COG_SLOPE_MIN): 1 line(s) used anyway' in run.output
    assert all(r[6] == '0.200' for r in rows(path / f'{STAR}.fit') if r[3] != '99.99')
    assert 'FLAG = -4  (REJ. COG SLOPE)' in (path / f'{STAR}.fit').read_text()
    assert [r[:6] + r[7:] for r in rows(path / f'{STAR}.fit')] == [r[:6] + r[7:] for r in rows(EXPECTED_SUBSET / f'{STAR}.fit')]
    assert rows(path / 'abundance_final.txt') == rows(EXPECTED_SUBSET / 'abundance_final.txt')


@pytest.mark.parametrize('slope', ['0.20', None])
def test_parameter_errors_of_saturated_lines(tmp_path, slope):
    # 13 columns in lista.stars: errors on the parameters. They are computed also for the lines with
    # flag -4 (before 1.4.0: 0.00). The stand-in has no synthetic of the changed models: every fit
    # stops and the error is 99.99, as for the accepted lines
    path = tmp_path / 'run'
    env = prepare_subset(path)
    stars = path / 'lista.stars'
    stars.write_text(stars.read_text().rstrip('\n') + ' 0.2\n')
    run = run_salvador(path, {**env, **({'FAKE_K2_SLOPE': slope} if slope else {})})
    assert run.returncode == 0, run.output
    header = [r for r in (path / f'{STAR}.fit').read_text().splitlines() if r.startswith('#SPECTRUM')][0].split()
    assert header[10:13] == ['d_Teff', 'd_logg', 'd_vt']
    for r in rows(path / f'{STAR}.fit'):
        if r[3] != '99.99':
            assert r[10:12] == ['99.99', '99.99'], r                     # Teff and logg: attempted
            assert r[12] == ('99.99' if float(r[4]) > 50 else '0.00'), r  # vt only for EW > 50 mA


def test_ewr_max_replaces_the_slope(tmp_path):
    path = tmp_path / 'run'
    env = prepare_subset(path)
    set_keyword(path, 'EWR_MAX', '-4.4')
    run = run_salvador(path, {**env, 'FAKE_K2_SLOPE': '0.20'})
    assert run.returncode == 0, run.output
    assert 'EWR from -6.00 to -4.40 (COG slope not used)' in run.output
    assert [r[:6] + r[7:] for r in rows(path / f'{STAR}.fit')] == [r[:6] + r[7:] for r in rows(EXPECTED_SUBSET / f'{STAR}.fit')]


def test_missing_slope_uses_the_ewr_limits(tmp_path):
    path = tmp_path / 'run'
    run = subset_run(path, '1', FAKE_K2_SLOPE='nan')
    assert run.returncode == 0, run.output
    assert 'K2 gave no COG slope' in run.output
    assert [r[-1] for r in rows(path / f'{STAR}.fit')] == [r[-1] for r in rows(EXPECTED_SUBSET / f'{STAR}.fit')]
    assert all(r[6] == '99.999' for r in rows(path / f'{STAR}.fit'))


def test_ewr_and_slope_columns_and_resolution_in_the_plot(tmp_path):
    path = tmp_path / 'run'
    run = subset_run(path)
    assert run.returncode == 0, run.output
    for r in rows(path / f'{STAR}.fit'):
        if r[4] != '99.99':
            assert float(r[5]) == pytest.approx(np.log10(float(r[4]) / 1000 / float(r[1])), abs=0.006)
    text = ''.join(page.extract_text() for page in PdfReader(path / f'{STAR}.pdf').pages)
    assert text.count('R = 18000') == 5          # one plot per line (resolution of the linelist)


def test_measured_resolution_in_the_plot(tmp_path):
    # ABU2 case: AUTO_RES, the legend shows the measured resolution
    path = tmp_path / 'run'
    run = run_salvador(path, prepare_case(KEYWORDS / 'ABU2', path, KEYWORDS / 'ABU2' / 'expected' / 'star10uc.fit'))
    assert run.returncode == 0, run.output
    text = ''.join(page.extract_text() for page in PdfReader(path / 'star10uc.pdf').pages)
    assert 'R = ' in text
