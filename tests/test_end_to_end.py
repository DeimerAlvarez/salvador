"""The salvador command on a 5-line subset of the tutorial (M54, GIRAFFE HR18), with the
stand-in for K2: the results must be those of the reference run (tests/data/.../expected)."""
import os
import subprocess
import sys

from pypdf import PdfReader

from conftest import SUBSET, prepare_subset, run_salvador, set_keyword

EXPECTED = SUBSET / 'expected'
STAR = 'M54_HR18_1500138'


def expected_rows(name):
    return [r.split() for r in (EXPECTED / name).read_text().splitlines() if r.strip() and not r.startswith('#')]


def test_parallel_run_reproduces_reference(parallel_run):
    assert parallel_run.returncode == 0, parallel_run.output
    assert parallel_run.data_rows(f'{STAR}.fit') == expected_rows(f'{STAR}.fit')
    assert parallel_run.data_rows('abundance_final.txt') == expected_rows('abundance_final.txt')


def test_sequential_run_equals_parallel(sequential_run, parallel_run):
    assert sequential_run.returncode == 0, sequential_run.output
    for name in (f'{STAR}.fit', 'abundance_final.txt'):
        assert sequential_run.data_rows(name) == parallel_run.data_rows(name)


def test_lines_outside_the_spectrum_are_excluded(parallel_run):
    assert 'Line 6390.455 outside the spectrum.' in parallel_run.output
    lines = [row[1] for row in parallel_run.data_rows(f'{STAR}.fit')]
    assert '6390.455' not in lines and '6393.600' not in lines


def test_flags(parallel_run):
    flags = {row[1]: row[-1] for row in parallel_run.data_rows(f'{STAR}.fit')}
    assert flags['7496.103'] == '-3'            # out of the limits of the automatic mode
    assert flags['7461.519'] == '1'


def test_pdf_with_all_the_plots(parallel_run):
    pdf = parallel_run.path / f'{STAR}.pdf'
    assert pdf.exists()
    # one page per analysed line (5) + the diagnostic plot of Fe
    assert len(PdfReader(pdf).pages) == 6


def test_temporary_files_removed(parallel_run):
    for name in ('tmp_plots', 'linelist_cut', 'info_ew.txt', 'abu_var_k2.tmp'):
        assert not (parallel_run.path / name).exists()
    assert (parallel_run.path / 'synth_out').is_dir()


def test_warning_for_incomplete_parameter_errors(parallel_run):
    # lista.stars of the tutorial has 12 columns: 2 of the 3 errors on the parameters
    assert 'has 12 columns' in parallel_run.output


def test_invalid_linelist_rows_are_skipped(tmp_path):
    path = tmp_path / 'run'
    env = prepare_subset(path)
    with open(path / 'linelist.txt', 'a') as f:
        f.write('XYZ   7600.000  26.00  18000  0  0  0\n')
        f.write('ABU2  7650.000  26.00  18000  7649.8  7650.2  0\n')
        f.write('ABU   7700.000  120.00 18000  0  0  0\n')
        f.write('ISOR  7711.721  26.01  18000  0  0  0\n')     # same wavelength as an ABU row
    run = run_salvador(path, env)
    assert run.returncode == 0, run.output
    assert "unknown keyword 'XYZ'" in run.output
    assert 'ABU2 needs 9 columns' in run.output
    assert 'element code 120.00 not in solar_abu.txt' in run.output
    assert 'has the same wavelength as an earlier row' in run.output
    assert run.data_rows(f'{STAR}.fit') == expected_rows(f'{STAR}.fit')


def test_missing_k2_stops_with_a_message(tmp_path):
    path = tmp_path / 'run'
    env = prepare_subset(path, k2_in_path=False)
    env['PATH'] = '/usr/bin:/bin'
    run = run_salvador(path, env)
    assert run.returncode != 0
    assert 'k2.py not found in PATH' in run.output


def test_missing_input_file(tmp_path):
    run = run_salvador(tmp_path, {'PATH': '/usr/bin:/bin', 'MPLBACKEND': 'Agg'})
    assert run.returncode != 0
    assert 'input.salvador does not exist' in run.output


def test_version_and_check(tmp_path):
    from salvador import __version__
    assert __version__ in run_salvador(tmp_path, {'PATH': '/usr/bin:/bin'}, '--version').output
    check = run_salvador(tmp_path, {'PATH': '/usr/bin:/bin'}, '--check')
    assert '[OK]   numpy' in check.output
    assert 'k2.py is not in the PATH' in check.output


def test_mc_error_for_elements_with_few_lines(parallel_run):
    # fewer than 10 lines of Fe here: the Monte Carlo error is computed (99.99 otherwise)
    emc = {row[1]: row[7] for row in parallel_run.data_rows(f'{STAR}.fit')}
    assert emc['7461.519'] != '99.99'


def test_set_keyword_helper(tmp_path):
    (tmp_path / 'input.salvador').write_text('OBS = a\nNPROC = 2\n')
    set_keyword(tmp_path, 'NPROC', '1')
    assert (tmp_path / 'input.salvador').read_text() == 'OBS = a\nNPROC = 1\n'


def test_merge_of_fit_files(tmp_path):
    # two .fit files of the same star (here the same file twice) merged into one, with the
    # flags recalculated and a wider EWR range
    fit = (EXPECTED / f'{STAR}.fit').read_text()
    (tmp_path / 'part1.fit').write_text(fit)
    (tmp_path / 'part2.fit').write_text(fit)
    (tmp_path / 'list.txt').write_text('part1.fit\npart2.fit\nmerged.fit\n')
    answers = 'list.txt\n1\n0\n3\n0\n10\n-6.1\n-4.4\n'      # list, solar ref, recalc flags, sclip, EP and EWR limits
    proc = subprocess.run([sys.executable, '-m', 'salvador.merge_fit'], cwd=tmp_path, input=answers,
                          capture_output=True, text=True, env={**os.environ, 'MPLBACKEND': 'Agg'}, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    merged = [r.split() for r in (tmp_path / 'merged.fit').read_text().splitlines() if r.strip() and not r.startswith('#')]
    assert len(merged) == 10
    final = {r.split()[1]: r.split() for r in (tmp_path / 'abundance_final_merged.txt').read_text().splitlines()
             if r.strip() and not r.startswith('#')}
    # Fe: the same line twice gives the same [Fe/H] as the single run, now from 2 lines
    fe_single = [r for r in expected_rows('abundance_final.txt') if r[1] == '26.00'][0]
    assert final['26.00'][2] == '2' and final['26.00'][3] == fe_single[3]
    assert (tmp_path / 'merged.pdf').exists()
