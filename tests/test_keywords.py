"""The keywords ABU2, ISOR, MOLEC and ROTAT, each on a real case (tests/data/keywords), with the
stand-in for K2. The expected results are those of the runs with salvador 1.0.0, identical to the
previous version of salvador (see the notes of each case), with the changes of 1.1.0 and 1.2.0
in the case ABU2: signed parameter errors (d_vt of the Fe line is -0.04), and the logg error is
computed for every line (before 1.1.0: only ionized species): the stand-in has no synthetic for
logg + 0.10, so d_logg of both lines is 99.99 (the fit stopped by the missing synthetic gives
99.99) and so are the total errors. 1.4.0, case ABU2: the new local RV correction (template fit,
NPIX_RV) finds the Fe line 0.04 pixels to the blue (before, from the centre of the line: 0.13 pixels
to the red): A(Fe) +0.01 dex, d_Teff and d_vt -0.01; the fits at the new abundances use synthetics
interpolated by the stand-in (FAKE_K2_INTERPOLATE)."""
import pytest

from conftest import KEYWORDS, prepare_case, run_salvador

# case: (interactive answers or None, result files)
CASES = {
    # NORM = 1, AUTO_RES, NPIX_RV, CALC_BEST = n, errors on the atmospheric parameters (13 columns).
    # EWR_MAX = -4.4 in input.salvador (1.4.0): the selection of the previous versions (with the COG slope
    # the Fe line, EW = 146 mA at 8805 A, slope 0.34, gets flag -4, and is used as the only Fe I line)
    'ABU2': (None, ['star10uc.fit', 'abundance_final.txt']),
    # interactive, NORM = 2, ABS abundance of C. Every line accepted with 'y': ratio 5 (the run of
    # the author, with interactive adjustments, gave 3); the ABU lines are outside the spectrum
    'ISOR': ('y\n' * 20, ['CS29497-034_C_rv_isor.fit', 'abundance_final_isor.txt']),
    # interactive, NORM = 2, ABS abundances of N and O: no EW, EP and log gf for molecules
    'MOLEC': ('y\n' * 20, ['M54_LR2_1500138_norm_rvc.fit', 'abundance_final.txt']),
    # automatic, NORM = 1, the optional columns of ROTAT (initial abundance, span, rotation range).
    # Since 1.3.0 (exact broadening for each v sin i, upper limits): all the lines are upper limits
    # (4 km/s = half the instrumental FWHM at R = 40000); before, 0-1 km/s.
    'ROTAT': (None, ['star37934_rot.fit']),
}


def rows(path):
    return [r.split() for r in path.read_text().splitlines() if r.strip() and not r.startswith('#')]


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """Each case is run once, at its first use, and kept for the other tests."""
    done = {}

    def get(name):
        if name not in done:
            source = KEYWORDS / name
            ew = next((f for f in (source / 'expected').glob('*.fit') if not f.stem.endswith(('_isor', '_rot'))), None)
            path = tmp_path_factory.mktemp(name) / 'run'
            env = prepare_case(source, path, ew)
            if name == 'ABU2':
                env['FAKE_K2_INTERPOLATE'] = '1'      # see the docstring (1.4.0)
            done[name] = run_salvador(path, env, answers=CASES[name][0])
        return done[name]
    return get


@pytest.mark.parametrize('name', sorted(CASES))
def test_results_as_expected(runs, name):
    run = runs(name)
    assert run.returncode == 0, run.output
    missing = [c for c in run.k2_calls if c.startswith('MISSING')]
    # every synthetic from the cache, except the ABU2 fit with logg + 0.10 (see the docstring)
    assert all(name == 'ABU2' and 'g087' in c for c in missing), missing
    assert name == 'ABU2' or not any(c.startswith('INTERPOLATED') for c in run.k2_calls)
    for result in CASES[name][1]:
        assert rows(run.path / result) == rows(KEYWORDS / name / 'expected' / result), result


def test_rotation_errors_are_integers(runs):
    for row in rows(runs('ROTAT').path / 'star37934_rot.fit'):
        assert row[3].isdigit() and row[4].isdigit(), row          # ROT and eROT in km/s


def test_parameter_errors_in_fit_file(runs):
    fit = runs('ABU2').path / 'star10uc.fit'
    header = [r for r in fit.read_text().splitlines() if r.startswith('#SPECTRUM')][0]
    assert 'd_Teff' in header and 'd_logg' in header and 'd_vt' in header
    assert all(len(r) == 15 for r in rows(fit))             # 1.4.0: + EWR and SLOPE; + SIGFIT


def test_molecules_have_no_ew(runs):
    row = rows(runs('MOLEC').path / 'M54_LR2_1500138_norm_rvc.fit')[0]
    assert row[4:7] + row[8:10] == ['99.99', '99.99', '99.999', '99.999', '99.999']    # EW, EWR, SLOPE, EP, log gf
    assert 0 < float(row[7]) < 1          # SIGFIT: the profile error is known also for a molecule (not used to select)


def test_rotat_without_vmac_stops_before_k2(tmp_path):
    # since 1.3.0 ROT_VMAC is mandatory when the linelist has ROTAT lines
    path = tmp_path / 'run'
    env = prepare_case(KEYWORDS / 'ROTAT', path)
    (path / 'input.salvador').write_text(
        ''.join(r for r in (path / 'input.salvador').read_text().splitlines(keepends=True) if not r.startswith('ROT_VMAC')))
    run = run_salvador(path, env)
    assert run.returncode != 0 and 'set the macroturbulence with ROT_VMAC' in run.output
    assert run.k2_calls == []
