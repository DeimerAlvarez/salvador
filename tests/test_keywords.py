"""The keywords ABU2, ISOR, MOLEC and ROTAT, each on a real case (tests/data/keywords), with the
stand-in for K2. The expected results are those of the runs with salvador 1.0.0, identical to the
previous version of salvador (see the notes of each case)."""
import pytest

from conftest import KEYWORDS, prepare_case, run_salvador

# case: (interactive answers or None, result files)
CASES = {
    # NORM = 1, AUTO_RES, NPIX_RV, CALC_BEST = n, errors on the atmospheric parameters (13 columns)
    'ABU2': (None, ['star10uc.fit', 'abundance_final.txt']),
    # interactive, NORM = 2, ABS abundance of C. Every line accepted with 'y': ratio 5 (the run of
    # the author, with interactive adjustments, gave 3); the ABU lines are outside the spectrum
    'ISOR': ('y\n' * 20, ['CS29497-034_C_rv_isor.fit', 'abundance_final_isor.txt']),
    # interactive, NORM = 2, ABS abundances of N and O: no EW, EP and log gf for molecules
    'MOLEC': ('y\n' * 20, ['M54_LR2_1500138_norm_rvc.fit', 'abundance_final.txt']),
    # automatic, NORM = 1, the optional columns of ROTAT (initial abundance, span, rotation range)
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
            done[name] = run_salvador(path, prepare_case(source, path, ew), answers=CASES[name][0])
        return done[name]
    return get


@pytest.mark.parametrize('name', sorted(CASES))
def test_results_as_expected(runs, name):
    run = runs(name)
    assert run.returncode == 0, run.output
    assert not [c for c in run.k2_calls if c.startswith('MISSING')]     # every synthetic from the cache
    for result in CASES[name][1]:
        assert rows(run.path / result) == rows(KEYWORDS / name / 'expected' / result), result


def test_rotation_errors_are_integers(runs):
    for row in rows(runs('ROTAT').path / 'star37934_rot.fit'):
        assert row[3].isdigit() and row[4].isdigit(), row          # ROT and eROT in km/s


def test_parameter_errors_in_fit_file(runs):
    fit = runs('ABU2').path / 'star10uc.fit'
    header = [r for r in fit.read_text().splitlines() if r.startswith('#SPECTRUM')][0]
    assert 'd_Teff' in header and 'd_logg' in header and 'd_vt' in header
    assert all(len(r) == 12 for r in rows(fit))


def test_molecules_have_no_ew(runs):
    row = rows(runs('MOLEC').path / 'M54_LR2_1500138_norm_rvc.fit')[0]
    assert row[4:7] == ['99.99', '99.999', '99.999']
