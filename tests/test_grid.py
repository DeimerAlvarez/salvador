"""Grid of precomputed synthetic spectra (PATH_SYN) and salvador-merge on results of grid runs.

The grid is built from the synthetic spectra of the tutorial subset (computed by K2 for single
lines), so that the EWs measured on the grid can be compared with those of K2 for the same lines.
"""
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from conftest import SUBSET, TESTS, prepare_subset, run_salvador, set_keyword

STAR = 'M54_HR18_1500138'
SYMBOL = {26: 'Fe', 22: 'Ti', 28: 'Ni'}
NAME = re.compile(r'^a[mp]\d{3}t\d+g\d+k\d+_cl_a[mp]\d{2}_([A-Za-z]+)_[mp]\d{3}_(\d+)_(\d+)\.nb$')


def rows(path):
    return [r.split() for r in Path(path).read_text().splitlines() if r.strip() and not r.startswith('#')]


def build_grid(cache: Path, grid: Path, linelist: Path) -> None:
    """Grid folder with, for each line of the linelist, the K2 synthetics of one wavelength range."""
    groups = defaultdict(list)
    for f in cache.glob('*.nb'):
        m = NAME.match(f.name)
        if m:
            groups[(m[1], int(m[2]), int(m[3]))].append(f)
    grid.mkdir()
    names = []
    for row in rows(linelist):
        line, code = float(row[1]), float(row[2])
        elem = SYMBOL.get(int(code))
        ranges = [k for k in groups if k[0] == elem and k[1] <= line <= k[2] and len(groups[k]) >= 3]
        if len(ranges) == 1:
            for f in groups[ranges[0]]:
                shutil.copy(f, grid / f.name)
                names.append(f.name)
    (grid / 'list.syn').write_text('\n'.join(sorted(names)) + '\n')


@pytest.fixture(scope='module')
def grid_run(tmp_path_factory):
    path = tmp_path_factory.mktemp('grid') / 'run'
    env = prepare_subset(path)
    build_grid(path.parent / 'run_k2_cache', path / 'grid', path / 'linelist.txt')
    set_keyword(path, 'PATH_SYN', 'grid/list.syn')     # relative path, with parallel workers (NPROC = 2)
    env['PATH'] = '/usr/bin:/bin:' + str(Path(sys.executable).parent)   # no K2 needed with a grid
    return run_salvador(path, env)


def test_grid_run_needs_no_k2(grid_run):
    assert grid_run.returncode == 0, grid_run.output
    assert grid_run.k2_calls == []


def test_grid_ew_is_close_to_k2(grid_run):
    # EW of the best fit on the normalized synthetic (continuum = 1): before, with the continuum
    # taken at the edges of the fit region, it could be much smaller or even negative
    k2 = {r[1]: float(r[4]) for r in rows(SUBSET / 'k2_ew.fit')}
    grid = {r[1]: r for r in rows(grid_run.path / f'{STAR}.fit')}
    compared = 0
    for line, ew_k2 in k2.items():
        if line in grid and grid[line][3] != '99.99':
            assert float(grid[line][4]) == pytest.approx(ew_k2, rel=0.2), line
            compared += 1
    assert compared >= 3


def test_grid_ew_never_negative(grid_run):
    assert all(float(r[4]) >= 0 for r in rows(grid_run.path / f'{STAR}.fit'))


def test_merge_of_grid_results(tmp_path):
    # .fit of a grid run (EP not available, some negative EWs of the previous version): the merged
    # file and its plot must be created (before: "Axis limits cannot be NaN or Inf")
    shutil.copy(TESTS / 'data' / 'merge_grid' / '47Tuc_12272_up.fit', tmp_path)
    (tmp_path / 'tomerge.txt').write_text('47Tuc_12272_up.fit\nmerged.fit\n')
    answers = 'tomerge.txt\n1\n0\n3\n0\n8\n-6\n-4.5\n'
    proc = subprocess.run([sys.executable, '-m', 'salvador.merge_fit'], cwd=tmp_path, input=answers, text=True,
                          capture_output=True, env={**os.environ, 'MPLBACKEND': 'Agg'}, timeout=300)
    assert proc.returncode == 0
    assert 'Error' not in proc.stdout + proc.stderr, proc.stdout + proc.stderr
    assert (tmp_path / 'merged.pdf').exists()
    final = {r[1]: r for r in rows(tmp_path / 'abundance_final_merged.txt')}
    assert final['26.00'][2] == '27' and final['20.00'][2] == '6'


def test_line_at_the_edge_of_the_grid_is_rejected(grid_run):
    # 7496.103 (Ti): chi2 minimum at the lowest abundance of the grid -> rejected, as with K2
    # (out of limit in the automatic mode)
    row = {r[1]: r for r in rows(grid_run.path / f'{STAR}.fit')}['7496.103']
    assert row[3] == '99.99' and row[-1] == '-3'
    logs = ''.join(f.read_text() for f in (grid_run.path / 'salvador_work').glob('*.log'))   # NPROC = 2
    assert 'edge of the grid' in grid_run.output + logs
