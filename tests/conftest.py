"""Shared fixtures of the salvador tests.

The end-to-end tests run the salvador command with a stand-in for K2 (tests/fake_k2/k2.py) that
returns the synthetic spectra and equivalent widths of real runs, so neither K2 nor the Kurucz
codes are needed. The data of those runs are in tests/data: a 5-line subset of the tutorial
(tutorial_subset) and one case for each of the keywords ABU2, ISOR, MOLEC and ROTAT (keywords/).
Each folder has the input files, synth_out.tar.gz (the synthetic spectra the run needs) and
expected/ (the results).
"""
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from salvador.config import Config, cfg

TESTS = Path(__file__).resolve().parent
SUBSET = TESTS / 'data' / 'tutorial_subset'
KEYWORDS = TESTS / 'data' / 'keywords'
FAKE_K2 = TESTS / 'fake_k2'


@pytest.fixture
def settings():
    """Default settings of a run (NORM = 2, automatic), without an input.salvador file."""
    previous = cfg._config
    config = Config(input_stars='lista.stars', type_norm=2, window=5, inter='n', sclip=3.0)
    cfg.use(config)
    yield config
    cfg.use(previous)


@dataclass
class Run:
    """A salvador run: its folder, the console output, the exit code and the calls to K2."""

    path: Path
    output: str
    returncode: int
    k2_calls: list[str]

    def data_rows(self, name: str) -> list[list[str]]:
        """Rows of an output file without the comment lines, split in fields."""
        return [r.split() for r in (self.path / name).read_text().splitlines() if r.strip() and not r.startswith('#')]


def prepare_subset(path: Path, k2_in_path: bool = True) -> dict[str, str]:
    """Copy the tutorial subset into path; return the environment for running salvador there."""
    return prepare_case(SUBSET, path, SUBSET / 'k2_ew.fit', k2_in_path)


def prepare_case(source: Path, path: Path, ew_file: Path | None = None, k2_in_path: bool = True) -> dict[str, str]:
    """Copy the input files of a test case into path; return the environment for running salvador
    there, with the stand-in for K2 (synthetic spectra from source/synth_out.tar.gz, equivalent
    widths from ew_file)."""
    shutil.copytree(source, path, ignore=shutil.ignore_patterns('expected', 'synth_out.tar.gz', 'k2_ew.fit'))
    cache = path.parent / f'{path.name}_k2_cache'
    with tarfile.open(source / 'synth_out.tar.gz') as tar:
        if hasattr(tarfile, 'data_filter'):        # Python >= 3.10.12
            tar.extractall(cache, filter='data')
        else:
            tar.extractall(cache)
    fake_bin = path.parent / f'{path.name}_k2_bin'
    fake_bin.mkdir()
    shutil.copy(FAKE_K2 / 'k2.py', fake_bin / 'k2.py')
    (fake_bin / 'k2.py').chmod(0o755)

    env = dict(os.environ, MPLBACKEND='Agg', FAKE_K2_SYNTH=str(cache), FAKE_K2_EW=str(ew_file or os.devnull),
               FAKE_K2_LOG=str(path / 'fake_k2.log'))
    if k2_in_path:
        env['PATH'] = f'{fake_bin}{os.pathsep}{env.get("PATH", "")}'
    return env


def run_salvador(path: Path, env: dict[str, str], *args: str, answers: str | None = None) -> Run:
    """Run the salvador command (python -m salvador) in path, in a new process. answers: what is
    typed at the prompts of the interactive mode."""
    proc = subprocess.run([sys.executable, '-m', 'salvador', *args], cwd=path, env=env, input=answers,
                          capture_output=True, text=True, timeout=600)
    log = path / 'fake_k2.log'
    calls = log.read_text().splitlines() if log.exists() else []
    return Run(path, proc.stdout + proc.stderr, proc.returncode, calls)


def set_keyword(path: Path, key: str, value: str) -> None:
    """Set (or add) a keyword of input.salvador."""
    file = path / 'input.salvador'
    rows = [r for r in file.read_text().splitlines() if r.split('=')[0].strip() != key]
    file.write_text('\n'.join(rows + [f'{key} = {value}']) + '\n')


@pytest.fixture(scope='session')
def parallel_run(tmp_path_factory) -> Run:
    """The tutorial subset analysed with NPROC = 2."""
    path = tmp_path_factory.mktemp('parallel') / 'run'
    env = prepare_subset(path)
    return run_salvador(path, env)


@pytest.fixture(scope='session')
def sequential_run(tmp_path_factory) -> Run:
    """The tutorial subset analysed with NPROC = 1."""
    path = tmp_path_factory.mktemp('sequential') / 'run'
    env = prepare_subset(path)
    set_keyword(path, 'NPROC', '1')
    return run_salvador(path, env)
