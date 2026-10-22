"""Parallel analysis of the lines (NPROC > 1): worker set-up, model preparation, clean-up."""
import importlib
import multiprocessing
import os
import shutil
import sys

from .config import WORK_DIR, cfg
from .k2_interface import _trim_cache, run_k2

### =============================================================================================
### PARALLEL ANALYSIS OF THE LINES (NPROC > 1, automatic mode only)
### =============================================================================================
### Each line (nominal fit + parameter-error fits) is analysed by a worker process in its own
### scratch directory WORK_DIR/worker_<pid>: K2 and its Fortran codes write fixed-name files
### (input.k2, fort.*, guess_model_*) in the current directory, so workers must not share it.
### Shared outputs use absolute paths (SYNTH_OUT, path_tmp). The results are written by the main
### process in the order of the linelist, so the output files are the same as with NPROC = 1.

def parallel_available(n_lines):
    """Number of worker processes to use (1 = sequential)."""
    if cfg.nproc <= 1 or n_lines < 2:
        return 1
    if cfg.inter != 'n':
        print('NPROC > 1 is used only in automatic mode (INTER = n): lines analysed one by one')
        return 1
    if 'fork' not in multiprocessing.get_all_start_methods():
        print('Parallel analysis not available on this system: lines analysed one by one')
        return 1
    return min(cfg.nproc, n_lines, os.cpu_count() or 1)


def init_parallel_worker():
    """Runs once in each worker: own scratch directory and own log file."""
    wdir = WORK_DIR / f'worker_{os.getpid()}'
    wdir.mkdir(parents=True, exist_ok=True)
    os.chdir(wdir)
    log = open(WORK_DIR / f'worker_{os.getpid()}.log', 'a', buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log.fileno(), 1)        # also the output of K2 and of the Fortran codes
    os.dup2(log.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)
    _trim_cache.clear()         # trimmed linelists of the parent are not used here


def ensure_models(models):
    """Compute (if missing) the ATLAS9 models needed by the parallel workers, before they start,
    so that two workers never build the same model at the same time. models: list of
    (teff, logg, met, alpha, vt) with the same values used in the K2 calls. K2's 'atlas9' keyword
    is used with the ODF microturbulence chosen as K2's synthe does (closevt(vt, 1)); with
    CHECKMOD = 1 (K2 default) existing models are not recomputed."""
    try:
        closevt = importlib.import_module('ancillary').under_the_carpet().closevt
    except Exception:
        closevt = lambda v, flag: max(i for i in [0, 1, 2, 4, 8] if i <= v)
    seen, rows = set(), []
    for teff, logg, met, alpha, vt in models:
        key = (int(teff), float(logg), float(met), int(alpha), int(closevt(float(vt), 1)))
        if key not in seen:
            seen.add(key)
            rows.append(f'atlas9 {key[0]}  {key[1]}  {key[2]}  {key[3]}  {key[4]} * *\n')
    with open('input.k2', 'w') as f:
        f.writelines(rows)
        f.write('VERBOSE 0\n')
    run_k2()


def cleanup_parallel_work(keep_logs=True):
    """Remove the workers' scratch directories; the logs (one per worker, with the detailed
    output of each line) are kept until the next run unless keep_logs is False."""
    if not WORK_DIR.is_dir():
        return
    if not keep_logs:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        return
    for d in WORK_DIR.glob('worker_*'):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
