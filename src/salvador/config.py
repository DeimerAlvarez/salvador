"""Settings of a run (input.salvador -> cfg), warnings policy, constants, paths and the
matplotlib backend. Other modules import cfg and plt from here.

Importing salvador has no side effects: input.salvador is read at the first use of a setting
(or by cfg.load), and the warnings policy and the matplotlib backend are set by the salvador
command (configure_warnings, select_backend), not at import.
"""
import matplotlib
import difflib
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

### WARNINGS
### Before, every warning was hidden (warnings.filterwarnings('ignore')), including the ones that
### signal a problem in a fit (division by zero, NaN, a fit that did not converge). Now each kind of
### warning is shown once (per place in the code) in one short line, and only harmless ones are
### hidden: library deprecation notices and the messages listed in HARMLESS_WARNINGS. To hide
### another warning you know is harmless, add the beginning of its message to the list.
HARMLESS_WARNINGS = [
    'No artists with labels found to put in legend',
]


def _short_warning(message, category, filename, lineno, line=None):
    """Format of the warnings: one short line with the type, the file and the line of code."""
    return f'WARNING ({category.__name__}, {os.path.basename(filename)}:{lineno}): {message}\n'


def configure_warnings() -> None:
    """Warnings policy of a salvador run (see HARMLESS_WARNINGS). Called by the salvador command;
    it changes the global warning filters, so it is not done at import."""
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    for msg in HARMLESS_WARNINGS:
        warnings.filterwarnings('ignore', message=re.escape(msg))
    warnings.formatwarning = _short_warning


# --------------- important values
c = 299792.458  # speed of light in km/s
ksize = 101 # kernel size for the median filter used to find the continuum points in the spectrum.
init_ans = 'zzz'

### CONFIGURATION: the keywords of input.salvador
###
### Every setting of a run lives in one object, cfg (class Config), read by load_config(). In the
### code a setting is always written cfg.<name> (cfg.window, cfg.type_norm, ...), so it cannot be
### confused with a local variable or a parameter with a similar name.
### To add a keyword: one row in KEYWORDS and one field in Config.

#       keyword in input.salvador: (field of Config, conversion, default; None = mandatory)
KEYWORDS = {
    # --- mandatory
    'OBS':          ('input_stars',    str,   None),    # file with the spectra and their parameters (lista.stars)
    'NORM':         ('type_norm',      int,   None),    # normalization: 0 none, 1 scaled to a synthetic, 2 continuum windows, 3 to be implemented
    'WINDOW':       ('window',         int,   None),    # width [A] of the region around each line
    'INTER':        ('inter',          str,   None),    # y = interactive, n = automatic
    'SCLIP':        ('sclip',          float, None),    # sigma-clipping of the line abundances in the final mean
    # --- optional
    'PATH_SYN':     ('path_syn',       str,   None),    # list of a grid of precomputed synthetic spectra (grid mode)
    'SOLAR_REF':    ('solar_ref',      int,   2),       # column of solar_abu.txt used as solar reference
    'LIMIT_ABU':    ('limit_abu',      int,   3),       # automatic mode: max variation [+-dex] before skipping a line
    'CALC_BEST':    ('calc_best',      str,   'y'),     # y = compute the +-0.2 dex synthetics for the plot
    'EWR_MIN':      ('ewr_min',        float, -6.0),    # minimum reduced EW, log(EW/lambda)
    'EWR_MAX':      ('ewr_max',        float, -4.4),    # maximum reduced EW
    'EP_MIN':       ('ep_min',         float, 0.0),     # minimum excitation potential [eV]
    'EP_MAX':       ('ep_max',         float, 10.0),    # maximum excitation potential [eV]
    'VERBOSE':      ('verbose',        int,   0),       # 0 or 1
    'AUTO_RES':     ('auto_res_limit', lambda v: float(v) / 100.0, 0.0),    # automatic resolution within +-X %
    'NPIX_RV':      ('n_pix_rv',       float, 0.0),     # local RV correction: tolerance [pixels]
    'PLOT_TYPE':    ('plot_type',      int,   0),       # style of the observed spectrum in the plots
    'NPROC':        ('nproc',          int,   1),       # lines analysed in parallel (automatic mode only)
    'ROT_ABU_SPAN': ('rot_abu_span',   float, 0.5),     # ROTAT: abundance range [+-dex] (0 = fixed)
    'ROT_MIN':      ('rot_min',        float, 0.0),     # ROTAT: minimum rotation [km/s]
    'ROT_MAX':      ('rot_max',        float, 60.0),    # ROTAT: maximum rotation [km/s]
    'ROT_NMC':      ('rot_nmc',        int,   100),     # ROTAT: Monte Carlo iterations
    'ROT_STEP':     ('rot_step',       float, 10.0),    # ROTAT: step of the coarse rotation grid [km/s]
}


@dataclass
class Config:
    """Settings of a salvador run (see KEYWORDS for the meaning of each one)."""
    input_stars: str
    type_norm: int
    window: int
    inter: str
    sclip: float
    path_syn: Optional[str] = None
    solar_ref: int = 2
    limit_abu: int = 3
    calc_best: str = 'y'
    ewr_min: float = -6.0
    ewr_max: float = -4.4
    ep_min: float = 0.0
    ep_max: float = 10.0
    verbose: int = 0
    auto_res_limit: float = 0.0
    n_pix_rv: float = 0.0
    plot_type: int = 0
    nproc: int = 1
    rot_abu_span: float = 0.5
    rot_min: float = 0.0
    rot_max: float = 60.0
    rot_nmc: int = 100
    rot_step: float = 10.0

    @property
    def grid(self):
        """'y' when a grid of precomputed synthetic spectra is used (PATH_SYN given)."""
        return 'y' if self.path_syn else 'n'

    @property
    def limit_iter(self):
        """Iterations of the automatic mode allowed by LIMIT_ABU (default +-3 dex = 11 iterations)."""
        return (self.limit_abu * 5) - 4


def load_config(path='input.salvador'):
    """Read, convert and check the keywords of input.salvador. Stops with a clear message if a
    mandatory keyword is missing or a value is not valid; warns about unknown keywords."""
    raw = {}
    try:
        with open(path, 'r') as file:
            for row in file:
                if not row.strip() or row.strip().startswith('#'):
                    continue
                parts = row.split('=', 1)
                if len(parts) == 2:
                    raw[parts[0].strip()] = parts[1].strip()
    except FileNotFoundError:
        sys.exit(f'Error: the input file {path} does not exist in {Path.cwd()}')

    # a misspelled keyword was silently ignored before (e.g. CAL_BEST for CALC_BEST)
    for key in raw:
        if key not in KEYWORDS:
            close = difflib.get_close_matches(key.upper(), list(KEYWORDS), n=1)
            hint = f' Did you mean {close[0]}?' if close else ''
            print(f"WARNING ({path}): unknown keyword '{key}' is ignored.{hint}")

    values = {}
    for key, (field, convert, default) in KEYWORDS.items():
        if key not in raw:
            if default is None and key != 'PATH_SYN':
                sys.exit(f"Error: mandatory keyword '{key}' not found in {path}!")
            values[field] = default
            continue
        try:
            values[field] = convert(raw[key])
        except ValueError:
            sys.exit(f"Error: {key} = '{raw[key]}' in {path} is not a valid value")
    cfg = Config(**values)
    # FIX: absolute path, so the grid is found also by the parallel workers (NPROC > 1), which run
    # in their own directories (before: every line was rejected with a relative PATH_SYN)
    if cfg.path_syn:
        cfg.path_syn = str(Path(cfg.path_syn).resolve())

    cfg.inter = cfg.inter.strip().lower()
    cfg.calc_best = cfg.calc_best.strip().lower()
    if cfg.inter not in ('y', 'n'):
        sys.exit(f"Error: INTER must be y or n in {path} (found '{cfg.inter}')")
    if cfg.calc_best not in ('y', 'n'):
        sys.exit(f"Error: CALC_BEST must be y or n in {path} (found '{cfg.calc_best}')")
    if cfg.type_norm not in (0, 1, 2, 3):
        sys.exit(f"Error: NORM must be 0, 1, 2 or 3 in {path} (found {cfg.type_norm})")
    if cfg.type_norm == 3:
        sys.exit('Error: NORM = 3 is still under development and cannot be used yet (use NORM = 0, 1 or 2)')
    return cfg


class LazyConfig:
    """The settings of the run (a Config), read from input.salvador at the first use.

    Every module imports this one object (from .config import cfg) and reads cfg.<setting>, so
    importing salvador needs no input file. cfg.load(path) reads a given file; cfg.use(config)
    sets the settings directly (e.g. in tests).
    """

    def __init__(self) -> None:
        object.__setattr__(self, '_config', None)

    def load(self, path: str | Path = 'input.salvador') -> Config:
        """Read the settings from path (see load_config) and use them."""
        object.__setattr__(self, '_config', load_config(path))
        return self._config

    def use(self, config: Config) -> None:
        """Use the given settings."""
        object.__setattr__(self, '_config', config)

    @property
    def loaded(self) -> bool:
        """True once the settings have been read or set."""
        return self._config is not None

    def __getattr__(self, name: str):
        if name.startswith('__'):       # copy/pickle protocols: no file reading for them
            raise AttributeError(name)
        if self._config is None:
            self.load()
        return getattr(self._config, name)

    def __setattr__(self, name: str, value) -> None:
        if self._config is None:
            self.load()
        setattr(self._config, name, value)

    def __repr__(self) -> str:
        return f'LazyConfig({self._config!r})'


cfg = LazyConfig()


def select_backend(inter: str) -> None:
    """Non-interactive matplotlib backend (Agg) for automatic runs (INTER = n). Called by the
    salvador command before any figure is created; interactive runs keep the default backend."""
    if inter == 'n':
        matplotlib.use('Agg')


import matplotlib.pyplot as plt # noqa: E402, F401 -- other modules import plt from config

### The default solar references are those by Caffau & Lodders (COL = 2). They can be changed.


### DIRECTORIES
path_tmp = Path.cwd() / 'tmp_plots'
### Working directory of the run. Shared files are addressed with absolute paths, so they stay
### correct when parallel workers run in their own scratch directories (see NPROC).
BASE_DIR = Path.cwd()
SYNTH_OUT = BASE_DIR / 'synth_out'
WORK_DIR = BASE_DIR / 'salvador_work'

plot_file = 'nomi_plot.txt'
path_plot_file = path_tmp / plot_file

### unnecessary files to be deleted at the end of the execution of the program
patterns_to_delete = ["guess*", "input.cog", "output.cog", "tmp.78", "interpol.odfbig", "interpol.ros", "interpol.sh", "xatlas9.sh", "*_cl_*", "hammer.nb", "iso.tmp", \
        "Abundances_log", "info_ew.txt", "single.cog*", "abu_var_k2.tmp"]

### temporary copy of the abundance-variation file passed to K2 (see autok)
abu_var_tmp_name = 'abu_var_k2.tmp'
