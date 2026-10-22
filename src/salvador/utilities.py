"""Compatibility module: `from salvador.utilities import *` gives all the functions of salvador
(the former `from utilities_salvador import *`), for your own scripts. They live in the modules of
the package salvador, one per topic (see the docstring at the top of each module):

    salvador/config.py         settings (input.salvador -> cfg), warnings, constants, paths
    salvador/elements.py       solar abundances, element symbols, model abundances
    salvador/k2_interface.py   K2: synthetic spectra, EWs, trimmed linelists, DELTA files
    salvador/spectra_io.py     reading observed spectra and synthetic grids
    salvador/fit_files.py      the .fit result files
    salvador/normalization.py  continuum normalization
    salvador/fitting.py        resolution, convolution, abundance fit, Monte Carlo
    salvador/rotation.py       rotational velocity
    salvador/plotting.py       plots of each line
    salvador/summary.py        final abundances (mean, sigma-clipping, errors)
    salvador/line_analysis.py  analysis of one line (measure_line, interactive menu)
    salvador/parallel.py       parallel analysis (NPROC)
"""
from .config import *
from .elements import *
from .spectra_io import *
from .k2_interface import *
from .fit_files import *
from .normalization import *
from .fitting import *
from .rotation import *
from .plotting import *
from .summary import *
from .line_analysis import *
from .parallel import *
