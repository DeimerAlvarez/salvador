# salvador

[![tests](https://github.com/DeimerAlvarez/salvador/actions/workflows/tests.yml/badge.svg)](https://github.com/DeimerAlvarez/salvador/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**salvador** measures chemical abundances of stars by spectral synthesis. For every line of a
linelist it finds the abundance that minimises the χ² between the observed line and synthetic
spectra computed with SYNTHE (Kurucz 2005) on ATLAS9 model atmospheres, in which only the
abundance of the element under analysis is varied. The Kurucz codes are run through
[K2](https://sites.google.com/view/koala-database/software), written by Alessio Mucciarelli.
salvador can also use a grid of precomputed synthetic spectra instead of K2.

Besides abundances (`ABU`, `ABU2` for line wings, `MOLEC`, `ABU_GF`, `ABUF`) it measures
isotopic ratios (`ISOR`) and projected rotational velocities (`ROTAT`). For each star it gives
[Fe/H], [X/Fe] and absolute abundances with their errors: internal (line-to-line scatter or a Monte
Carlo on the observed SNR) and, optionally, those due to the atmospheric parameters.
The atmospheric parameters must be known: salvador does not determine them.

It works in automatic or interactive mode, analyses several lines in parallel (automatic mode),
and writes a PDF with the fit of every line.

> **Status:** 1.0.0. The full manual is [docs/salvador_instructions.pdf](docs/salvador_instructions.pdf)
> (its installation section, Sect. 2, still describes the previous scripts: use the instructions below).

## Requirements

* Python ≥ 3.10 on Linux (macOS should work but is not tested).
* **K2 and the Kurucz codes**, to compute the synthetic spectra (not needed if you only use a
  grid of precomputed synthetics, keyword `PATH_SYN`). K2 will be distributed from the
  [KOALA software page](https://sites.google.com/view/koala-database/software); follow its
  instructions. The command `k2.py` must be in your `PATH`.
* The Python packages (numpy, scipy, pandas, matplotlib, astropy, specutils ≥ 2.0, PyAstronomy,
  pypdf) are installed automatically by `pip`.

## Installation

In the Python environment you use (a virtual environment is recommended):

```bash
pip install git+https://github.com/DeimerAlvarez/salvador.git
salvador --check          # Python, packages, K2 found in the PATH
```

For development:

```bash
git clone https://github.com/DeimerAlvarez/salvador.git
cd salvador
pip install -e ".[test]"
pytest
```

To uninstall: `pip uninstall salvador-spec`.

## Quick start: the tutorial

The folder [`tutorial/`](tutorial) contains a complete example: a spectrum of the red giant
1500138 of the globular cluster M54 (GIRAFFE HR18 setup), its parameters, a linelist of 27
lines and the continuum windows.

1. Edit `tutorial/linelist.xkur`: write the paths of the Kurucz atomic and molecular linelists
   on your computer (see Sect. 3.3 of the manual).
2. Run salvador in the folder:
   ```bash
   cd tutorial
   salvador
   ```
3. Compare your results with those in `tutorial/expected/`:
   ```bash
   mkdir regression_ref && cp expected/M54_HR18_1500138.fit expected/abundance_final.txt regression_ref/
   salvador-regression          # ALL RESULTS IDENTICAL
   ```

The first run computes all the synthetic spectra and takes a few minutes; they are kept in
`synth_out/` and reused by the following runs.

## How it works

salvador is always run in the folder of your data, where it reads the input files and writes the
results.

| Input file | Content |
| --- | --- |
| `input.salvador` | the settings of the run (`KEYWORD = value`): file of the stars (`OBS`), normalization (`NORM`), window (`WINDOW`), interactive or automatic (`INTER`), sigma clipping (`SCLIP`) and optional keywords (`NPROC`, `PATH_SYN`, `SOLAR_REF`, ...) |
| `lista.stars` (any name, `OBS`) | one row per spectrum: name, T<sub>eff</sub>, log g, [M/H], alpha code (1–5 for [α/Fe] = +0.4 … −0.4), v<sub>t</sub>, RV, abundance-variation file, linelist, continuum windows, and optionally the errors on T<sub>eff</sub>, log g, v<sub>t</sub> |
| linelist | one row per line: keyword (`ABU`, `ABU2`, `MOLEC`, `ABU_GF`, `ABUF`, `ISOR`, `ROTAT`), wavelength, species code (26.00 = Fe I, 26.01 = Fe II), resolution, edges of the fit (0 0 = automatic), initial abundance variation |
| continuum windows | λ<sub>1</sub> λ<sub>2</sub> per row, for `NORM = 2` (you can select them with `bob cows` of [bobspec](https://github.com/DeimerAlvarez/bobspec)) |
| `linelist.xkur` | the Kurucz linelists used by K2 |

| Output | Content |
| --- | --- |
| `<spectrum>.fit` | the results of every line (abundance, EW, EP, log gf, Monte Carlo error, parameter errors, flag) |
| `<spectrum>_isor.fit`, `<spectrum>_rot.fit` | isotopic ratios; rotational velocities |
| `abundance_final.txt` (`_isor.txt`) | [Fe/H], [X/Fe], absolute abundances and errors of every star |
| `<spectrum>.pdf` | the fit of every line and the Fe diagnostic plots |
| `synth_out/` | the synthetic spectra computed by K2 (reused by later runs) and the best fit of every line |

All the keywords and files are described in the [manual](docs/salvador_instructions.pdf).

## Commands

| Command | |
| --- | --- |
| `salvador [input file]` | the analysis (default input file: `input.salvador`); also `python -m salvador` |
| `salvador --check` | checks the installation |
| `salvador-merge` | merges the `.fit` files of the same star and computes its abundances again, optionally with an external [Fe/H] (example list: [docs/example_merge_fit.txt](docs/example_merge_fit.txt)) |
| `salvador-regression [save]` | saves the results of a run as a reference / compares a new run with it |
| `salvador-check-k2 [N]` | checks that the fast K2 calls of salvador give the same spectra and EWs as the standard ones |

The functions are in the package `salvador`, one module per topic
([docs/FUNCTION_MAP.txt](docs/FUNCTION_MAP.txt)); `from salvador.utilities import *` gives all of them.

## Solar abundances

The table `solar_abu.txt` installed with salvador contains the solar abundances of Caffau et al.
(2011) and Lodders (2010) (default), Grevesse & Sauval (1998), Magg et al. (2022) and Asplund et
al. (2009); the keyword `SOLAR_REF` chooses the reference.

## Data

The spectra of the tutorial and of the tests are taken from the ESO archive.

## Citing salvador

A paper describing salvador is in preparation. Until then, please cite this repository (see
[CITATION.cff](CITATION.cff)), as well as K2 and the Kurucz codes.

## License

MIT, see [LICENSE](LICENSE). Bug reports and contributions are welcome: see
[CONTRIBUTING.md](CONTRIBUTING.md).
