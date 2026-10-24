# Changelog

## 1.2.0

### Changed

* The error due to logg (d_logg) is computed for every line, also for the neutral species (before:
  only for ionized species and for `MOLEC`, `ABU2` and `ISOR` lines; 0.00 for the others). The
  errors of [Fe/H] and [X/Fe] due to the atmospheric parameters change, and a run with the
  parameter errors needs one more fit for each line of a neutral species.

## 1.1.0

### Fixed

* Errors due to the atmospheric parameters:
  * they keep their sign (result with the parameter increased − nominal result). The errors of
    [X/Fe] take into account species that change in the opposite way to Fe I (e.g. ionized
    species); before, with absolute values, they were underestimated for those species. The
    d_Teff, d_logg and d_vt columns of the `.fit` files can now be negative.
  * a changed fit that could not be measured (99.99) no longer enters the mean of the species
    (one line in four gave an error of 25 dex); the errors that depend on a missing value are 99.99.
  * a line rejected in the interactive mode (`r`) has the three error columns (99.99), so all the
    rows of the `.fit` file have the same columns.
* Grid of synthetic spectra (`PATH_SYN`):
  * the parameter errors were always 0.00 (the nominal synthetics were used again). The synthetics
    with Teff + ΔT, logg + Δlogg, vt + Δvt are taken from the grid; without them the error is 99.99
    with a message.
  * only the synthetics with the [M/H] and the alpha-enhancement of the star are used (before,
    spectra of different model atmospheres were mixed); if a line is in more than one wavelength
    range of the grid, one range is used.
* Names of the synthetic spectra in `synth_out/`: a change of log gf (`ABU_GF`, `GF`) and a change of
  abundance had the same name, and the ISOR (and ISOF) rows of the abundance-variation file were not
  in the name, so synthetics computed with another isotopic ratio were reused. The names of the
  synthetics with only DELTA and ABS rows are unchanged (they are still reused).
* `ISOR`: a rejected line was written as 100 and entered the mean ratio; it is now 99.99 with flag −3.
* Pixels with NaN or infinite values in the observed spectrum are not used (before, the run
  stopped, or the abundance was searched in the wrong direction).
* Spectra given with a folder or with dots in the name (`./star.fits`, `../data/star.v2.fits`): the
  results are named after the spectrum (`star.fit`, `star.v2.fit`) in the working folder; before,
  `.fit` and `.pdf` (empty name) or a crash with K2.
* An unexpected error in a line skips that line with a message (before, the whole run stopped); the
  run stops after 3 lines in a row stopped by an error.
* Interactive mode: `i` restores the spectrum of the first fit, with the local RV correction; `dw`
  accepts only the number of a window and keeps at least 2 windows; `aw`, `res` and `c` accept only
  valid values; the χ² panel of MOLEC lines has its labels.
* `salvador-merge`: with an external [Fe/H] and parameter errors, all the errors of [X/Fe] were 99.99:
  the error of the external [Fe/H] is now used (third column of the list file, or asked); without
  Fe I lines and without an external [Fe/H], [X/Fe] is 99.99 (before, meaningless values); `.fit`
  files with and without parameter errors can be merged; the EW limits are not applied to lines
  without an EW (molecules).

### Changed

* The error due to logg is computed also for `MOLEC` and `ABU2` lines (not only for ionized species).
* The lines are analysed in order of wavelength: the rows of the `.fit` files and the pages of the PDF
  are in order of wavelength (before: in the order of the linelist and of the file names).
* `salvador-merge` uses the same functions as `salvador` for the abundances, the errors and the Fe
  plot (before, a copy of them): the two programs give the same numbers. The settings can be given on
  the command line (`--solar`, `--flags`, `--sclip`, `--ep`, `--ewr`); the missing ones are asked, and
  wrong choices are refused.
* Warnings: when the linelists changed since the synthetic spectra of `synth_out/` were computed (they
  would be reused), and when all the continuum windows near a line (`NORM = 2`) are on one side of it.

## 1.0.1

### Fixed

* Grid of synthetic spectra (`PATH_SYN`): the EW of a line, measured on the best-fit synthetic,
  used as continuum the flux at the two edges of the fit region, so it was underestimated and could
  be negative (edges in the wings of the line or on nearby lines). The synthetic spectra are
  normalized: the continuum is now 1. On the tutorial the EWs agree with those of K2 within a few
  per cent.
* `salvador-merge`: the Fe plot was not created, with the error "Axis limits cannot be NaN or Inf",
  when the excitation potentials were not available (results of a grid run); lines with a negative
  or null EW are left out of the reduced-EW panel.
* The Fe plot of `salvador` no longer stops the run if no excitation potential is available.

### Changed

* Grid of synthetic spectra, automatic mode: a line whose χ² minimum is at the first or the last
  abundance of the grid is rejected (abundance 99.99, flag −3), because the real minimum may be
  outside the grid, as the lines "out of limit" in the automatic mode with K2.

## 1.0.0

First public release: the scripts `salvador.py`, `utilities_salvador.py` and `salvador_lib/` become
the installable package `salvador` (`pip install salvador-spec`). The analysis is unchanged: on the
tutorial the results are identical to those of the previous version.

### Changed

* Installation with `pip`; the `Makefile` and `requirements.txt` are removed and nothing is added
  to the shell configuration files.
* Commands: `salvador` (was `salvador.py`), `salvador-merge` (`merge_fit.py`),
  `salvador-regression` (`regression_check.py`), `salvador-check-k2` (`check_k2_speedup.py`),
  `salvador --check` (`make check`), `salvador --version`. `salvador` accepts the name of the input
  file (default `input.salvador`).
* `from salvador.utilities import *` replaces `from utilities_salvador import *`.
* `solar_abu.txt` is installed with the package.
* Importing salvador has no side effects: `input.salvador` is read at the first use of a setting,
  and the warnings policy and the matplotlib backend are set by the `salvador` command.
* Python ≥ 3.10 is required (specutils ≥ 2.0).
* `ROTAT`: the error of the rotational velocity (eROT in `<spectrum>_rot.fit`, on the terminal and
  in the plots) is an integer in km/s, like the velocity itself (rounded half up).
* `cows.py` is no longer part of salvador: continuum windows can be selected with `bob cows` of
  [bobspec](https://github.com/DeimerAlvarez/bobspec).

### Fixed

* The Monte Carlo error was 0.00 when no continuum window (`NORM = 2`) was within ±20 Å of the
  line (undefined SNR). The SNR is now estimated from the flux around the line, and the error is
  99.99 if the SNR cannot be estimated.
* Grid mode (`PATH_SYN`): the synthetic spectra are selected by exact match of T<sub>eff</sub>,
  log g, v<sub>t</sub> and element in their names (before, by substring: for instance a C line also
  used the Ca spectra, and log g 1.10 matched v<sub>t</sub> 1.10).
* Grid mode with `NPROC > 1`: every line was rejected when `PATH_SYN` was a relative path.
* Grid mode with `NPROC > 1` no longer needs K2.
* Lines with an unknown keyword, an element missing from `solar_abu.txt`, `ABU2` without the
  second pair of edges, or `ROTAT` in grid mode are skipped with a message (before, the run
  stopped with an error); rows with the same wavelength as an earlier row are reported.
* The errors due to the atmospheric parameters were also appended to the last row of the other
  `.fit` file (`.fit` / `_isor.fit`) in some interactive sessions.
* The clean-up at the end of a run deletes only the temporary files written during the run
  (before, also files of the user named like `guess*` or `*_cl_*`).
* `salvador` stops at once with a clear message if `k2.py` is not in the `PATH` (and no grid is used).
* Stars with an alpha code outside 1–5 are skipped with a message; errors on T<sub>eff</sub>
  written as `50.0` are accepted.
* The Fe diagnostic figure of each star is closed after saving.
* `salvador-merge`: when the flags are recalculated, lines rejected with the previous EWR/EP limits
  are considered again; a species without valid lines no longer stops the analysis of the star.
