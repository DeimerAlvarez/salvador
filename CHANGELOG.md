# Changelog

## 1.3.0

### Added

* `ROTAT`: macroturbulence (radial-tangential profile), keyword `ROT_VMAC` in `input.salvador`,
  **mandatory when the linelist has `ROTAT` lines** (the run stops before any K2 call otherwise):
  a value in km/s, or a relation evaluated for each star from its Teff and logg: `VF05` (Valenti &
  Fischer 2005, dwarfs and subgiants), `D14` (Doyle et al. 2014, eq. 8, dwarfs), `B16` (Brewer et
  al. 2016, eq. 2, dwarfs and subgiants; 4 km/s for logg < 3), `C08` (Carney et al. 2008, eq. 5, red
  giants, also metal-poor; not RHB stars). A warning is printed outside the calibration of the
  relation. Before, no macroturbulence: the v sin i also contained it (`ROT_VMAC = 0` gives the same).
* `ROTAT`: optional keyword `ROT_VMAC_ERR` (km/s; default: the scatter of the relation, 0.8 km/s
  for `C08` and 0.73 km/s for `D14`, otherwise 0, with a printed note). The fit is repeated with
  vmac -/+ its error, and the resulting error of v sin i is in the new column `eROTv` of
  `<spectrum>_rot.fit`; it is added in quadrature to the Monte Carlo error to decide the upper
  limits. At R <= 40000 v sin i and vmac cannot be separated: below ~10-15 km/s the v sin i depends
  on the vmac assumed (simulations of a metal-poor giant with the Mg b lines: with vmac too small by
  5 km/s, a star without rotation gives v sin i = 6-7 km/s).
* `ROTAT`: the macroturbulence used for each star is printed at the start of its analysis.
* `ROTAT`: keyword `ROT_LD` for the linear limb-darkening coefficient (default 0.6, as before).
* `ROTAT`: upper and lower limits. A v sin i smaller than its error, or than `ROT_UL_FWHM` (default
  0.5) times the instrumental FWHM, is an upper limit; a v sin i at `ROT_MAX` is a lower limit. New
  columns in `<spectrum>_rot.fit`: `eROTv` (above), `VMAC` (the macroturbulence used) and `LIM`
  (`D`, `UL`, `LL`).
* `ROTAT`: a warning when the fitted v sin i is larger than the half-width of the fit region, and a
  note when the best fit is at v sin i = 0 (resolution or macroturbulence possibly too large).

### Changed

* **Action needed**: an `input.salvador` used with `ROTAT` lines must now contain `ROT_VMAC`
  (`ROT_VMAC = 0` reproduces the treatment of the previous versions, without macroturbulence).
* `ROTAT`: the broadening (instrumental, macroturbulence, rotation) is computed on a fine grid uniform
  in velocity (0.25 km/s), with the kernels integrated over the pixels, for every v sin i of the grid
  (`ROT_STEP`, now 1 km/s by default): no interpolation in v sin i. The model is compared with the
  observed pixels. Before, the rotation was applied on the observed pixels (kernel sampled at a few
  points for small v sin i) every 10 km/s and interpolated: in simulations at R = 40000 a v sin i of
  4-6 km/s was underestimated by about 2 km/s. The best-fit spectrum of the plot is computed from the
  same models (one K2 call less per line).

### Fixed

* The tests could not be collected with pytest 9.2 (it reads private attributes of the objects
  of `conftest.py`, and the settings tried to read `input.salvador`): the settings are now read
  only for public names.

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
