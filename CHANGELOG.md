# Changelog

## 1.5.0

### Added

* Selection of the lines by the sensitivity of their profile to the abundance: keyword
  `SIGMA_FIT_MAX` (default 0.20 dex; 0 = not used). For each line salvador computes the error of the
  abundance due to the noise of the profile, sigma_fit = (1/SNR) / sqrt(sum_i (dF_i/dA)^2), from the
  synthetics at the best abundance +- 0.2 dex (K2, with `CALC_BEST = y`; otherwise those of the fit,
  interpolated in abundance) on the pixels of the chi2 region, and the SNR measured near the line. A
  line with sigma_fit > `SIGMA_FIT_MAX` is not used for the abundances (new flag -5, REJ. PROFILE
  SENSITIVITY): its profile hardly changes with the abundance. It works also with a grid of
  synthetics. It is applied to the lines of `ABU`, `ABU_GF`, and `ABUF`, as the other selections.
* `.fit` files: new column `SIGFIT` (sigma_fit, 99.999 when not available) after `SLOPE`, the flag
  -5 and the selection in the header. The files of the previous versions are still read and merged.
* For Fe I, salvador prints the mean abundance of the saturated lines (COG slope < 0.35) minus that
  of the other usable lines, with its error, and a warning when they differ by more than twice the
  error: the saturated lines are more sensitive to the microturbulence and to the damping.
* `salvador-merge`: `--sigma-fit-max` (default 0.20, 0 = not used) for `--flags recalc`;
  `--slope-min 0` (or 0 as the answer) does not use the COG slope.

### Changed

* `COG_SLOPE_MIN` is optional, without default (before: 0.35): the saturated lines whose profile
  still changes with the abundance are used. In a giant of M54 (Teff 4248 K, logg 0.77, [M/H]
  -1.54, R = 18000, HR18) the six Fe I lines rejected by the slope in 1.4.0 (S = 0.20-0.30) have
  sigma_fit = 0.012-0.017 dex, as the unsaturated ones: their core still changes by ~0.05 (about 10
  times the noise) for +-0.2 dex. They give an abundance lower by 0.16 +- 0.07 dex than the other lines
  (a sign of a slightly too high vt, or of the damping): the new check prints it. With
  `COG_SLOPE_MIN = 0.35` the selection of 1.4.0 is obtained (plus `SIGMA_FIT_MAX`).

### Fixed

* SNR near a line (used by the Monte Carlo errors and by sigma_fit): DER_SNR (Stoehr et al. 2008),
  the median flux divided by the noise from the median of |2 f_i - f_(i-2) - f_(i+2)|, within each
  continuum window with `NORM = 2`, and on the flux within +-20 A with the other normalizations (or
  when no window is within +-20 A); the mean/std of the 3-sigma-clipped pixels only with fewer than
  10 differences. Before: the mean/std of the pixels, so a bad pixel or a structure in a window (a
  weak line, a wave of the normalization), or the lines within +-20 A with the other
  normalizations, gave a much lower SNR, hence larger Monte Carlo errors and sigma_fit. The
  abundances do not change. In the M54 giant the SNR goes from 110-160 to 175-250 (from 23 to 150
  at 7852-7855 A: Monte Carlo error of Ti I 7852 from 0.09 to 0.01 dex). In the PEPSI giant 1r of
  NGC 6749 the window 8779.7-8789.8 A holds a structure (flux 0.70-1.15) and the SNR of Fe I 8793
  goes from 9.5 to 79 (sigma_fit from 0.26 to 0.03 dex: the line is used). In the test cases ISOR
  (CS 29497-034, `NORM = 2`) and ABU2 (`NORM = 1`) the SNR goes from 4 to 50 and from 33 to 146.
  The noise is assumed uncorrelated beyond 2 pixels: in strongly oversampled spectra the SNR is
  overestimated.

## 1.4.0

Needs the version of K2 of November 2026 (or later): its `eqw` writes in `info_ew.txt` the slope
of the curve of growth of the line, S = d log(EW/lambda) / d A(X), computed with A(X) +- 0.1 dex
(5 columns: wavelength, EW, EP, log gf, S). With an older K2 the run stops with the message
"salvador needs EW file with the COG slope".

### Added

* Selection of the strong lines by the slope of the curve of growth: keyword `COG_SLOPE_MIN`
  (default 0.35). A line with S < `COG_SLOPE_MIN` is not used for the abundances (new flag -4, REJ.
  COG SLOPE), unless its species has no other usable line (see below). The default comes from the
  curves of growth of 954 Fe I lines of a metal-poor giant computed with K2 (Teff 4500 K, logg
  0.70, [M/H] -1.5, vt 2.0 km/s): a line leaves the linear part
  between S ~ 0.9 and 0.35 (largest curvature at S ~ 0.67); S = 0.35 corresponds to log(EW/lambda)
  ~ -4.79 (~80 mA at 5000 A, 0.39 dex below the linear curve of growth), with a dependence on the
  wavelength that a fixed EWR limit does not follow. Lines with hyperfine or isotopic splitting are
  less saturated than single lines of the same EW, and keep a larger S.
* `EWR_MAX` is now optional, without default: if it is given in `input.salvador`, the lines are
  selected by `EWR_MIN` and `EWR_MAX` and the slope is not used (as in the previous versions).
  Without `EWR_MAX`, the selection is `EWR_MIN` and `COG_SLOPE_MIN`, plus a fixed limit EWR <= -4.4
  against the very strong lines, whose slope rises again in the damping part (S = 0.4-0.5 at EW ~
  450-700 mA). That limit is `EWR_MAX_DEFAULT` in `src/salvador/config.py`. With a grid of
  synthetics (`PATH_SYN`, no slope), `EWR_MAX` (default -4.4) is used. A line without slope (K2
  could not compute it) is selected by its EWR only, with a note.
* `.fit` files: new columns `EWR` = log(EW/lambda) and `SLOPE` after `EW` (99.99/99.999 when not
  available), the flag -4 and a header line with the selection used. The selection is also printed
  at the start of the run.
* Saturated lines of the species without other lines: when all the lines of a species that have an
  abundance are flagged -4 (no line with flag 0), they are used anyway (e.g. K I 7699, the Ba II
  lines of a cool giant), with a note in the output; for a species with unsaturated lines the
  saturated ones stay rejected. The same in `salvador-merge --flags recalc`.
* The parameter errors (`d_Teff`, `d_logg`, `d_vt`) are computed also for the lines with flag -4:
  for the saturated lines used as above they give the larger uncertainty of the abundance (e.g. K I
  7699 in the tutorial star, S = 0.18: Teff +-100 K -> +-0.18 dex, vt +-0.2 km/s -> +-0.13 dex);
  for the others they show how sensitive they are.
* Plots: the main panel has a legend with the resolution used for the line (with `AUTO_RES`, the
  measured one).

### Changed

* `salvador-merge` reads the `.fit` files by the names of their columns, so files of any version
  (with or without `EWR` and `SLOPE`) can be merged; the merged file has the columns of 1.4.0 (EWR
  computed from EW for the old files, SLOPE 99.999). With `--flags recalc` the selection is the
  same as in salvador: new options `--ewr-min` and `--slope-min`, or `--ewr MIN MAX` for the EWR
  limits only; when asked, the choice between the two.
* `salvador-regression` compares the columns with the same name when the columns of a file changed
  (e.g. a reference made with 1.3.0), and lists the columns present in only one of the two files.
* `salvador-check-k2` also compares the slopes.
* Local RV correction (`NPIX_RV`): the shift is now measured by fitting the broadened synthetic
  spectrum of the starting abundance to the observed line, c + a (T(lambda - d) - 1), with the
  continuum level c and the depth scale a free, d scanned in steps of 0.02 pixels and refined with
  a parabola (error from the curvature of the chi2). The template contains the blends and the
  hyperfine/isotopic structure, so blended, weak and asymmetric lines no longer bias the shift. The
  shift is applied only if it is within `NPIX_RV` pixels, the minimum is not at the edge of the scan
  and the line is detected (depth > 3 times the noise); otherwise a message says why it was not
  applied (printed also without `VERBOSE`). Before, the centre of the observed line (Gaussian fit
  and half depth) was compared with the laboratory wavelength, and the shift was applied whenever
  it was within `NPIX_RV`. Not used for `ABU2` lines (their fit regions exclude the core).
* Plots: in the main panel, windows wider than ~20 A (up to 50 A) have a label every 5 or 10 A (at
  most 10 labels, at multiples of the step) and minor ticks every 1 A: before, the labels overlapped.
  Smaller windows are as before. In the zoom panel the first and last labels are aligned inside the
  panel.
* The shortcut `K2_FAST_MODEL_CHECK` (check of the existing model atmosphere before the ODF
  interpolation, done by wrapping a K2 method) was removed: K2 now does it itself.
* A line without EW (99.99, e.g. K2 failed) is not judged by the EW limits (before: its EWR was
  computed from 99.99 mA).

### Fixed

* When all the Fe I lines of a star were rejected (e.g. by the EW limits), [Fe/H] was 99.99 and
  entered the [X/Fe] (e.g. [Mg/Fe] = -100.51): now the [M/H] of the model is used, as for a star
  without Fe I lines (salvador-merge: [X/Fe] not available).
* The trimmed linelists ignore, as K2 does, the comment lines of `linelist.xkur` (lines starting
  with `#` that are not section markers) and recognise the markers in lower case (before, the
  comments were taken as file names).

### Tests

* The stand-in for K2 writes the slope (from the median curve of growth of the Fe I lines, or
  `FAKE_K2_SLOPE`), can write the old 4-column file (`FAKE_K2_EW_COLUMNS = 4`), and with
  `FAKE_K2_INTERPOLATE = 1` interpolates in the abundance a synthetic missing from the cache.
* Case ABU2: `EWR_MAX = -4.4` in its `input.salvador` (selection of the previous versions); with the
  new RV correction the Fe line is found 0.04 pixels to the blue (before: 0.13 pixels to the red):
  A(Fe) +0.01 dex, d_Teff and d_vt -0.01 dex (expected files updated; the fits at the new
  abundances use interpolated synthetics). In the cases ISOR and MOLEC the template fit gives
  shifts larger than `NPIX_RV`, so no correction is applied: results unchanged.

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
