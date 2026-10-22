# Changelog

## 1.0.0 (unreleased)

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
