# Contributing to salvador

Bug reports, questions and contributions are welcome.

## Reporting a problem

Open an issue at <https://github.com/DeimerAlvarez/salvador/issues> with:

* what you ran and the full output of salvador (with `NPROC > 1`, also the logs in `salvador_work/`);
* the output of `salvador --check` (versions of salvador, Python and the packages);
* if possible, the input files that reproduce the problem (`input.salvador`, the list of stars,
  the linelist, the spectrum).

## Development

```bash
git clone https://github.com/DeimerAlvarez/salvador.git
cd salvador
pip install -e ".[test]"
pytest
```

The tests do not need K2: a stand-in (`tests/fake_k2/k2.py`) returns the synthetic spectra and the
equivalent widths of real runs: a 5-line subset of the tutorial (`tests/data/tutorial_subset`) and one
case for each of the keywords ABU2, ISOR, MOLEC and ROTAT (`tests/data/keywords`). To add a case, put
its input files, `synth_out.tar.gz` (the synthetic spectra it needs) and `expected/` (the results) in
a new folder of `tests/data/keywords` and add it to `CASES` in `tests/test_keywords.py`.

Before and after a change that should not alter the results, run salvador on your own data and
compare the results with `salvador-regression save` / `salvador-regression`.

## Code organisation

* `src/salvador/cli.py`: the `salvador` command (loop over the stars and the lines, final abundances).
* `src/salvador/line_analysis.py`: the analysis of one line (automatic iterations, interactive menu).
* One module per topic for the rest (configuration, K2, spectra, normalization, fitting, rotation,
  plots, summary, parallel analysis); see `docs/FUNCTION_MAP.txt`.
* New settings: one row in `KEYWORDS` and one field in `Config` (`src/salvador/config.py`), and a
  description in the manual.
