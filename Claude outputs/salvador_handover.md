# salvador: handover notes (state on 17 Nov 2026)

These notes let a new conversation pick up the salvador project where this one stopped. They cover the code, everything changed from 1.0.0 to 1.4.0, the decisions taken (and why), how the work was done, and what is still open.

**For the new conversation:** read these notes first, then `CHANGELOG.md`, `README.md` and `docs/FUNCTION_MAP.txt` in the repository. The manual (`docs/salvador_instructions.pdf`, source `salvador_v7.tex`) describes every keyword and file.

---

## 1. People, roles and working rules

- **User:** Deimer Antonio Alvarez Garay (call him Deimer). Postdoc at INAF – Arcetri, working in stellar spectroscopy (normalization, RVs, chemical abundances, model atmospheres). His main subject is globular clusters (M 54, ω Cen, ...).
- **salvador:** his own code. Claude has been improving, debugging and packaging it, with the A&A paper on salvador as the final goal.
- **K2:** written by Alessio Mucciarelli, a collaborator. It is the Python manager of the Kurucz codes (ATLAS9/12, SYNTHE). Deimer works with him on K2 and has permission to try to improve it.

**Working rules (standing instructions):**

- Before creating, modifying, moving or deleting any file: explain the plan, show the diff for code, and **wait for an explicit OK**. Don't install packages system-wide or change the environment without asking.
- State the design plan before writing code; ask when something is ambiguous.
- Code style:
  - PEP 8, type hints, NumPy/Google-style docstrings
  - no bare `except`
  - pytest tests for every change
- Prefer numpy, scipy, astropy and matplotlib; ask before adding a dependency.
- Keep existing behaviour unless told otherwise.
- Ubuntu Linux (26.04), Python only, **venv + pip, never conda**. The main venv is `.work` and is active by default.
- Be concise. For science: state assumptions and approximations, flag uncertainty, and **never invent references, values or citations**.
- Reply in English, or in the language Deimer writes in.
- His email is private: never send it to external services without explicit approval.

---

## 2. The code today

### Repository and installation

- **GitHub:** `DeimerAlvarez/salvador`, branch `main`.
  - Last pushed commit known here: `245fe85` (tag `v1.3.0`).
  - Version 1.4.0 is applied and working in Deimer's local clone. Whether it is committed, tagged and pushed is to be checked (see §6).
- **pip name:** `salvador-spec`; import package `salvador`; MIT licence; Python ≥ 3.10.
- **Dependencies:** numpy, scipy, pandas, matplotlib, astropy, specutils ≥ 2.0, PyAstronomy, pypdf. Tests use pytest and pytest-cov.
- **Development install:** `pip install -e ".[test]"`, then `pytest`. A GitHub Actions workflow (`tests.yml`) runs the tests.
- **Repository folders:**

  | Folder | Content |
  |---|---|
  | `src/salvador/` | the package |
  | `tests/` | tests and test data |
  | `tutorial/` | M54 star 1500138, GIRAFFE HR18, 27 lines, `expected/` results |
  | `docs/` | manual PDF, `FUNCTION_MAP.txt`, `example_merge_fit.txt` |
  | `Claude outputs/` | Deimer keeps the patches and manual sources from these sessions here |

- **Local paths on the laptop** (from earlier notes; verify them):
  - K2: `/home/deimer/softw/K2/k2.py`
  - salvador: `~/softw/salvador`, editable install in `.work`
  - ODFs: `/home/deimer/softw/ATMOS/CL_ap04/odf/`
  - The keyword test cases were originally run in `/home/deimer/Desktop/Postdoc/Software/salvador/tests_keywords/`.
  - Drive folder with the K2 curves of growth used for the slope calibration: `/work/salvador/cogs`.

### Commands

| Command | Purpose |
|---|---|
| `salvador [input file]` | the analysis (default `input.salvador`); also `python -m salvador`; `--check`, `--version` |
| `salvador-merge [list] [options]` | merges the `.fit` files of a star and recomputes its abundances, optionally with an external [Fe/H] and its error |
| `salvador-regression [save]` | saves the results of a run as a reference / compares a new run with it (by column names) |
| `salvador-check-k2 [N]` | checks that salvador's fast K2 calls give the same spectra, EWs and slopes as standard K2 |

`salvador-merge` options: `--solar 1-4`, `--flags keep|recalc`, `--sclip`, `--ep MIN MAX`. The strong lines are selected with either `--ewr-min` + `--slope-min`, or `--ewr MIN MAX`. Settings that are not given are asked.

### Modules (`src/salvador/`)

| Module | Content |
|---|---|
| `cli.py` | `salvador` command: loop over stars and lines, `analyse_line`, `_analyse_line` (selection flags, parameter errors), `write_line_result`, final abundances |
| `config.py` | `KEYWORDS` table, `Config`, `cfg` (lazy), `EWR_MAX_DEFAULT = -4.4`, `ew_selection()` |
| `k2_interface.py` | K2 in-process, `autok` (synthetics), `eqw_k2`/`read_info_ew` (EW, EP, log gf, slope), trimmed linelists, `K2FormatError`; switches `K2_IN_PROCESS`, `TRIM_LINELIST` |
| `line_analysis.py` | `measure_line`: normalization, local RV, first fit, interactive menu, automatic iterations, MC errors, plot |
| `fitting.py` | χ² abundance determination, convolution, resolution (`new_res_est`), MC, `template_shift`/`ShiftResult` (1.4.0) |
| `summary.py` | means, σ-clipping, errors, `star_abundance_rows`, selection flags (`ew_selection_flag`, `use_saturated_species`, ...) |
| `fit_files.py` | `.fit` header and columns (`FIT_COLS`), reading any version by column names (`read_fit_rows`, `read_fit_table`) |
| `merge_fit.py` | `salvador-merge` |
| `regression_check.py` | `salvador-regression` |
| `check_k2_speedup.py` | `salvador-check-k2` |
| `rotation.py` | `ROTAT`: broadening kernels, macroturbulence relations, `analysis_rotat` |
| `plotting.py` | `PlotManager` (per-line plot), `NullPlotter`, `wavelength_ticks` (1.4.0) |
| `normalization.py` | NORM 1/2 (3 under development) |
| `spectra_io.py` | observed spectra, synthetics, grid (`PATH_SYN`) |
| `elements.py` | solar abundances (`data/solar_abu.txt`), symbols (26.00 → `Fe`, 26.01 → `FeII`) |
| `parallel.py` | `NPROC` workers |
| `utilities.py` | `from salvador.utilities import *` compatibility |

### Input keywords (`input.salvador`)

- **Mandatory:** `OBS`, `NORM`, `WINDOW`, `INTER`, `SCLIP`.
- **Optional (defaults):**
  - general: `PATH_SYN`, `SOLAR_REF` (2), `LIMIT_ABU` (3), `CALC_BEST` (y), `VERBOSE` (0), `AUTO_RES` (%), `NPIX_RV` (0), `PLOT_TYPE` (0), `NPROC` (1)
  - line selection: `EWR_MIN` (−6.0), `EWR_MAX` (none, 1.4.0), `COG_SLOPE_MIN` (0.35, 1.4.0), `EP_MIN` (0), `EP_MAX` (10)
  - ROTAT: `ROT_ABU_SPAN` (0.5), `ROT_MIN` (0), `ROT_MAX` (60), `ROT_NMC` (100), `ROT_STEP` (1), `ROT_VMAC` (mandatory with ROTAT lines: a value in km/s or `VF05`, `D14`, `B16`, `C08`), `ROT_VMAC_ERR`, `ROT_LD` (0.6), `ROT_UL_FWHM` (0.5)

**Linelist keywords:** `ABU`, `ABU2` (wings, 9 columns), `MOLEC`, `ABU_GF`, `ABUF`, `ISOR`, `ROTAT` (optional columns 8–11).

**`lista.stars`:** 13 columns turn on the parameter errors (ΔTeff, Δlogg, Δvt); with 12 columns a warning is printed and no errors are computed.

### Flags in the `.fit` files

| Flag | Meaning |
|---|---|
| 1 | used |
| 0 | rejected by σ-clipping |
| −1 | rejected by the EW limits (EWR) or EW = 0 |
| −2 | rejected by EP |
| −3 | rejected by abundance (out of limits / failed) |
| −4 | rejected by COG slope (1.4.0) |

---

## 3. Version history (what was done in these sessions)

### Refactor (Oct 2026, before 1.0.0)

- The original scripts `salvador.py` + `utilities_salvador.py` were split into modules. The code was translated to English, every function got a docstring, and all settings moved into one `cfg` object.
- Renames: `gestore_keyword` → `measure_line`, `leggi_file_delta` → `read_delta_file`.
- Speed: K2 runs in-process; linelists are trimmed to the window; the Monte Carlo is seeded (reproducible); `NPROC` parallelism; no plots for the parameter-error reruns; grid cache. A 26-line test run went from 21 min to 4 min 20 s (sequential).

### 1.0.0 (first public release)

- Installable package and commands as above. `cows.py` was dropped in favour of `bob cows` from bobspec. `solar_abu.txt` is installed with the package. Importing salvador has no side effects.
- `ROTAT` error rounded to an integer in km/s.
- Tutorial results identical to the scripts.

### 1.0.1 (fixes)

- **Grid mode:** the EW on the best-fit synthetic used the fit-region edges as continuum, so EWs were underestimated or negative. The continuum is now 1.
- **Grid mode:** a χ² minimum at the grid edge now gives flag −3.
- **merge:** the Fe plot crashed when no EP was available.

### 1.1.0 (fixes)

- **Parameter errors:**
  - they are signed (result with the parameter increased − nominal), so species that change opposite to Fe are handled in [X/Fe];
  - a 99.99 no longer enters the means;
  - lines rejected in interactive mode keep the three error columns.
- **Grid:** the perturbed synthetics are taken from the grid, and only the star's [M/H] and α are used.
- **Synthetic names:** log gf changes, ISOR and ISOF are now in the names (wrong reuse before).
- **ISOR:** a rejected line is 99.99 (was 100).
- **Robustness:** NaN pixels are ignored; spectrum paths with folders or dots work; an error in one line skips that line only (the run stops after 3 in a row).
- **Interactive commands** (`i`, `dw`, `aw`, `res`, `c`) were fixed.
- **salvador-merge** uses the same functions as salvador. It got command-line options and the external [Fe/H] error, and can merge files with and without errors.
- Lines are processed in wavelength order.
- New warnings: changed linelists vs `synth_out`; continuum windows all on one side of a line.

### 1.2.0

- d_logg is computed for **every** line (before: ionized species, MOLEC, ABU2 and ISOR only).

### 1.3.0: ROTAT revision

Details are in the project doc `claude/rotat_evaluation_1.3.0.md`. Summary:

- **New keywords:**
  - `ROT_VMAC` (radial-tangential macroturbulence; a value or the relations VF05/D14/B16/C08) is mandatory with ROTAT lines; `ROT_VMAC = 0` gives the old behaviour.
  - `ROT_VMAC_ERR`: the fit is repeated with vmac ± its error, giving the column `eROTv`.
  - `ROT_LD`, `ROT_UL_FWHM`.
- **Exact broadening:** on a 0.25 km/s velocity grid with pixel-integrated kernels, for each v sin i. `ROT_STEP` is now 1 km/s; there is no interpolation in v sin i.
- **Limits and output:**
  - upper limit if v sin i < its error, or < 0.5 × the instrumental FWHM; lower limit at `ROT_MAX`;
  - new `_rot.fit` columns `eROTv`, `VMAC`, `LIM`;
  - warnings when v sin i > half the fit region, and when the best fit is at 0.
- **Simulations** (ROTAT test case, dwarf with Teff 5970, logg 4.10, [M/H] −1.30, S/N 100):
  - the old method underestimated v sin i by ~2 km/s at 4–6 km/s; a missing vmac biases v sin i upwards;
  - at R ≤ 40000, vmac and v sin i cannot be separated below ~10–15 km/s.
- Also fixed: pytest 9.2 collection error (`conftest`/`cfg` private attributes).

### 1.4.0: curve-of-growth slope, plus the later additions

Needs **K2 of November 2026**. Its `eqw`/`eqwb` write `info_ew.txt` with 5 columns, `λ EW EP loggf S` (e.g. `6300.310 36.10 0.000 -9.776 0.935`). S = d log(EW/λ)/dA(X) is computed by centred differences at ±0.1 dex, and is `nan` when undefined. With a 4-column file, salvador stops with exactly: `salvador needs EW file with the COG slope`.

**Calibration of the default** (954 Fe I lines, 4000–7000 Å, K2 COGs of a giant: Teff 4500, logg 0.70, [M/H] −1.5, vt 2.0, R 30000; abundances within ±0.4 dex of the model):

| S | log(EW/λ) median | EW at 5000 Å | below linear COG | note |
|---|---|---|---|---|
| 0.80 | −5.24 | 29 mÅ | 0.05 dex | leaves the linear part |
| 0.67 | −5.09 | 41 mÅ | 0.10 dex | maximum curvature |
| 0.40 | −4.84 | 73 mÅ | 0.32 dex | |
| 0.35 | −4.79 | 80 mÅ | 0.39 dex | **default** |
| 0.30 | −4.75 | 89 mÅ | 0.49 dex | flat part |

- The transition is S ≈ 0.91 → 0.35. S = 0.35 corresponds to −4.78 at 4000–4600 Å and −4.84 at 5400–7000 Å, so a fixed EWR cannot follow it.
- An EW error of 10% gives about 0.041/S dex in the abundance.
- In the damping part S rises again to 0.4–0.5 (EW ~450–700 mÅ). Hence a fixed safety limit EWR ≤ −4.4 always applies in slope mode.
- Lines with hfs or isotopic structure keep a larger S than single lines of the same EW.
- Median relation (logistic, used by the test stand-in): S = 0.0401 + 0.9599 / (1 + exp((y + 4.9518)/0.2117)), with y = log(EW/λ).
- Figure: `cog_slope.pdf`, made by `figure_centred.py` (scratch script, see §8).

**Selection rules** (`Config.ew_selection()` → (ewr_min, ewr_max, slope_min)):

- **Default:** flag −1 if EWR < `EWR_MIN`, EW ≤ 0, or EWR > `EWR_MAX_DEFAULT` (−4.4); flag −4 if S < `COG_SLOPE_MIN`. To change −4.4, edit `EWR_MAX_DEFAULT` near the top of `src/salvador/config.py`.
- **`EWR_MAX = X` in `input.salvador`:** the old behaviour (EWR range only; the slope is not used).
- **Grid mode (no slope):** EWR range, with `EWR_MAX` defaulting to −4.4.
- **Missing slope** (nan, stored as 99.999): judged by EWR only, with a note.
- **Not judged:** EW 99.99 (molecules, ABU2, ...). The selection applies to ABU, ABU_GF and ABUF (eqwb), not to ABU2, MOLEC, ISOR or ROTAT.
- **Species rule (added 17 Nov):** if all lines of a species that have an abundance are flagged −4 and none has flag 0, they are used anyway (flag 0, then σ-clipping), with the note `Note (<star>): K (19.00): all its lines are saturated (COG slope < COG_SLOPE_MIN): N line(s) used anyway`. `salvador-merge --flags recalc` applies the same rule (`summary.use_saturated_species`).
- **Errors for −4 lines (added 17 Nov):** d_Teff, d_logg and d_vt are computed for −4 lines too. Other rejected lines get 0.00.
- **Fe fix:** if all Fe I lines are rejected, [Fe/H] = [M/H] of the model (merge: not available). Before, [Mg/Fe] came out as −100.51.

**`.fit` format (1.4.0):**

- Columns: `SPECTRUM LINE ELEM ABU EW EWR SLOPE EP log_gf eMC [d_Teff d_logg d_vt] flag_line`; EWR is 99.99 and SLOPE 99.999 when not available.
- The header gives the flags, a `SELECTION:` line and an EWR/SLOPE explanation. The default text is `EWR >= -6.00 and COG slope >= 0.35 (EWR <= -4.40 for the very strong lines; saturated lines used only for the species without other lines)`.
- salvador-merge and salvador-regression read files of any version by column names; for old files, EWR is computed and SLOPE is set to 99.999.

**Local RV correction `NPIX_RV` (rewritten 17 Nov, `fitting.template_shift`):**

- **Model:** obs = c + a·(T(λ−d) − 1), where T is the synthetic at the starting abundance, convolved to the line resolution. The template is `autok(param1, delta, window)`; in grid mode it is the middle synthetic of the grid.
  - The fit region is fit1 − λ/R … fit2 + λ/R.
  - d is scanned over ±(NPIX_RV + 0.5) pixels in steps of 0.02 pixels, and c, a are solved by linear least squares at each step.
  - The minimum is refined with a parabola; the error is sqrt(2σ²/curvature).
- **Applied only if** the minimum is not at the edge of the scan, |d| ≤ NPIX_RV pixels, a > 0, and depth/noise ≥ 3. Otherwise a message (always printed) gives the reason and the best shift.
- Not used for ABU2 lines (their regions exclude the core).
- The old method compared the line centre (Gaussian fit and half depth) with the laboratory wavelength and applied any shift within the tolerance.
- **Tests:**
  - tutorial: a line-to-line scatter of the shifts similar to the old method (≈0.3–0.4 pixel), and it flags the unreliable weak Ti lines;
  - ABU2 case: the Fe line is found 0.04 pixel to the blue (old method: 0.13 pixel to the red) → A(Fe) +0.01 dex;
  - MOLEC 4323.5 and ISOR 4230.3: the template finds shifts of 1.4 ± 0.4 and 1.2 ± 0.1 pixels, so nothing is applied with NPIX_RV = 0.5 (the old method applied 0.28 pixel in MOLEC). **To be checked on real data.**
- A smooth whole-spectrum shift map was tried and rejected: A-band and sky residuals, and the real shifts vary on 10–20 Å scales.

**Plots:**

- The legend of the main panel shows `R = <resolution used>` (measured, with AUTO_RES).
- `wavelength_ticks`: windows up to ~20 Å are unchanged (1 Å steps, or 2 Å when ≥ 10 ticks). Wider windows (up to 50 Å) get labels every 5 or 10 Å, at multiples of the step and at most 10 labels, with minor ticks every 1 Å. Fixes the label overlap seen in `ngc6352_09.pdf`.
- Zoom panel: the first label is left-aligned and the last right-aligned.

**Other 1.4.0 changes:**

- Removed the model-check shortcut `K2_FAST_MODEL_CHECK` (K2 now checks itself).
- The `linelist.xkur` parser skips `#` comments and accepts lower-case markers.
- A line with EW 99.99 is not judged by the EWR limits.
- `salvador-check-k2` compares the slopes too.

**Results with the real new K2** (Deimer's tutorial run, before the species rule):

- Fe I goes from 14 to 8 lines; [Fe/H] from −1.58 to −1.51.
- The rejected saturated Fe lines average A(Fe) = 5.85 vs 6.01 for the used ones (difference 0.16 ± 0.07, 2.3σ).
- K I 7699 has S = 0.183: EW ±10% → ±0.23 dex, vt ±0.2 km/s → ±0.13 dex, Teff ±100 K → ±0.18 dex. With the species rule it is now used (it is the only K line).

---

## 4. K2: facts that matter for salvador

- salvador calls `k2.main()` in-process, on `./input.k2`. The `synthe`, `eqw`, `eqwb` and `atlas9` syntax is unchanged; `VERBOSE`/`OUTPUT`/`CHECKSYN` are accepted.
- K2 catches command errors itself and prints `@@@ ERROR @@@`. salvador detects a failure from the missing output (e.g. a missing `.nb` file → RuntimeError with hints).
- K2 (Nov 2026) checks whether the model atmosphere exists before interpolating ODFs.
- In the abundance file, when DELTA/ABS/GF rows are given for the same element, **the last row wins**. salvador appends the line's own variation to a temporary copy of the user's file.
- atlas9 requires an **integer vt** code in the model names (e.g. `k195`).
- Synthetic names in `synth_out/`: `am154t4248g077k195_cl_ap04_Fe_p004_7709_7715.nb` = model, element, delta code (p004 = +0.04; m037 = −0.37), window. ABS and ISOR codes are also in the name.

---

## 5. Tests

- 293 tests: `test_config`, `test_elements_and_names`, `test_files`, `test_numerics`, `test_rotation`, `test_fixes_1_1`, `test_cog_slope`, `test_end_to_end`, `test_grid`, `test_keywords`.
- **Running them in the cloud workspace:** the container was memory-limited, so the suite ran in two groups (the first six files, then the other four), about 3 min in total. Locally they can run with a single `pytest`.
- **K2 stand-in** (`tests/fake_k2/k2.py`, put in the PATH by `conftest.prepare_case`):
  - `synthe` copies the cached synthetics (`FAKE_K2_SYNTH`, from each case's `synth_out.tar.gz`);
  - `eqw` writes EW, EP and log gf from a `.fit` file (`FAKE_K2_EW`), plus a slope from the logistic relation;
  - environment variables:
    - `FAKE_K2_SLOPE` (a number or `nan`): fixed slope for all lines
    - `FAKE_K2_EW_COLUMNS=4`: write the old K2 format
    - `FAKE_K2_INTERPOLATE=1`: interpolate a missing synthetic in abundance, or extrapolate by ≤ 0.05 dex; used only for the ABU2 case
    - `FAKE_K2_LOG`: log of the K2 calls
- **Test data:** `tests/data/tutorial_subset` (5 lines of the tutorial star), `tests/data/keywords/{ABU2,ISOR,MOLEC,ROTAT}` (expected from real runs of 1.0.0, updated where versions changed results), `tests/data/merge_grid`.
- **Known, harmless:** the ISOR expected `.fit` header lacks the line `FLAG = -3 (REJ. ISOR)` that runs write; tests compare data rows only.
- **Regression checks:** every release was compared with the previous one on all keyword cases and the tutorial subset, and differences were explained (e.g. ABU2 +0.01 dex from the new RV method). `pyflakes` is clean except the expected star-import warnings in `utilities.py` and an unused `plt` import in `config.py` (both pre-existing).

---

## 6. Status and next steps

### Finish the 1.4.0 release (Deimer)

1. Check the local clone: `git status` / `git log`. The 1.4.0 changes were applied from patches; are they committed?
2. Regenerate `tutorial/expected/` with the real K2: run `salvador` in `tutorial/`, check the results, then copy `M54_HR18_1500138.fit` and `abundance_final.txt` to `expected/` (or use `salvador-regression save`).
3. Manual: put `cog_slope.pdf` in the manual's `figures/` folder, compile `salvador_v7.tex` (AASTeX), and replace `docs/salvador_instructions.pdf`.
4. Commit, tag `v1.4.0`, push, and make a GitHub release.

### Good first tasks now that Claude can run salvador and K2 locally

- Run the full test suite and the tutorial with the real K2; run `salvador-check-k2`.
- Re-run the keyword test cases (ABU2, ISOR, MOLEC, ROTAT) with the real K2 to refresh their `synth_out.tar.gz` caches and expected files. The ABU2 case would then no longer need `FAKE_K2_INTERPOLATE`.
- Check the new template RV correction on real spectra: compare it with the old method on several stars and setups. Look in particular at the >1-pixel shifts found for the MOLEC (CH band, LR2) and ISOR cases: is the star's RV off, or does the fit region or template need care for molecular bands?
- Check the wide-window ticks on a real 40–50 Å window (the `ngc6352_09` case).
- Check the species rule on stars where only strong lines exist for some elements (K I 7699, Ba II, Na D, Mg b in cool giants).

### Longer term

- **A&A paper on salvador**, edited in Overleaf; the name is always lowercase "salvador".
  - Cite K2 as "Mucciarelli et al., in prep." for now; salvador was earlier cited as "Alvarez Garay et al., in prep.".
  - Scope: mainly FGK stars, optical and IR; M giants possible but slow.
  - Points other codes don't offer:
    - abundances from log gf variation (consistency with the model atmosphere)
    - automatic resolution per line
    - ABU2 (saturated lines from their wings)
    - isotopic ratios
    - rotation with abundance free or fixed
    - errors from the atmospheric parameters (signed)
    - the COG-slope selection
  - Validation: benchmark stars and simulations with synthetic spectra.
  - A section, written with Alessio Mucciarelli, on a new GALA version that derives atmospheric parameters.
  - The introduction must cite existing codes. The literature is in the project files: `fama.pdf`, `Blanco_Cuaresma_14.pdf`, `Blanco_Cuaresma_19.pdf`, `GALA.pdf`, `mygisfos.pdf`, `Mucciarelli_26.pdf`. Also `salvador_instructions.pdf` and `abu_var.txt` (an example abundance-variation file).
  - The 1.3.0 ROTAT simulations and the 1.4.0 COG calibration (table above) are paper material.
- **Improve K2** (separate thread, with Alessio Mucciarelli).
- **NORM = 3** is still under development.

---

## 7. Decisions and ideas that were declined (and why)

- **COG_SLOPE_MIN default 0.35, not 0.40:** 0.35 is where the flat part begins, 0.39 dex below the linear COG (see the table). Users can change it.
- **Lowering COG_SLOPE_MIN to 0.15** to keep K I 7699 (declined): it would also admit the saturated lines of species that have good unsaturated lines. The species rule keeps K, Ba II etc. only when nothing better exists, and the −4 errors show their uncertainty.
- **A "depth sensitivity" criterion** from the ±0.2 dex synthetics, removing `CALC_BEST` (declined): S already *is* the sensitivity of the line strength to the abundance. A saturated line is still sensitive, but its abundance is less precise (errors scale as 1/S). A second criterion would duplicate S and add complexity. `CALC_BEST` stays.
- **Smooth whole-spectrum RV map** (declined): see §3, 1.4.0 RV.
- **Old K2 (4-column `info_ew.txt`):** not supported in 1.4.0, as Deimer decided; salvador stops with the message above.
- **Version stays 1.4.0** for all the November additions (the species rule, −4 errors, template RV and ticks).

---

## 8. Deliverables and how the work was done

- **Workflow in this account:**
  - Claude worked in a scratch clone of the GitHub repo in the cloud workspace, with separate venvs for the current and the previous version, so results could be compared.
  - Deimer applied the changes himself as `git apply` patches, after reviewing them, then ran `pip install -e ".[test]"` and `pytest`.
  - Every patch was also checked by applying it to a fresh clone and running the full test suite there.
- **Files delivered** (in Deimer's downloads / `Claude outputs/`):
  - `salvador_1.0.1.patch`, `_1.1.0`, `_1.2.0`, `_1.3.0` (+ review diffs, vmac proposal)
  - `salvador_1.4.0.patch` (full patch against 1.3.0, version of 17 Nov)
  - `salvador_1.4.0_update.patch` (on top of the 16 Nov version)
  - `salvador_1.4.0_files.zip` (the 27 final files)
  - manual sources `salvador_v3.tex` … `salvador_v7.tex`
  - `cog_slope.pdf`, `cog_slope_threshold.png`
- **Manual:** v7 was generated from v6 by two scripted, verified find-and-replace passes:
  - new section "Selection of the lines by their strength" (`sect:cog_slope`), with the calibration table and the figure;
  - updated keywords (`COG_SLOPE_MIN`, `EWR_MAX`, `NPIX_RV`), `.fit` columns, flags, speed section, grid section, tools, troubleshooting messages and the salvador-merge appendix.

  It compiles with no new overfull boxes.
- **Scripts that exist only in the cloud workspace** (they will be lost with this account unless exported):
  - the COG analysis (`load.py`, `pairs.py`, `relation.py`, `centred.py`) and `figure_centred.py` for `cog_slope.pdf`
  - the RV tests (`rv_compare.py`, `rv_map.py`)
  - the manual update scripts
  - the FUNCTION_MAP regenerator (`regen_map.py`)

  The numbers they produced are recorded above. The figure can be remade from the K2 COG files in `/work/salvador/cogs`.
