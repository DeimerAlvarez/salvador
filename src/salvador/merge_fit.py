"""salvador-merge: merge the .fit files of the same star (e.g. from different settings or
spectral ranges) and compute its final abundances again, optionally with an external [Fe/H].

The input file lists the .fit files of each star, then the name of the merged file, optionally
followed by an external [Fe/H] and its error; stars are separated by '###' (see
docs/example_merge_fit.txt). Results: the merged .fit files, their 3-panel Fe plots and
abundance_final_merged.txt.

The abundances, the errors, the flags and the Fe plot are computed with the same functions as
salvador (salvador.summary, salvador.plotting): the two programs give the same numbers.

Usage (the settings not given on the command line are asked, as before):

    salvador-merge                                   # everything asked
    salvador-merge list.txt --solar 1 --flags keep
    salvador-merge list.txt --solar 1 --flags recalc --sclip 2.5 --ep 0 10 --ewr-min -6 --slope-min 0.35
    salvador-merge list.txt --solar 1 --flags recalc --sclip 2.5 --ep 0 10 --ewr -6 -4.4

NEW (1.4.0): the .fit files are read by the names of their header, so files of any version of
salvador (with or without the columns EWR and SLOPE) can be merged; the merged file has the columns
of this version. With --flags recalc the strong lines are rejected by the slope of the curve of
growth (--slope-min, as COG_SLOPE_MIN) or, with --ewr MIN MAX, by the reduced EW (as EWR_MAX).
"""
import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .config import EWR_MAX_DEFAULT, Config, cfg
from .fit_files import ERR_COLS, FIT_COLS, fit_file_header, read_fit_rows, read_fit_table, replace_flag_column
from .plotting import fe_diagnostic_plot
from .summary import describe_ew_selection, ew_selection_flag, star_abundance_rows, use_saturated_species

### solar references: choice of salvador-merge -> name (column of solar_abu.txt = choice + 1, the
### value of SOLAR_REF in input.salvador)
SOLAR_REFERENCES = {1: 'CL', 2: 'GS98', 3: 'Magg22', 4: 'Asp09'}

COLS_BASE = list(FIT_COLS)         # NEW (1.4.0): with EWR and SLOPE
COLS_ERR = list(ERR_COLS)
DEFAULT_SLOPE_MIN = 0.35            # as COG_SLOPE_MIN

FINAL_HEADER = ('#STAR\t\t= name of the spectrum \n'
                '#CODE\t\t= element code in kurucz format \n'
                '#N_LIN\t\t= number of lines used to compute the mean \n'
                '#ABU\t\t= abundance in the format [Fe/H] or [X/Fe] \n'
                '#ERROR\t\t= total error of ABU. Sum in quadrature of of E_INT + E_PARAM \n'
                '#E_INT\t\t= internal error. Sigma / sqrt(N_LIN) or MC error \n'
                '#E_PARAM\t= atmospheric params error. Sum in quadrature of error due to Teff, logg, and vt \n'
                '#A_ABS\t\t= absolute abundance \n'
                '#E_X/H\t\t= total error of abundance in [X/H] format. Sum in quadrature of of E_X/H_INT + E_X/H_PARAM \n'
                '#E_X/H_INT\t= internal error. Sigma / sqrt(N_LIN) or MC error \n'
                '#E_X/H_PARAM\t= atmospheric params error. Sum in quadrature of error due to Teff, logg, and vt \n'
                '#' + '-' * 110 + '\n'
                '#STAR                            CODE  N_LIN    ABU  ERROR  E_INT  E_PARAM     A_ABS   E_XH E_XH_INT E_XH_PARAM\n')


@dataclass
class MergeSettings:
    """Settings of salvador-merge.

    Attributes:
        solar: Solar reference, 1-4 (see SOLAR_REFERENCES).
        recalc: True: flags recalculated (EW, EP limits and sigma-clipping); False: the flags of
            the .fit files are used.
        sclip: Sigma of the clipping (recalc only).
        ep: (min, max) excitation potential [eV] (recalc only).
        ewr_min: Minimum reduced EW, log(EW/lambda) (recalc only).
        ewr_max: Maximum reduced EW (recalc only; with slope_min it only removes the very strong
            lines, as EWR_MAX_DEFAULT in salvador).
        slope_min: Minimum slope of the curve of growth (recalc only); None: selection by ewr_min
            and ewr_max only (NEW 1.4.0, as EWR_MAX and COG_SLOPE_MIN in salvador).
    """
    solar: int
    recalc: bool
    sclip: float = 3.0
    ep: tuple[float, float] = (0.0, 10.0)
    ewr_min: float = -6.0
    ewr_max: float = EWR_MAX_DEFAULT
    slope_min: float | None = DEFAULT_SLOPE_MIN

    def selection(self) -> str:
        """The selection of the lines by their strength, as written in the merged .fit file."""
        if not self.recalc:
            return 'flags of the merged .fit files (not recalculated)'
        return describe_ew_selection(self.ewr_min, self.ewr_max, self.slope_min)


def get_type_input(prompt: str, target_type: type, valid: Callable | None = None, hint: str = '') -> object:
    """Ask for a value until it has the right type and, if valid is given, valid(value) is True.
    NEW: wrong choices are refused (before, e.g. a solar reference 7 stopped the program later)."""
    while True:
        try:
            value = target_type(input(prompt))
        except ValueError:
            print(f"Error: please enter a valid value (required type: {target_type.__name__}).")
            continue
        if valid is None or valid(value):
            return value
        print(f"Error: {hint}")


def _ask_strength_selection(args: argparse.Namespace, settings: MergeSettings) -> None:
    """Selection of the lines by their strength (recalc): from the command line, otherwise asked.
    NEW (1.4.0): the COG slope (as COG_SLOPE_MIN, with EWR_MAX_DEFAULT against the very strong
    lines) or the reduced EW only (as EWR_MAX)."""
    if args.ewr is not None:
        settings.ewr_min, settings.ewr_max = args.ewr
        settings.slope_min = None
        return
    if args.slope_min is None and args.ewr_min is None:
        use_ewr = get_type_input('Reject the strong lines by the COG slope (0) or by a maximum EWR (1)?:  ', int,
                                 lambda v: v in (0, 1), 'choose 0 or 1.') == 1
        if use_ewr:
            settings.ewr_min, settings.ewr_max = _ask_range('EWR', '-6.0', str(EWR_MAX_DEFAULT))
            settings.slope_min = None
            return
    settings.ewr_min = args.ewr_min if args.ewr_min is not None else get_type_input('Insert EWR min (es. -6.0): ', float)
    settings.slope_min = args.slope_min if args.slope_min is not None else get_type_input(
        f'Insert COG slope min (es. {DEFAULT_SLOPE_MIN}): ', float, lambda v: v >= 0, 'the value cannot be negative.')
    settings.ewr_max = EWR_MAX_DEFAULT


def _ask_range(name: str, low_default: str, high_default: str) -> tuple[float, float]:
    """Ask for the minimum and the maximum of a range (minimum < maximum)."""
    low = get_type_input(f"Insert {name} min (es. {low_default}): ", float)
    high = get_type_input(f"Insert {name} max (es. {high_default}): ", float, lambda v: v > low,
                          f"the maximum must be larger than the minimum ({low}).")
    return low, high


def merge_files(input_filenames: list[str], output_filename: str, selection: str = '') -> bool:
    """Write the merged .fit file, with the columns of this version (NEW 1.4.0: the files are read
    by the names of their header, see fit_files.read_fit_rows, so files of older versions, without
    EWR and SLOPE, can be merged). FIX: all the rows have the same columns: rows without the
    parameter errors (other runs, or rejected in the interactive mode) get 99.99. Returns False if
    nothing was read."""
    rows, any_errors, ok, without_slope = [], False, False, 0
    for fname in input_filenames:
        if not os.path.exists(fname):
            print(f"Attention: File {fname} not found.")
            continue
        ok = True
        file_rows, file_has_errors = read_fit_rows(fname)
        any_errors |= file_has_errors
        rows += file_rows
        without_slope += sum(r['slope'] == '99.999' and r['ew'] != '99.99' for r in file_rows)
        if not file_has_errors:
            print(f"Note: {fname} has no errors due to the atmospheric parameters (99.99 in the merged file)")
    if not ok:
        return False
    if without_slope and 'COG slope >=' in selection:      # only when the slope is used (recalc)
        print(f"Note: {without_slope} line(s) with an EW but without the COG slope (files of salvador < 1.4.0, "
              f"grid mode, or K2 could not compute it): with --flags recalc only the EWR limits are used for them")

    cols = COLS_BASE + (COLS_ERR if any_errors else []) + ['flag']
    with open(output_filename, 'w') as out:
        out.write(fit_file_header(selection).rstrip('\n') + ('  d_Teff d_logg  d_vt' if any_errors else '') + '  flag_line\n')
        for row in rows:
            out.write("\t".join(row[c] for c in cols) + "\n")
    return True


def _external_feh(parts: list[str], output_filename: str) -> tuple[float, float] | None:
    """External [Fe/H] and its error from the last row of a group ('merged.fit -1.50 0.05').
    NEW: the error is asked when it is not in the file (before, it was taken as 99.99, so all the
    errors of [X/Fe] were 99.99)."""
    if len(parts) < 2:
        return None
    try:
        feh = float(parts[1])
    except ValueError:
        print(f"Attention: [Fe/H] not valid for {output_filename}")
        return None
    try:
        error = float(parts[2])
        if error < 0:
            raise ValueError
    except IndexError:
        try:
            error = get_type_input(f"Error of the external [Fe/H] = {feh} of {output_filename} (dex): ", float,
                                   lambda v: v >= 0, "the error cannot be negative.")
        except EOFError:
            sys.exit(f"Error: give the error of the external [Fe/H] of {output_filename} in the list file "
                     f"(e.g. '{output_filename}  {feh}  0.10')")
    except ValueError:
        sys.exit(f"Error: the error of the external [Fe/H] of {output_filename} is not valid: '{parts[2]}'")
    return feh, error


def process_group(group: list[str], settings: MergeSettings) -> None:
    """Merge the .fit files of one star and analyse the merged file."""
    if len(group) < 2:
        return
    last_line_parts = group[-1].split()
    output_filename = last_line_parts[0]
    external_feh = _external_feh(last_line_parts, output_filename)

    if not merge_files(group[:-1], output_filename, settings.selection()):
        return
    print(f"{output_filename} created.")
    if external_feh is not None:
        print(f"Using external [Fe/H]: {external_feh[0]} +- {external_feh[1]}")
    print('#' * 40, '\n')
    analyze_abundances(output_filename, settings, external_feh=external_feh)


def merge_stellar_files(input_list_file: str, settings: MergeSettings) -> None:
    """Process the groups of the list file (one star each, separated by '###')."""
    if not os.path.exists(input_list_file):
        print(f"Error: File {input_list_file} does not exist.")
        return
    with open(input_list_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    group = []
    for line in lines:
        if line == '###':
            process_group(group, settings)
            group = []
        else:
            group.append(line)
    if group:
        process_group(group, settings)


def analyze_abundances(abu_file: str, settings: MergeSettings, external_feh: tuple[float, float] | None = None) -> None:
    """Flags (recalculated or kept), abundances of the merged .fit file (appended to
    abundance_final_merged.txt) and its Fe plot."""
    try:
        with open(abu_file) as f:
            has_errors = 'd_Teff' in next((r for r in f if r.startswith('#SPECTRUM')), '')
        df = read_fit_table(abu_file)
    except (OSError, ValueError) as e:
        print(f"Error during the analysis of {abu_file}: {e}")
        return
    if df.empty:
        return

    ID = Path(abu_file).stem
    nelem = df['nelem'].to_numpy(dtype=float, copy=True)
    abu = df['abu'].to_numpy(dtype=float, copy=True)
    abu_mc = df['abu_mc'].to_numpy(dtype=float, copy=True)
    flag_line = df['flag'].to_numpy(dtype=int, copy=True)

    if settings.recalc:
        EP = df['EP'].to_numpy(dtype=float)
        # start again from all the lines with an abundance (the -1/-2/-4 flags of the old limits are reset)
        flag_line = np.where(abu != 99.99, 0, flag_line)
        has_abu = abu != 99.99
        # NEW (1.4.0): the same selection as salvador (summary.ew_selection_flag): reduced EW and
        # COG slope (-4), or reduced EW only with --ewr; lines without EW (99.99) are not judged
        for i in np.where(has_abu)[0]:
            flag_line[i] = ew_selection_flag(df['ew'].iat[i], df['line'].iat[i], df['slope'].iat[i],
                                             settings.ewr_min, settings.ewr_max, settings.slope_min)
        flag_line[has_abu & (EP != 99.999) & ((EP < settings.ep[0]) | (EP > settings.ep[1]))] = -2
        if settings.slope_min is not None:      # saturated lines of the species without other lines
            flag_line = use_saturated_species(nelem, flag_line, label=ID)

    param_errors = tuple(df[c].to_numpy(dtype=float, copy=True) for c in COLS_ERR) if has_errors else None
    rows, flag_line = star_abundance_rows(ID, nelem, abu, abu_mc, flag_line, param_errors, feh_model=None,
                                          external_feh=external_feh, select_flag=0 if settings.recalc else 1,
                                          clip=settings.recalc)

    with open('abundance_final_merged.txt', 'a+') as f:
        for row in rows:
            f.write(row + '\n')
        f.write('#' + '-' * 39 + '\n')

    replace_flag_column(abu_file, flag_line)
    fe_diagnostic_plot(abu_file, str(Path(abu_file).with_suffix('.pdf')))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options of salvador-merge (all optional: what is missing is asked)."""
    parser = argparse.ArgumentParser(
        prog='salvador-merge',
        description='Merge the .fit files of the same star and compute its abundances again. '
                    'The settings not given here are asked.')
    parser.add_argument('list', nargs='?', help='file with the .fit files to merge (see docs/example_merge_fit.txt)')
    parser.add_argument('--solar', type=int, choices=sorted(SOLAR_REFERENCES),
                        help='solar reference: ' + ', '.join(f'{k} = {v}' for k, v in SOLAR_REFERENCES.items()))
    parser.add_argument('--flags', choices=['recalc', 'keep'],
                        help='recalc: recalculate the flags (EW, EP limits and sigma-clipping); keep: use the flags of the .fit files')
    parser.add_argument('--sclip', type=float, help='sigma of the sigma-clipping (with --flags recalc)')
    parser.add_argument('--ep', type=float, nargs=2, metavar=('MIN', 'MAX'), help='excitation potential limits [eV] (with --flags recalc)')
    parser.add_argument('--ewr', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='reduced EW limits, log(EW/lambda), instead of the COG slope (as EWR_MAX; with --flags recalc)')
    parser.add_argument('--ewr-min', type=float, metavar='MIN', help='minimum reduced EW (with --slope-min; with --flags recalc)')
    parser.add_argument('--slope-min', type=float, metavar='S',
                        help=f'minimum slope of the curve of growth (as COG_SLOPE_MIN, e.g. {DEFAULT_SLOPE_MIN}; with --flags recalc)')
    parser.add_argument('--version', action='version', version=f'salvador {__version__}')
    args = parser.parse_args(argv)
    if args.sclip is not None and args.sclip <= 0:
        parser.error('--sclip must be positive')
    for name in ('ep', 'ewr'):
        value = getattr(args, name)
        if value is not None and value[0] >= value[1]:
            parser.error(f'--{name}: MIN must be smaller than MAX')
    if args.ewr is not None and (args.slope_min is not None or args.ewr_min is not None):
        parser.error('--ewr (selection by the reduced EW) cannot be used with --slope-min or --ewr-min')
    if args.slope_min is not None and args.slope_min < 0:
        parser.error('--slope-min cannot be negative')
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point of salvador-merge. NEW: the settings can be given on the command line; the
    missing ones are asked, as before."""
    args = _parse_args(argv)
    to_merge = args.list or get_type_input('Filename containing the .fit files to merge: ', str)
    print('')
    solar = args.solar or get_type_input('Which solar reference? (1 -> CL | 2 -> GS98 | 3 -> Magg22 | 4 -> Asp09 ):  ', int,
                                         lambda v: v in SOLAR_REFERENCES, 'choose 1, 2, 3 or 4.')
    print('')
    if args.flags is None:
        recalc = get_type_input('Recalculate the flags (0) or use the flags present in the .fit files (1)?:  ', int,
                                lambda v: v in (0, 1), 'choose 0 or 1.') == 0
        print('')
    else:
        recalc = args.flags == 'recalc'

    settings = MergeSettings(solar=solar, recalc=recalc)
    if recalc:
        if args.sclip is None or args.ep is None or (args.ewr is None and (args.slope_min is None or args.ewr_min is None)):
            print("\n--- Parameters for sigma, EP and line-strength (EWR, COG slope) clipping ---")
        settings.sclip = args.sclip if args.sclip is not None else get_type_input(
            'Insert value for sigma clipping? ', float, lambda v: v > 0, 'the value must be positive.')
        settings.ep = tuple(args.ep) if args.ep is not None else _ask_range('EP', '0.0', '10.0')
        _ask_strength_selection(args, settings)
        print('')

    # settings used by the shared functions: sigma-clipping and solar reference (column = choice + 1)
    cfg.use(Config(input_stars='', type_norm=0, window=0, inter='n', sclip=settings.sclip, solar_ref=solar + 1))

    with open('abundance_final_merged.txt', 'w') as f:
        f.write(FINAL_HEADER)

    merge_stellar_files(to_merge, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
