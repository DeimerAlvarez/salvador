"""The .fit result files: header, error columns, flags, reading them back (also by the names of
the header, for the files of any version: read_fit_table)."""
import numpy as np
import pandas as pd

# columns of the .fit file (names used when it is read back; with the parameter errors they are
# followed by abu_pT, abu_pg, abu_pv). NEW (1.4.0): ewr and slope.
FIT_COLS = ('ID', 'line', 'nelem', 'abu', 'ew', 'ewr', 'slope', 'EP', 'loggf', 'abu_mc')
FIT_HEADER_NAMES = '#SPECTRUM                           LINE   ELEM    ABU      EW     EWR   SLOPE      EP  log_gf    eMC'


def fit_file_header(selection: str) -> str:
    """Header of the .fit file: meaning of the flags, selection of the lines by their strength
    (NEW 1.4.0, e.g. from summary.describe_ew_selection) and the names of the columns."""
    return ('#   FLAG =  1  (USED LINE)\n'
            '#   FLAG =  0  (REJ. SIGMA-CLIPPING)\n'
            '#   FLAG = -1  (REJ. EW)\n'
            '#   FLAG = -2  (REJ. E.P.)\n'
            '#   FLAG = -3  (REJ. ABU)\n'
            '#   FLAG = -4  (REJ. COG SLOPE)\n'
            f'#   SELECTION: {selection}\n'
            '#   EWR = log(EW/lambda); SLOPE = d log(EW/lambda) / d A(X) from K2 (99.999 = not available)\n'
            '#' + '-' * 39 + '\n'
            + FIT_HEADER_NAMES + '\n')


# names of the header of the .fit files (all versions) -> names of the columns in salvador
HEADER_TO_COL = {'SPECTRUM': 'ID', 'LINE': 'line', 'ELEM': 'nelem', 'ABU': 'abu', 'EW': 'ew', 'EWR': 'ewr',
                 'SLOPE': 'slope', 'EP': 'EP', 'log_gf': 'loggf', 'eMC': 'abu_mc',
                 'd_Teff': 'abu_pT', 'd_logg': 'abu_pg', 'd_vt': 'abu_pv', 'flag_line': 'flag'}
OLD_FIT_COLS = ('ID', 'line', 'nelem', 'abu', 'ew', 'EP', 'loggf', 'abu_mc')     # before 1.4.0
ERR_COLS = ('abu_pT', 'abu_pg', 'abu_pv')
MISSING_ERRORS = ('99.99', '99.99', '99.99')


def read_fit_rows(fit_file) -> tuple[list[dict[str, str]], bool]:
    """Rows of a .fit file of ANY version of salvador, as text, with the columns of this version:
    FIT_COLS, the three parameter errors and the flag. NEW (1.4.0): the columns are found from the
    names of the #SPECTRUM header (before, from their position), so the files written before 1.4.0
    (without EWR and SLOPE) can be read and merged with the new ones. Missing values: EWR computed
    from EW (99.99 if not available), SLOPE 99.999, errors 99.99, flag 0.

    A data row may have, after the base columns, 3 errors and the flag, only the errors, only the
    flag, or nothing (e.g. rejected lines of the interactive mode before 1.1.0). A row that does
    not fit is skipped with a message.

    Returns:
        (rows, True if the header has the parameter errors). ([], False) if the file is missing.
    """
    try:
        with open(fit_file) as f:
            lines = f.readlines()
    except OSError:
        print(f"Attention: File {fit_file} not found.")
        return [], False
    header = next((r for r in lines if r.startswith('#SPECTRUM')), None)
    names = [HEADER_TO_COL.get(tok, tok) for tok in header.replace('#', ' ').split()] if header else list(OLD_FIT_COLS)
    base = names[:names.index('abu_mc') + 1] if 'abu_mc' in names else list(OLD_FIT_COLS)
    has_errors = 'abu_pT' in names
    rows = []
    for k, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        extra = len(fields) - len(base)
        if extra not in (0, 1, 3, 4):
            print(f"Attention: row {k} of {fit_file} has {len(fields)} columns: skipped")
            continue
        row = dict(zip(base, fields[:len(base)]))
        errors = fields[len(base):len(base) + 3] if extra >= 3 else list(MISSING_ERRORS)
        row.update(zip(ERR_COLS, errors))
        row['flag'] = fields[-1] if extra in (1, 4) else '0'
        if 'ewr' not in row:
            from .summary import reduced_ew
            try:
                ewr = reduced_ew(float(row['ew']), float(row['line']))
            except (KeyError, ValueError):
                ewr = 99.99
            row['ewr'] = '99.99' if ewr == 99.99 else f'{ewr:.2f}'
        row.setdefault('slope', '99.999')
        rows.append({col: row[col] for col in FIT_COLS + ERR_COLS + ('flag',)})
    return rows, has_errors


def read_fit_table(fit_file) -> pd.DataFrame:
    """A .fit file of any version (see read_fit_rows) as a DataFrame with the columns FIT_COLS, the
    parameter errors and the flag (numbers as float, the flag as int). Empty if there are no rows."""
    rows, _ = read_fit_rows(fit_file)
    columns = list(FIT_COLS + ERR_COLS + ('flag',))
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    for col in columns[1:]:
        df[col] = df[col].astype(float)
    df['flag'] = df['flag'].astype(int)
    return df


def add_errors_to_file(abu_file, new_headers, len_cols, delta_pT, delta_pg, delta_pv):
    """Add the three errors due to the parameters (d_Teff, d_logg, d_vt) to the LAST row of a .fit
    file, and their names to its #SPECTRUM header (only once). The row is changed only if it has
    len_cols columns, i.e. no errors yet. Returns the number of columns declared by the header."""
    num_cols = len_cols

    # --- 1. UPDATE OF THE .fit FILE ---
    try:
        # Read the whole file
        with open(abu_file, "r") as f:
            lines = f.readlines()

        if lines:
            # Find the index of the right header
            header_index = -1
            for i, line in enumerate(lines):
                if line.startswith("#SPECTRUM"):
                    header_index = i
                    break

            if header_index != -1:
                # Update the header if it does not contain the new names yet
                if not all(header in lines[header_index] for header in new_headers):
                    current_header = lines[header_index].strip()
                    lines[header_index] = f"{current_header}  {' '.join(new_headers)}\n"

                # Count the columns from the right header (needed later by pandas)
                header_line_clean = lines[header_index].strip().replace('#', '')
                num_cols = len(header_line_clean.split())

                # Update the last data row
                last_data_index = -1
                for i in range(len(lines) - 1, -1, -1):
                    line_text = lines[i].strip()
                    if line_text and not line_text.startswith("#"):
                        last_data_index = i
                        break

                if last_data_index != -1:
                    parts = lines[last_data_index].strip().split()
                    # Add the extra values only if the row still has the original columns
                    if len(parts) == len_cols:
                        lines[last_data_index] = lines[last_data_index].strip() + f"  {delta_pT}  {delta_pg}  {delta_pv}\n"
                # Write the changes to the file
                with open(abu_file, "w") as f:
                    f.writelines(lines)

            else:
                print(f"Warning: Header '#SPECTRUM' not found in {abu_file}")

    except FileNotFoundError:
        print(f"Error: File '{abu_file}' not found.")
        return None

    return num_cols


def count_header_cols(fit_file):
    """Number of columns declared in the '#SPECTRUM' header of a .fit file (0 if not found)."""
    with open(fit_file, 'r') as f:
        for row in f:
            if row.startswith('#SPECTRUM'):
                return len(row.replace('#', '').split())
    return 0


def read_fit_file(total_cols_in_file, col_names_base, col_names_err, fit_file):
    """Read a .fit file with pandas, with the error columns when the file has them.
    Returns (ID, nelem, abu, abu_mc, list of species, DataFrame), or six None if empty or unreadable."""
    try:
        expected_cols_with_errors = len(col_names_base) + len(col_names_err)

        if total_cols_in_file == expected_cols_with_errors:
            all_col_names = col_names_base + col_names_err
        else:
            all_col_names = col_names_base

        df = pd.read_csv(fit_file, sep=r"\s+", comment='#', header=None, names=all_col_names)

        if df.empty:
            print(f"File '{fit_file}' is empty. Skipping.")
            return None, None, None, None, None, None

    except (FileNotFoundError, pd.errors.EmptyDataError):
        # If the file cannot be read, stop and return None
        print(f"File '{fit_file}' not found or is empty. Skipping.")
        return None, None, None, None, None, None

    # Extract the data from the columns
    ID = df['ID'].iloc[0]
    nelem = df['nelem'].values
    abu = df['abu'].values
    abu_mc = df['abu_mc'].values

    unique_nelems = np.unique(nelem)
    other_nelems = unique_nelems[unique_nelems != 26.00]

    return ID, nelem, abu, abu_mc, other_nelems, df


def check_if_has_data(filepath, keyword):
    """True if the file has at least one data row after the header line starting with keyword."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 1. Find the row with the #SPECTRUM header
        spectrum_index = -1
        for i, line in enumerate(lines):
            if line.startswith(keyword):
                spectrum_index = i
                break

        # 2. If the header was found, check what follows it
        if spectrum_index != -1:
            # Go through all the rows after the header
            for i in range(spectrum_index + 1, len(lines)):
                line_clean = lines[i].strip()

                # A row that is neither empty nor a comment: there are data
                if line_clean and not line_clean.startswith('#'):
                    return True

        # No valid data after the header: the file is empty
        return False

    except FileNotFoundError:
        print(f"Warning: the file '{filepath}' was not found.")
        return False


def append_flag_column(abu_file_path, flag_abu):
    # --- 1. read the original file ---
    """Add the FLAG column (one value per data row, in order) to a .fit file and its name to the
    header. Nothing is written if the number of flags differs from the number of data rows."""
    try:
        with open(abu_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file {abu_file_path} not found.")
        return

    # --- 2. check of lengths ---
    num_data_rows = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))

    if len(flag_abu) != num_data_rows:
        print(f"Error: number of flags ({len(flag_abu)}) not corresponds to the number of line data ({num_data_rows})!")
        return

    # --- 3. process lines ---
    new_lines_content = []
    data_row_index = 0 # Counter over the flag_abu array

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            # Keep empty rows
            new_lines_content.append('\n')
            continue

        if line_stripped.startswith('#SPECTRUM'):
            # Header row
            new_header = f"{line_stripped}  flag_line\n"
            new_lines_content.append(new_header)

        elif line_stripped.startswith('#'):
            new_lines_content.append(f"{line_stripped}\n")

        else:
            # Data row
            flag_value = int(flag_abu[data_row_index]) # Convert to an integer (1, not 1.0)
            new_data_line = f"{line_stripped}  {flag_value:>2}\n"
            new_lines_content.append(new_data_line)

            # Increase the counter only for data rows
            data_row_index += 1

    # --- 4. Rewrite the original file ---
    try:
        with open(abu_file_path, 'w') as f:
            f.writelines(new_lines_content)
    except IOError as e:
        print(f"Error writing file: {e}")


def get_header_names(file_path):
    """Names of the columns of a file, from its LAST comment (#) row. None if the file is missing."""
    header_line = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('#'):
                    # Keep overwriting: we want the LAST comment row
                    header_line = line

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None

    if header_line is None:
        print("Error: no header line (starting with #) found in the file.")
        return None

    # Clean the header row: remove '#' and extra spaces, then split
    column_names = header_line.lstrip('#').strip().split()
    return column_names


def replace_flag_column(abu_file_path, flag_abu):
    """Replace the flag (last column) of each data row of a .fit file, in order (salvador-merge).
    Nothing is written if the number of flags differs from the number of data rows."""
    try:
        with open(abu_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file {abu_file_path} not found.")
        return

    num_data_rows = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
    if len(flag_abu) != num_data_rows:
        print(f"Error: number of flags ({len(flag_abu)}) not corresponds to the number of line data ({num_data_rows})!")
        return

    new_lines, k = [], 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append('\n')
        elif stripped.startswith('#'):
            new_lines.append(line)
        else:
            parts = stripped.split()
            parts[-1] = str(int(flag_abu[k]))
            new_lines.append("\t".join(parts) + "\n")
            k += 1
    try:
        with open(abu_file_path, 'w') as f:
            f.writelines(new_lines)
    except OSError as e:
        print(f"Error writing file: {e}")
