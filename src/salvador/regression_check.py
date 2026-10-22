"""
salvador-regression: compares the results of a run with a stored reference.
Run it in the salvador working directory, after a salvador run.

    salvador-regression save            # store the current results as the reference
    salvador-regression                 # compare the current results with it
    salvador-regression --tol 0.01      # numbers may differ by up to 0.01

Compared files: abundance_final.txt, abundance_final_isor.txt and every *.fit file of the
working directory. The reference is kept in ./regression_ref/. Rows are compared in order,
field by field: numbers within the tolerance (default 0: identical), text exactly.
Exit code 0 if everything matches, 1 otherwise.
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

REF_DIR = Path('regression_ref')
MAX_REPORTED = 15       # differences printed per file


def result_files(directory):
    names = ['abundance_final.txt', 'abundance_final_isor.txt'] + sorted(glob.glob(str(Path(directory) / '*.fit')))
    return sorted({Path(n).name for n in names if (Path(directory) / Path(n).name).is_file()})


def read_table(path):
    """Header names (last '#' line before the data) and data rows (lists of fields)."""
    header, rows = None, []
    with open(path) as f:
        for row in f:
            if not row.strip():
                continue
            if row.lstrip().startswith('#'):
                if not rows:
                    header = row.replace('#', ' ').split()
                continue
            rows.append(row.split())
    return header, rows


def same_field(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except ValueError:
        return a == b


def compare_file(name, tol):
    header, ref = read_table(REF_DIR / name)
    _, new = read_table(name)
    problems = []
    if len(ref) != len(new):
        problems.append(f'number of rows: reference {len(ref)}, now {len(new)}')
    for i, (r, n) in enumerate(zip(ref, new)):
        label = ' '.join(r[:2])
        if len(r) != len(n):
            problems.append(f'row {i + 1} ({label}): {len(r)} fields in the reference, {len(n)} now')
            continue
        for j, (a, b) in enumerate(zip(r, n)):
            if not same_field(a, b, tol):
                col = header[j] if header and len(header) == len(r) else f'column {j + 1}'
                problems.append(f'row {i + 1} ({label}) {col}: reference {a}, now {b}')
    return len(ref), problems


def main():
    parser = argparse.ArgumentParser(description='salvador regression check')
    parser.add_argument('action', nargs='?', default='compare', choices=['compare', 'save'])
    parser.add_argument('--tol', type=float, default=0.0, help='allowed difference for numbers (default 0)')
    args = parser.parse_args()

    current = result_files('.')
    if args.action == 'save':
        if not current:
            sys.exit('No result files found here: run salvador first.')
        if REF_DIR.exists():
            shutil.rmtree(REF_DIR)
        REF_DIR.mkdir()
        for name in current:
            shutil.copy2(name, REF_DIR / name)
        print(f'Reference saved in {REF_DIR}/: ' + ', '.join(current))
        return 0

    if not REF_DIR.is_dir():
        sys.exit(f'No reference found ({REF_DIR}/). First run: salvador-regression save')
    reference = result_files(REF_DIR)
    all_ok = True
    for name in sorted(set(reference) | set(current)):
        if name not in current:
            print(f'MISSING   {name}: in the reference but not produced by this run')
            all_ok = False
            continue
        if name not in reference:
            print(f'NEW       {name}: not in the reference')
            all_ok = False
            continue
        nrows, problems = compare_file(name, args.tol)
        if problems:
            all_ok = False
            print(f'DIFFERENT {name}: {len(problems)} difference(s)')
            for p in problems[:MAX_REPORTED]:
                print(f'          {p}')
            if len(problems) > MAX_REPORTED:
                print(f'          ... and {len(problems) - MAX_REPORTED} more')
        else:
            print(f'OK        {name} ({nrows} rows)')
    tol_txt = 'identical' if args.tol == 0 else f'within {args.tol}'
    print(f'\nALL RESULTS {tol_txt.upper()}' if all_ok else '\nSOME RESULTS DIFFER')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
