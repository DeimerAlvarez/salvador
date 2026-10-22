#!/usr/bin/env python3
"""Stand-in for K2, used by the tests: salvador runs without the Kurucz codes.

salvador finds K2 as the command k2.py in the PATH and runs it on ./input.k2. This module
answers the requests of input.k2 from files of a real run:

* ``synthe``: copies the synthetic spectrum <name>.nb from the directory FAKE_K2_SYNTH;
* ``eqw``/``eqwb``: writes info_ew.txt with EW, EP and log gf of the line, read from the
  .fit file FAKE_K2_EW;
* ``atlas9``: nothing (the synthetic spectra are already computed).

A request that cannot be answered ends with exit code 1, as a failing K2 would. Every call is
written to the file FAKE_K2_LOG (if set).
"""
import os
import shutil
import sys
from pathlib import Path


def _log(message: str) -> None:
    log = os.environ.get('FAKE_K2_LOG')
    if log:
        with open(log, 'a') as f:
            f.write(message + '\n')


def _ew_of_line(line: float) -> list[str] | None:
    """EW, EP and log gf of the line from the .fit file FAKE_K2_EW, or None."""
    for row in Path(os.environ['FAKE_K2_EW']).read_text().splitlines():
        fields = row.split()
        if fields and not row.startswith('#') and abs(float(fields[1]) - line) < 1e-4:
            return fields[4:7]
    return None


def main() -> None:
    """Answer the requests of ./input.k2 (see the module docstring)."""
    for tok in (r.split() for r in Path('input.k2').read_text().splitlines() if r.strip()):
        key = tok[0]
        if key == 'synthe':
            # synthe teff logg met alpha vt linelist abu_var w1 w2 res 0 <name> 0
            name = tok[12] + '.nb'
            src = Path(os.environ['FAKE_K2_SYNTH']) / name
            if not src.exists():
                _log(f'MISSING synthe {name}')
                sys.exit(1)
            shutil.copyfile(src, name)
            _log(f'synthe {name}')
        elif key in ('eqw', 'eqwb'):
            # eqw teff logg met alpha vt linelist delta_file line code delta
            line = float(tok[8])
            values = _ew_of_line(line)
            if values is None:
                _log(f'MISSING eqw {line}')
                sys.exit(1)
            Path('info_ew.txt').write_text(f'{line} {" ".join(values)}\n')
            _log(f'eqw {line}')
        elif key == 'atlas9':
            _log('atlas9 ' + ' '.join(tok[1:6]))


if __name__ == '__main__':
    main()
