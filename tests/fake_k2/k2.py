#!/usr/bin/env python3
"""Stand-in for K2, used by the tests: salvador runs without the Kurucz codes.

salvador finds K2 as the command k2.py in the PATH and runs it on ./input.k2. This module
answers the requests of input.k2 from files of a real run:

* ``synthe``: copies the synthetic spectrum <name>.nb from the directory FAKE_K2_SYNTH. With
  FAKE_K2_INTERPOLATE = 1, a synthetic missing there is interpolated linearly in the abundance between
  the two nearest synthetics of the same model, element and window, or extrapolated from the two
  nearest ones by at most 0.05 dex (written in the log as 'INTERPOLATED'): used when a change of
  salvador moves a fit to an abundance not in the cache;
* ``eqw``/``eqwb``: writes info_ew.txt as K2 (version of November 2026) does: wavelength, EW, EP,
  log gf and slope of the curve of growth. EW, EP and log gf are read from the .fit file
  FAKE_K2_EW (by the names of its header, so .fit files of any version can be used). The slope is
  NOT from a real run: it is computed from log(EW/lambda) with the median relation of the curve of
  growth of 954 Fe I lines of a giant computed with K2 (logistic fit, centred slope A +- 0.1 dex),
  unless FAKE_K2_SLOPE gives a value for every line (a number or 'nan'). With FAKE_K2_EW_COLUMNS = 4
  the file is written as the previous K2 did (without the slope);
* ``atlas9``: nothing (the synthetic spectra are already computed).

A request that cannot be answered ends with exit code 1, as a failing K2 would. Every call is
written to the file FAKE_K2_LOG (if set).
"""
import math
import os
import re
import shutil
import sys
from pathlib import Path

# S(y) = S_MIN + (1 - S_MIN) / (1 + exp((y - Y0) / WIDTH)), y = log(EW/lambda): median relation
# of the K2 curves of growth of a giant (Teff 4500 K, logg 0.70, [M/H] -1.5, vt 2.0 km/s)
S_MIN, Y0, WIDTH = 0.0401, -4.9518, 0.2117


def _log(message: str) -> None:
    log = os.environ.get('FAKE_K2_LOG')
    if log:
        with open(log, 'a') as f:
            f.write(message + '\n')


def _ew_of_line(line: float) -> list[str] | None:
    """EW, EP and log gf of the line from the .fit file FAKE_K2_EW, or None."""
    names = None
    for row in Path(os.environ['FAKE_K2_EW']).read_text().splitlines():
        if row.startswith('#SPECTRUM'):
            names = row.replace('#', ' ').split()
            continue
        fields = row.split()
        if not fields or row.startswith('#'):
            continue
        if abs(float(fields[1]) - line) < 1e-4:
            cols = names or ['SPECTRUM', 'LINE', 'ELEM', 'ABU', 'EW', 'EP', 'log_gf', 'eMC']
            return [fields[cols.index(name)] for name in ('EW', 'EP', 'log_gf')]
    return None


def _slope(ew: float, line: float) -> str:
    """Slope of the curve of growth written in info_ew.txt (see the module docstring)."""
    fixed = os.environ.get('FAKE_K2_SLOPE')
    if fixed is not None:
        return 'nan' if fixed.strip().lower() == 'nan' else f'{float(fixed):.3f}'
    if not ew > 0:
        return 'nan'
    y = math.log10(ew / 1000.0 / line)
    return f'{S_MIN + (1 - S_MIN) / (1 + math.exp((y - Y0) / WIDTH)):.3f}'


def _delta(code: str) -> float:
    """Abundance change of the code in the name of a synthetic spectrum (p004 -> 0.04, m037 -> -0.37)."""
    return (-1 if code[0] == 'm' else 1) * int(code[1:]) / 100


def _interpolate(name: str, cache: Path) -> bool:
    """Write the synthetic ``name`` by linear interpolation in the abundance between the two nearest
    synthetics of the cache of the same model, element and window (or extrapolation by at most
    0.05 dex from the two nearest ones). False if not possible."""
    match = re.fullmatch(r'(.+_)([pm]\d{3})(_\d+_\d+\.nb)', name)
    if os.environ.get('FAKE_K2_INTERPOLATE') != '1' or not match:
        return False
    head, target, tail = match.group(1), _delta(match.group(2)), match.group(3)
    known = sorted((_delta(f.name[len(head):-len(tail)]), f) for f in cache.glob(f'{head}*{tail}')
                   if re.fullmatch(r'[pm]\d{3}', f.name[len(head):-len(tail)]))
    below = [k for k in known if k[0] < target]
    above = [k for k in known if k[0] > target]
    if below and above:
        (d0, f0), (d1, f1) = below[-1], above[0]
    elif len(known) >= 2 and min(abs(d - target) for d, _ in known) <= 0.05 + 1e-9:
        (d0, f0), (d1, f1) = below[-2:] if below else above[:2]
    else:
        return False
    w = (target - d0) / (d1 - d0)
    rows0, rows1 = f0.read_text().splitlines(), f1.read_text().splitlines()
    out = []
    for r0, r1 in zip(rows0, rows1):
        if r0.startswith('#'):
            out.append(r0)
            continue
        v0, v1 = [float(v) for v in r0.split()], [float(v) for v in r1.split()]
        v = [v0[0]] + [a + w * (b - a) for a, b in zip(v0[1:], v1[1:])]
        out.append(f'{v[0]:11.4f}{v[1]:13.6f}' + ''.join(f'{x:20.8E}' for x in v[2:]))
    Path(name).write_text('\n'.join(out) + '\n')
    return True


def main() -> None:
    """Answer the requests of ./input.k2 (see the module docstring)."""
    for tok in (r.split() for r in Path('input.k2').read_text().splitlines() if r.strip()):
        key = tok[0]
        if key == 'synthe':
            # synthe teff logg met alpha vt linelist abu_var w1 w2 res 0 <name> 0
            name = tok[12] + '.nb'
            src = Path(os.environ['FAKE_K2_SYNTH']) / name
            if src.exists():
                shutil.copyfile(src, name)
                _log(f'synthe {name}')
            elif _interpolate(name, src.parent):
                _log(f'INTERPOLATED synthe {name}')
            else:
                _log(f'MISSING synthe {name}')
                sys.exit(1)
        elif key in ('eqw', 'eqwb'):
            # eqw teff logg met alpha vt linelist delta_file line code delta
            line = float(tok[8])
            values = _ew_of_line(line)
            if values is None:
                _log(f'MISSING eqw {line}')
                sys.exit(1)
            row = f'{line} {" ".join(values)}'
            if os.environ.get('FAKE_K2_EW_COLUMNS') != '4':
                row += ' ' + _slope(float(values[0]), line)
            Path('info_ew.txt').write_text(row + '\n')
            _log(f'eqw {line}')
        elif key == 'atlas9':
            _log('atlas9 ' + ' '.join(tok[1:6]))


if __name__ == '__main__':
    main()
