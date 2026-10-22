"""
salvador-check-k2: check of the K2 speed-up in salvador. Run it in your salvador working directory
(where input.salvador, lista.stars and linelist.xkur are):

    salvador-check-k2            # first 5 lines of the first star
    salvador-check-k2 10         # first 10 lines

For each line it computes the same synthetic spectrum and the same equivalent width twice
with the real K2:
  OLD: external 'k2.py' with the full linelist.xkur (as salvador did before)
  NEW: in-process K2 with the trimmed linelist and the model check (as salvador does now)
then compares spectra (number by number, headers excluded) and EWs, and reports the times.
"""
import os
import sys
import time

import numpy as np

from . import utilities as us


def main(argv: list[str] | None = None) -> int:
    """Entry point of salvador-check-k2."""
    argv = sys.argv[1:] if argv is None else argv
    N_LINES = int(argv[0]) if argv else 5
    us.cfg.load()
    us.select_backend('n')

    # first star of lista.stars: name teff logg met alpha vt rv abu_var linelist ...
    with open(us.cfg.input_stars) as f:
        star = next(r.split() for r in f if r.strip() and not r.lstrip().startswith('#'))
    teff, logg, met, alpha, vt = int(star[1]), float(star[2]), float(star[3]), int(star[4]), float(star[5])
    lines, codes = np.genfromtxt(star[8], comments='#', usecols=(1, 2), dtype=float, unpack=True)
    lines, codes = np.atleast_1d(lines)[:N_LINES], np.atleast_1d(codes)[:N_LINES]
    win = 50 if us.cfg.type_norm == 3 else us.cfg.window
    print(f'Star {star[0]}: Teff={teff} logg={logg} [M/H]={met} alpha={alpha} vt={vt} | {len(lines)} lines, window {win} A\n')

    open('abu_empty_check.txt', 'w').close()

    def run(label, linelist_file, w1, w2, external):
        name = f'check_{label}_{w1}_{w2}'
        with open('input.k2', 'w') as f:
            f.write(f'synthe {teff}  {logg}  {met}  {alpha}  {vt} {linelist_file} abu_empty_check.txt  {w1}  {w2}  30000 0 {name} 0\n')
            f.write('VERBOSE 0\nOUTPUT 0\nCHECKSYN 0\n')
        t = time.perf_counter()
        if external:
            os.system('k2.py')
        else:
            us.run_k2()
        dt = time.perf_counter() - t
        data = np.loadtxt(name + '.nb', comments='#')
        os.remove(name + '.nb')
        return data, dt

    def run_eqw(linelist_file, line, code, external):
        with open('input.k2', 'w') as f:
            f.write(f'eqw {teff}  {logg}  {met}  {alpha}  {vt} {linelist_file} abu_empty_check.txt  {line}  {code:.2f}  0.00\n')
            f.write('VERBOSE 0\n')
        if os.path.exists('info_ew.txt'):
            os.remove('info_ew.txt')
        t = time.perf_counter()
        if external:
            os.system('k2.py')
        else:
            us.run_k2()
        dt = time.perf_counter() - t
        try:
            values = np.atleast_1d(np.loadtxt('info_ew.txt'))
        except (OSError, ValueError):
            values = None
        return values, dt

    t_old, t_new, e_old, e_new, all_ok = [], [], [], [], True
    for k, (line, code) in enumerate(zip(lines, codes)):
        w1, w2 = int(np.floor(line - win / 2)), int(np.ceil(line + win / 2))
        old, dt_old = run('old', us.K2_LINELIST, w1, w2, external=True)
        new, dt_new = run('new', us.k2_linelist(w1, w2), w1, w2, external=False)
        same = old.shape == new.shape and np.array_equal(old, new)
        all_ok &= same
        diff = 'identical' if same else (f'max |diff| = {np.max(np.abs(old - new)):.2e}' if old.shape == new.shape else f'different sizes {old.shape} vs {new.shape}')
        t_old.append(dt_old)
        t_new.append(dt_new)
        print(f'{line:10.3f} A  [{w1}-{w2}]  synthe: old {dt_old:6.2f} s   new {dt_new:6.2f} s   spectra: {diff}')
        ew_o, de_old = run_eqw(us.K2_LINELIST, line, code, external=True)
        ew_n, de_new = run_eqw(us.k2_linelist(w1, w2), line, code, external=False)
        same_ew = ew_o is not None and ew_n is not None and np.array_equal(ew_o, ew_n)
        all_ok &= same_ew
        e_old.append(de_old)
        e_new.append(de_new)
        ew_txt = f'EW {ew_o[1]:.2f} mA, identical' if same_ew else f'EW differ: old {ew_o} new {ew_n}'
        print(f'{"":24s}  eqw   : old {de_old:6.2f} s   new {de_new:6.2f} s   {ew_txt}')

    us.cleanup_linelist_cut()
    os.remove('abu_empty_check.txt')
    print(f'\nMean time per synthetic: old {np.mean(t_old):.2f} s, new {np.mean(t_new):.2f} s')
    print(f'Mean time per EW       : old {np.mean(e_old):.2f} s, new {np.mean(e_new):.2f} s')
    print('(the first "new" call also imports K2 and indexes the linelists once)')
    print('ALL SPECTRA AND EWs IDENTICAL' if all_ok else 'SOME RESULTS DIFFER: send me the output')
    if os.path.exists('info_ew.txt'):
        os.remove('info_ew.txt')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
