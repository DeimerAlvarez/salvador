"""Everything that talks to K2: running it (in-process), synthetic spectra (autok), equivalent widths
(eqw_k2), trimmed linelists, the model-atmosphere check, abundance-variation (DELTA) files and the
names of the synthetic spectra. The switches K2_IN_PROCESS, TRIM_LINELIST and K2_FAST_MODEL_CHECK are here."""
import importlib
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import astropy.units as u
from specutils import Spectrum

from .config import BASE_DIR, SYNTH_OUT, WORK_DIR, abu_var_tmp_name, cfg, plt
from .elements import get_element_symbol, strip_ion, symbol_map
from .spectra_io import read_two_columns


### =============================================================================================
### SPEED: calling K2
### =============================================================================================
### 1) K2 is run inside salvador's Python process. Before, every synthetic spectrum started a new
###    interpreter that re-imported numpy, scipy, astropy, specutils, matplotlib... (~2.5 s per
###    call). k2.main() reads input.k2 exactly as the command 'k2.py' does: same code path, same
###    results. Set K2_IN_PROCESS = False to go back to the external call.
### 2) K2 receives a trimmed linelist for each window. Before, for every synthetic K2 read all
###    the atomic files of linelist.xkur in full (only to check their first and last line) and
###    passed all the molecular files to rmolecasc (~800 MB of reading per synthetic for a few A).
###    Now, once per window, the lines within [w1 - LINELIST_MARGIN_A, w2 + LINELIST_MARGIN_A]
###    are copied into small files in LINELIST_CUT_DIR. Files not sorted in wavelength are
###    passed unchanged (no cut). Set TRIM_LINELIST = False to use linelist.xkur as before.
K2_IN_PROCESS = True
TRIM_LINELIST = True
K2_FAST_MODEL_CHECK = True      # see _install_k2_model_check below
K2_LINELIST = str(BASE_DIR / 'linelist.xkur')
LINELIST_MARGIN_A = 50.0
LINELIST_CUT_DIR = Path('linelist_cut')
LINELIST_INDEX = Path.home() / '.cache' / 'salvador' / 'linelist_index.json'
_WL_WIDTH = {'#ATOMS': 11, '#MOLECULES': 10}     # width of the wavelength field (nm) at line start
_K2_MARKERS = ('#ATOMS', '#MOLECULES', '#TIO', '#H2O', '#PREDICTED')
_k2_module = None
_trim_cache = OrderedDict()


def run_k2():
    """Run K2 on ./input.k2. Returns 0 on success, like os.system('k2.py')."""
    global _k2_module, K2_IN_PROCESS
    if K2_IN_PROCESS and _k2_module is None:
        try:
            exe = shutil.which('k2.py')
            if exe is None:
                raise ImportError('k2.py not found in PATH')
            k2_dir = str(Path(exe).resolve().parent)
            if k2_dir not in sys.path:
                sys.path.insert(0, k2_dir)
            _k2_module = importlib.import_module('k2')
        except Exception as e:
            print(f'Warning: cannot import K2 ({e}). Using the external call k2.py')
            K2_IN_PROCESS = False
        if K2_IN_PROCESS and K2_FAST_MODEL_CHECK:
            try:
                _install_k2_model_check()
            except Exception as e:      # only the shortcut is lost, K2 still runs in-process
                print(f'Warning: K2 model check not installed ({type(e).__name__}: {e})')

    if not K2_IN_PROCESS:
        return os.system('k2.py')

    # K2 may create figures (new ATLAS9 models): no pop-up windows, and close what it leaves open.
    # kgraph.graph_atlas calls plt.close(''), which closes the FIRST figure with an empty label:
    # salvador's open figures get a temporary label during the call so they cannot be closed.
    figs_before = set(plt.get_fignums())
    relabelled = []
    try:
        from matplotlib._pylab_helpers import Gcf
        for manager in Gcf.get_all_fig_managers():
            fig = manager.canvas.figure
            if fig.get_label() == '':
                fig.set_label('_salvador_protected_')
                relabelled.append(fig)
    except Exception:
        pass
    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        _k2_module.main()
        return 0
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Warning: K2 stopped: {e.code}')
            return 1
        return 0
    except Exception as e:
        print(f'Warning: K2 failed: {type(e).__name__}: {e}')
        return 1
    finally:
        for num in set(plt.get_fignums()) - figs_before:
            plt.close(num)
        for fig in relabelled:
            fig.set_label('')
        if was_interactive:
            plt.ion()


def _install_k2_model_check():
    """SPEED: in K2's RunAtlas9.atlas9(), the ODFs and Rosseland tables are interpolated for a
    [M/H] that is not a grid value (interpol_odf.exe, ~0.26 s per call) BEFORE checking whether the
    model atmosphere already exists; when it exists the interpolation is thrown away. Since K2 runs
    in this process, this wrapper does the existence check first, with K2's own naming method and
    the same directory rule, and does what K2 does in that case (copy the model to
    guess_model_4other.dat and return 1). In every other case K2's original method runs unchanged.
    The skipped steps only prepare files for an ATLAS9 run (interpol.*, guess_model_4atlas9.dat)
    and erase fort.* files, which synthe() erases anyway before SYNTHE. No K2 file is modified."""
    import atlas9 as k2_atlas9
    from path import path_model
    original = k2_atlas9.RunAtlas9.atlas9
    if getattr(original, '_salvador_model_check', False):
        return

    def atlas9(self, pset, ross_set, name):
        """Wrapper of K2's RunAtlas9.atlas9: if the model atmosphere already exists, copy it as K2 does and
        skip the ODF interpolation; otherwise run K2's original method (see _install_k2_model_check)."""
        if len(pset) == 5 and self.checkmod == 1:
            try:
                path0, strname = path_model().path1(pset[3])[4:6]
                name_mod = self.name_model(pset, strname, 1) if name == '*' else name
                if self.moddir == '@#@':
                    where = path0 + '/models-new/'
                elif self.moddir == '.':
                    where = './'
                else:
                    where = str(self.moddir) + '/'
                outmod = where + name_mod + '.dat'
                if os.path.exists(outmod):
                    if self.verbose > 0:
                        print('Model atmosphere already exists')
                    shutil.copyfile(outmod, 'guess_model_4other.dat')   # FIX: safe with spaces in the paths
                    return 1
            except Exception:
                pass                # anything unexpected: K2 decides, as before
        return original(self, pset, ross_set, name)

    atlas9._salvador_model_check = True
    k2_atlas9.RunAtlas9.atlas9 = atlas9


def _parse_k2_linelist(path):
    """Sections of a K2 linelist file, read as K2 does (first token of each line):
    [(marker, [file, ...]), ...]."""
    sections = []
    with open(path, 'r') as f:
        for row in f:
            tok = row.split()
            if not tok:
                continue
            if tok[0] in _K2_MARKERS:
                sections.append((tok[0], []))
            elif sections:
                sections[-1][1].append(tok[0])
    return sections


def _scan_linelist(path, width):
    """Check once that a linelist file is sorted in wavelength. Returns (sorted, wmin, wmax) in nm.
    A file with unparsable or blank lines (except at the end) counts as not sorted."""
    prev, wmin, blank = -np.inf, np.inf, False
    with open(path, 'rb') as f:
        for row in f:
            if not row.strip():
                blank = True
                continue
            if blank:
                return False, None, None
            try:
                w = float(row[:width])
            except ValueError:
                return False, None, None
            if w < prev:
                return False, None, None
            if w < wmin:
                wmin = w
            prev = w
    return True, float(wmin), float(prev)


def _linelist_record(path, width, index):
    """Sortedness and range of a linelist file, from the index (rebuilt if the file changed)."""
    st = os.stat(path)
    key = os.path.abspath(path)
    rec = index.get(key)
    if rec and rec['size'] == st.st_size and rec['mtime'] == st.st_mtime and rec['width'] == width:
        return rec, False
    if cfg.verbose == 1:
        print(f'Indexing linelist {path} (only once)')
    ok, wmin, wmax = _scan_linelist(path, width)
    rec = {'size': st.st_size, 'mtime': st.st_mtime, 'width': width, 'sorted': ok, 'wmin': wmin, 'wmax': wmax}
    index[key] = rec
    return rec, True


def _line_start_at(f, pos, width):
    """(byte offset, wavelength) of the first line starting at or after byte 'pos'. inf at the end."""
    if pos <= 0:
        f.seek(0)
    else:
        f.seek(pos - 1)
        f.readline()
    start = f.tell()
    row = f.readline()
    if not row.strip():
        return start, np.inf
    return start, float(row[:width])


def _first_offset(f, size, width, condition):
    """Byte offset of the first line whose wavelength satisfies 'condition' (file sorted)."""
    lo, hi = 0, size
    while lo < hi:
        mid = (lo + hi) // 2
        if condition(_line_start_at(f, mid, width)[1]):
            hi = mid
        else:
            lo = mid + 1
    return _line_start_at(f, lo, width)[0]


def _cut_linelist(src, dst, lo_nm, hi_nm, width):
    """Copy the lines of a sorted file with lo_nm <= wavelength <= hi_nm. False if there are none."""
    size = os.path.getsize(src)
    with open(src, 'rb') as f:
        start = _first_offset(f, size, width, lambda w: w >= lo_nm)
        end = _first_offset(f, size, width, lambda w: w > hi_nm)
        if end <= start:
            return False
        f.seek(start)
        data = f.read(end - start)
    with open(dst, 'wb') as g:
        g.write(data)
    return True


def _write_trimmed_linelist(w1, w2):
    """Write the trimmed linelist of the window [w1, w2] (see k2_linelist) and the cut files it
    points to. Returns (path of the linelist file, list of the cut files created)."""
    LINELIST_CUT_DIR.mkdir(exist_ok=True)
    try:
        index = json.loads(LINELIST_INDEX.read_text())
    except (OSError, ValueError):
        index = {}
    changed = False
    lo_nm = (w1 - LINELIST_MARGIN_A) / 10.0
    hi_nm = (w2 + LINELIST_MARGIN_A) / 10.0

    rows, created = [], []
    for marker, files in _parse_k2_linelist(K2_LINELIST):
        rows.append(marker)
        width = _WL_WIDTH.get(marker)
        for src in files:
            if width is None or not os.path.isfile(src) or os.path.getsize(src) == 0:
                rows.append(src)            # left to K2 exactly as before
                continue
            rec, new = _linelist_record(src, width, index)
            changed = changed or new
            if not rec['sorted']:
                rows.append(src)            # cannot cut safely: full file
                continue
            if rec['wmax'] < lo_nm or rec['wmin'] > hi_nm:
                continue                    # no lines near the window
            dst = LINELIST_CUT_DIR / f"{Path(src).stem}_{w1}_{w2}{Path(src).suffix}"
            if _cut_linelist(src, dst, lo_nm, hi_nm, width):
                rows.append(str(dst.resolve()))
                created.append(dst)

    if changed:
        LINELIST_INDEX.parent.mkdir(parents=True, exist_ok=True)
        tmp_index = LINELIST_INDEX.with_suffix(f'.{os.getpid()}.tmp')
        tmp_index.write_text(json.dumps(index, indent=1))
        os.replace(tmp_index, LINELIST_INDEX)       # atomic: safe with parallel workers

    out = (LINELIST_CUT_DIR / f"linelist_{w1}_{w2}.xkur").resolve()
    out.write_text('\n'.join(rows) + '\n')
    return str(out), created


def k2_linelist(w1, w2):
    """Linelist file to pass to K2 for the window [w1, w2] (A). The last 3 windows are kept."""
    if not TRIM_LINELIST:
        return K2_LINELIST
    key = (int(w1), int(w2))
    if key in _trim_cache:
        _trim_cache.move_to_end(key)
        return _trim_cache[key][0]
    try:
        out, created = _write_trimmed_linelist(*key)
    except Exception as e:
        print(f'Warning: linelist trimming failed ({type(e).__name__}: {e}). Using {K2_LINELIST}')
        return K2_LINELIST
    _trim_cache[key] = (out, created)
    while len(_trim_cache) > 3:
        _, (old_out, old_created) = _trim_cache.popitem(last=False)
        for p in [Path(old_out)] + old_created:
            try:
                p.unlink()
            except OSError:
                pass
    return out


def cleanup_linelist_cut():
    """Remove the trimmed linelists (end of run)."""
    _trim_cache.clear()
    if LINELIST_CUT_DIR.is_dir():
        shutil.rmtree(LINELIST_CUT_DIR, ignore_errors=True)

def add_delta_entry(type_abu, abu_var, zcode, var):
    """Add to the abundance-variation file passed to K2 the entry of the line: DELTA (abundance),
    GF (log gf) or ISOR (isotopic ratio) for the element zcode, with the value var."""
    filename = abu_var
    abu_str = f"{var:.2f}"

    if type_abu in ['ABU', 'ABU2', 'MOLEC', 'ROTAT', 'ABUF']:
        new_entry = f"DELTA {zcode} {abu_str}\n"
    elif type_abu == 'ABU_GF':
        new_entry = f"GF {zcode} {abu_str}\n"
    elif type_abu == 'ISOR':
        abu_str = f"{var:.0f}"
        new_entry = f"ISOR {zcode} {abu_str}\n"
    else:
        return # Leave the function if the keyword is not recognized

    # Whether a newline is needed before the new entry
    add_newline = False

    # --- CHECK OF THE FILE ---
    if os.path.exists(filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        if lines:
            if lines[0].strip() == "":
                has_text = any(entry.strip() != "" for entry in lines)

                if has_text:
                    print(f"Santarelli Error: the file '{filename}' starts with an empty row. Execution not possible.")
                    sys.exit(1)
                else:
                    open(filename, 'w').close()
            else:
                if not lines[-1].endswith('\n'):
                    add_newline = True

    with open(filename, "a") as f:
        if add_newline:
            f.write("\n")
        f.write(new_entry)


def read_delta_file(file_name):
    """Read an abundance-variation file: {element code: (value, keyword)} for its DELTA, GF and ABS
    rows. An empty dictionary if the file does not exist."""
    extracted_data = {}

    if not os.path.exists(file_name):
        print(f"Warning: file '{file_name}' does not exist.")
        return extracted_data

    with open(file_name, "r") as f:
        for text_line in f:
            clean_text_line = text_line.strip()

            if not clean_text_line:
                continue

            fields = clean_text_line.split()

            if len(fields) >= 3 and fields[0].upper() in ["DELTA", "GF", "ABS"]:
                try:
                    type_line = str(fields[0])
                    code_num = int(fields[1])
                    delta_num = float(fields[2])

                    extracted_data[code_num] = delta_num, type_line

                except ValueError:
                    print(f"Warning: Unable to convert line: '{clean_text_line}' to numbers.")  # FIX: was undefined 'clean_line'

    return extracted_data

def write_name_syn(met, alpha, teff, logg, vt, l1, l2, file_abu_var, type_abu, zcode_str, quantity):
    """Name of a synthetic spectrum in synth_out/, built from all the parameters that define it
    (metallicity, alpha, Teff, logg, vt, wavelength range, abundance variations), so that a
    synthetic computed once is found and reused."""
    prefix = 'm' if met < 0 else 'p'
    met_abs_scaled = int(round(abs(met) * 100))
    met1 = f"{prefix}{met_abs_scaled:03d}"

    teff1 = str(int(teff))
    logg1 = f"{logg:.2f}".replace('.', '')
    vt1 = f"{vt:.2f}".replace('.', '')

    alpha_codes = {1: 'p04', 2: 'p02', 3: 'p00', 4: 'm02', 5: 'm04'}    # K2 alpha code -> [alpha/Fe]
    if alpha not in alpha_codes:
        raise ValueError(f'alpha = {alpha}: must be 1-5 (K2 code of the alpha-enhancement)')
    alpha1 = alpha_codes[alpha]

    l1 = str(l1)
    l2 = str(l2)

    name_syn = 'a'+met1+'t'+teff1+'g'+logg1+'k'+vt1+'_cl_a'+alpha1

    values = read_delta_file(file_abu_var)
    for zcode, (delta, type_line) in sorted(values.items()):
        elem_tmp = get_element_symbol(zcode, symbol_map)
        elem_tmp = strip_ion(elem_tmp)

        delta_arr = np.round(delta, 2)

        #
        # FIX: round() instead of int() truncation. abs(0.29)*100 = 28.999999999999996,
        # so int() gave '028' and 0.28/0.29 shared the same cache name in synth_out.
        prefix = 'm' if delta < 0 else 'p'
        delta1 = f"{prefix}{int(round(abs(delta_arr) * 100)):03}"

        #
        delta2 = f"{int(round(abs(delta_arr) * 100)):03}"

        #
        if type_line == 'ABS':
            name_syn += f"_{elem_tmp}_abs{delta2}"
        else:
            name_syn += f"_{elem_tmp}_{delta1}"

    if type_abu == 'ISOR':
        elem_tmp = get_element_symbol(zcode_str, symbol_map)
        elem_tmp = strip_ion(elem_tmp)
        quantity = f"{quantity:.0f}"

        name_syn += f"_ISOR_{elem_tmp}_{quantity}"

    name_syn += f"_{l1}_{l2}"
    return name_syn


def autok(param1, delta, window):
    """Synthetic spectrum computed by K2 (SYNTHE) for the line of param1, with the abundance of the
    element changed by delta, in a window around the line. A spectrum already in synth_out/ is
    reused. Returns (wavelength, flux), not broadened."""
    teff = int(param1['teff'])
    logg = float(param1['logg'])
    metal = float(param1['met'])
    alpha = int(param1['alpha'])
    vturb = float(param1['vt'])
    resol = int(param1['res'])
    line = float(param1['line'])
    zcode = int(param1['nelem'])
    abu_var = str(param1['abu_var'])
    type_abu = str(param1['tana'])
    abu0 = float(param1['abu0'])

    if cfg.type_norm == 3:
        window = 50

    sep = '  '
    w1 = int(np.floor(line - (window/2)))
    w2 = int(np.ceil(line + (window/2)))

    # FIX: work on a temporary copy of the user's abundance-variation file, so the original
    # is never modified (before, a crash or an exit between add_delta_entry and
    # removing it again left a stray DELTA line in the user's input file).
    abu_var_k2 = abu_var_tmp_name
    if os.path.exists(abu_var):
        shutil.copyfile(abu_var, abu_var_k2)
    else:
        open(abu_var_k2, 'w').close()

    # Add the abundance variation into the temporary abu_var file
    add_delta_entry(type_abu, abu_var_k2, zcode, delta)

    # writing the syn name
    name_syn = write_name_syn(metal, alpha, teff, logg, vturb, w1, w2, abu_var_k2, type_abu, zcode, delta)

    # LANCIA K2
    SYNTH_OUT.mkdir(parents=True, exist_ok=True)

    name_syn = name_syn+'.nb'

    # 1. Define all paths as Path objects at the beginning for clarity.
    output_dir = SYNTH_OUT
    source_file_path = Path(name_syn)
    destination_file_path = output_dir / name_syn

    # 2. Ensure the output directory exists before doing anything else.
    output_dir.mkdir(exist_ok=True)

    # 3. Check if the FINAL file already exists at the destination.
    if destination_file_path.exists():
        if cfg.verbose == 1:
            print("Synthetic spectrum already exists.")
    else:
        written_el = get_element_symbol(zcode, symbol_map)
        if cfg.verbose == 1:    #TO_ADJUST
            print(f"Calling k2: A({written_el}) = {abu0 + delta - float(param1['delta']):.2f}")

        # SCRIVI INPUT.K2 (only when K2 runs). SPEED: trimmed linelist for this window
        with open('input.k2', 'w') as inpk:
            inpk.write('synthe ' + str(teff) + sep + str(logg) + sep + \
                str(metal) + sep + str(alpha) + sep + str(vturb) + \
                ' ' + k2_linelist(w1, w2) + ' ' + abu_var_k2 + sep + str(w1) + sep + str(w2) + \
                sep + str(resol)  + ' 0 ' + name_syn[:-3] + ' 0' + '\n')
            inpk.write('VERBOSE 0\n')
            inpk.write('OUTPUT 0\n')
            inpk.write('CHECKSYN 1\n')

        run_k2()    # SPEED: in-process call (was os.system('k2.py'))

        # 5. Move the file using pathlib's .rename() method, which is robust.
        if source_file_path.exists():
            source_file_path.rename(destination_file_path)
        else:
            # FIX: stop with a clear message (before: a bare FileNotFoundError on the next line)
            raise RuntimeError(f"K2 did not create the synthetic spectrum '{source_file_path.name}'. "
                f"See the K2 messages above (in parallel mode: {WORK_DIR.name}/worker_*.log) and the "
                f"K2 input in {Path.cwd() / 'input.k2'}; common causes: a wrong path in linelist.xkur, "
                f"a missing model atmosphere, or the Kurucz executables not found.")


    xs, ys0 = read_two_columns(SYNTH_OUT / name_syn)    # SPEED: faster reader

    spec_tmp = Spectrum(spectral_axis=xs*u.AA, flux=ys0*u.count)
    xs, ys0 = spec_tmp.spectral_axis, spec_tmp.flux

    # FIX: remove the temporary copy instead of editing the user's file
    if os.path.exists(abu_var_k2):
        os.remove(abu_var_k2)

    return xs, ys0

def eqw_k2(param1, delta, window):
    """Equivalent width of the line computed by K2 (eqw; eqwb for ABUF) with the abundance changed by
    delta. Returns (EW [mA], EP [eV], log gf); 99.99, 99.999, 99.999 if K2 gives no result."""
    teff = int(param1['teff'])
    logg = float(param1['logg'])
    metal = float(param1['met'])
    alpha = int(param1['alpha'])
    vturb = float(param1['vt'])
    line = float(param1['line'])
    zcode = format(param1['nelem'], ".2f")

    delta = format(delta, ".2f")

    if str(param1['tana']) == 'ABUF':
        first_key = 'eqwb '
    else:
        first_key = 'eqw '

    sep = '  '
    # SPEED: same window as autok, so the trimmed linelist of this line is reused
    win = 50 if cfg.type_norm == 3 else window
    w1 = int(np.floor(line - (win/2)))
    w2 = int(np.ceil(line + (win/2)))
    # SCRIVI INPUT.K2
    inpk = open('input.k2','w')
    inpk.write(first_key + str(teff) + sep + str(logg) + sep + \
        str(metal) + sep + str(alpha) + sep + str(vturb) + \
        ' ' + k2_linelist(w1, w2) + ' ' + str(BASE_DIR / 'delta.tmp') + ' ' + sep + str(line) + \
        sep + zcode  + sep + delta + '\n')
    inpk.write('VERBOSE 0\n')

    inpk.close()

    # LANCIA K2
    # FIX: remove the EW of the previous line, so a silent K2 failure cannot reuse it
    if os.path.exists('info_ew.txt'):
        os.remove('info_ew.txt')
    exit_code = run_k2()    # SPEED: in-process call (was os.system('k2.py'))

    if exit_code == 0:
        try:
            wave, ew_k2, EP, loggf = np.loadtxt('info_ew.txt', unpack=True)
            print('EW = %.2f mA' % ew_k2, '\n')
        except (FileNotFoundError, IOError, ValueError) as e:
            print(f'Warning: the EW of K2 could not be read ({type(e).__name__}): EW set to 99.99')
            ew_k2, EP, loggf = 99.99, 99.999, 99.999
    else:
        print('Warning: eqw of k2 did not calculate the EW!!!')
        ew_k2, EP, loggf = 99.99, 99.999, 99.999

    return ew_k2, EP, loggf
