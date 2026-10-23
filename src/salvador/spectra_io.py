"""Reading spectra: observed spectra, two-column synthetic spectra, grids of precomputed synthetics;
cleaning of temporary files."""
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import astropy.io.fits as pyfits
import astropy.units as u
from PyAstronomy import pyasl

from .config import cfg

def cleaning_tmp_files(patterns: list[str], directory: str | Path = '.', newer_than: float | None = None) -> None:
    """Delete the files of the directory matching the patterns (temporary files of a run).

    Args:
        patterns: Glob patterns of the temporary files.
        directory: Directory to clean.
        newer_than: A time.time() value (e.g. the start of the run): only files created or modified
            after it (1 s tolerance) are deleted, so the user's own files with similar names
            (guess*, *_cl_*) are never touched. None deletes every match (behaviour before this option).
    """
    search_path = Path(directory)
    for pattern in patterns:
        for filepath in search_path.glob(pattern):
            try:
                if filepath.is_file() and (newer_than is None or filepath.stat().st_mtime >= newer_than - 1.0): # 1 s: coarse file-system clock
                    filepath.unlink()
            except OSError as e:
                print(f"Error deleting file {filepath.name}: {e}")


def read_two_columns(path):
    """SPEED: fast reader for 2-column ASCII spectra. Same arrays as
    np.genfromtxt(path, comments='#', usecols=(0,1), unpack=True), ~6x faster on large files."""
    try:
        data = pd.read_csv(path, sep=r'\s+', comment='#', header=None, usecols=[0, 1], dtype=float).to_numpy()
        return data[:, 0], data[:, 1]
    except (ValueError, pd.errors.ParserError):
        return np.genfromtxt(path, comments='#', usecols=(0, 1), dtype='float', unpack=True)


### SPEED: grid files are read once and kept in memory (the same files are needed for every
### line of an element and for every parameter-error rerun). Each entry costs about
### 16 bytes x number of rows; lower GRID_CACHE_SIZE if memory becomes an issue.
GRID_CACHE_SIZE = 32

@lru_cache(maxsize=GRID_CACHE_SIZE)
def read_grid_file(path_str):
    """Read a grid synthetic (cached, read-only arrays: see GRID_CACHE_SIZE)."""
    w, f = read_two_columns(path_str)
    w.setflags(write=False)
    f.setflags(write=False)
    return w, f


### names of the synthetics: a<met>t<teff>g<logg>k<vt>_cl_a<alpha>_<Elem>_<var>_<w1>_<w2>.nb
### e.g. am080t4060g110k160_cl_ap04_Ca_m020_6000_6810.nb (see write_name_syn)
_GRID_NAME = re.compile(r'^a(?P<met>[mp]\d{3})t(?P<teff>\d+)g(?P<logg>\d+)k(?P<vt>\d+)_cl_a(?P<alpha>[mp]\d{2})_(?P<elem>[A-Za-z]+)_[mp]\d{3}_')

### K2 code of the alpha-enhancement -> [alpha/Fe] as written in the names of the synthetics
ALPHA_CODES = {1: 'p04', 2: 'p02', 3: 'p00', 4: 'm02', 5: 'm04'}


def model_name_codes(teff: float, logg: float, vt: float, met: float, alpha: int) -> dict[str, str]:
    """Atmospheric parameters as they are written in the names of the synthetic spectra.

    Args:
        teff: Effective temperature [K].
        logg: Surface gravity.
        vt: Microturbulence [km/s].
        met: [M/H] of the model atmosphere.
        alpha: K2 code of the alpha-enhancement (1-5).

    Returns:
        {'teff': '4060', 'logg': '110', 'vt': '160', 'met': 'm080', 'alpha': 'p04'}.

    Raises:
        ValueError: If alpha is not 1-5.
    """
    if alpha not in ALPHA_CODES:
        raise ValueError(f'alpha = {alpha}: must be 1-5 (K2 code of the alpha-enhancement)')
    prefix = 'm' if met < 0 else 'p'
    return {'teff': str(int(teff)), 'logg': f"{logg:.2f}".replace('.', ''), 'vt': f"{vt:.2f}".replace('.', ''),
            'met': f"{prefix}{int(round(abs(met) * 100)):03d}", 'alpha': ALPHA_CODES[alpha]}


def grid_name_fields(name: str) -> dict[str, str] | None:
    """Fields of the name of a grid synthetic.

    Args:
        name: File name, e.g. 'am080t4060g110k160_cl_ap04_Ca_m020_6000_6810.nb'.

    Returns:
        {'met': 'm080', 'teff': '4060', 'logg': '110', 'vt': '160', 'alpha': 'p04', 'elem': 'Ca'},
        or None if the name does not follow the scheme.
    """
    match = _GRID_NAME.match(Path(str(name)).name)
    return match.groupdict() if match else None


def grid_name_matches(name: str, teff1: str, logg1: str, vt1: str, elem: str,
                      met1: str | None = None, alpha1: str | None = None) -> bool:
    """True if a grid synthetic has exactly these Teff, logg, vt and element (and [M/H] and
    alpha-enhancement, if given).

    FIX: before, the fields were searched as substrings of the whole name, so logg '110' also
    matched 'k110' or a variation 'm110', and element 'C' matched 'Ca', 'Co', 'Cr', ...

    Args:
        name: File name of the synthetic.
        teff1: Teff as in the names ('4060').
        logg1: logg x 100 as in the names ('110' for 1.10).
        vt1: vt x 100 as in the names ('160' for 1.60).
        elem: Bare element symbol ('Ca').
        met1: [M/H] as in the names ('m080'); None: any.
        alpha1: [alpha/Fe] as in the names ('p04'); None: any.
    """
    fields = grid_name_fields(name)
    if fields is None or (fields['teff'], fields['logg'], fields['vt'], fields['elem']) != (teff1, logg1, vt1, elem):
        return False
    return (met1 is None or fields['met'] == met1) and (alpha1 is None or fields['alpha'] == alpha1)


def _grid_window(name: str) -> tuple[float, float] | None:
    """Wavelength range (w1, w2) at the end of the name of a grid synthetic (..._w1_w2.nb)."""
    try:
        parts = str(name).removesuffix('.nb').split('_')
        return float(parts[-2]), float(parts[-1])
    except (IndexError, ValueError):
        return None


def select_grid_synthetics(names_all, codes: dict[str, str], elem: str, line: float) -> tuple[list[str], np.ndarray, str | None]:
    """Grid synthetics for one line and one set of atmospheric parameters.

    Only the synthetics with exactly the Teff, logg, vt, [M/H] and alpha-enhancement of codes
    (see model_name_codes) and the element are used. FIX: before, [M/H] and alpha were not
    checked, so spectra computed with different model atmospheres were mixed. If the line is in
    more than one wavelength range of the grid, the range where it is farthest from the edges is
    used (before, the ranges were mixed); synthetics with the same abundance are used once.

    Args:
        names_all: Names of all the synthetics of the grid (list of PATH_SYN).
        codes: Parameters as in the names (model_name_codes).
        elem: Bare element symbol.
        line: Wavelength of the line [A].

    Returns:
        (names, abundance offsets, reason): names and offsets sorted by abundance (see
        process_names_grid); when no synthetic can be used, empty lists and the reason.
    """
    names_all = [str(n) for n in np.atleast_1d(names_all)]
    same_star = [n for n in names_all if grid_name_matches(n, codes['teff'], codes['logg'], codes['vt'], elem)]
    selected = [n for n in same_star if grid_name_matches(n, codes['teff'], codes['logg'], codes['vt'], elem,
                                                          codes['met'], codes['alpha'])]
    if not selected:
        if same_star:
            found = sorted({f"[M/H] {grid_name_fields(n)['met']} alpha {grid_name_fields(n)['alpha']}" for n in same_star})
            return [], np.array([]), (f"no synthetic spectra with [M/H] {codes['met']} and alpha {codes['alpha']} "
                                      f"(the grid has {', '.join(found)})")
        return [], np.array([]), (f"no synthetic spectra for Teff {codes['teff']}, logg {codes['logg']}, "
                                  f"vt {codes['vt']}, element {elem}")

    windows = {}
    for n in selected:
        window = _grid_window(n)
        if is_line_in_grid(n, line):
            windows.setdefault(window, []).append(n)
    if not windows:
        return [], np.array([]), 'no synthetic spectra covering the line'
    best = max(windows, key=lambda w: min(line - w[0], w[1] - line))
    if len(windows) > 1:
        print(f"Note: line {line:.3f} is in {len(windows)} wavelength ranges of the grid: "
              f"the synthetics of {best[0]:.0f}-{best[1]:.0f} A are used")

    names, values = process_names_grid(windows[best])
    values, first = np.unique(values, return_index=True)
    if first.size < len(names):
        print(f"Note: synthetics with the same abundance for line {line:.3f}: each abundance is used once")
    return [str(names[k]) for k in first], values, None


def process_names_grid(filenames):
    """Abundance offsets encoded in the names of grid synthetics (Fe: [M/H] + variation; alpha
    elements: variation + alpha enhancement; others: variation). Returns the valid names and their
    offsets, sorted by offset (abu_absolute turns them into absolute abundances)."""

    alpha_elements = {'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti'}
    final_values = []
    pattern = re.compile(r'^(a[mp]\d{3}).*_cl_(a[mp]\d{2})_([A-Za-z]+)_([mp]\d{3})_')

    for file_name in filenames:
        match = pattern.search(file_name)
        if match:
            met_str, alpha_str, element, abundance_str = match.groups()

            # --- Numerical values ---
            met_sign = -1.0 if met_str[1] == 'm' else 1.0
            metallicity = (float(met_str[2:]) / 100.0) * met_sign

            alpha_sign = -1.0 if alpha_str[1] == 'm' else 1.0
            alpha_val = (float(alpha_str[2:]) / 10.0) * alpha_sign

            abundance_sign = -1.0 if abundance_str[0] == 'm' else 1.0
            abundance_var = (float(abundance_str[1:]) / 100.0) * abundance_sign

            if element == 'Fe':
                computed_value = metallicity + abundance_var
            elif element in alpha_elements:
                computed_value = abundance_var + alpha_val
            else:
                computed_value = abundance_var

            final_values.append(computed_value)
        else:
            final_values.append(np.nan)

    names_array = np.array(filenames)
    values_array = np.array(final_values)

    valid_indices = ~np.isnan(values_array)
    valid_names, valid_values = names_array[valid_indices], values_array[valid_indices]

    sorted_indices = np.argsort(valid_values)
    sorted_names, sorted_values = valid_names[sorted_indices], valid_values[sorted_indices]

    return sorted_names, sorted_values


def _finite_pixels(wave: np.ndarray, flux: np.ndarray, name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Remove the pixels with a NaN or infinite wavelength or flux (bad pixels).

    FIX: before, a NaN pixel near a line gave a NaN chi2: the run stopped (grid mode) or the
    abundance was searched in the wrong direction (K2).

    Args:
        wave: Wavelengths.
        flux: Fluxes.
        name: Name of the spectrum, for the message.

    Returns:
        (wave, flux) without those pixels; (None, None) if no pixel is left.
    """
    wave, flux = np.atleast_1d(wave), np.atleast_1d(flux)
    good = np.isfinite(wave) & np.isfinite(flux)
    if not np.all(good):
        print(f"Warning: {np.count_nonzero(~good)} pixels of {name} with a NaN or infinite value are not used")
    if not np.any(good):
        print(f"Error: no valid pixel in {name}")
        return None, None
    return wave[good], flux[good]


def read_spec(spec_file):
    """Read an observed spectrum: FITS (wavelengths from CRVAL1, CDELT1 or CD1_1, CRPIX1) or ASCII
    (two columns). Returns (wavelength, flux), or (None, None) if the file cannot be read."""
    filepath = Path(spec_file)

    if not filepath.exists():
        print(f"Error: File not found: {spec_file}")
        return None, None

    try:
        # Check whether the extension is .fits (case insensitive)
        if filepath.suffix.lower() == '.fits':
            with pyfits.open(filepath) as hdulist:
                header = hdulist[0].header
                flux = np.array(hdulist[0].data, dtype=np.float64).ravel() # .ravel() flattens the array

                crval = header['CRVAL1']
                cdelt = header.get('CDELT1', header.get('CD1_1', 1.0)) # Compatibility with CD1_1
                pix = header['CRPIX1']  # Reference Pixel
                wave = crval - (cdelt * pix - cdelt) + np.arange(len(flux)) * cdelt
                return _finite_pixels(wave, flux, filepath.name)

        else:
            wave, flux = np.genfromtxt(filepath, comments='#', usecols=(0, 1), dtype=float, unpack=True, autostrip=True)
            return _finite_pixels(wave, flux, filepath.name)

    except KeyError as e:
        print(f"Error: Missing key in the header of FITS file {filepath.name}: {e}")
        return None, None
    except (ValueError, IndexError):
        print("Error: Format of the ASCII file not valid or empty file.")
        return None, None
    except Exception as e:
        print(f"Error while reading {filepath.name}: {e}")
        return None, None


def is_line_in_grid(filename, line_value):
    """True if the wavelength range in the name of a grid synthetic (..._w1_w2.nb) contains the line."""
    window = _grid_window(filename)
    return window is not None and window[0] <= line_value <= window[1]

def readsyn(name, res, wcen):
    """Read and broaden the grid synthetics. Returns wl_syn, the tuple of broadened spectra and
    the indices (in 'name') of the files actually used.
    FIX: a file that could not be read or had no points within +-50 A of the line was skipped
    (or reset the whole list), so the spectra no longer matched the abundance array. The
    caller now uses 'used' to keep spectra and abundances aligned."""
    list_syn_br = ()
    used = []
    wl_syn = np.array([])
    path_syn_tmp = Path(cfg.path_syn).parent
    for k, filename in enumerate(name):
        try:
            w, f = read_grid_file(str(path_syn_tmp / filename)) # SPEED: cached + faster reader
        except Exception as e:
            print(f"Error reading file {path_syn_tmp / filename}: {e}")
            continue

        ind = np.where((w>(wcen-50)) & (w<(wcen+50)))
        wl_tmp, f = w[ind], f[ind]
        if f.size != 0:
            bbb, fwhm = pyasl.instrBroadGaussFast(wl_tmp, f, res, edgeHandling='firstlast', fullout=True, maxsig=5., equid=True)

            wl_syn, bbb = wl_tmp * u.AA, bbb * u.count
            list_syn_br += (bbb,)
            used.append(k)
        else:
            print(f"Warning: no points within +-50 A of {wcen} in {filename}, file skipped")

    return wl_syn, list_syn_br, used

