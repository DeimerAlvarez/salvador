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
_GRID_NAME = re.compile(r'^a(?P<met>[mp]\d{3})t(?P<teff>\d+)g(?P<logg>\d+)k(?P<vt>\d+)_cl_a[mp]\d{2}_(?P<elem>[A-Za-z]+)_[mp]\d{3}_')


def grid_name_fields(name: str) -> dict[str, str] | None:
    """Fields of the name of a grid synthetic.

    Args:
        name: File name, e.g. 'am080t4060g110k160_cl_ap04_Ca_m020_6000_6810.nb'.

    Returns:
        {'met': 'm080', 'teff': '4060', 'logg': '110', 'vt': '160', 'elem': 'Ca'}, or None if the
        name does not follow the scheme.
    """
    match = _GRID_NAME.match(Path(str(name)).name)
    return match.groupdict() if match else None


def grid_name_matches(name: str, teff1: str, logg1: str, vt1: str, elem: str) -> bool:
    """True if a grid synthetic has exactly these Teff, logg, vt and element.

    FIX: before, the fields were searched as substrings of the whole name, so logg '110' also
    matched 'k110' or a variation 'm110', and element 'C' matched 'Ca', 'Co', 'Cr', ...

    Args:
        name: File name of the synthetic.
        teff1: Teff as in the names ('4060').
        logg1: logg x 100 as in the names ('110' for 1.10).
        vt1: vt x 100 as in the names ('160' for 1.60).
        elem: Bare element symbol ('Ca').
    """
    fields = grid_name_fields(name)
    return fields is not None and (fields['teff'], fields['logg'], fields['vt'], fields['elem']) == (teff1, logg1, vt1, elem)


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
                return wave, flux

        else:
            wave, flux = np.genfromtxt(filepath, comments='#', usecols=(0, 1), dtype=float, unpack=True, autostrip=True)
            return wave, flux

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
    try:
        parts = filename.removesuffix('.nb').split('_')
        w_start = float(parts[-2])
        w_end = float(parts[-1])

        return w_start <= line_value <= w_end
    except (IndexError, ValueError):
        return False

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

