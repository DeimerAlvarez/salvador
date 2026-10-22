"""Chemical elements: solar abundances (solar_abu.txt), element symbols, model and absolute abundances."""
from collections.abc import Iterator, Mapping
from importlib import resources
from pathlib import Path
import numpy as np

from .config import cfg

SOLAR_FILE = 'solar_abu.txt'


def solar_table_path() -> Path:
    """The table of solar abundances, solar_abu.txt, installed with salvador (salvador/data/)."""
    return Path(str(resources.files('salvador') / 'data' / SOLAR_FILE))


###
def read_solar_data(path=None):
    """Read the table of solar abundances (default: solar_table_path()): one list of fields per
    row, comments (#) skipped. Returns None if the file does not exist."""
    path = solar_table_path() if path is None else Path(path)
    try:
        data = []
        with open(path, 'r') as file:
            for line in file:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if parts:
                        data.append(parts)
        return data
    except FileNotFoundError:
        print(f"Error: file '{path}' not found.")
        return None

def abu_dict(solar_data, column=2):
    """Dictionary {element symbol: solar abundance}, using the given column of solar_abu.txt
    (SOLAR_REF). Rows with too few columns or a non-numeric value are skipped with a warning."""
    abundance_dict = {}
    for row in solar_data:
        if len(row) >= column + 1:  # Check that the requested column exists
            element_name = row[1]
            try:
                abundance = float(row[column])
                abundance_dict[element_name] = abundance
            except ValueError:
                print(f"Warning: Skipping invalid abundance in row: {row}")
        else:
            print(f"Warning: Skipping row with insufficient columns: {row}")
    return abundance_dict


def get_element_symbol(nelem, symbol_map):
    """Symbol of a species from its code: 26.00 -> 'Fe', 26.01 -> 'FeII', 26.02 -> 'FeIII'.
    Returns None for an unknown atomic number."""
    int_nelem = int(nelem)
    symbol = symbol_map.get(int_nelem)
    if symbol:
        if "{:.2f}".format(nelem).endswith("00"):
            return symbol
        elif "{:.2f}".format(nelem).endswith("01"):
            return f"{symbol}II"
        else:
            return f"{symbol}III"
    else:
        return None


def strip_ion(elem):
    """Return the bare element symbol: 'FeII' -> 'Fe', 'FeIII' -> 'Fe', 'Fe' -> 'Fe'.
    FIX: elem.replace('II', '') turned 'FeIII' into 'FeI'. The check against the solar
    table also keeps iodine ('I', 'III' = I II) correct."""
    for suffix in ('III', 'II'):
        if elem.endswith(suffix) and elem[:-len(suffix)] in abundance_link:
            return elem[:-len(suffix)]
    return elem


def define_abu0(nelem, alpha, met, delta, elem):
    """Abundance of the element in the model atmosphere: solar value + [M/H], + the alpha-enhancement
    of the chemical mixture (alpha = 1: +0.4, 2: +0.2, ...) for the alpha elements, + delta (the
    var column of the linelist)."""
    elem_tmp = strip_ion(elem)
    abu_sol = abundance_link.get(elem_tmp)

    target_elements = {8, 10, 12, 14, 16, 18, 20, 22}
    abu0 = abu_sol + met
    if alpha == 1:
        if nelem in target_elements:
            abu0 = abu0 + 0.4
    elif alpha == 2:
        if nelem in target_elements:
            abu0 = abu0 + 0.2
    elif alpha == 3:
        if nelem in target_elements:
            abu0 = abu0 + 0.0
    elif alpha == 4:
        if nelem in target_elements:
            abu0 = abu0 - 0.2
    elif alpha == 5:
        if nelem in target_elements:
            abu0 = abu0 - 0.4

    return abu0 + delta


def abu_absolute(elem, abu_fin, met):
    """Absolute abundance from a value relative to the Sun: + solar value + [M/H] (Fe: + solar
    value only). Rounded to 0.01 dex. Used for the abundances in the names of grid synthetics."""
    elem_tmp = strip_ion(elem)
    abu_sol = abundance_link.get(elem_tmp)
    if elem_tmp != 'Fe':
        abu_abs = abu_fin + abu_sol + met
    else:
        abu_abs = abu_fin + abu_sol

    return np.round(abu_abs, 2)

class _SolarAbundances(Mapping):
    """{element symbol: solar abundance} from the SOLAR_REF column of solar_abu.txt. Built at the
    first use, because the column is a setting of the run (cfg)."""

    def __init__(self) -> None:
        self._data: dict[str, float] | None = None

    def _table(self) -> dict[str, float]:
        if self._data is None:
            self._data = abu_dict(solar_data, column=cfg.solar_ref)
        return self._data

    def __getitem__(self, key: str) -> float:
        return self._table()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table())

    def __len__(self) -> int:
        return len(self._table())


### solar data
solar_data = read_solar_data()
if not solar_data:
    raise FileNotFoundError(f'Mandatory file {SOLAR_FILE} is missing or empty ({solar_table_path()})')

abundance_link = _SolarAbundances()

# dictionary: atomic number -> element symbol
symbol_map = {}
for line in solar_data:
    num, symbol = line[0], line[1]
    symbol_map[int(num)] = symbol
