"""Elements and solar abundances; names of synthetic spectra (K2 cache and grids); DELTA files."""
import numpy as np
import pytest

from salvador.elements import (abu_absolute, abundance_link, define_abu0, get_element_symbol, read_solar_data,
                               solar_table_path, strip_ion, symbol_map)
from salvador.k2_interface import add_delta_entry, read_delta_file, write_name_syn
from salvador.spectra_io import (grid_name_fields, grid_name_matches, is_line_in_grid, model_name_codes,
                                 process_names_grid, select_grid_synthetics)


# ------------------------------------------------------------------------- elements
def test_solar_table_is_installed():
    path = solar_table_path()
    assert path.is_file() and path.name == 'solar_abu.txt'
    rows = read_solar_data()
    assert rows[0][:2] == ['1', 'H'] and len(rows) >= 92


@pytest.mark.parametrize('code, symbol', [(26.00, 'Fe'), (26.01, 'FeII'), (26.02, 'FeIII'), (3.00, 'Li'), (999.0, None)])
def test_element_symbol(code, symbol):
    assert get_element_symbol(code, symbol_map) == symbol


@pytest.mark.parametrize('name, bare', [('FeII', 'Fe'), ('FeIII', 'Fe'), ('Fe', 'Fe'), ('I', 'I'), ('III', 'I'), ('Ti', 'Ti')])
def test_strip_ion(settings, name, bare):
    assert strip_ion(name) == bare


def test_solar_reference_column(settings):
    assert abundance_link['Fe'] == pytest.approx(7.52)         # Caffau & Lodders (default, SOLAR_REF = 2)


def test_model_abundance_with_alpha_enhancement(settings):
    # Mg is an alpha element: + 0.4 dex with alpha code 1, - 0.4 with code 5; Fe is not
    assert define_abu0(12, 1, -1.5, 0.0, 'Mg') == pytest.approx(abundance_link['Mg'] - 1.5 + 0.4)
    assert define_abu0(12, 5, -1.5, 0.0, 'Mg') == pytest.approx(abundance_link['Mg'] - 1.5 - 0.4)
    assert define_abu0(26, 1, -1.5, 0.2, 'FeII') == pytest.approx(abundance_link['Fe'] - 1.5 + 0.2)


def test_absolute_abundance(settings):
    assert abu_absolute('Fe', -1.5, -1.5) == pytest.approx(abundance_link['Fe'] - 1.5)
    assert abu_absolute('Ti', 0.3, -1.5) == pytest.approx(round(abundance_link['Ti'] + 0.3 - 1.5, 2))


# ------------------------------------------------------------------------- K2 names and DELTA files
def test_name_of_synthetic(settings, tmp_path):
    abu_var = tmp_path / 'abu_var.txt'
    add_delta_entry('ABU', abu_var, 26, 0.29)
    name = write_name_syn(-1.54, 1, 4248, 0.77, 1.95, 7459, 7465, abu_var, 'ABU', 26, 0.29)
    # 0.29 must not become 028 (int(28.999...) truncation)
    assert name == 'am154t4248g077k195_cl_ap04_Fe_p029_7459_7465'


def test_name_of_synthetic_with_invalid_alpha(settings, tmp_path):
    with pytest.raises(ValueError, match='alpha'):
        write_name_syn(-1.5, 7, 4000, 1.0, 1.5, 7000, 7010, tmp_path / 'none.txt', 'ABU', 26, 0.0)


def test_delta_file_round_trip(tmp_path):
    path = tmp_path / 'abu_var.txt'
    path.write_text('DELTA 12 0.30')             # no final newline: a newline must be added
    add_delta_entry('ABU_GF', path, 22, -0.1)
    add_delta_entry('ISOR', path, 6, 12.0)
    add_delta_entry('UNKNOWN', path, 8, 1.0)     # ignored
    assert path.read_text() == 'DELTA 12 0.30\nGF 22 -0.10\nISOR 6 12\n'
    assert read_delta_file(path) == {12: (0.30, 'DELTA'), 22: (-0.10, 'GF')}


def test_missing_delta_file(tmp_path, capsys):
    assert read_delta_file(tmp_path / 'none.txt') == {}
    assert 'does not exist' in capsys.readouterr().out


# ------------------------------------------------------------------------- grids
GRID = ['am080t4060g110k160_cl_ap04_Ca_m020_6000_6810.nb',
        'am080t4060g110k160_cl_ap04_Ca_p040_6000_6810.nb',
        'am080t4060g110k160_cl_ap04_Fe_m020_4750_6850.nb',
        'am080t4060g110k160_cl_ap04_Fe_p010_4750_6850.nb',
        'am080t4060g160k110_cl_ap04_Fe_m020_4750_6850.nb',    # logg 1.60, vt 1.10
        'am080t4060g110k160_cl_ap04_C_p010_4000_5000.nb',
        'not_a_grid_name.nb']


def test_grid_name_fields():
    assert grid_name_fields(GRID[0]) == {'met': 'm080', 'teff': '4060', 'logg': '110', 'vt': '160', 'alpha': 'p04', 'elem': 'Ca'}
    assert grid_name_fields('not_a_grid_name.nb') is None


def test_grid_selection_is_exact():
    def select(elem):
        return [n for n in GRID if grid_name_matches(n, '4060', '110', '160', elem)]
    assert select('Fe') == GRID[2:4]             # not the file with logg 1.60 and vt 1.10
    assert select('C') == [GRID[5]]              # not the Ca files
    assert select('Ca') == GRID[:2]


def test_process_names_grid_sorts_by_abundance():
    names, values = process_names_grid([GRID[3], GRID[2], 'not_a_grid_name.nb'])
    assert list(names) == [GRID[2], GRID[3]]
    # Fe: [M/H] of the name + variation
    assert np.allclose(values, [-0.80 - 0.20, -0.80 + 0.10])
    _, values = process_names_grid([GRID[1]])
    assert np.allclose(values, [0.40 + 0.4])     # alpha element: variation + alpha enhancement


@pytest.mark.parametrize('line, inside', [(6500.0, True), (6000.0, True), (6900.0, False)])
def test_line_in_grid(line, inside):
    assert is_line_in_grid(GRID[0], line) is inside
    assert is_line_in_grid('bad.nb', line) is False


# ------------------------------------------------------------------- grid selection (1.1.0)
CODES = model_name_codes(4060, 1.10, 1.60, -0.80, 1)


def test_model_name_codes():
    assert CODES == {'teff': '4060', 'logg': '110', 'vt': '160', 'met': 'm080', 'alpha': 'p04'}
    assert model_name_codes(5000, 4.5, 1.0, 0.0, 3)['met'] == 'p000'
    with pytest.raises(ValueError):
        model_name_codes(5000, 4.5, 1.0, 0.0, 6)


def test_grid_selection_checks_metallicity_and_alpha():
    other_met = 'am150t4060g110k160_cl_ap04_Ca_p000_6000_6810.nb'
    other_alpha = 'am080t4060g110k160_cl_ap00_Ca_p020_6000_6810.nb'
    names, values, why = select_grid_synthetics(GRID + [other_met, other_alpha], CODES, 'Ca', 6500.0)
    assert why is None and names == GRID[:2]
    assert np.allclose(values, [-0.20 + 0.4, 0.40 + 0.4])
    _, _, why = select_grid_synthetics([other_met], CODES, 'Ca', 6500.0)
    assert 'm080' in why and 'm150' in why             # the [M/H] available are reported
    _, _, why = select_grid_synthetics(GRID, CODES, 'Ti', 6500.0)
    assert why.startswith('no synthetic spectra for')
    _, _, why = select_grid_synthetics(GRID, CODES, 'Ca', 7000.0)
    assert why == 'no synthetic spectra covering the line'


def test_grid_selection_uses_one_wavelength_range(capsys):
    second = ['am080t4060g110k160_cl_ap04_Ca_m020_6800_7600.nb', 'am080t4060g110k160_cl_ap04_Ca_p040_6800_7600.nb']
    names, values, _ = select_grid_synthetics(GRID[:2] + second, CODES, 'Ca', 6809.0)
    assert names == second                              # 6809 is farther from the edges of 6800-7600
    assert 'wavelength ranges' in capsys.readouterr().out
    names, _, _ = select_grid_synthetics(GRID[:2] + [GRID[0]], CODES, 'Ca', 6500.0)
    assert len(names) == 2                              # the same abundance is used once
