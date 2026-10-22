"""input.salvador: keywords, defaults, checks, and the lazy settings object."""
import subprocess
import sys
from pathlib import Path

import pytest

from salvador.config import Config, LazyConfig, load_config

MANDATORY = 'OBS = lista.stars\nNORM = 2\nWINDOW = 5\nINTER = n\nSCLIP = 3\n'


def write(tmp_path, text):
    path = tmp_path / 'input.salvador'
    path.write_text(text)
    return path


def test_defaults(tmp_path):
    config = load_config(write(tmp_path, MANDATORY))
    assert (config.input_stars, config.type_norm, config.window, config.inter, config.sclip) == ('lista.stars', 2, 5, 'n', 3.0)
    assert config.grid == 'n' and config.path_syn is None
    assert config.solar_ref == 2 and config.nproc == 1 and config.calc_best == 'y'
    assert config.limit_iter == 11            # LIMIT_ABU = 3 dex


def test_optional_keywords_and_conversions(tmp_path):
    config = load_config(write(tmp_path, MANDATORY + 'AUTO_RES = 5\nINTER = Y\nCALC_BEST = N\nNPROC = 4\n'))
    assert config.auto_res_limit == pytest.approx(0.05)
    assert config.inter == 'y' and config.calc_best == 'n' and config.nproc == 4


def test_comments_and_blank_lines(tmp_path):
    config = load_config(write(tmp_path, '# a comment\n\n' + MANDATORY))
    assert config.window == 5


@pytest.mark.parametrize('missing', ['OBS', 'NORM', 'WINDOW', 'INTER', 'SCLIP'])
def test_missing_mandatory_keyword(tmp_path, missing):
    text = ''.join(r + '\n' for r in MANDATORY.splitlines() if not r.startswith(missing))
    with pytest.raises(SystemExit, match=missing):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize('row, message', [
    ('NORM = 7', 'NORM must be'),
    ('NORM = 3', 'under development'),
    ('INTER = maybe', 'INTER must be y or n'),
    ('WINDOW = five', 'not a valid value'),
])
def test_invalid_values(tmp_path, row, message):
    key = row.split()[0]
    text = ''.join(r + '\n' for r in MANDATORY.splitlines() if not r.startswith(key)) + row + '\n'
    with pytest.raises(SystemExit, match=message):
        load_config(write(tmp_path, text))


def test_unknown_keyword_warns_with_suggestion(tmp_path, capsys):
    load_config(write(tmp_path, MANDATORY + 'CAL_BEST = n\n'))
    assert "unknown keyword 'CAL_BEST' is ignored. Did you mean CALC_BEST?" in capsys.readouterr().out


def test_missing_file(tmp_path):
    with pytest.raises(SystemExit, match='does not exist'):
        load_config(tmp_path / 'input.salvador')


def test_path_syn_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(write(tmp_path, MANDATORY + 'PATH_SYN = grid/list.syn\n'))
    assert config.grid == 'y'
    assert Path(config.path_syn) == tmp_path / 'grid' / 'list.syn'


def test_lazy_config_reads_the_file_at_first_use(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lazy = LazyConfig()
    assert not lazy.loaded
    write(tmp_path, MANDATORY)
    assert lazy.window == 5 and lazy.loaded


def test_lazy_config_use_and_set():
    lazy = LazyConfig()
    lazy.use(Config(input_stars='x', type_norm=0, window=3, inter='n', sclip=2.5))
    lazy.window = 7
    assert lazy.window == 7 and lazy.type_norm == 0


def test_import_has_no_side_effects(tmp_path):
    # importing every module in a folder without input.salvador must work and read nothing
    code = ('import warnings; import salvador.utilities as u; '
            'assert not u.cfg.loaded; '
            'import salvador.config as c; assert warnings.formatwarning is not c._short_warning; '
            'assert not any(f[1] is not None and "No artists" in f[1].pattern for f in warnings.filters); '
            'print("ok")')
    proc = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True)
    assert proc.stdout.strip() == 'ok', proc.stderr
