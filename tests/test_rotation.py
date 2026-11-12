"""ROTAT (salvador 1.3.0): broadening kernels, macroturbulence, and recovery of v sin i on mock
spectra built from the K2 synthetics of the ROTAT test case (tests/data/keywords/ROTAT)."""
import tarfile

import numpy as np
import pytest
from scipy.integrate import quad

import salvador.rotation as rot
from conftest import KEYWORDS
from salvador.config import Config, cfg, load_config
from salvador.elements import define_abu0

STAR = dict(teff=5970, logg=4.10, met=-1.30, alpha=1, vt=1.30)
FE_LINE, FE_FIT = 6137.691, (6137.41, 6138.13)       # Fe I line of the test case and its fit region
R = 40000


# ------------------------------------------------------------------------------- kernels
@pytest.mark.parametrize('kernel, args, span', [
    (rot.rotation_kernel, (10.0, 0.6), 10.0),
    (rot.rotation_kernel, (3.0, 0.0), 3.0),
    (rot.macroturbulence_kernel, (4.0,), 80.0),
    (rot.gaussian_kernel, (7.5,), 40.0),
])
def test_kernels_are_normalized(kernel, args, span):
    assert quad(lambda v: kernel(np.array([v]), *args)[0], -span, span, limit=200)[0] == pytest.approx(1.0, abs=1e-6)


def test_pixel_kernel_narrower_than_a_pixel():
    k = rot.pixel_kernel(1.0, 0.3, lambda v: rot.rotation_kernel(v, 0.3, 0.6))
    assert k.sum() == pytest.approx(1.0) and k[k.size // 2] == pytest.approx(1.0)


def test_rotation_like_pyastronomy():
    from PyAstronomy import pyasl
    w = np.arange(6000.0, 6010.0, 0.005)
    f = 1 - 0.5 * np.exp(-0.5 * ((w - 6005) / 0.05) ** 2)
    dv = 0.005 / 6005 * rot.C_KMS
    ours = rot.convolve_rows(f, rot.pixel_kernel(dv, 20.0, lambda v: rot.rotation_kernel(v, 20.0, 0.6)))[0]
    theirs = pyasl.rotBroad(w, f, 0.6, 20.0)
    assert np.abs(ours - theirs)[200:-200].max() < 2e-3


def test_convolution_keeps_continuum_and_size():
    f = np.ones((2, 50))
    out = rot.convolve_rows(f, rot.pixel_kernel(0.25, 5.0, lambda v: rot.macroturbulence_kernel(v, 5.0)))
    assert out.shape == (2, 50) and np.allclose(out, 1.0)


# ------------------------------------------------------------------------------- macroturbulence
def test_valenti_fischer():
    assert rot.vmac_valenti_fischer(5770) == pytest.approx(3.98)
    assert rot.vmac_valenti_fischer(5000) == pytest.approx(2.80, abs=0.01)    # their Fig. 3
    assert rot.vmac_valenti_fischer(2000) == 0.0


@pytest.mark.parametrize('name, teff, logg, expected', [
    ('D14', 5777, 4.44, 3.21),              # Doyle et al. 2014 at the solar values
    ('B16', 5777, 4.44, 2.202 + 1.30),      # Brewer et al. 2016, dwarfs
    ('B16', 5777, 3.50, 1.166 + 3.30),      # subgiants
    ('B16', 4500, 2.00, 4.0),               # giants: constant
    ('C08', 4351, 0.96, 7.33 - 0.80 * 0.96),  # Carney et al. 2008, eq. 5
])
def test_vmac_relations(name, teff, logg, expected):
    assert rot.VMAC_RELATIONS[name].function(teff, logg) == pytest.approx(expected, abs=1e-3)


def test_star_macroturbulence(capsys):
    assert rot.star_macroturbulence(5970, 4.1, 'VF05') == pytest.approx((4.29, 0.0), abs=0.01)
    assert capsys.readouterr().out == ''
    rot.star_macroturbulence(4300, 1.5, 'VF05')                  # a giant: outside the calibration
    assert 'is calibrated on' in capsys.readouterr().out
    assert rot.star_macroturbulence(4500, 2.0, 'C08') == pytest.approx((5.73, 0.8))   # error: scatter of the relation
    assert capsys.readouterr().out == ''
    assert rot.star_macroturbulence(4300, 1.5, 5.5) == (5.5, 0.0)


def test_star_macroturbulence_error_from_keyword():
    previous = cfg._config
    cfg.use(Config(input_stars='x', type_norm=1, window=10, inter='n', sclip=3.0, rot_vmac=6.0, rot_vmac_err=1.5))
    try:
        assert rot.star_macroturbulence(4500, 2.0) == (6.0, 1.5)
        assert rot.star_macroturbulence(4500, 2.0, 'C08', 1.5)[1] == 1.5
    finally:
        cfg.use(previous)


@pytest.mark.parametrize('settings, vmac, note', [
    (dict(rot_vmac='C08'), (6.56, 0.8), False),                    # error: scatter of Carney et al. 2008
    (dict(rot_vmac=5.0), (5.0, 0.0), True),                        # no error: note
    (dict(rot_vmac='VF05'), (1.80, 0.0), True),
    (dict(rot_vmac='VF05', rot_vmac_err=2.0), (1.80, 2.0), False),
    (dict(rot_vmac=5.0, rot_vmac_err=0.0), (5.0, 0.0), False),     # 0 given explicitly: no note
])
def test_check_rotat_settings(capsys, settings, vmac, note):
    previous = cfg._config
    cfg.use(Config(input_stars='x', type_norm=1, window=10, inter='n', sclip=3.0, **settings))
    try:
        assert rot.check_rotat_settings(4351, 0.96) == pytest.approx(vmac, abs=0.01)
    finally:
        cfg.use(previous)
    out = capsys.readouterr().out
    assert 'ROTAT: vmac =' in out and ('Note: the macroturbulence has no error' in out) == note
    assert ('calibrated on' in out) == (settings['rot_vmac'] == 'VF05')    # the giant is inside the C08 range


def test_rotat_needs_vmac():
    previous = cfg._config
    cfg.use(Config(input_stars='x', type_norm=1, window=10, inter='n', sclip=3.0))
    try:
        with pytest.raises(SystemExit, match='ROT_VMAC'):
            rot.check_rotat_settings(5970, 4.1)
        with pytest.raises(ValueError):
            rot.star_macroturbulence(5970, 4.1)
    finally:
        cfg.use(previous)


def test_rotat_keywords(tmp_path):
    base = 'OBS = a\nNORM = 1\nWINDOW = 10\nINTER = n\nSCLIP = 3\n'
    (tmp_path / 'in1').write_text(base + 'ROT_VMAC = vf05\nROT_LD = 0.7\n')
    config = load_config(tmp_path / 'in1')
    assert config.rot_vmac == 'VF05' and config.rot_ld == 0.7 and config.rot_step == 1.0 and config.rot_ul_fwhm == 0.5
    assert config.rot_vmac_err is None
    (tmp_path / 'in4').write_text(base + 'ROT_VMAC = c08\nROT_VMAC_ERR = 1.2\n')
    config = load_config(tmp_path / 'in4')
    assert config.rot_vmac == 'C08' and config.rot_vmac_err == 1.2
    (tmp_path / 'in2').write_text(base)
    assert load_config(tmp_path / 'in2').rot_vmac is None              # not set: needed only with ROTAT lines
    for bad in ('ROT_VMAC = -1', 'ROT_VMAC = abc', 'ROT_LD = 1.5', 'ROT_VMAC_ERR = -0.5'):
        (tmp_path / 'in3').write_text(base + bad + '\n')
        with pytest.raises(SystemExit):
            load_config(tmp_path / 'in3')


# ------------------------------------------------------------------------------- v sin i recovery
@pytest.fixture(scope='module')
def fe_synthetics(tmp_path_factory):
    """Unbroadened K2 synthetics of the Fe I line of the test case, by abundance offset."""
    cache = tmp_path_factory.mktemp('rotat_cache')
    with tarfile.open(KEYWORDS / 'ROTAT' / 'synth_out.tar.gz') as tar:
        tar.extractall(cache, filter='data') if hasattr(tarfile, 'data_filter') else tar.extractall(cache)
    spectra = {}
    for path in cache.rglob('*_Fe_*_6132_6143.nb'):
        offset = int(path.name.split('_Fe_')[1][1:4]) / 100 * (-1 if path.name.split('_Fe_')[1][0] == 'm' else 1)
        data = np.loadtxt(path, comments='#', usecols=(0, 1))
        spectra[round(offset, 2)] = (data[:, 0], data[:, 1])
    return spectra


@pytest.fixture
def rotat(fe_synthetics, monkeypatch):
    """analysis_rotat with the cached synthetics instead of K2; returns a function fitting a mock."""
    previous = cfg._config
    cfg.use(Config(input_stars='x', type_norm=1, window=10, inter='n', sclip=3.0))
    monkeypatch.setattr(rot, 'autok', lambda param1, delta, window: fe_synthetics[round(float(delta), 2)])
    abu_model = define_abu0(26, STAR['alpha'], STAR['met'], 0.0, 'Fe')
    param1 = dict(STAR, res=R, line=FE_LINE, nelem=26.0, abu_var='none', delta=0.0, tana='ROTAT', abu0=abu_model,
                  name_star='mock')

    def mock(vsini: float, vmac: float, snr: float = 100.0, seed: int = 1):
        w, f = fe_synthetics[0.0]
        w_v, f_v = rot.velocity_grid(w, f)
        fwhm = rot.C_KMS / R
        spec = rot.convolve_rows(f_v, rot.pixel_kernel(rot.FINE_DV, 4 * fwhm, lambda v: rot.gaussian_kernel(v, fwhm)))
        if vmac:
            spec = rot.convolve_rows(spec, rot.pixel_kernel(rot.FINE_DV, 6 * vmac, lambda v: rot.macroturbulence_kernel(v, vmac)))
        if vsini:
            spec = rot.convolve_rows(spec, rot.pixel_kernel(rot.FINE_DV, vsini, lambda v: rot.rotation_kernel(v, vsini, 0.6)))
        w_obs = np.arange(w[0] + 0.5, w[-1] - 0.5, 0.0352)
        return w_obs, np.interp(w_obs, w_v, spec[0]) + np.random.default_rng(seed).normal(0, 1 / snr, w_obs.size)

    def fit(w_obs, f_obs, **kwargs):
        options = dict(abu_init=abu_model, abu_span=0.3, rot_min=0, rot_max=60) | kwargs
        return rot.analysis_rotat(26.0, STAR['alpha'], STAR['met'], 0.0, 'Fe', w_obs, f_obs, FE_LINE, 10, dict(param1),
                                  R, *FE_FIT, **options)
    yield mock, fit
    cfg.use(previous)


@pytest.mark.parametrize('vsini', [6.0, 10.0, 15.0])
def test_vsini_recovered_with_the_right_macroturbulence(rotat, vsini):
    mock, fit = rotat
    result = fit(*mock(vsini, 3.0), vmac=3.0)
    assert result['limit'] == 'D'
    assert abs(result['best_rot_fine'] - vsini) <= max(1.5, 2 * result['err_rot'])
    assert abs(result['best_abu_fine'] - define_abu0(26, 1, -1.3, 0.0, 'Fe')) <= 0.05


def test_macroturbulence_ignored_overestimates_vsini(rotat):
    # the bias of the previous versions: the macroturbulence is absorbed by the rotation
    mock, fit = rotat
    w, f = mock(2.0, 5.0, snr=300)
    assert fit(w, f, vmac=0.0)['best_rot_fine'] > fit(w, f, vmac=5.0)['best_rot_fine'] + 1.5


def test_no_rotation_gives_an_upper_limit(rotat, capsys):
    mock, fit = rotat
    result = fit(*mock(0.0, 3.0), vmac=3.0)
    assert result['limit'] == 'UL'
    assert result['rot_value'] >= 0.5 * rot.C_KMS / R - 1e-9            # at least half the instrumental FWHM
    assert 'upper limit' in capsys.readouterr().out


def test_lower_limit_and_narrow_fit_region(rotat, capsys):
    mock, fit = rotat
    result = fit(*mock(40.0, 3.0), vmac=3.0, rot_max=20)
    assert result['limit'] == 'LL' and result['rot_value'] == 20
    assert 'half-width of the fit region' in capsys.readouterr().out


def test_error_due_to_macroturbulence(rotat):
    # v sin i and vmac are degenerate: an error on vmac gives an error on v sin i, larger for slow rotators
    mock, fit = rotat
    w, f = mock(6.0, 3.0, snr=300)
    assert fit(w, f, vmac=3.0)['err_rot_vmac'] == 0.0
    slow = fit(w, f, vmac=3.0, vmac_err=1.0)
    fast = fit(*mock(25.0, 3.0, snr=300), vmac=3.0, vmac_err=1.0)
    assert slow['err_rot_vmac'] >= 0.5 and slow['err_rot_vmac'] > fast['err_rot_vmac']
    assert abs(slow['best_rot_fine'] - 6.0) <= 1.5          # the best fit does not change
