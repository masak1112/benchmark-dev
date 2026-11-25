"""CPU tests of s41eval.bing_modes (synthetic data) + the blocking reference tests (read from the fitted ERA5 outputs).

    PYTHONPATH=envs/bing_pylib:<code> python -m pytest -q tests/test_bing_modes.py
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from s41eval import bing_modes as BM


def test_prev120_excludes_current_day():
    a = np.arange(300, dtype=float)[:, None] * np.ones((1, 3))
    f = BM.prev120(a)
    assert np.isnan(f[:120]).all()
    # mean of days i-120..i-1 of 0,1,2,... is i - 60.5 -> filtered value 60.5
    assert np.allclose(f[120:], 60.5)
    a[200, 1] = np.nan
    f = BM.prev120(a)
    assert np.isnan(f[200]).all() and np.isnan(f[201:321]).all() and np.isfinite(f[199]).all()


def test_window_alignment():
    x = np.arange(46, dtype=float)                       # value = lead day
    w = BM.window_mean_daily(x)
    assert w[BM.WK.index('w1')] == np.mean(range(1, 8))
    assert w[BM.WK.index('w34')] == np.mean(range(15, 29))
    assert w[BM.WK.index('w6')] == np.mean(range(36, 43))


def _synthetic_mjo(n=6000, period=45.0, seed=0):
    """eastward-propagating wavenumber-1 signal in OLR/u850/u200 + noise and an annual cycle."""
    rng = np.random.default_rng(seed)
    days = [date(1979, 1, 1) + timedelta(days=i) for i in range(n)]
    t = np.arange(n)
    lon = np.deg2rad(BM.LON15)
    ph = 2 * np.pi * t[:, None] / period - lon[None]     # eastward: phase decreases with lon at fixed t
    ann = 10 * np.cos(2 * np.pi * t / 365.0)[:, None]
    s = {'olr': 240 - 15 * np.cos(ph + np.deg2rad(125)) + ann + rng.normal(0, 4, (n, 240)),
         'u850': 3 * np.cos(ph + np.deg2rad(125) + np.pi / 2) + rng.normal(0, 1, (n, 240)),
         'u200': -6 * np.cos(ph + np.deg2rad(125) + np.pi / 2) + rng.normal(0, 2, (n, 240))}
    return s, days


def test_mjo_index_rotates_counterclockwise_and_sign():
    s, days = _synthetic_mjo()
    ix = BM.MJOIndex(s, days, eof_years=(1979, 1994))
    r = ix.rmm_fit
    ok = np.isfinite(r).all(1)
    ang = np.unwrap(np.arctan2(r[ok, 1], r[ok, 0]))
    assert (ang[-1] - ang[0]) > 0                         # phases advance 1 -> 8 (eastward propagation)
    assert abs(np.std(r[ok, 0]) - 1) < 0.05 and abs(np.std(r[ok, 1]) - 1) < 0.05
    # sign rule: EOF1 has negative OLR over 100-150E (RMM1 > 0 = convection over the Maritime Continent); a pure wave-1 signal
    # leaves the EOF1/EOF2 rotation degenerate, so only the sign is checked here (the real orientation is checked against BoM)
    mc = (BM.LON15 >= 100) & (BM.LON15 <= 150)
    assert ix.eof[0, :240][mc].mean() < 0


def test_mjo_perfect_forecast_matches_obs():
    s, days = _synthetic_mjo(n=1200)
    ix = BM.MJOIndex(s, days, eof_years=(1979, 1981))
    ix.add_series('day', s, days)
    init = days[500]
    i0 = 500
    band = {v: s[v][i0:i0 + BM.NLEAD][None].repeat(3, 0) for v in ix.fields}
    F = ix.forecast('day', init, band)
    O = ix.obs('day', BM.valid_dates(init))
    assert np.allclose(F[:, 1:], O[None, 1:], atol=1e-9)
    cor, rmse, n = BM.bivar(F[0, 1:], O[1:])
    assert cor > 0.999999 and rmse < 1e-8


def test_bivar_skill_day_and_cor_components():
    rng = np.random.default_rng(1)
    o = rng.normal(size=(200, 2))
    assert BM.bivar(o, o)[0] == pytest.approx(1.0)
    assert BM.skill_day([0.9, 0.7, 0.4], [1, 2, 3]) == pytest.approx(2 + 0.2 / 0.3)
    f = o[:, None, :] + rng.normal(size=(200, 5, 2))
    oo = np.repeat(o[:, None, :], 5, 1)
    c = BM.mjo_cor_from_components(f, oo)
    assert c[0] == pytest.approx(BM.bivar(f[:, 0], oo[:, 0])[0])


def test_rmm_phase_wh04():
    # phase 1: angle -180..-135 (RMM1 < 0, RMM2 small negative); phase 5: angle 0..45
    assert BM.rmm_phase(np.array([-1.0]), np.array([-0.1]))[0] == 1
    assert BM.rmm_phase(np.array([1.0]), np.array([0.1]))[0] == 5
    assert BM.rmm_phase(np.array([-0.1]), np.array([1.0]))[0] == 7
    assert BM.rmm_phase(np.array([0.1]), np.array([-1.0]))[0] == 3


def test_amp_phase_error_sign():
    o = np.array([[1.0, 0.0]] * 10)
    f = np.array([[np.cos(0.2), np.sin(0.2)]] * 10) * 0.5
    ae, pe = BM.amp_phase_err(f, o)
    assert ae == pytest.approx(-0.5) and pe == pytest.approx(np.rad2deg(0.2))      # forecast ahead (counter-clockwise) > 0


def test_loyo_excludes_target_year():
    starts = [date(y, 6, 1) for y in range(2019, 2027)]
    err = np.array([float(d.year) for d in starts])[:, None]
    b = BM.loyo_bias(err, starts)
    for i, d in enumerate(starts):
        others = [y for y in range(2019, 2027) if y != d.year]
        assert b[i, 0] == pytest.approx(np.mean(others))


def test_tele_pattern_recovery():
    rng = np.random.default_rng(2)
    nao = BM.nao_template()
    pna = BM.pna_template()
    from scipy.ndimage import gaussian_filter
    pat_n = gaussian_filter(nao, 4) * 400
    pat_p = gaussian_filter(pna, 4) * 400
    n = 480
    months = [(1979 + i // 12, i % 12 + 1) for i in range(n)]
    a, b = rng.normal(size=n), rng.normal(size=n)
    X = a[:, None, None] * pat_n + b[:, None, None] * pat_p + rng.normal(0, 3, (n, 121, 240))
    TP = BM.TelePatterns(X.astype(np.float32), months, n_rot=4)
    r = TP.raw_indices(X[:24], 1)
    assert np.corrcoef(r['nao'], a[:24])[0, 1] > 0.9
    assert np.corrcoef(r['pna'], b[:24])[0, 1] > 0.9


def test_damped_persistence_ar1():
    import importlib.util
    import sys
    from pathlib import Path
    sp = Path(__file__).resolve().parents[1] / 'scripts' / 'bing_modes_score.py'
    spec = importlib.util.spec_from_file_location('bms', sp)
    m = importlib.util.module_from_spec(spec)
    sys.modules['bms'] = m
    spec.loader.exec_module(m)
    rng = np.random.default_rng(3)
    n = 365 * 30
    days = [date(1990, 1, 1) + timedelta(days=i) for i in range(n)]
    phi = 0.97
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    r = m.damped_fit(x, days, years=(1991, 2018))
    j = BM.WK.index('w1')
    expect = np.mean([phi ** k for k in range(1, 8)])
    assert abs(np.nanmean(r[:, j]) - expect) < 0.03


def test_fast15_matches_to15():
    from s41eval import bing_scores as BS
    x = np.random.default_rng(4).normal(size=(3, 180, 360)).astype(np.float32)
    x[0, 5, 5] = np.nan
    a, b = BM.fast15(x), BS.to15(x)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-5) and (np.isnan(a) == np.isnan(b)).all()


# ---------------------------------------------------------------- blocking reference tests (need the fitted ERA5 outputs)
def _load(name):
    f = BM.OUT / 'tests' / name
    if not f.exists():
        pytest.skip(f'{f} not produced yet (scripts/bing_modes.py mjo_fit / tele_fit)')
    return json.loads(f.read_text())


def test_blocking_mjo_vs_bom():
    t = _load('mjo_test.json')['test']
    assert t['r_rmm1'] >= 0.95 and t['r_rmm2'] >= 0.95, t
    assert t['phase_match_active'] >= 0.90, t


def test_blocking_nao_vs_cpc():
    t = _load('tele_test.json')
    assert t['nao_r_all'] >= 0.9, t
