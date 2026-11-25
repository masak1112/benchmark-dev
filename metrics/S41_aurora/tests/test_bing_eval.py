"""Blocking CPU tests of the S41 evaluation package (brief 2026-11-11, "Tests"): perfect forecast, climatology-as-forecast, fair CRPS
ensemble-size invariance, LOYO by construction, cos-lat weights, lead/valid alignment per file format, OLR convention, RPSS,
reliability, bootstrap/grading. Real-data checks (OLR day-1 global mean vs ERA5, references for every field) are skipped when the
data are absent. Run: ``PYTHONPATH=envs/bing_pylib:<worktree> python -m pytest -q tests/test_bing_eval.py``."""
from __future__ import annotations

import importlib.util
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from s41eval.bing_eval import core as K
from s41eval.bing_eval import data as D
from s41eval.bing_eval import pipeline as P
from s41eval.bing_eval import stats as ST

R = Path('/lus/flare/projects/S2Storm/s2storm')
CFG = Path(__file__).resolve().parents[1] / 'configs' / 'bing_eval' / 'aurora.yaml'


def _starts():
    s = []
    for y in (2019, 2020, 2021, 2025):
        s += [date(y, 5, 6) + timedelta(days=7 * i) for i in range(3)]
    return s


def _field(S, W, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(S, W, 11, 12)) * scale
    return np.kron(base, np.ones((1, 1, 11, 20)))[..., :121, :240].astype(np.float32)


def _V(O, C, ens_ref):
    """verification dict as task_verif writes it, from obs O, climatology C and a reference ensemble [S, n, W, y, x]."""
    cc = np.stack([K.fair_crps(e, o) for e, o in zip(ens_ref, O)]).astype(np.float32)
    T = np.stack([K.tercile_edges(e) for e in ens_ref])
    return dict(O=O, Cfixed=C, Ctrend=C, CCfixed=cc, CCtrend=cc, Tfixed=T, Ttrend=T)


def _rows(rows, **kw):
    return {(r['metric'], r['reference'], r['members']): r['value'] for r in rows
            if all(r[k] == v for k, v in dict(dict(period='pooled', season='all', region='global', window='w34', correction='raw'),
                                                  **kw).items())}


def test_perfect_forecast():
    st = _starts()
    S, W = len(st), len(K.WK)
    O = _field(S, W, 1)
    C = np.zeros_like(O)
    ref = np.stack([_field(30, W, 100 + i) for i in range(S)])               # [S, 30, W, y, x]
    F = np.repeat(O[:, None], 12, 1)
    rows, *_ = P.Scorer(F, _V(O, C, ref), st, 'g15').run(log=lambda *a: None, n_draws=2, do_infl=False, do_maps=False)
    g = _rows(rows)
    assert abs(g[('crps', '', 12)]) < 1e-6 and abs(g[('rmse', '', 12)]) < 1e-6
    assert abs(g[('acc', 'trend', 12)] - 1) < 1e-5 and abs(g[('acc_pattern', 'fixed', 12)] - 1) < 1e-5
    assert g[('crpss', 'fixed', 12)] == pytest.approx(1.0)
    assert g[('rpss', 'fixed', 12)] > 0.9
    gd = _rows(rows, correction='dc')
    assert abs(gd[('crps', '', 12)]) < 1e-5                                   # no drift: dc leaves a perfect forecast perfect


def test_climatology_as_forecast_crpss_zero():
    st = _starts()
    S, W = len(st), len(K.WK)
    O = _field(S, W, 2)
    ref = np.stack([_field(30, W, 200 + i) for i in range(S)])
    V = _V(O, ref.mean(1), ref)
    rows, *_ = P.Scorer(ref, V, st, 'g15').run(log=lambda *a: None, n_draws=0, do_infl=False, do_maps=False)
    for w in K.WK:
        g = _rows(rows, window=w)
        assert abs(g[('crpss', 'fixed', 30)]) <= 1e-9 and abs(g[('crpss', 'trend', 30)]) <= 1e-9
    # RPSS of the reference itself: its tercile probabilities are exactly 10/10/10 members, so RPS = RPS_clim and the debiased
    # skill score is D / (RPS_clim + D) > 0 (Weigel 2007 corrects for a random finite ensemble, not a quantile-matched one)
    assert 0 < _rows(rows)[('rpss', 'fixed', 30)] < 0.05


def test_fair_crps_ensemble_size_invariance():
    rng = np.random.default_rng(3)
    n = 400000
    obs = rng.normal(size=n)
    c11 = K.fair_crps(rng.normal(size=(11, n)), obs).mean()
    c30 = K.fair_crps(rng.normal(size=(30, n)), obs).mean()
    exact = 1 / np.sqrt(np.pi)                                                  # E CRPS(N(0,1), y ~ N(0,1)) = 1/sqrt(pi)
    se = 0.6 / np.sqrt(n) * 3
    assert abs(c11 - exact) < 5 * se and abs(c30 - exact) < 5 * se and abs(c11 - c30) < 5 * se
    e = rng.normal(size=(11, n))
    unfair11 = np.abs(e - obs[None]).mean(0).mean() - np.abs(e[:, None] - e[None]).mean((0, 1)).mean() / 2
    assert unfair11 - exact > 0.03                                               # the M^2 (unfair) version is biased at M = 11


def test_loyo_never_uses_target_year():
    st = _starts() + [date(2019, 5, 27), date(2020, 6, 1)]
    S, W = len(st), 2
    O = np.zeros((S, W, 121, 240), np.float32)
    em = _field(S, W, 4)
    valid = np.ones((S, W), bool)
    b0 = K.drift_bias(em, O, st, valid)
    Wt = K.loyo_weights(st)
    for s in range(S):
        assert all(st[j].year != st[s].year for j in np.flatnonzero(Wt[s]))
        em2 = em.copy()
        em2[[j for j in range(S) if st[j].year == st[s].year]] += 1000.0          # change the whole target year
        b2 = K.drift_bias(em2, O, st, valid)
        assert np.allclose(b2[s], b0[s])
    # constant bias c removed exactly
    c = 2.5
    b = K.drift_bias(O + c, O, st, valid)
    assert np.allclose(b, c)


def test_coslat_weights_and_regrid():
    m = (K.LAT['g15'][:, None] >= 60) * np.ones((1, 240))
    frac = K.amean(m, np.ones((121, 240)), 'g15')
    assert abs(frac - (1 - np.sin(np.deg2rad(60))) / 2) < 0.005               # area of the cap north of 60N
    x = _field(1, 1, 5, 10.0)[0, 0] + 3.0
    y = K.regrid(x, 'g15', 'g30')
    assert y.shape == (61, 120)
    assert abs(K.amean(x, np.ones((121, 240)), 'g15') - K.amean(y, np.ones((61, 120)), 'g30')) < 0.05
    one = K.regrid(np.ones((121, 240)), 'g15', 'g30')
    assert np.allclose(one, 1.0)
    Wl, _ = K.cons_weights('g15', 'g30')
    assert np.allclose(Wl.sum(0), np.diff(K._cell_edges_lat(K.LAT['g15'], 1.5)))   # every source cell fully assigned


def _write_h25(path, grid, lat_desc, init, olr_val=-240.0):
    import netCDF4
    ny, nx = grid
    lat = np.linspace(-89.5, 89.5, 180) if ny == 180 else np.linspace(-90, 90, 121)
    if lat_desc:
        lat = lat[::-1]
    lon = np.arange(nx) * (360 / nx)
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('time', 46)
        ds.createDimension('lat', ny)
        ds.createDimension('lon', nx)
        ds.createVariable('lat', 'f4', ('lat',))[:] = lat
        ds.createVariable('lon', 'f4', ('lon',))[:] = lon
        ds.createVariable('lead_day', 'i2', ('time',))[:] = np.arange(46)
        t = ds.createVariable('time', 'i8', ('time',))
        t.units, t.calendar = f'days since {init:%Y-%m-%d} 00:00:00', 'proleptic_gregorian'
        t[:] = np.arange(46)
        lead = np.arange(46, dtype=np.float32)[:, None, None]
        latg = lat[None, :, None].astype(np.float32)
        ds.createVariable('t2m', 'f4', ('time', 'lat', 'lon'))[:] = 280 + lead + 0 * latg + np.zeros((1, 1, nx))
        ds.createVariable('z500', 'f4', ('time', 'lat', 'lon'))[:] = 5e4 + latg + 0 * lead + np.zeros((1, 1, nx))
        ds.createVariable('tp24', 'f4', ('time', 'lat', 'lon'))[:] = 0.001 * lead + np.zeros((1, ny, nx))
        ds.createVariable('olr', 'f4', ('time', 'lat', 'lon'))[:] = olr_val + 0 * lead + np.zeros((1, ny, nx))
        ds.createVariable('u150', 'f4', ('time', 'lat', 'lon'))[:] = 10.0 + np.zeros((46, ny, nx))
        ds.createVariable('u250', 'f4', ('time', 'lat', 'lon'))[:] = 20.0 + np.zeros((46, ny, nx))


@pytest.mark.parametrize('fmt,grid,desc', [('ucs2s_h25', (180, 360), False), ('fuxi_h25', (121, 240), True)])
def test_lead_alignment_h25(tmp_path, fmt, grid, desc):
    init = date(2025, 1, 6)
    f = tmp_path / 'm00_h25.nc'
    _write_h25(f, grid, desc, init)
    out, info = D.FORMATS[fmt](str(f), init, ['t2m', 'z500', 'tp24', 'olr', 'u200'])
    assert out['t2m'].shape == (8, 121, 240)
    exp = {w: 280 + (a + b) / 2 for w, (a, b) in K.WINDOWS.items()}             # lead 0 = start day; week 1 = days 1-7 -> mean 4
    for i, w in enumerate(K.WK):
        assert np.allclose(out['t2m'][i], exp[w], atol=1e-3), w
    assert np.allclose(out['tp24'][0], 4.0, atol=1e-3)                          # mm/day (m * 1000)
    assert np.allclose(out['olr'], 240.0, atol=1e-3)                            # net TOA LW -240 -> OLR +240 positive up
    assert np.allclose(out['u200'], K.W150 * 10 + (1 - K.W150) * 20) and info['u200_src'] == 'u150/u250 log-p'
    z = out['z500'][0]
    assert z[-1, 0] > z[0, 0] and abs(z[60, 0] - 5e4) < 1.0                     # lat ascending internally (north last)


def test_olr_zero_file_refused(tmp_path):
    init = date(2025, 1, 6)
    f = tmp_path / 'm00_h25.nc'
    _write_h25(f, (180, 360), False, init, olr_val=0.0)
    with pytest.raises(D.ReadError):
        D.FORMATS['ucs2s_h25'](str(f), init, ['olr'])


def test_ifs_alignment_synthetic_grib(tmp_path):
    spec = importlib.util.spec_from_file_location('tbs', Path(__file__).with_name('test_bing_scores.py'))
    tbs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbs)
    lat = K.LAT['g15'][:, None] * np.ones((1, 240))
    msgs = []
    for n in (0, 1):
        for k in range(0, 47):
            msgs.append(('msl', n, 0, 24 * k, 'instant', 1e5 + 10 * k + 0 * lat))
            msgs.append(('ttr', n, 0, 24 * k, 'accum', -250.0 * 86400 * k + 0 * lat))
    d = tmp_path / '20250106'
    d.mkdir()
    tbs._write_grib(d / 'p1sfc.grib', msgs)
    recs, info = D.FORMATS['ifs_ecds_grib'](str(d), date(2025, 1, 6), ['msl', 'olr'])
    assert len(recs) == 2
    exp = {w: 1e5 + 10 * (a + b) / 2 for w, (a, b) in K.WINDOWS.items()}        # step 24k = lead k
    for i, w in enumerate(K.WK):
        assert np.allclose(recs[0]['msl'][i], exp[w], rtol=1e-6)
    assert np.allclose(recs[1]['olr'], 250.0, rtol=1e-3)                        # positive up


def test_rps_and_reliability():
    edges = np.array([[-0.43], [0.43]])
    obs = np.array([-1.0])
    assert K.rps3(np.array([1.0]), np.array([0.0]), obs, edges)[0] == 0
    assert np.isclose(K.rps_clim3(obs, edges)[0], 5 / 9)
    assert np.isclose(K.rpss_debias_D(30), 4 / 270)
    p = np.array([0.0, 0.1, 0.1, 1.0])
    o = np.array([0.0, 1.0, 0.0, 1.0])
    c = K.reliability_counts(p, o, np.ones(4))
    assert c[0, 0] == 1 and c[1, 0] == 2 and c[1, 2] == 1 and c[10, 2] == 1


def test_rank_histogram_flat_for_exchangeable():
    rng = np.random.default_rng(7)
    ens = rng.normal(size=(10, 200000))
    obs = rng.normal(size=200000)
    h = np.bincount(K.obs_rank(ens, obs, rng), minlength=11) / 200000
    assert np.abs(h - 1 / 11).max() < 0.004


def test_bootstrap_and_grade():
    idx = ST.mbb_indices(10, 4, 5, 0)
    assert idx.shape == (5, 10)
    for row in idx:
        assert all(row[i + 1] - row[i] == 1 for i in range(0, 3))               # first block consecutive
    rng = np.random.default_rng(0)
    n = 200
    cr = np.ones(n)
    ca = 0.9 + 0.01 * rng.normal(size=n)
    cb = 1.0 + 0.01 * rng.normal(size=n)
    full, ci, _ = ST.boot_scalar(ST.crpss_diff_stat(ca, cb, cr), n, n_draws=300)
    assert ci[0] > 0 and abs(full - 0.1) < 0.01
    assert ST.grade(ci, 0.1, 0.08) == 'robust' and ST.grade(ci, 0.1, -0.02) == 'suggestive' and ST.grade((-0.1, 0.2), 1, 1) == 'noise'
    # paired ACC: A = obs + small noise beats B = obs + big noise
    S = 60
    oa = rng.normal(size=(S, 6, 8))
    mk = np.zeros((121, 240))
    mk[50:56, 10:18] = 1
    big = lambda a: np.pad(a, ((0, 0), (50, 65), (10, 222)))                   # noqa: E731
    r = ST.paired_acc_diff(big(oa + 0.3 * rng.normal(size=oa.shape)), big(oa + 1.5 * rng.normal(size=oa.shape)), big(oa), mk, 'g15',
                           n_draws=200)
    assert r['a'] > r['b'] and r['ci'][0] > 0


# ------------------------------------------------------------------ real data (skipped if absent)
REAL = R / 'data' / 'era5_bing_1p5'


@pytest.mark.skipif(not (REAL / 'olr' / '2025.npy').exists(), reason='verification store not staged')
def test_olr_day1_global_mean_vs_era5():
    """UC-S2S (both) and FuXi day-1 OLR global mean within 10 W m-2 of ERA5 (same sampling as the scorer)."""
    cfg = P.load_config(CFG)
    era = D.Era5(cfg['verification']['store'])
    init = date(2025, 1, 6)
    res = {}
    for sy in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
        sc = cfg['systems'][sy]
        (m, p), = D.member_paths(sc, init)[:1]
        if p is None:
            pytest.skip(f'{sy} file missing')
        daily, ld, _ = D.read_h25_daily(p, init, ['olr'])
        x = daily['olr'][list(ld).index(1)]
        o = era.day('olr', D.sampling(sc, 'olr'), init + timedelta(days=1))
        one = np.ones((121, 240))
        res[sy] = (float(K.amean(x, one, 'g15')), float(K.amean(o, one, 'g15')))
    print(res)
    for sy, (f, o) in res.items():
        assert f > 0 and abs(f - o) < 10.0, (sy, f, o)
