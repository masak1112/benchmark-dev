"""CPU tests of s41eval.bing_scores (synthetic cases with known answers + an end-to-end run on runs/bing/writer_test).

    python -m pytest -q tests/test_bing_scores.py
"""
from __future__ import annotations

import importlib.util
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from s41eval import bing_scores as B

R = Path('/lus/flare/projects/S2Storm/s2storm')
WRITER = R / 'runs' / 'bing' / 'writer_test'
Y, X = 121, 240


def _starts():
    # 2025 and 2026 Mondays at matching calendar dates (training sets exist) + Nov 2025 (no 2026 partner -> no dc)
    s = [date(2025, 5, 5) + timedelta(days=7 * i) for i in range(4)] + [date(2026, 5, 4) + timedelta(days=7 * i) for i in range(4)]
    return s + [date(2025, 11, 17)]


def _fields(S, W=3, seed=0, sy=2, sx=3):
    """small smooth random fields broadcast to the 1.5-degree grid (cheap; per-start variation)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(S, W, sy, sx))
    return np.kron(base, np.ones((1, 1, int(np.ceil(Y / sy)), X // sx)))[..., :Y, :X].astype(np.float32)


def test_fair_crps_matches_bruteforce():
    rng = np.random.default_rng(1)
    for m in (2, 5, 11):
        ens = rng.normal(size=(m, 4))
        obs = rng.normal(size=4)
        for j in range(4):
            e = ens[:, j]
            brute = np.abs(e - obs[j]).mean() - np.abs(e[:, None] - e[None, :]).sum() / (2 * m * (m - 1))
            assert abs(B.fair_crps(ens, obs)[j] - brute) < 1e-12
    # fair CRPS of a 2-member ensemble at the obs with symmetric members: |d| - |2d|/2 = 0
    assert abs(B.fair_crps(np.array([[1.0], [-1.0]]), np.array([0.0]))[0] - 0.0) < 1e-12


def _clim_ens(O, rng, n=10, scale=1.0):
    return (O[:, None] + rng.normal(scale=scale, size=(O.shape[0], n) + O.shape[1:])).astype(np.float32)


def test_perfect_forecast():
    st = _starts()
    S, M, W = len(st), 5, 3
    O = _fields(S, W)
    C = np.zeros_like(O)
    E = np.repeat(O[:, None], M, 1)
    CE = _clim_ens(O, np.random.default_rng(2))
    CC = np.stack([B.fair_crps(CE[i], O[i]) for i in range(S)]).astype(np.float32)
    rows, maps = B.score_var(E, O, C, CC, st)
    g = {(r['correction'], r['metric']): r['value'] for r in rows if r['region'] == 'global' and r['window'] == 0}
    assert abs(g[('raw', 'crps')]) < 1e-6
    assert abs(g[('raw', 'crpss')] - 1) < 1e-6
    assert abs(g[('raw', 'acc')] - 1) < 1e-5
    assert abs(g[('raw', 'acc_spatial')] - 1) < 1e-5
    assert g[('raw', 'rmse')] < 1e-6
    assert abs(g[('dc', 'rmse')]) < 1e-5                    # bias of a perfect forecast is 0
    assert np.nanmax(np.abs(maps['raw']['acc'][0] - 1)) < 1e-4


def test_climatology_forecast_crpss_zero():
    st = _starts()
    S, W = len(st), 3
    O = _fields(S, W, seed=3)
    rng = np.random.default_rng(4)
    CE = _clim_ens(np.zeros_like(O), rng, n=12)
    CC = np.stack([B.fair_crps(CE[i], O[i]) for i in range(S)]).astype(np.float32)
    rows, maps = B.score_var(CE, O, np.zeros_like(O), CC, st, corrections=('raw',))
    for r in rows:
        if r['metric'] == 'crpss':
            assert abs(r['value']) < 1e-6, r
    assert np.nanmax(np.abs(maps['raw']['crpss'])) < 1e-5


def test_det_members_subsample():
    """IFS: ACC/RMSE/spread from members 0-10 only; CRPS uses all members."""
    st = _starts()
    S, W = len(st), 2
    rng = np.random.default_rng(5)
    O = _fields(S, W, seed=6)
    E11 = (O[:, None] + rng.normal(scale=0.5, size=(S, 11, W, 1, 1))).astype(np.float32)
    junk = (1e3 + rng.normal(scale=50, size=(S, 40, W, 1, 1)) + 0 * O[:, None, :, :1, :1]).astype(np.float32)
    E = np.concatenate([E11, np.broadcast_to(junk, (S, 40, W, Y, X))], 1)
    CC = np.ones_like(O)
    r51, _ = B.score_var(E, O, np.zeros_like(O), CC, st, det_members=11, corrections=('raw',))
    r11, _ = B.score_var(E11 + 0 * O[:, None], O, np.zeros_like(O), CC, st, corrections=('raw',))
    a = {(r['region'], r['window'], r['metric']): r for r in r51}
    b = {(r['region'], r['window'], r['metric']): r for r in r11}
    for k in a:
        if k[2] in ('acc', 'acc_spatial', 'rmse', 'spread', 'ssr', 'bias'):
            assert abs(a[k]['value'] - b[k]['value']) < 1e-4 * max(1, abs(b[k]['value'])), k
            assert a[k]['members'] == 11
        if k[2] == 'crps':
            assert a[k]['value'] > 10 * b[k]['value'] and a[k]['members'] == 51       # junk members count in CRPS


def test_drift_correction_removes_constant_bias():
    st = _starts()
    S, M, W = len(st), 4, 3
    rng = np.random.default_rng(7)
    O = _fields(S, W, seed=8)
    E = (O[:, None] + 2.5 + rng.normal(scale=0.1, size=(S, M, W, 1, 1))).astype(np.float32)
    CC = np.ones_like(O)
    rows, maps = B.score_var(E, O, np.zeros_like(O), CC, st)
    g = {(r['correction'], r['metric']): r for r in rows if r['region'] == 'global' and r['window'] == 1}
    assert abs(g[('raw', 'bias')]['value'] - 2.5) < 0.1
    assert abs(g[('dc', 'bias')]['value']) < 0.1
    assert g[('raw', 'rmse')]['n_starts'] == S
    assert g[('dc', 'rmse')]['n_starts'] == S - 1 == g[('raw_dcsub', 'rmse')]['n_starts']   # Nov 2025 start: no training


def test_window_means_and_missing_day():
    x = np.arange(46, dtype=np.float32)[:, None, None] * np.ones((1, 2, 3), np.float32)
    w = B.window_means(x)
    assert w.shape == (8, 2, 3)
    assert np.allclose(w[0], 4.0) and np.allclose(w[B.WK.index('w34')], 21.5) and np.allclose(w[B.WK.index('w6')], 39.0)
    x[20] = np.nan
    w = B.window_means(x)
    assert np.isnan(w[B.WK.index('w3')]).all() and np.isnan(w[B.WK.index('w34')]).all() and np.isfinite(w[0]).all()
    # lead days 1..42 only (FuXi files without lead 0)
    w = B.window_means(x[1:43], lead_days=np.arange(1, 43))
    assert np.allclose(w[0], 4.0)


def test_regrid_and_masks():
    c = np.full((3, 180, 360), 7.0)
    assert np.allclose(B.to15(c), 7.0)
    mk = B.masks()
    assert abs(B.amean(np.ones((Y, X)), mk['global']) - 1) < 1e-12
    lat = B.LAT15
    f = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, X))
    assert B.amean(f, mk['tropics']) > B.amean(f, mk['nh_extratropics'])
    assert mk['conus'].sum() == ((lat >= 25) & (lat <= 50)).sum() * ((B.LON15 >= 235) & (B.LON15 <= 295)).sum()


def test_rmm_perfect_forecast():
    """synthetic eastward-propagating MJO in olr/u850/u200 band series: a forecast equal to ERA5 reproduces the ERA5 RMM (COR 1)."""
    rng = np.random.default_rng(9)
    d0, d1 = date(1991, 1, 1), date(1996, 12, 31)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    days = [d for d in days if not (d.month == 2 and d.day == 29)]
    t = np.arange(len(days))
    lon = np.deg2rad(B.LON15)
    ph = 2 * np.pi * t / 45.0
    wave = np.cos(lon[None] - ph[:, None])
    ser = {'olr': {}, 'u850': {}, 'u200': {}}
    for i, d in enumerate(days):
        ann = 5 * np.cos(2 * np.pi * B.doy365(d) / 365)
        ser['olr'][d] = 240 + ann - 20 * wave[i] + rng.normal(scale=2, size=240)
        ser['u850'][d] = ann / 5 + 3 * np.roll(wave[i], 20) + rng.normal(scale=0.5, size=240)
        ser['u200'][d] = -ann / 5 - 6 * np.roll(wave[i], 20) + rng.normal(scale=1, size=240)
    rm = B.RMM(ser, eof_years=(1991, 1995))
    assert sum(rm.explained) > 0.5
    inits = [date(1996, 3, 4) + timedelta(days=7 * i) for i in range(10)]
    F, O = [], []
    for s in inits:
        band = {v: np.stack([ser[v][s + timedelta(days=k)] for k in range(46)]) for v in ser}
        band['olr'][0] = np.nan                                   # FuXi-like: no lead 0 -> ERA5 init day used
        F.append(rm.forecast(s, band))
        O.append(np.stack([rm.obs_rmm[s + timedelta(days=k)] for k in range(46)]))
    cor, rmse, n = B.rmm_skill(np.stack(F), np.stack(O))
    assert np.nanmin(cor[1:]) > 0.999 and np.nanmax(rmse[1:]) < 1e-6
    # amplitude ~ sqrt(2) for a pure wave, and eastward propagation = counter-clockwise rotation in (RMM1, RMM2)
    r = np.stack([rm.obs_rmm[d] for d in days[400:800]])
    ang = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    assert ang[-1] - ang[0] > 0


def test_drift_training_sets_loyo():
    from s41eval.biascorr import training_sets
    st = _starts()
    tr = training_sets(st)
    assert all(st[j].year != st[i].year for i, js in enumerate(tr) for j in js)
    assert tr[-1] == []


@pytest.mark.skipif(importlib.util.find_spec('xarray') is None or not (WRITER / '1p5_cl4.nc').exists(),
                    reason='needs xarray and runs/bing/writer_test')
def test_end_to_end_writer_sample(tmp_path):
    """the S40 writer sample (one member, native 1 deg and 1.5 deg copies) scored as a 2-member 'ensemble' against ERA5."""
    import xarray as xr
    with xr.open_dataset(WRITER / '1p5_cl4.nc') as ds:
        init = date.fromisoformat(str(ds.attrs['init'])[:10])
        a = ds['t2m'].values
    with xr.open_dataset(WRITER / 'native_cl4.nc') as ds:
        b = B.to15(ds['t2m'].values)
    assert np.nanmax(np.abs(a - b)) < 1e-3                       # writer's 1.5-deg output = our remap of its native output
    d = tmp_path / 'hc' / 'sstanom' / f'{init:%Y%m%d}'
    d.mkdir(parents=True)
    os.symlink(WRITER / 'native_cl4.nc', d / 'm00_h25.nc')
    os.symlink(WRITER / '1p5_cl4.nc', d / 'm01_h25.nc')
    vv = ['t2m', 'z500', 'tp24', 'msl', 'u850']
    out = B.run(['ucs2s_sstanom'], vv, [init], 'e2e', roots={'ucs2s_sstanom': str(tmp_path / 'hc' / 'sstanom')}, procs=2,
                mjo=False, figures=False, out_root=tmp_path)
    import csv
    rows = list(csv.DictReader(open(out / 'scores.csv')))
    g = {(r['variable'], r['window'], r['metric']): float(r['value']) for r in rows
         if r['region'] == 'global' and r['correction'] == 'raw'}
    print({k: round(v, 3) for k, v in g.items() if k[1] in ('w1', 'w34') and k[2] in ('crpss', 'acc_spatial', 'rmse')})
    assert g[('z500', 'w1', 'acc_spatial')] > 0.8                  # week-1 z500 pattern ACC of a real forecast
    assert g[('t2m', 'w1', 'rmse')] < 3.0
    assert -1.0 < g[('tp24', 'w1', 'crpss')] < 1.0                   # same units (mm/day) for forecast, ERA5 and climatology
    assert g[('z500', 'w1', 'crpss')] > 0.2 and g[('msl', 'w1', 'crpss')] > 0.0
    assert (out / 'maps_ucs2s_sstanom.nc').exists() and (out / 'summary.md').exists()


def test_rmm_winds_only_aligned():
    """winds-only RMM of a synthetic MJO is aligned (order and sign) with the OLR RMM."""
    rng = np.random.default_rng(10)
    d0 = date(1991, 1, 1)
    days = [d0 + timedelta(days=i) for i in range(365 * 6 + 1)]
    days = [d for d in days if not (d.month == 2 and d.day == 29)]
    lon = np.deg2rad(B.LON15)
    ser = {'olr': {}, 'u850': {}, 'u200': {}}
    for i, d in enumerate(days):
        w = np.cos(lon - 2 * np.pi * i / 45.0)
        ser['olr'][d] = 240 - 20 * w + rng.normal(scale=2, size=240)
        ser['u850'][d] = 3 * np.roll(w, 20) + rng.normal(scale=0.5, size=240)
        ser['u200'][d] = -6 * np.roll(w, 20) + rng.normal(scale=1, size=240)
    rm = B.RMM(ser, eof_years=(1991, 1995))
    rw = B.RMM(ser, eof_years=(1991, 1995), fields=('u850', 'u200'), ref=rm)
    a = np.stack([rm.obs_rmm[d] for d in days[500:900]])
    b = np.stack([rw.obs_rmm[d] for d in days[500:900]])
    assert np.corrcoef(a[:, 0], b[:, 0])[0, 1] > 0.9 and np.corrcoef(a[:, 1], b[:, 1])[0, 1] > 0.9


def _write_grib(path, msgs):
    import eccodes
    with open(path, 'wb') as f:
        for sn, num, step0, step1, stype, field in msgs:
            h = eccodes.codes_grib_new_from_samples('regular_ll_sfc_grib2')
            for k, v in (('Ni', 240), ('Nj', 121), ('latitudeOfFirstGridPointInDegrees', 90.0),
                         ('longitudeOfFirstGridPointInDegrees', 0.0), ('latitudeOfLastGridPointInDegrees', -90.0),
                         ('longitudeOfLastGridPointInDegrees', 358.5), ('iDirectionIncrementInDegrees', 1.5),
                         ('jDirectionIncrementInDegrees', 1.5), ('productDefinitionTemplateNumber', 11 if stype != 'instant' else 1),
                         ('shortName', sn), ('number', num), ('stepType', stype)):
                eccodes.codes_set(h, k, v)
            if stype == 'instant':
                eccodes.codes_set(h, 'step', step1)
            else:
                eccodes.codes_set(h, 'startStep', step0)
                eccodes.codes_set(h, 'endStep', step1)
            eccodes.codes_set_values(h, field[::-1].ravel().astype(np.float64))          # GRIB rows north -> south
            eccodes.codes_write(h, f)
            eccodes.codes_release(h)


@pytest.mark.skipif(importlib.util.find_spec('eccodes') is None, reason='needs eccodes')
def test_ifs_adapter_synthetic_grib(tmp_path):
    """msl instantaneous, tp and ttr accumulated from step 0, 2t daily means: leads, signs and units as documented."""
    lat = B.LAT15[:, None] * np.ones((1, 240))
    msgs = []
    for n in (0, 1, 2):
        for k in range(0, 47):
            msgs.append(('msl', n, 0, 24 * k, 'instant', 1e5 + 10 * k + n + lat))
            msgs.append(('tp', n, 0, 24 * k, 'accum', 2.0 * k + 0 * lat))                 # 2 kg m-2 per day
            msgs.append(('ttr', n, 0, 24 * k, 'accum', -250.0 * 86400 * k + 0 * lat))     # OLR 250 W m-2
            if k:
                msgs.append(('2t', n, 24 * (k - 1), 24 * k, 'avg', 280.0 + k + 0 * lat))
    d = tmp_path / '20250106'
    d.mkdir()
    _write_grib(d / 'p1sfc.grib', msgs)
    mem, msl, kind = B.ifs_var_daily(d, 'msl')
    assert mem == [0, 1, 2] and kind == 'inst'
    assert np.allclose(msl[1, 3], 1e5 + 30 + 1 + lat, atol=1.0)                     # lat ascending, lead 3 = step 72 h
    _, tp, _ = B.ifs_var_daily(d, 'tp24')
    units = B._grib_var(d / 'p1sfc.grib', ('tp',), 0)[1]
    per_day = 0.002 if 'kg' in units else 2.0                                     # ECDS s2s tp is kg m-2 (-> m); sample paramId 228 is m
    assert np.isnan(tp[0, 0]).all() and np.allclose(tp[:, 1:], per_day, rtol=1e-5)
    _, olr, _ = B.ifs_var_daily(d, 'olr')
    assert np.allclose(olr[:, 1:], 250.0, rtol=1e-3)
    _, t2, k2 = B.ifs_var_daily(d, 't2m')
    assert k2 == 'mean' and np.allclose(t2[2, 5], 285.0, atol=0.01)
    recs = B.read_ifs(d, date(2025, 1, 6), ['msl', 'tp24', 't2m'])
    assert len(recs) == 3 and np.allclose(recs[0]['tp24'][0], per_day * 1000, rtol=1e-4)   # mm/day window mean
    assert np.allclose(recs[0]['mjo']['olr'][1:], 250.0, rtol=1e-3) and np.isnan(recs[0]['mjo']['u850']).all()
