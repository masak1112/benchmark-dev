"""Trend-adjusted climatology on Aurora (redesign brief a2f0bbb; DSI S33 method = ``s2storm/data/clim_detrended.py`` @ aurora-bundle-v4):
for each variable, hour h in {00, 06, 12, 18}, calendar day d (365-day calendar, Feb 29 dropped) and grid point:
pooled least squares over years 1991-2025 and the 15 days d-7..d+7 (circular) of x = a(d) + b(d) (y - 2008); a, b smoothed by the mean
+ 6 annual harmonics; evaluated at Y: C_Y(h, d) = a_s(d) + b_s(d) (Y - 2008) (precipitation clipped at 0); points with < 50 % of the
maximum sample count -> NaN. The fit code below is the DSI code's algebra (``cmd_fit``) applied to our staged arrays.

Data (differs from DSI, which read Bing's 1-degree h5 stores): 1.5-degree 6-hourly ERA5 on the WeatherBench2 grid (240 x 121,
lat -90..90, lon 0..358.5): 1991-2022 from WB2 ``1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`` (conservative
regrid by WB2); 2023-2025 from ARCO 0.25-degree ERA5/ERA5T, area-averaged to the same cells (cos-lat weighted block mean over the 0.25-degree
points within each 1.5-degree cell, edge points half weight) -- consistency checked on 2022 dates (``check``).
Variables: t2m, msl, z500, swvl1 (00/06/12/18 UTC) and tp24 (24-h total ending 00 UTC of day d+1 = ERA5 daily precipitation of day d,
stored as hour 0 of day d).

    python -m s41eval.sci.clim_trend stage --years 1991 ... --workers 8      # -> data/clim_trend/stage/<var>_<year>.npy [365, 4, 121, 240]
    python -m s41eval.sci.clim_trend check                                    # WB2 vs ARCO-regrid on 2022 dates
    python -m s41eval.sci.clim_trend fit                                      # -> data/clim_trend/<var>/clim_2025.npy, clim_2026.npy, coef
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

R = Path(os.environ.get('S2STORM_DATA', '/lus/flare/projects/S2Storm/s2storm'))
OUT = R / 'data' / 'clim_trend'
WB2 = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
VARS = {'t2m': '2m_temperature', 'msl': 'mean_sea_level_pressure', 'z500': 'geopotential', 'swvl1': 'volumetric_soil_water_layer_1',
        'tp24': 'total_precipitation_24hr'}
ARCO_NAME = {'t2m': '2m_temperature', 'msl': 'mean_sea_level_pressure', 'z500': 'geopotential',
             'swvl1': 'volumetric_soil_water_layer_1'}
Y0, NH, HALF = 2008, 6, 7
LAT15 = np.linspace(-90, 90, 121)
LON15 = np.arange(240) * 1.5


def days365(year):
    d0 = date(year, 1, 1)
    out = []
    for k in range(366):
        d = d0 + timedelta(days=k)
        if d.year != year:
            break
        if not (d.month == 2 and d.day == 29):
            out.append(d)
    return out


_W = None


def _wb2():
    global _W
    if _W is None:
        import fsspec
        import zarr
        _W = zarr.open_group(fsspec.get_mapper(WB2, token='anon'), mode='r')
    return _W


_RM = None


def regrid_mats():
    """0.25-degree (721 x 1440, lat 90..-90) -> 1.5-degree (121 x 240, lat -90..90) block means (cos-lat weighted)"""
    global _RM
    if _RM is None:
        lat = np.linspace(90, -90, 721)
        lon = np.arange(1440) * 0.25
        A = np.zeros((121, 721))
        for j, c in enumerate(LAT15):
            d = np.abs(lat - c)
            w = np.where(d < 0.75 - 1e-9, 1.0, np.where(np.abs(d - 0.75) < 1e-9, 0.5, 0.0)) * np.cos(np.deg2rad(lat))
            w = np.maximum(w, 0)
            if w.sum() == 0:                         # poles: nearest row
                w = (d == d.min()).astype(float)
            A[j] = w / w.sum()
        B = np.zeros((240, 1440))
        for i, c in enumerate(LON15):
            d = np.abs(((lon - c) + 180) % 360 - 180)
            w = np.where(d < 0.75 - 1e-9, 1.0, np.where(np.abs(d - 0.75) < 1e-9, 0.5, 0.0))
            B[i] = w / w.sum()
        _RM = (A, B)
    return _RM


def to15(x):
    A, B = regrid_mats()
    ok = np.isfinite(x)
    num = A @ np.where(ok, x, 0.0) @ B.T
    den = A @ ok.astype(float) @ B.T
    with np.errstate(invalid='ignore', divide='ignore'):
        y = num / den
    return np.where(den > 0.5, y, np.nan).T.astype(np.float32)          # -> [240 lon, 121 lat] (WB2 layout)


def stage_year(arg):
    var, year = arg.split('|')
    year = int(year)
    f = OUT / 'stage' / f'{var}_{year}.npy'
    if f.exists():
        return 'cached'
    days = days365(year)
    out = np.full((365, 4, 240, 121), np.nan, np.float32)
    if year <= 2022:
        g = _wb2()
        a = g[VARS[var]]
        t0 = datetime(1959, 1, 1)
        i0 = int((datetime(year, 1, 1) - t0).total_seconds() // 21600)
        i1 = int((datetime(year + 1, 1, 2) - t0).total_seconds() // 21600)
        if var == 'z500':
            lev = [int(x) for x in g['level'][:]]
            blk = np.asarray(a[i0:i1, lev.index(500)])
        else:
            blk = np.asarray(a[i0:i1])
        for k, d in enumerate(days):
            j = int((datetime(d.year, d.month, d.day) - datetime(year, 1, 1)).total_seconds() // 21600)
            if var == 'tp24':
                out[k, 0] = blk[j + 4]                                   # 24-h total ending 00 UTC of d + 1
            else:
                out[k] = blk[j:j + 4]
    else:
        from s41eval.sci import prep as P
        g = P._group()
        levels = [int(x) for x in g['level'][:]]
        for k, d in enumerate(days):
            if var == 'tp24':
                out[k, 0] = to15(P.read_accum(datetime(d.year, d.month, d.day) + timedelta(days=1), 24))
                continue
            for h in range(4):
                t = datetime(d.year, d.month, d.day, 6 * h)
                for r in range(4):
                    try:
                        x = np.asarray(g[ARCO_NAME[var]][P._tidx(t)])
                        break
                    except Exception:  # noqa: BLE001
                        time.sleep(5 * (r + 1))
                if var == 'z500':
                    x = x[levels.index(500)]
                out[k, h] = to15(np.asarray(x, np.float64))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(f) + '.partial.npy', out)
    os.replace(str(f) + '.partial.npy', f)
    return 'ok'


def harmonics(nd=365, nh=NH):
    t = 2 * np.pi * np.arange(nd) / nd
    cols = [np.ones(nd)]
    for k in range(1, nh + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    return np.stack(cols, 1)


def fit_var(var, years=range(1991, 2026), eval_years=(2025, 2026)):
    nd = 365
    shp = (nd, 4, 240, 121)
    n = np.zeros(shp, np.float32)
    sy = np.zeros(shp, np.float32)
    syy = np.zeros(shp, np.float32)
    sx = np.zeros(shp, np.float64)
    sxy = np.zeros(shp, np.float64)
    used = []
    for y in years:
        f = OUT / 'stage' / f'{var}_{y}.npy'
        if not f.exists():
            continue
        x = np.load(f)
        used.append(y)
        yy = float(y - Y0)
        ok = np.isfinite(x)
        xv = np.where(ok, x, 0.0)
        n += ok
        sy += ok * yy
        syy += ok * yy * yy
        sx += xv
        sxy += xv * yy
    H = harmonics()
    Pm = np.linalg.pinv(H)

    def win(z):
        c = np.concatenate([z[-HALF:], z, z[:HALF]], 0).astype(np.float64)
        cs = np.cumsum(c, 0)
        cs = np.concatenate([np.zeros((1,) + z.shape[1:]), cs], 0)
        return cs[2 * HALF + 1:] - cs[:-(2 * HALF + 1)]
    N, SY, SYY, SX, SXY = (win(z) for z in (n, sy, syy, sx, sxy))
    with np.errstate(invalid='ignore', divide='ignore'):
        ybar = SY / N
        vy = SYY - SY * SY / N
        b = (SXY - SY * SX / N) / vy
        a = SX / N - b * ybar
    bad = N < 0.5 * N.max()
    a[bad] = np.nan
    b[bad] = np.nan
    ca = np.einsum('kd,d...->k...', Pm, np.nan_to_num(a))
    cb = np.einsum('kd,d...->k...', Pm, np.nan_to_num(b))
    badany = bad.any(0)
    od = OUT / var
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / 'coef_a.npy', ca.astype(np.float32))
    np.save(od / 'coef_b.npy', cb.astype(np.float32))
    for Y in eval_years:
        c = np.einsum('dk,k...->d...', H, ca + cb * (Y - Y0))
        if var == 'tp24':
            c = np.maximum(c, 0)
        c[:, badany] = np.nan
        np.save(od / f'clim_{Y}.npy', c.astype(np.float32))           # [365, 4 hours, 240 lon, 121 lat]
    # 1991-2020 plain mean (same window/harmonics, no trend) for the DSI cross-check
    m = np.einsum('dk,k...->d...', H, np.einsum('kd,d...->k...', Pm, np.nan_to_num(SX / N)))
    yrs = [y for y in used if y <= 2020]
    (od / 'meta.json').write_text(json.dumps(dict(var=var, years=used, n_max=float(N.max()), y0=Y0, harmonics=NH, half_window=HALF,
                                                  grid='WB2 240x121 (lon 0..358.5, lat -90..90)', layout='[doy-1, hour/6, lon, lat]',
                                                  note='tp24 stored at hour 0 only (daily total of day d)')))
    return used


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage')
    s.add_argument('--years', nargs='+', type=int, default=list(range(1991, 2026)))
    s.add_argument('--vars', nargs='+', default=list(VARS))
    s.add_argument('--workers', type=int, default=8)
    sub.add_parser('check')
    f = sub.add_parser('fit')
    f.add_argument('--vars', nargs='+', default=list(VARS))
    a = p.parse_args(argv)
    if a.cmd == 'stage':
        items = [f'{v}|{y}' for y in sorted(a.years, key=lambda y: (y > 2022, y)) for v in a.vars]
        t0 = time.time()
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for it, r in zip(items, pool.imap(stage_year, items)):
                print(f'stage {it}: {r} {time.time() - t0:.0f}s', flush=True)
    elif a.cmd == 'check':
        from s41eval.sci import prep as P
        g = _wb2()
        res = {}
        for t in (datetime(2022, 7, 1, 0), datetime(2022, 12, 31, 12)):
            i = int((t - datetime(1959, 1, 1)).total_seconds() // 21600)
            for v in ('t2m', 'msl', 'swvl1'):
                w = np.asarray(g[VARS[v]][i])
                x = to15(P.read_vars(t, [{'t2m': 't2m', 'msl': 'msl', 'swvl1': 'swvl1'}[v]])[v].astype(np.float64))
                d = x - w
                ok = np.isfinite(d)
                res[f'{v}_{t:%Y%m%d%H}'] = dict(mean=float(d[ok].mean()), rms=float(np.sqrt((d[ok] ** 2).mean())),
                                                 scale=float(np.nanstd(w)))
        print(json.dumps(res, indent=1))
        (OUT / 'check_wb2_vs_arco.json').parent.mkdir(parents=True, exist_ok=True)
        (OUT / 'check_wb2_vs_arco.json').write_text(json.dumps(res, indent=1))
    else:
        for v in a.vars:
            print(v, 'years', fit_var(v), flush=True)


if __name__ == '__main__':
    main()
