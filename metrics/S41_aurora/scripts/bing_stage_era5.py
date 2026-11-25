"""Login-node staging of the ERA5 pieces that ``s41eval.bing_scores`` needs and that do not exist yet on Aurora.

1. ``stage``: 1.5-degree 6-hourly arrays per (variable, year), same layout and method as ``s41eval.sci.clim_trend stage``
   ([365 days, 4 hours, 240 lon, 121 lat], Feb 29 dropped) -> ``data/clim_trend_bing/stage/<var>_<year>.npy``:
     1991-2022 from WeatherBench2 (conservative 1.5 deg, all four hours), except olr (WB2 olr is all NaN) -> ARCO every year;
     2023-2025 from ARCO 0.25 deg (cos-lat block mean to the same cells, ``clim_trend.to15``), 00 UTC only for the pressure-level
     variables (one 37-level chunk per time; 06/12/18 UTC left NaN, so the trend fit at those hours uses 1991-2022), all four hours for
     sst and olr.
   Variables: u850 u200 u250 (one u read), v850 v250, t850, sst, olr. olr = daily mean of the 4 samples ending 06, 12, 18, 24 UTC of day d
   (WB2 / ARCO ``mean_top_net_long_wave_radiation_flux``, hourly means; sign flipped to positive-up), stored at hour 0 only (as tp24).
2. ``fit``: ``clim_trend.fit_var`` on those arrays -> ``data/clim_trend_bing/<var>/{coef_a,coef_b,clim_2025,clim_2026}.npy`` (same
   algebra: 1991-2025, +-7 days, linear trend about 2008, 6 harmonics).
3. ``obs``: ERA5 verification not in ``era5_verif00``/``era5_verif`` -> ``data/era5_bing_obs/<YYYYMMDD>.npz`` (1.5 deg, lat -90..90,
   lon 0..358.5): ``olr_day`` (as above), ``sst_00``, ``sst_day``; with ``--ua_dmean`` also daily means (00/06/12/18 UTC) ``<v>_day`` of
   u850 u200 u250 v850 v250 t850 (for FuXi-S2S daily means).

    python scripts/bing_stage_era5.py stage --workers 8 [--vars ...] [--years 1991 2025]
    python scripts/bing_stage_era5.py fit [--vars ...]
    python scripts/bing_stage_era5.py obs --first 2025-01-01 --last 2026-10-29 --workers 16 [--ua_dmean]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = Path(os.environ.get('S2STORM_ROOT', '/lus/flare/projects/S2Storm/s2storm'))
OUT = R / 'data' / 'clim_trend_bing'
OBS = R / 'data' / 'era5_bing_obs'
WB2 = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
ARCO = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
GROUPS = {'u': ('u_component_of_wind', {'u850': 850, 'u200': 200, 'u250': 250}),
          'v': ('v_component_of_wind', {'v850': 850, 'v250': 250}),
          't': ('temperature', {'t850': 850}),
          'sst': ('sea_surface_temperature', {'sst': None}),
          'olr': ('mean_top_net_long_wave_radiation_flux', {'olr': None})}
UA = ['u850', 'u200', 'u250', 'v850', 'v250', 't850']
_G = {}


def grp(url):
    if url not in _G:
        import fsspec
        import zarr
        _G[url] = zarr.open_group(fsspec.get_mapper(url, token='anon'), mode='r')
    return _G[url]


def _retry(f, n=5):
    for k in range(n):
        try:
            return f()
        except Exception:  # noqa: BLE001
            if k == n - 1:
                raise
            time.sleep(5 * (k + 1))


def arco_t(t: datetime) -> int:
    return int((t - datetime(1900, 1, 1)).total_seconds() // 3600)


def arco_field(name, t: datetime, levels=None):
    """ARCO 0.25 deg at time t -> {level or None: [240, 121] 1.5 deg (clim_trend layout)}."""
    from s41eval.sci.clim_trend import to15
    g = grp(ARCO)
    x = _retry(lambda: np.asarray(g[name][arco_t(t)]))
    if levels is None:
        return {None: to15(x.astype(np.float64))}
    lev = [int(v) for v in g['level'][:]]
    return {L: to15(x[lev.index(L)].astype(np.float64)) for L in levels}


def stage_item(arg):
    gname, year = arg.split('|')
    year = int(year)
    from s41eval.sci.clim_trend import days365
    name, outs = GROUPS[gname]
    files = {v: OUT / 'stage' / f'{v}_{year}.npy' for v in outs}
    if all(f.exists() for f in files.values()):
        return 'cached'
    days = days365(year)
    arr = {v: np.full((365, 4, 240, 121), np.nan, np.float32) for v in outs}
    t0 = time.time()
    if year <= 2022 and gname != 'olr':                    # WB2 1.5-deg OLR is all NaN (checked 1960-2022) -> ARCO for olr
        g = grp(WB2)
        a = g[name]
        i0 = int((datetime(year, 1, 1) - datetime(1959, 1, 1)).total_seconds() // 21600)
        i1 = int((datetime(year + 1, 1, 2) - datetime(1959, 1, 1)).total_seconds() // 21600)
        if a.ndim == 4:
            lev = [int(x) for x in g['level'][:]]
            li = [lev.index(L) for L in outs.values()]
            blk = _retry(lambda: a.get_orthogonal_selection((slice(i0, i1), li, slice(None), slice(None))))
        else:
            blk = _retry(lambda: np.asarray(a[i0:i1]))[:, None]
        for k, d in enumerate(days):
            j = int((datetime(d.year, d.month, d.day) - datetime(year, 1, 1)).total_seconds() // 21600)
            for iv, v in enumerate(outs):
                if v == 'olr':
                    arr[v][k, 0] = -blk[j + 1:j + 5, iv].mean(0)            # samples 06, 12, 18, 24 UTC
                else:
                    arr[v][k] = blk[j:j + 4, iv]
    else:
        levels = None if list(outs.values())[0] is None else list(outs.values())
        for k, d in enumerate(days):
            base = datetime(d.year, d.month, d.day)
            if gname == 'olr':
                arr['olr'][k, 0] = -np.mean([arco_field(name, base + timedelta(hours=h))[None] for h in (6, 12, 18, 24)], 0)
                continue
            hours = (0, 6, 12, 18) if gname == 'sst' else (0,)
            for h in hours:
                f = arco_field(name, base + timedelta(hours=h), levels)
                for v, L in outs.items():
                    arr[v][k, h // 6] = f[L]
    for v, f in files.items():
        np.save(str(f) + '.partial.npy', arr[v])
        os.replace(str(f) + '.partial.npy', f)
    return f'ok {time.time() - t0:.0f}s'


def obs_item(arg):
    day, ua = arg.split('|')
    d = date.fromisoformat(day)
    f = OBS / f'{d:%Y%m%d}.npz'
    if f.exists():
        return 'cached'
    base = datetime(d.year, d.month, d.day)
    out = {}
    tr = lambda x: x.T.astype(np.float32)  # noqa: E731   [240, 121] -> [121 lat, 240 lon]
    out['olr_day'] = tr(-np.mean([arco_field(GROUPS['olr'][0], base + timedelta(hours=h))[None] for h in (6, 12, 18, 24)], 0))
    s = [arco_field(GROUPS['sst'][0], base + timedelta(hours=h))[None] for h in (0, 6, 12, 18)]
    out['sst_00'], out['sst_day'] = tr(s[0]), tr(np.mean(s, 0))
    if ua == '1':
        for g in ('u', 'v', 't'):
            name, outs = GROUPS[g]
            fs = [arco_field(name, base + timedelta(hours=h), list(outs.values())) for h in (0, 6, 12, 18)]
            for v, L in outs.items():
                out[f'{v}_day'] = tr(np.mean([x[L] for x in fs], 0))
                out[f'{v}_00'] = tr(fs[0][L])
    tmp = f.with_suffix('.partial.npz')
    np.savez(tmp, **out, meta=np.str_(f'{d} ERA5 ARCO 1.5 deg (clim_trend.to15); olr_day = -mean of hourly mean_top_net_long_wave_'
                                      f'radiation_flux at 06/12/18/24 UTC; *_day = mean of 00/06/12/18 UTC'))
    os.replace(tmp, f)
    return 'ok'


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage')
    s.add_argument('--groups', nargs='+', default=list(GROUPS))
    s.add_argument('--years', nargs=2, type=int, default=[1991, 2025])
    s.add_argument('--workers', type=int, default=8)
    f = sub.add_parser('fit')
    f.add_argument('--vars', nargs='+', default=[v for g in GROUPS.values() for v in g[1]])
    o = sub.add_parser('obs')
    o.add_argument('--first', default='2025-01-01')
    o.add_argument('--last', default='2026-10-29')
    o.add_argument('--workers', type=int, default=16)
    o.add_argument('--ua_dmean', action='store_true')
    a = p.parse_args(argv)
    t0 = time.time()
    from s41eval.bing_scores import mkdir_striped       # every output dir striped away from the hung OST 9
    mkdir_striped(OUT)
    mkdir_striped(OUT / 'stage')
    mkdir_striped(OBS)
    if a.cmd == 'stage':
        yrs = range(a.years[0], a.years[1] + 1)
        # cheap 2-D groups first, ARCO (slow) years last
        items = [f'{g}|{y}' for y in sorted(yrs, key=lambda y: (y > 2022, y)) for g in sorted(a.groups, key=lambda g: g not in ('olr', 'sst'))]
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for it, r in zip(items, pool.imap(stage_item, items)):
                print(f'{datetime.now(timezone.utc):%H:%M:%S} stage {it}: {r} ({time.time() - t0:.0f}s)', flush=True)
    elif a.cmd == 'fit':
        from s41eval.sci import clim_trend as CT
        CT.OUT = OUT
        for v in a.vars:
            mkdir_striped(OUT / v)
            print(v, 'years', CT.fit_var(v), f'{time.time() - t0:.0f}s', flush=True)
    else:
        d0, d1 = date.fromisoformat(a.first), date.fromisoformat(a.last)
        items = [f'{d0 + timedelta(days=i)}|{int(a.ua_dmean)}' for i in range((d1 - d0).days + 1)]
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for i, (it, r) in enumerate(zip(items, pool.imap(obs_item, items))):
                if i % 20 == 0 or not r.startswith(('ok', 'cached')):
                    print(f'{datetime.now(timezone.utc):%H:%M:%S} obs {it}: {r} ({time.time() - t0:.0f}s)', flush=True)
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
