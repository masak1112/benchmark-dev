"""Login-node ERA5 staging for the modes evaluation (``s41eval.bing_modes``), into NEW dirs only.

mjo    OLR, u850, u200 (u250 comes with the same read) for years not yet in ``data/clim_trend_bing/stage`` (1979-1990 by default),
       same layout and method as ``scripts/bing_stage_era5.py stage`` ([365, 4, 240, 121], Feb 29 dropped; u from WB2 1.5 deg
       6-hourly; olr from ARCO = -mean of the hourly mean_top_net_long_wave_radiation_flux at 06/12/18/24 UTC of day d, stored at
       hour 0) -> ``data/era5_modes/stage/<var>_<year>.npy``.
strat  ERA5 daily 00 UTC stratospheric indices -> ``data/era5_modes/strat/<year>.npz``: zonal-mean u [levels, 121 lat] and
       polar-cap (60-90N, cos-lat) mean z [levels] at 10, 50, 100, 250, 500 hPa (u: 10, 50 only). Source <= 2022: 10 hPa from WB2
       512x256 37-level hourly (48-h chunks, one level per chunk), 50-500 hPa from WB2 1.5-degree 13-level 6-hourly; >= 2023: ARCO
       0.25 deg 37 levels (one 00 UTC chunk per day).

    python scripts/bing_modes_stage.py mjo --years 1979 1990 --workers 12
    python scripts/bing_modes_stage.py strat --years 1991 2026 --workers 12
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

R = Path(os.environ.get('S2STORM_ROOT', '/lus/flare/projects/S2Storm/s2storm'))
OUT = R / 'data' / 'era5_modes'
WB512 = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr'
STRAT_U = (10, 50)
STRAT_Z = (10, 50, 100, 250, 500)
LAT15 = np.linspace(-90, 90, 121)


def mjo_item(arg):
    import bing_stage_era5 as B
    B.OUT = OUT                                          # same code, new output tree (spawned worker: set here)
    return B.stage_item(arg)


def _zm_to15(zm, lat):
    """zonal mean on lat (any order) -> 121 WB2 latitudes (-90..90) by linear interpolation in lat."""
    o = np.argsort(lat)
    return np.interp(LAT15, np.asarray(lat)[o], np.asarray(zm)[o])


def _cap(x, lat):
    """polar-cap mean 60-90N, cos-lat weighted, of a [lat, lon] field (zonal mean first)."""
    zm = x.mean(-1)
    m = lat >= 60
    w = np.cos(np.deg2rad(lat[m]))
    return float((zm[m] * w).sum() / w.sum())


def strat_item(arg):
    year = int(arg)
    f = OUT / 'strat' / f'{year}.npz'
    if f.exists():
        return 'cached'
    import bing_stage_era5 as B
    days = [date(year, 1, 1) + timedelta(days=k) for k in range(366)]
    days = [d for d in days if d.year == year]
    nd = len(days)
    zmu = np.full((nd, len(STRAT_U), 121), np.nan, np.float32)
    cap = np.full((nd, len(STRAT_Z)), np.nan, np.float32)
    t0 = time.time()
    if year <= 2022:
        tunits0 = datetime(1959, 1, 1)
        hrs = [int((datetime(d.year, d.month, d.day) - tunits0).total_seconds() // 3600) for d in days]
        # 10 hPa: WB2 512x256 37-level hourly (48-h chunks, one level per chunk)
        g = B.grp(WB512)
        lev = [int(x) for x in g['level'][:]]
        lat = np.asarray(g['latitude'][:], float)
        for name, kind in (('u_component_of_wind', 'u'), ('geopotential', 'z')):
            a = g[name]
            li = lev.index(10)
            ch = int(a.chunks[0])
            for c in sorted({h // ch for h in hrs}):
                blk = B._retry(lambda: np.asarray(a[ch * c:ch * c + ch, li]))       # [48, lon 512, lat 256]
                for k, h in enumerate(hrs):
                    if h // ch != c:
                        continue
                    x = blk[h - ch * c].T.astype(np.float64)
                    if kind == 'u':
                        zmu[k, STRAT_U.index(10)] = _zm_to15(x.mean(-1), lat)
                    else:
                        cap[k, STRAT_Z.index(10)] = _cap(x, lat)
        # 50-500 hPa: WB2 1.5-degree 13-level 6-hourly (8-time chunks, all levels)
        g = B.grp(B.WB2)
        lev = [int(x) for x in g['level'][:]]
        lat = np.asarray(g['latitude'][:], float)
        t0h = int(np.asarray(g['time'][:1])[0])                                     # hours since 1959-01-01 of index 0
        h6 = [(h - t0h) // 6 for h in hrs]
        for name, kind, levels in (('u_component_of_wind', 'u', [50]), ('geopotential', 'z', [50, 100, 250, 500])):
            a = g[name]
            li = [lev.index(L) for L in levels]
            ch = int(a.chunks[0])
            for c in sorted({i // ch for i in h6}):
                blk = B._retry(lambda: a.get_orthogonal_selection((slice(ch * c, ch * c + ch), li, slice(None), slice(None))))
                for k, i in enumerate(h6):
                    if i // ch != c:
                        continue
                    for j, L in enumerate(levels):
                        x = blk[i - ch * c, j].T.astype(np.float64)                 # [lat 121, lon 240]
                        if kind == 'u':
                            zmu[k, STRAT_U.index(L)] = _zm_to15(x.mean(-1), lat)
                        else:
                            cap[k, STRAT_Z.index(L)] = _cap(x, lat)
    else:
        g = B.grp(B.ARCO)
        lev = [int(x) for x in g['level'][:]]
        lat = np.asarray(g['latitude'][:], float)
        for k, d in enumerate(days):
            t = datetime(d.year, d.month, d.day)
            if t > datetime(2026, 11, 5):          # ARCO ERA5T end on 11-11 (attrs valid_time_stop_era5t); re-run later for more
                break
            it = B.arco_t(t)
            try:
                u = B._retry(lambda: np.asarray(g['u_component_of_wind'][it]))
                z = B._retry(lambda: np.asarray(g['geopotential'][it]))
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite(u[lev.index(10)]).any():
                continue
            for il, L in enumerate(STRAT_U):
                zmu[k, il] = _zm_to15(u[lev.index(L)].astype(np.float64).mean(-1), lat)
            for il, L in enumerate(STRAT_Z):
                cap[k, il] = _cap(z[lev.index(L)].astype(np.float64), lat)
    tmp = f.with_suffix('.partial.npz')
    np.savez(tmp, days=np.array([d.isoformat() for d in days]), zm_u=zmu, cap_z=cap, u_levels=np.array(STRAT_U),
             z_levels=np.array(STRAT_Z), lat=LAT15,
             meta=np.str_('ERA5 00 UTC; zm_u = zonal-mean u interpolated to the WB2 1.5-deg latitudes; cap_z = cos-lat mean of zonal-'
                          'mean geopotential (m2 s-2) over 60-90N; source WB2 512x256 37-level (<=2022) / ARCO 0.25 deg (>=2023)'))
    os.replace(tmp, f)
    return f'ok {time.time() - t0:.0f}s, {int(np.isfinite(zmu[:, 0, 0]).sum())} days'


def stl1_item(arg):
    """ERA5 soil temperature level 1, 00 UTC, ARCO 0.25 deg -> 1.5 deg (clim_trend.to15) in the stage layout (hour 0 only)."""
    year = int(arg)
    import bing_stage_era5 as B
    from s41eval.sci.clim_trend import days365
    f = OUT / 'stage' / f'stl1_{year}.npy'
    if f.exists():
        return 'cached'
    arr = np.full((365, 4, 240, 121), np.nan, np.float32)
    t0 = time.time()
    n = 0
    for k, d in enumerate(days365(year)):
        t = datetime(d.year, d.month, d.day)
        if t > datetime(2026, 11, 5):
            break
        try:
            arr[k, 0] = B.arco_field('soil_temperature_level_1', t)[None]
            n += 1
        except Exception:  # noqa: BLE001
            pass
    np.save(str(f) + '.partial.npy', arr)
    os.replace(str(f) + '.partial.npy', f)
    return f'ok {n} days {time.time() - t0:.0f}s'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['mjo', 'strat', 'stl1'])
    p.add_argument('--years', nargs=2, type=int, default=[1979, 1990])
    p.add_argument('--groups', nargs='+', default=['olr', 'u'])
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args(argv)
    from s41eval.bing_scores import mkdir_striped
    t0 = time.time()
    yrs = list(range(a.years[0], a.years[1] + 1))
    if a.cmd == 'mjo':
        mkdir_striped(OUT / 'stage')
        items = [f'{g}|{y}' for g in a.groups for y in yrs]
        fn = mjo_item
    elif a.cmd == 'stl1':
        mkdir_striped(OUT / 'stage')
        items = [str(y) for y in sorted(yrs, key=lambda y: -y)]
        fn = stl1_item
    else:
        mkdir_striped(OUT / 'strat')
        items = [str(y) for y in sorted(yrs, key=lambda y: -y)]
        fn = strat_item
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for it, r in zip(items, pool.imap(fn, items)):
            print(f'{datetime.now(timezone.utc):%H:%M:%S} {a.cmd} {it}: {r} ({time.time() - t0:.0f}s)', flush=True)
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
