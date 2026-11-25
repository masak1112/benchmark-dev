"""Login-node staging of the ERA5 data the S41 evaluation package (``s41eval.bing_eval``) needs (brief 2026-11-11 task 4).

All on the common 1.5-degree WeatherBench2 grid (121 x 240, cells centred on lat -90..90 step 1.5, lon 0..358.5): WB2 1991-2022 (WB2's
conservative regrid) and ARCO 0.25-degree ERA5/ERA5T area-averaged to the same cells (``clim_trend.to15``), as the existing
``data/clim_trend*/stage`` arrays. New directories only (never overwrites existing files):

1. ``wb2``   z50, u50 for 1991-2022 from WB2 (all four hours) -> ``data/clim_trend_bing_v1/stage/<var>_<year>.npy`` (clim_trend layout
             [365 days (Feb 29 dropped), 4 hours, 240 lon, 121 lat]).
2. ``verif`` verification store 2019-01-01 .. last ERA5T day -> ``data/era5_bing_1p5/<var>/<year>.npy`` [days of the year incl. Feb 29,
             4 slots, 121 lat (ascending), 240 lon] float32. States: slots = 00/06/12/18 UTC instantaneous of day d. ``tp24``: slot 0 =
             total (m) over (d 00, d+1 00]. ``olr``: slot 0 = daily mean OLR, **positive up** (W m-2) = -mean of the hourly
             ``mean_top_net_long_wave_radiation_flux`` at 06/12/18/24 UTC of d (the convention of the existing olr stage), slots 1-3 NaN.
             Copied from the existing stage arrays where they have the slot (finite), the rest (Feb 29, 06/12/18 UTC of the
             2023-2025 pressure-level fields, z50/u50 2023+, all of 2026) read from ARCO, one task per date.
3. ``clim``  z50/u50 2023-2025 stage arrays from the verification store, then the trend fit (``clim_trend.fit_var``, 1991-2025) ->
             ``data/clim_trend_bing_v1/<var>/{coef_a,coef_b,clim_2025,clim_2026}.npy``.
4. ``lsm``   ERA5 land-sea mask fraction (ARCO, time 0) -> ``data/era5_bing_1p5/lsm.npy`` [121, 240].

Every new directory gets ``lfs setstripe -c 1 -i k`` with k not in the hung OSTs {9, 13}.

    python scripts/bing_eval_stage.py wb2 --workers 8
    python scripts/bing_eval_stage.py verif --workers 16 [--vars ...] [--first 2019-01-01 --last 2026-11-04]
    python scripts/bing_eval_stage.py clim
    python scripts/bing_eval_stage.py lsm
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = Path(os.environ.get('S2STORM_ROOT', '/lus/flare/projects/S2Storm/s2storm'))
VER = R / 'data' / 'era5_bing_1p5'
CLIM1 = R / 'data' / 'clim_trend_bing_v1'
STAGES = [R / 'data' / 'clim_trend' / 'stage', R / 'data' / 'clim_trend_bing' / 'stage', CLIM1 / 'stage']
WB2 = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
ARCO = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
# var -> (ARCO/WB2 name, level or None)
SRC = {'t2m': ('2m_temperature', None), 'msl': ('mean_sea_level_pressure', None), 'sst': ('sea_surface_temperature', None),
       'swvl1': ('volumetric_soil_water_layer_1', None), 'tp24': ('total_precipitation', None),
       'olr': ('mean_top_net_long_wave_radiation_flux', None),
       'z500': ('geopotential', 500), 'z50': ('geopotential', 50), 'u850': ('u_component_of_wind', 850),
       'u250': ('u_component_of_wind', 250), 'u200': ('u_component_of_wind', 200), 'u50': ('u_component_of_wind', 50),
       't850': ('temperature', 850)}
VARS = list(SRC)
ONE_SLOT = ('tp24', 'olr')
GOOD_OSTS = [3, 17, 29, 44, 58, 71, 86, 102, 117, 133, 148]
_K = [0]


def mkdir(d: Path) -> Path:
    new = not d.exists()
    d.mkdir(parents=True, exist_ok=True)
    if new and str(d).startswith('/lus/'):
        k = GOOD_OSTS[_K[0] % len(GOOD_OSTS)]
        _K[0] += 1
        subprocess.run(['lfs', 'setstripe', '-c', '1', '-i', str(k), str(d)], capture_output=True, timeout=60)
    return d


_G = {}


def grp(url):
    if url not in _G:
        import fsspec
        import zarr
        _G[url] = zarr.open_group(fsspec.get_mapper(url, token='anon'), mode='r')
    return _G[url]


def _retry(f, n=6):
    for k in range(n):
        try:
            return f()
        except Exception:  # noqa: BLE001
            if k == n - 1:
                raise
            time.sleep(5 * (k + 1))


def tix(t: datetime) -> int:
    return int((t - datetime(1900, 1, 1)).total_seconds() // 3600)


def days_of(year):
    d, out = date(year, 1, 1), []
    while d.year == year:
        out.append(d)
        d += timedelta(days=1)
    return out


def doy365(d: date) -> int:
    if d.month == 2 and d.day == 29:
        d = d - timedelta(days=1)
    return (date(2001, d.month, d.day) - date(2001, 1, 1)).days


# ------------------------------------------------------------------ 1. WB2 z50/u50
def wb2_item(arg):
    var, year = arg.split('|')
    year = int(year)
    f = CLIM1 / 'stage' / f'{var}_{year}.npy'
    if f.exists():
        return 'cached'
    from s41eval.sci.clim_trend import days365
    name, lev = SRC[var]
    g = grp(WB2)
    a = g[name]
    levs = [int(x) for x in g['level'][:]]
    i0 = int((datetime(year, 1, 1) - datetime(1959, 1, 1)).total_seconds() // 21600)
    i1 = int((datetime(year + 1, 1, 2) - datetime(1959, 1, 1)).total_seconds() // 21600)
    blk = _retry(lambda: np.asarray(a.get_orthogonal_selection((slice(i0, i1), levs.index(lev), slice(None), slice(None)))))
    out = np.full((365, 4, 240, 121), np.nan, np.float32)
    for k, d in enumerate(days365(year)):
        j = int((datetime(d.year, d.month, d.day) - datetime(year, 1, 1)).total_seconds() // 21600)
        out[k] = blk[j:j + 4]
    np.save(str(f) + '.partial.npy', out)
    os.replace(str(f) + '.partial.npy', f)
    return 'ok'


# ------------------------------------------------------------------ 2. verification store
def _stage_file(var, year):
    for d in STAGES:
        f = d / f'{var}_{year}.npy'
        if f.exists():
            return f
    return None


def _store(var, year, mode='r+'):
    f = VER / var / f'{year}.npy'
    if not f.exists():
        mkdir(VER / var)
        a = np.lib.format.open_memmap(str(f) + '.partial.npy', mode='w+', dtype=np.float32, shape=(len(days_of(year)), 4, 121, 240))
        a[:] = np.nan
        a.flush()
        del a
        os.replace(str(f) + '.partial.npy', f)
    return np.load(f, mmap_mode=mode)


def plan_verif(vars_, first: date, last: date, log=print):
    """copy stage slots into the store; -> {date: [(var, slot), ...]} still missing."""
    need = {}
    for v in vars_:
        slots = [0] if v in ONE_SLOT else [0, 1, 2, 3]
        for y in range(first.year, last.year + 1):
            A = _store(v, y)
            sf = _stage_file(v, y)
            S = np.load(sf, mmap_mode='r') if sf is not None else None
            n_copy = 0
            for i, d in enumerate(days_of(y)):
                if not first <= d <= last:
                    continue
                for h in slots:
                    if np.isfinite(A[i, h]).any():
                        continue
                    if S is not None and not (d.month == 2 and d.day == 29):
                        x = np.asarray(S[doy365(d), h])
                        if np.isfinite(x).any():
                            A[i, h] = x.T
                            n_copy += 1
                            continue
                    need.setdefault(d, []).append((v, h))
            A.flush()
            del A
            log(f'plan {v} {y}: copied {n_copy} slots from {sf}')
    return need


def arco_item(arg):
    """one date: read the missing (var, slot) fields from ARCO -> (date, {(var, slot): [121, 240]}) or (date, 'ERROR ...')."""
    d, items = arg
    try:
        from s41eval.sci.clim_trend import to15
        g = grp(ARCO)
        levs = [int(x) for x in g['level'][:]]
        base = datetime(d.year, d.month, d.day)
        out, cache = {}, {}

        def field(name, t):
            key = (name, t)
            if key not in cache:
                cache[key] = _retry(lambda: np.asarray(g[name][tix(t)]))
            return cache[key]
        for v, h in items:
            name, lev = SRC[v]
            if v == 'tp24':
                x = np.zeros((721, 1440))
                for k in range(1, 25):
                    x += field(name, base + timedelta(hours=k))
            elif v == 'olr':
                x = -np.mean([field(name, base + timedelta(hours=k)) for k in (6, 12, 18, 24)], 0)
            else:
                x = field(name, base + timedelta(hours=6 * h))
                if lev is not None:
                    x = x[levs.index(lev)]
            out[(v, h)] = to15(np.asarray(x, np.float64)).T
            if len(cache) > 8:
                cache.clear()
        return d, out
    except Exception as e:  # noqa: BLE001
        return d, f'ERROR {type(e).__name__}: {e}'


def cmd_verif(a):
    mkdir(VER)
    first, last = date.fromisoformat(a.first), date.fromisoformat(a.last)
    need = plan_verif(a.vars, first, last)
    todo = sorted(need.items())
    print(f'{len(todo)} dates need ARCO reads ({sum(len(x) for _, x in todo)} fields)', flush=True)
    t0 = time.time()
    fails = []
    opened = {}
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for c, (d, r) in enumerate(pool.imap_unordered(arco_item, todo, chunksize=1)):
            if isinstance(r, str):
                fails.append((str(d), r))
                print(d, r, flush=True)
                continue
            for (v, h), x in r.items():
                key = (v, d.year)
                if key not in opened:
                    opened[key] = _store(v, d.year)
                opened[key][(d - date(d.year, 1, 1)).days, h] = x
            if (c + 1) % 50 == 0:
                for A in opened.values():
                    A.flush()
                print(f'{datetime.now(timezone.utc):%H:%M:%S} {c + 1}/{len(todo)} dates ({time.time() - t0:.0f}s)', flush=True)
    for A in opened.values():
        A.flush()
    meta = dict(grid='WB2 1.5 deg 121 x 240, lat -90..90 ascending, lon 0..358.5', layout='[day of year incl. Feb 29, slot, lat, lon]',
                slots='states: 00/06/12/18 UTC; tp24: slot 0 = total (m) over (d 00, d+1 00]; olr: slot 0 = daily mean OLR positive up '
                      '(W m-2) = -mean of hourly mean_top_net_long_wave_radiation_flux at 06/12/18/24 UTC',
                sources='existing data/clim_trend*/stage arrays (WB2 conservative 1991-2022, ARCO block mean 2023-2025) where finite; '
                        'otherwise ARCO 0.25 deg ERA5/ERA5T area-averaged (clim_trend.to15)', first=str(first), last=str(last),
                vars=a.vars, failed=fails, written=f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}')
    mf = VER / 'meta.json'
    old = json.loads(mf.read_text()) if mf.exists() else {}
    old.setdefault('runs', []).append(meta)
    mf.write_text(json.dumps(old, indent=1))
    print('done', f'{time.time() - t0:.0f}s', 'failed dates', len(fails), flush=True)


# ------------------------------------------------------------------ 3. clim stage 2023-2025 + fit
def cmd_clim(a):
    from s41eval.sci import clim_trend as CT
    mkdir(CLIM1 / 'stage')
    for v in a.vars:
        for y in range(2023, 2026):
            f = CLIM1 / 'stage' / f'{v}_{y}.npy'
            if f.exists():
                continue
            A = np.load(VER / v / f'{y}.npy', mmap_mode='r')
            out = np.full((365, 4, 240, 121), np.nan, np.float32)
            for i, d in enumerate(days_of(y)):
                if d.month == 2 and d.day == 29:
                    continue
                out[doy365(d)] = np.swapaxes(np.asarray(A[i]), 1, 2)
            np.save(str(f) + '.partial.npy', out)
            os.replace(str(f) + '.partial.npy', f)
            print('stage', v, y, 'from', VER / v, flush=True)
        CT.OUT = CLIM1
        mkdir(CLIM1 / v)
        print(v, 'fit years', CT.fit_var(v), flush=True)


def cmd_lsm(a):
    from s41eval.sci.clim_trend import to15
    mkdir(VER)
    g = grp(ARCO)
    x = np.asarray(g['land_sea_mask'][tix(datetime(2020, 1, 1))], np.float64)
    np.save(VER / 'lsm.npy', to15(x).T.astype(np.float32))
    print('lsm', VER / 'lsm.npy', 'land fraction >= 0.5:', float((to15(x).T >= 0.5).mean()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('wb2')
    w.add_argument('--vars', nargs='+', default=['z50', 'u50'])
    w.add_argument('--years', nargs=2, type=int, default=[1991, 2022])
    w.add_argument('--workers', type=int, default=8)
    v = sub.add_parser('verif')
    v.add_argument('--vars', nargs='+', default=VARS, choices=VARS)
    v.add_argument('--first', default='2019-01-01')
    v.add_argument('--last', default='2026-11-04')
    v.add_argument('--workers', type=int, default=16)
    c = sub.add_parser('clim')
    c.add_argument('--vars', nargs='+', default=['z50', 'u50'])
    sub.add_parser('lsm')
    a = p.parse_args(argv)
    if a.cmd == 'wb2':
        mkdir(CLIM1)
        mkdir(CLIM1 / 'stage')
        items = [f'{v}|{y}' for y in range(a.years[0], a.years[1] + 1) for v in a.vars]
        t0 = time.time()
        with mp.get_context('spawn').Pool(a.workers) as pool:
            for it, r in zip(items, pool.imap(wb2_item, items)):
                print(f'{datetime.now(timezone.utc):%H:%M:%S} wb2 {it}: {r} ({time.time() - t0:.0f}s)', flush=True)
    elif a.cmd == 'verif':
        cmd_verif(a)
    elif a.cmd == 'clim':
        cmd_clim(a)
    else:
        cmd_lsm(a)


if __name__ == '__main__':
    main()
