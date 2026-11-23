"""Build UC-S2S ERA5 1 deg h5 input files from ARCO ERA5 (for Aurora, or years missing on DSI).

Layout (learned from utils/data_loader_multifiles.py and the DSI store /net/spaces/monsoon/S2S/h5data):
  file  <out>/<YYYY>_<NNNN>.h5, NNNN = 6-hourly index from Jan 1 00 UTC (0000 = Jan 1 00 UTC);
  group /input, one float32 dataset [180, 360] per variable (rows lat -89.5..89.5 ascending, cols lon 0..359),
  upper-air names '<var>_<level>.0' (17 levels 5..1000 hPa + 200), plus /input/time (bytes ISO string).
Only the variables UC-S2S inference reads are written (model state, diagnostics, boundary, conditioning):
see VARS_SFC / VARS_UA. Not written: nino34_*, max/min 2 m temperature, 2 m dewpoint, subgrid orography.

Regridding 0.25 -> 1 deg: ``--regrid point`` (sample the 0.25 deg point at the 1 deg centre), ``area``
(cell-area mean, trapezoid 0.5,1,1,1,0.5 per axis) or ``hybrid`` (default; the convention of Bing's DSI store,
found by job 2280336 on 2020-06-01 00 UTC: POINT_VARS are point samples, identical to the DSI file, with soil
fields NaN wherever SST is defined; all other fields are area means, rel. RMS 2e-5..2e-3 vs the DSI file,
not bitwise [inferred: a conservative regrid with exact cell areas]). ``--compare`` checks the built file against an existing DSI
file and reports, per variable, max |diff| and relative RMS, so the regrid / accumulation convention that
reproduces Bing's store can be chosen.

Accumulations at valid time t (ERA5 hourly fields in ARCO are accumulations over the hour ending at t):
  total_precipitation_24hr = sum of total_precipitation over the 24 h ending at t;
  toa_incident_solar_radiation = hourly accumulation at t (J m-2; DSI max 4.75e6 at 2020-06-01 00 UTC);
  mean_top_net_long_wave_radiation_flux = value at t (hourly mean, W m-2; negative upward, ERA5 convention).

    python -m s2storm.data.era5_h5 --times 2020-06-01T00 --out <dir> [--regrid hybrid] [--compare <h5dir>]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ARCO = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
LEVELS = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
VARS_UA = ['temperature', 'u_component_of_wind', 'v_component_of_wind', 'specific_humidity', 'geopotential']
VARS_SFC = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
            'surface_pressure', 'volumetric_soil_water_layer_1', 'soil_temperature_level_1', 'skin_temperature',
            'sea_surface_temperature', 'land_sea_mask', 'geopotential_at_surface', 'snow_depth', 'sea_ice_cover',
            'mean_top_net_long_wave_radiation_flux', 'toa_incident_solar_radiation']
POINT_VARS = ['land_sea_mask', 'geopotential_at_surface', 'sea_surface_temperature', 'sea_ice_cover',
              'volumetric_soil_water_layer_1', 'soil_temperature_level_1']
SOIL_VARS = ['volumetric_soil_water_layer_1', 'soil_temperature_level_1']
ACCUM24 = {'total_precipitation_24hr': 'total_precipitation'}
LAT1 = np.arange(-89.5, 90, 1.0)
LON1 = np.arange(0.0, 360.0, 1.0)


def _open():
    import xarray as xr
    return xr.open_zarr(ARCO, storage_options={'token': 'anon'}, consolidated=None, chunks={})


def regrid(a, lat, lon, how):
    """a [..., 721, 1440] on ARCO grid (lat 90..-90, lon 0..359.75) -> [..., 180, 360] UC-S2S grid."""
    a = np.asarray(a, np.float64)[..., ::-1, :]          # lat ascending -90..90
    lat = np.asarray(lat)[::-1]
    if how == 'point':
        i = np.searchsorted(lat, LAT1)
        j = np.round(LON1 / 0.25).astype(int)
        return a[..., i, :][..., :, j].astype(np.float32)
    # area: lat cells [c-.5, c+.5]; lon cells [c-.5, c+.5] (wrap)
    out_lat = np.zeros(a.shape[:-2] + (180, a.shape[-1]))
    wl = np.cos(np.deg2rad(lat))
    for k, c in enumerate(LAT1):
        sel = np.where((lat >= c - 0.5 - 1e-9) & (lat <= c + 0.5 + 1e-9))[0]
        w = np.where(np.isclose(np.abs(lat[sel] - c), 0.5), 0.5, 1.0) * wl[sel]
        if w.sum() == 0:
            w = np.ones_like(w)
        out_lat[..., k, :] = np.nansum(a[..., sel, :] * w[:, None], -2) / w.sum()
    out = np.zeros(a.shape[:-2] + (180, 360))
    for j, c in enumerate(LON1):
        idx = [(int(round(c / 0.25)) + o) % 1440 for o in (-2, -1, 0, 1, 2)]
        w = np.array([0.5, 1, 1, 1, 0.5])
        out[..., j] = (out_lat[..., idx] * w).sum(-1) / w.sum()
    return out.astype(np.float32)


def build_one(ds, t: datetime, out_dir: Path, how: str):
    import h5py
    import xarray as xr  # noqa: F401
    T = np.datetime64(t.strftime('%Y-%m-%dT%H'))
    idx = (t - datetime(t.year, 1, 1)).days * 4 + t.hour // 6
    path = out_dir / f'{t.year}_{idx:04d}.h5'
    tmp = path.with_suffix('.h5.partial')
    lat, lon = ds.latitude.values, ds.longitude.values
    with h5py.File(tmp, 'w') as f:
        g = f.create_group('input')
        sfc = {}
        for v in VARS_SFC:
            x = ds[v].sel(time=T).values
            h = ('point' if v in POINT_VARS else 'area') if how == 'hybrid' else how
            sfc[v] = regrid(x, lat, lon, h)
        if how == 'hybrid':
            for v in SOIL_VARS:
                sfc[v][np.isfinite(sfc['sea_surface_temperature'])] = np.nan
        for v in VARS_SFC:
            g.create_dataset(v, data=sfc[v])
        for v, src in ACCUM24.items():
            x = ds[src].sel(time=slice(T - np.timedelta64(23, 'h'), T)).sum('time').values
            g.create_dataset(v, data=regrid(x, lat, lon, 'area' if how == 'hybrid' else how))
        for v in VARS_UA:
            x = ds[v].sel(time=T, level=LEVELS).values           # [L, 721, 1440]
            r = regrid(x, lat, lon, 'area' if how == 'hybrid' else how)
            for k, L in enumerate(LEVELS):
                g.create_dataset(f'{v}_{L}.0', data=r[k])
        g.create_dataset('time', data=np.bytes_(str(T.astype('datetime64[ns]'))))
    os.replace(tmp, path)
    return path


def compare(built: Path, ref: Path):
    import h5py
    out = {}
    with h5py.File(built, 'r') as a, h5py.File(ref, 'r') as b:
        for k in a['input']:
            if k == 'time' or k not in b['input']:
                continue
            x, y = a['input'][k][:].astype(np.float64), b['input'][k][:].astype(np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            d = x[m] - y[m]
            out[k] = dict(max_abs=float(np.abs(d).max()), rel_rms=float(np.sqrt((d ** 2).mean()) / (y[m].std() + 1e-30)),
                          allclose=bool(np.allclose(x[m], y[m], rtol=1e-5, atol=1e-6)),
                          nan_mismatch=int((np.isfinite(x) != np.isfinite(y)).sum()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--times', nargs='+', default=None, help='YYYY-MM-DDTHH (00/06/12/18)')
    p.add_argument('--start', default=None, help='YYYY-MM-DD (with --end): build a range instead of --times')
    p.add_argument('--end', default=None, help='YYYY-MM-DD inclusive')
    p.add_argument('--hours', default='0', help='comma list of UTC hours for --start/--end (default 0)')
    p.add_argument('--task', type=int, default=0, help='shard index (e.g. SLURM_ARRAY_TASK_ID)')
    p.add_argument('--ntasks', type=int, default=1, help='number of shards')
    p.add_argument('--skip_existing', action='store_true', help='skip times whose h5 file already exists')
    p.add_argument('--out', required=True)
    p.add_argument('--regrid', choices=['point', 'area', 'hybrid'], default='hybrid')
    p.add_argument('--compare', default=None, help='existing h5 dir to compare against')
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    times = list(a.times or [])
    if a.start:
        d, d1 = datetime.strptime(a.start, '%Y-%m-%d'), datetime.strptime(a.end, '%Y-%m-%d')
        hrs = [int(h) for h in a.hours.split(',')]
        while d <= d1:
            times += [(d + timedelta(hours=h)).strftime('%Y-%m-%dT%H') for h in hrs]
            d += timedelta(days=1)
    times = times[a.task::a.ntasks]
    ds = _open()
    rep = {}
    import time as _time
    for s in times:
        t = datetime.strptime(s, '%Y-%m-%dT%H')
        idx = (t - datetime(t.year, 1, 1)).days * 4 + t.hour // 6
        if a.skip_existing and (out / f'{t.year}_{idx:04d}.h5').exists() and not a.compare:
            print('exists', s, flush=True)
            continue
        t0 = _time.time()
        path = build_one(ds, t, out, a.regrid)
        print('wrote', path, f'{_time.time() - t0:.1f}s', flush=True)
        if a.compare:
            rep[s] = compare(path, Path(a.compare) / path.name)
    if rep:
        (out / f'compare_{a.regrid}.json').write_text(json.dumps(rep, indent=1))
        for s, r in rep.items():
            for k in sorted(r, key=lambda k: -r[k]['rel_rms'])[:12]:
                print(s, k, r[k])
            print(s, 'allclose:', sum(v['allclose'] for v in r.values()), '/', len(r))


if __name__ == '__main__':
    main()
