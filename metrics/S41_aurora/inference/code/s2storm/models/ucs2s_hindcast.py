"""Hindcast output of UC-S2S members for Bing's evaluation (DECISIONS 2026-11-09 update + 2026-11-10 item 3): 27 global daily 00 UTC fields
+ daily zonal-mean u, t at 5-150 hPa, one file per (init, member). Member 0 of every start also keeps the full upstream file
(``--save_mode hindcast --full_members 0``).

Fields (physical units, lead 0..45, time = valid time 00 UTC; tp24 = 24 h ending at the valid time, clipped at 0 as in
``ucs2s_save``):
  surface  t2m, tp24, msl, olr, sst, skt, swvl1, stl1
  850 hPa  u850, v850, t850, z850, q850
  500 hPa  z500, t500
  250 hPa  u250, v250, z250
  150 hPa  u150, v150               (DECISIONS 2026-11-10 item 3: with 250 hPa, for the 200 hPa interpolation in Bing's MJO code)
  100 hPa  z100
  50 hPa   u50, z50, t50
  10 hPa   u10, z10, t10            (u10 = 10 hPa zonal wind, not 10 m wind)
  zm_u, zm_t [time, level_zm, lat]  zonal means at 5, 10, 20, 30, 50, 70, 100, 150 hPa (on the native 1 deg latitudes)
Grid: ``native`` = UC-S2S 1 deg (lat -89.5..89.5, lon 0..359); ``1p5`` = conservative to the WeatherBench2 1.5 deg grid
(lat -90..90 step 1.5 = 121, lon 0..358.5 = 240; pole cells half height), area weights in sin(lat) x lon; NaNs (sst on land,
swvl1/stl1 over sea) are left out of the weights, a target cell is NaN if < 50 % of its area is valid.
Compression: zlib (``complevel``) + shuffle, chunks of one lead; optional ``keepbits`` = mantissa bits kept by IEEE round-to-nearest
bit rounding (lossy; None = lossless).
File name keeps the historical suffix ``_h25.nc`` (27 fields since 2026-11-11).
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np

SFC = {"t2m": "2m_temperature", "msl": "mean_sea_level_pressure", "sst": "sea_surface_temperature",
       "skt": "skin_temperature", "swvl1": "volumetric_soil_water_layer_1", "stl1": "soil_temperature_level_1"}
DIAG = {"tp24": "total_precipitation_24hr", "olr": "mean_top_net_long_wave_radiation_flux"}
UAV = {"u": "u_component_of_wind", "v": "v_component_of_wind", "t": "temperature", "z": "geopotential",
       "q": "specific_humidity"}
UA_SPEC = {850: "uvtzq", 500: "zt", 250: "uvz", 150: "uv", 100: "z", 50: "uzt", 10: "uzt"}
ORDER = ["t2m", "tp24", "msl", "olr", "sst", "skt", "swvl1", "stl1"] + \
        [f"{v}{L}" for L, vs in UA_SPEC.items() for v in vs]
ZM_LEVELS = [5, 10, 20, 30, 50, 70, 100, 150]
UNITS = {"t": "K", "u": "m s-1", "v": "m s-1", "z": "m2 s-2", "q": "kg kg-1", "t2m": "K", "skt": "K", "sst": "K", "stl1": "K",
         "msl": "Pa", "swvl1": "m3 m-3", "tp24": "m", "olr": "W m-2"}
assert len(ORDER) == 27, len(ORDER)


def _units(k):
    return UNITS.get(k) or UNITS[k.rstrip("0123456789")]


def bitround(x: np.ndarray, keepbits: int) -> np.ndarray:
    """round float32 mantissa to ``keepbits`` bits (round to nearest, ties to even); NaN/inf kept"""
    x = np.ascontiguousarray(x, np.float32)
    b = x.view(np.uint32)
    drop = 23 - int(keepbits)
    if drop <= 0:
        return x
    half = np.uint32(1 << (drop - 1)) - np.uint32(1)
    lsb = (b >> np.uint32(drop)) & np.uint32(1)
    r = (b + half + lsb) & ~np.uint32((1 << drop) - 1)
    out = r.view(np.float32).copy()
    bad = ~np.isfinite(x)
    out[bad] = x[bad]
    return out


def _overlap(src_edges, dst_edges):
    """W[i, j] = length of overlap of dst cell i with src cell j (1-D, monotone edges)"""
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.clip(hi - lo, 0, None)


def conservative_1p5_weights():
    lat_e1 = np.sin(np.deg2rad(np.arange(-90.0, 90.1, 1.0)))
    c = np.arange(-90.0, 90.1, 1.5)
    lat_e2 = np.sin(np.deg2rad(np.clip(np.concatenate([c - 0.75, [90.75]]), -90, 90)))
    Wlat = _overlap(lat_e1, lat_e2)                                     # [121, 180]
    # lon: src cells [c-0.5, c+0.5] for c = 0..359; dst [c-0.75, c+0.75] for c = 0..358.5 (wrap via a 3x periodic copy)
    s = np.arange(-360.5, 720.0, 1.0)                                    # edges of 3 periods
    d = np.concatenate([np.arange(0.0, 360.0, 1.5) - 0.75, [359.25]])
    W3 = _overlap(s, d)                                                  # [240, 1080]
    Wlon = W3[:, :360] + W3[:, 360:720] + W3[:, 720:]
    return Wlat, Wlon, c.astype(np.float32), np.arange(0.0, 360.0, 1.5).astype(np.float32)


_W = None


def to_1p5(x: np.ndarray) -> np.ndarray:
    """x [..., 180, 360] on the UC-S2S 1 deg grid -> [..., 121, 240] conservative"""
    global _W
    if _W is None:
        _W = conservative_1p5_weights()
    Wlat, Wlon = _W[0], _W[1]
    x = np.asarray(x, np.float64)
    m = np.isfinite(x)
    num = np.einsum("ia,...ab,jb->...ij", Wlat, np.where(m, x, 0.0), Wlon, optimize=True)
    den = np.einsum("ia,...ab,jb->...ij", Wlat, m.astype(np.float64), Wlon, optimize=True)
    full = Wlat.sum(1)[:, None] * Wlon.sum(1)[None, :]
    out = num / np.where(den > 0, den, 1.0)
    out[den < 0.5 * full] = np.nan
    return out.astype(np.float32)


def collect(runner, sfc, ua, diag):
    """dict short name -> [T, 180, 360] float32; zonal means [T, 8, 180] (u, t). Inputs as in ``ucs2s_save.save_crops``."""
    ds_ = runner.valid_dataset
    levels = [int(l) for l in runner.params.levels]
    sv, dv, uv = list(ds_.surface_variables), list(ds_.diagnostic_variables or []), list(ds_.upper_air_variables)
    f = {k: np.asarray(sfc[:, sv.index(v)], np.float32) for k, v in SFC.items()}
    for k, v in DIAG.items():
        x = np.asarray(diag[:, dv.index(v)], np.float32)
        f[k] = np.clip(x, 0, None) if k == "tp24" else x
    for L, vs in UA_SPEC.items():
        for v in vs:
            f[f"{v}{L}"] = np.asarray(ua[:, uv.index(UAV[v]), levels.index(L)], np.float32)
    zi = [levels.index(L) for L in ZM_LEVELS]
    zm = {k: np.asarray(ua[:, uv.index(UAV[k])][:, zi], np.float32).mean(-1) for k in ("u", "t")}
    return {k: f[k] for k in ORDER}, zm


def save_hindcast(runner, sfc, ua, diag, start, path, grid="native", complevel=4, keepbits=None, attrs=None):
    """sfc [T, C, H, W], ua [T, V, L, H, W], diag [T, D, H, W] (physical units, numpy) -> netCDF4 ``path`` (atomic)."""
    import xarray as xr
    p = runner.params
    lat, lon = np.asarray(p.lat, np.float32), np.asarray(p.lon, np.float32)
    fields, zm = collect(runner, sfc, ua, diag)
    T = next(iter(fields.values())).shape[0]
    dt = int(p.timedelta_hours)
    times = np.array([np.datetime64(start + timedelta(hours=dt * i), "ns") for i in range(T)])
    if grid == "1p5":
        _, _, glat, glon = conservative_1p5_weights()
        fields = {k: to_1p5(v) for k, v in fields.items()}
    elif grid == "native":
        glat, glon = lat, lon
    else:
        raise ValueError(grid)
    if keepbits is not None:
        fields = {k: bitround(v, keepbits) for k, v in fields.items()}
        zm = {k: bitround(v, keepbits) for k, v in zm.items()}
    data = {k: (("time", "lat", "lon"), v, {"units": _units(k)}) for k, v in fields.items()}
    for k, v in zm.items():
        data[f"zm_{k}"] = (("time", "level_zm", "lat_zm"), v, {"units": _units(k), "long_name": f"zonal-mean {k}"})
    ds = xr.Dataset(data, coords=dict(time=times, lat=glat, lon=glon, level_zm=np.array(ZM_LEVELS, np.float32),
                                      lat_zm=lat, lead_day=("time", np.arange(T, dtype=np.int16))))
    ds.attrs = dict(description="UC-S2S hindcast, 27 fields + zonal means (s2storm.models.ucs2s_hindcast)",
                    init=str(start), grid=grid, keepbits=str(keepbits), run=str(getattr(p, "run_num", "")),
                    tp24="24 h accumulation ending at the valid time, clipped at 0", **(attrs or {}))
    enc = {}
    for k in ds.data_vars:
        sh = ds[k].shape
        enc[k] = {"zlib": complevel > 0, "complevel": complevel, "shuffle": True, "chunksizes": (1,) + sh[1:]}
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    tmp = str(path) + ".partial"
    ds.to_netcdf(tmp, encoding=enc, engine="netcdf4")
    os.replace(tmp, path)
    return path
