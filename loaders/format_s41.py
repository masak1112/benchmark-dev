"""Loader for the S41 hindcast layout (UC-S2S SST-anomaly / plain and FuXi-S2S, Aurora 2026-11), for use from this repository's metrics.

Layout (one netCDF4 file per start x member, 27 daily 00 UTC fields + zonal means, lead_day 0..45):
  UC-S2S  <root>/hc/{sstanom/S2S_SSTANOM,plain/S2S}/S41u_<model>_<YYYYMMDD>/predictions/
          pangu_plasim_S41u_<model>_<YYYYMMDD>_24h_45step_<YYYYMMDD>00_ens_<m>_h25.nc        native 1 deg (lat -89.5..89.5 asc)
  FuXi    <root>/hc/fuxi/<YYYYMMDD>/m<mm>_h25.nc                                             1.5 deg, daily means labelled 00 UTC
Fields: t2m tp24 msl olr sst skt swvl1 stl1 u850 v850 t850 z850 q850 z500 t500 u250 v250 z250 u150 v150 z100 u50 z50 t50 u10 z10 t10
(u10 = 10 hPa wind), zm_u / zm_t [time, level_zm, lat_zm]. Units: K, m (tp24 = 24 h total ending at the valid time), Pa, m2 s-2, m s-1,
kg kg-1, W m-2. **olr in the files is ERA5's net TOA long-wave flux (negative)**; this loader returns OLR positive up. UC-S2S writes sst over
land and swvl1/stl1 over sea as fill values (270 K / 0): this loader masks them (static ERA5 1 deg masks), else a remap leaks the fill into
coastal cells. **Seeds:** in these runs member m used the same random seed at every start (one start per run); see README §Caveats.

    import sys; sys.path.insert(0, "<repo>")
    from loaders.format_s41 import open_start, member_paths
    ds = open_start("ucs2s_sstanom", "2025-01-06", fields=["t2m", "olr"], regrid="1p5")   # xr.Dataset [member, lead, lat, lon]

Everything is read through ``metrics/S41_aurora/s41eval/bing_eval/data.read_h25_daily`` (the reader the S41 evaluation uses).
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1] / "metrics" / "S41_aurora"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ROOT = Path(os.environ.get("S41_ROOT", "/lus/flare/projects/S2Storm/s2storm/runs/bing"))
TEMPLATES = {
    "ucs2s_sstanom": "hc/sstanom/S2S_SSTANOM/S41u_sstanom_{d}/predictions/pangu_plasim_S41u_sstanom_{d}_24h_45step_{d}00_ens_{m}_h25.nc",
    "ucs2s_plain": "hc/plain/S2S/S41u_plain_{d}/predictions/pangu_plasim_S41u_plain_{d}_24h_45step_{d}00_ens_{m}_h25.nc",
    "fuxi": "hc/fuxi/{d}/m{m:02d}_h25.nc",
}


def _date(x):
    return x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


def member_paths(system, start, members=30, root=None):
    """existing member files of one start, in member order (missing members are skipped)"""
    d = _date(start)
    root = Path(root) if root else ROOT
    out = []
    for m in range(members):
        p = root / TEMPLATES[system].format(d=f"{d:%Y%m%d}", m=m)
        if p.exists():
            out.append((m, p))
    return out


def open_start(system, start, fields=("t2m",), members=30, regrid="native", root=None):
    """-> xarray.Dataset with one variable per field, dims (member, lead, lat, lon); olr positive up; sst / swvl1 / stl1 masked.
    regrid: 'native' (UC-S2S 1 deg, FuXi 1.5 deg) or '1p5' (conservative to the WeatherBench2 121 x 240 grid)."""
    import xarray as xr
    from s41eval.bing_eval import data as D
    d = _date(start)
    mp = member_paths(system, d, members, root)
    if not mp:
        raise FileNotFoundError(f"no member files for {system} {d}")
    arrs, ld = {f: [] for f in fields}, None
    for _, p in mp:
        out, ld, _ = D.read_h25_daily(str(p), d, list(fields), regrid=(regrid == "1p5"))
        for f in fields:
            arrs[f].append(out[f])
    ny, nx = arrs[fields[0]][0].shape[-2:]
    if (ny, nx) == (121, 240):
        lat, lon = np.arange(-90.0, 90.1, 1.5), np.arange(0.0, 360.0, 1.5)
    else:
        lat, lon = np.arange(-89.5, 90.0, 1.0), np.arange(0.0, 360.0, 1.0)
    ds = xr.Dataset({f: (("member", "lead", "lat", "lon"), np.stack(arrs[f])) for f in fields},
                    coords=dict(member=[m for m, _ in mp], lead=np.asarray(ld), lat=lat, lon=lon))
    ds.attrs = dict(system=system, init=str(d), note="olr positive up; tp24 in mm/day (package units); sst/swvl1/stl1 masked")
    return ds
