#!/usr/bin/env python
"""
Daily ERA5 monsoon indices on the same 1-deg grid and with the same operators
as postprocess_ucs2s.py (verification reference for the raw-forecast metrics).

Per day D (labelled D 00 UTC):
  wy, imi, llj, mhi      instantaneous 00 UTC winds; U200/V200 interpolated in
                         ln(p) from 150/250 hPa exactly as for UC-S2S
  wy_true200, mhi_true200  same but with ERA5's real 200 hPa level (for NGCM,
                         whose pre-computed WY uses the real 200 hPa)
  olr_cmz                OLR (= -mtnlwrf) 24 h mean ending at D 00 UTC
                         (06,12,18 UTC of D-1 and 00 UTC of D), 2-deg CMZ mean;
                         matches the UC-S2S olr lead convention (checked: lead-1
                         RMSE 26.6 W/m2 vs 33.4 for the 24 h mean ending at init)
  olr_cmz_00z            instantaneous 00 UTC value (diagnostic only)
  <index>_day            calendar-day means of the same quantities (00/06/12/18 UTC winds; OLR
                         06..24 UTC of day D) -- the verification for FuXi-S2S (daily-mean model)

Usage: python era5_indices.py --years 1991 2024 --workers 34
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

import config as C
import common as U

LEVELS = [850, 250, 200, 150]


def _open(var, year):
    fn = C.ERA5_DIR / var / f"{year}_180x360.nc"
    return xr.open_dataset(fn)[var]


def process_year(year):
    t0 = time.time()
    u = _open("u_component_of_wind", year)
    v = _open("v_component_of_wind", year)
    lat, lon = u.lat.values.astype(float), u.lon.values.astype(float)
    flip = lat[0] > lat[-1]
    if flip:
        lat = lat[::-1]
    t = pd.DatetimeIndex(u.time.values)
    i00 = np.where(t.hour == 0)[0]
    days = t[i00].normalize()

    A = dict(wy=U.box_averager(lat, lon, C.BOX_WY), imi_s=U.box_averager(lat, lon, C.BOX_IMI_S),
             imi_n=U.box_averager(lat, lon, C.BOX_IMI_N), mhi=U.box_averager(lat, lon, C.BOX_MHI))
    R = U.india_regridder(lat, lon)
    cmz = U.cmz_mask().values

    # every 6-hourly field once; box means are linear, so daily means of box means = box means of daily means
    tday = t.normalize()
    box6 = {}
    for name, da in (("u", u), ("v", v)):
        for lev in LEVELS:
            x = da.sel(level=lev).values.astype("float64")
            x = x[:, ::-1, :] if flip else x
            for k, avg in A.items():
                box6[(name, lev, k)] = avg(x)[:, 0, 0]
            del x
    w = C.P200_WEIGHT_150

    def indices(box):
        def p200(name, k, true200):
            if true200:
                return box[(name, 200, k)]
            return box[(name, 250, k)] + w * (box[(name, 150, k)] - box[(name, 250, k)])
        return dict(
            wy=box[("u", 850, "wy")] - p200("u", "wy", False),
            wy_true200=box[("u", 850, "wy")] - p200("u", "wy", True),
            llj=box[("u", 850, "imi_s")],
            imi=box[("u", 850, "imi_s")] - box[("u", 850, "imi_n")],
            mhi=box[("v", 850, "mhi")] - p200("v", "mhi", False),
            mhi_true200=box[("v", 850, "mhi")] - p200("v", "mhi", True),
        )

    i6 = indices(box6)
    out = {k: v6[i00] for k, v6 in i6.items()}                          # 00 UTC states (UC-S2S)
    for k, v6 in i6.items():                                            # calendar-day means (FuXi)
        out[k + "_day"] = pd.Series(v6, index=tday).groupby(level=0).mean().reindex(days).values

    # OLR: 6-hourly; prepend the last 3 steps of the previous year for 1 Jan
    o = _open("mean_top_net_long_wave_radiation_flux", year)
    ot = pd.DatetimeIndex(o.time.values)
    ox = -o.values.astype("float64")
    if flip:
        ox = ox[:, ::-1, :]
    try:
        op = _open("mean_top_net_long_wave_radiation_flux", year - 1)
        prev = -op.isel(time=slice(-3, None)).values.astype("float64")
        prev = prev[:, ::-1, :] if flip else prev
        prev_t = pd.DatetimeIndex(op.time.values[-3:])
    except FileNotFoundError:
        prev, prev_t = np.full((3,) + ox.shape[1:], np.nan), ot[:3] - pd.Timedelta(days=1)
    ox = np.concatenate([prev, ox])
    ot = prev_t.append(ot)
    o2 = R(ox)[:, cmz].mean(-1)                                  # (time,) CMZ mean
    s = pd.Series(o2, index=ot)
    if not (np.diff(ot.values).astype("timedelta64[h]") == np.timedelta64(6, "h")).all():
        raise ValueError(f"OLR {year}: not uniformly 6-hourly")
    roll = s.rolling(4).mean()                                   # mean of 4 samples ending at t
    out["olr_cmz"] = roll.reindex(days).values
    out["olr_cmz_00z"] = s.reindex(days).values
    out["olr_cmz_day"] = roll.reindex(days + pd.Timedelta(days=1)).values     # UTC day D (06..24 UTC samples)

    ds = xr.Dataset({k: ("time", np.asarray(val, "float32")) for k, val in out.items()},
                    coords=dict(time=days.values))
    print(f"ERA5 {year}: {len(days)} days in {time.time() - t0:.0f}s", flush=True)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[1991, 2024])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    years = list(range(a.years[0], a.years[1] + 1))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        parts = list(ex.map(process_year, years))
    ds = xr.concat(parts, dim="time").sortby("time")
    for k in ds.data_vars:
        base = k.replace("_day", "").replace("_true200", "").replace("_00z", "")
        ds[k].attrs = dict(units=C.INDEX_INFO.get(base, {}).get("units", ""),
                           long_name=C.INDEX_INFO.get(base, {}).get("label", k))
    ds.attrs = dict(source=str(C.ERA5_DIR), note=__doc__.strip().splitlines()[0],
                    created=pd.Timestamp.now().isoformat(timespec="seconds"))
    U.ensure_dir(C.OBS_INDEX_FILE.parent)
    ds.to_netcdf(C.OBS_INDEX_FILE)
    print("wrote", C.OBS_INDEX_FILE, dict(ds.sizes))


if __name__ == "__main__":
    main()
