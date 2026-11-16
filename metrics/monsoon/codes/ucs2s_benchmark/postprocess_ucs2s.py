#!/usr/bin/env python
"""
Postprocess raw UC-S2S hindcasts (hindcasts_bing format) for the monsoon benchmark.

Input  (one directory per init, 30 members; FuXi-S2S: <raw_dir>/<YYYYMMDD>/mNN_h25.nc, 1.5 deg,
        daily means of the UTC day labelled by time, lead 0 = ERA5 start day):
  <raw_dir>/<run_prefix>_<YYYYMMDD>/predictions/
      pangu_plasim_<run>_24h_45step_<YYYYMMDD>00_ens_<m>_h25.nc
  native 1-deg grid (lat -89.5..89.5, lon 0..359), time = valid time 00 UTC,
  lead_day 0..45, tp24 [m] = 24 h total ending at the valid time (0 at lead 0).

Output (per year, all inits of that year):
  <tp_dir>/<year>.nc           tp(init_time, member, day, lat, lon) [mm/day]
                               on the 2-deg India grid, day 0..45 (day 0 = 0,
                               kept so the legacy loaders also work on it)
  <tp_dir>/indices/<year>.nc   wy, imi, llj, mhi, olr_cmz (init_time, member, day)

Regridding is exact first-order conservative (area-weighted overlap; equivalent
to `cdo remapcon,grid_2deg_india.txt`).  Run with --check-cdo once to compare
against CDO on one member.

Usage:
  python postprocess_ucs2s.py --model ucs2s_plain --years 2019 2020 --workers 64
  python postprocess_ucs2s.py --model ucs2s_sstanom --years ALL --workers 100
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

import config as C
import common as U

READ_VARS = ["tp24", "olr", "u850", "v850", "u250", "v250", "u150", "v150"]
INDEX_VARS = ["wy", "imi", "llj", "mhi", "olr_cmz"]


# ----------------------------------------------------------------------------
def list_inits(model_key):
    m = C.MODELS[model_key]
    pre = "" if m.get("raw_layout") == "fuxi" else m["run_prefix"] + "_"
    out = []
    for d in os.listdir(m["raw_dir"]):
        if d.startswith(pre) and len(d) - len(pre) == 8 and d[len(pre):].isdigit():
            out.append(pd.Timestamp(d[len(pre):]))
    return pd.DatetimeIndex(sorted(out))


def member_path(model_key, init, mem):
    m = C.MODELS[model_key]
    s = init.strftime("%Y%m%d")
    if m.get("raw_layout") == "fuxi":           # FuXi-S2S: <raw_dir>/<YYYYMMDD>/mNN_h25.nc
        return m["raw_dir"] / s / f"m{mem:02d}_h25.nc"
    run = f"{m['run_prefix']}_{s}"
    return m["raw_dir"] / run / "predictions" / f"pangu_plasim_{run}_24h_45step_{s}00_ens_{mem}_h25.nc"


# ----------------------------------------------------------------------------
_AVG = {}
_CMZ = U.cmz_mask().values


def _averagers(lat, lon):
    key = (lat.size, float(lat[0]), lon.size, float(lon[0]))
    if key not in _AVG:
        _AVG[key] = dict(
            india=U.india_regridder(lat, lon),
            wy=U.box_averager(lat, lon, C.BOX_WY),
            imi_s=U.box_averager(lat, lon, C.BOX_IMI_S),
            imi_n=U.box_averager(lat, lon, C.BOX_IMI_N),
            mhi=U.box_averager(lat, lon, C.BOX_MHI),
        )
    return _AVG[key]


def compute_fields(lat, lon, v):
    """v: dict of (time, lat, lon) arrays on the native grid -> 2-deg tp + indices."""
    A = _averagers(lat, lon)
    box = lambda k, f: A[k](f)[..., 0, 0]
    w = C.P200_WEIGHT_150
    u200 = v["u250"] + w * (v["u150"] - v["u250"])
    v200 = v["v250"] + w * (v["v150"] - v["v250"])
    tp2 = A["india"](v["tp24"]) * 1000.0                 # m -> mm (per 24 h)
    olr2 = A["india"](-v["olr"])                          # net-down LW -> OLR (up, +)
    olr_cmz = olr2[:, _CMZ].mean(axis=-1)
    olr_cmz[0] = np.nan                                   # lead 0 is a zero placeholder
    llj = box("imi_s", v["u850"])
    out = dict(
        tp=tp2.astype("float32"),
        wy=(box("wy", v["u850"]) - box("wy", u200)).astype("float32"),
        imi=(llj - box("imi_n", v["u850"])).astype("float32"),
        llj=llj.astype("float32"),
        mhi=(box("mhi", v["v850"]) - box("mhi", v200)).astype("float32"),
        olr_cmz=olr_cmz.astype("float32"),
    )
    return out


def process_member(args):
    model_key, init, mem = args
    path = member_path(model_key, init, mem)
    if not path.exists():
        return init, mem, None, f"missing {path}"
    try:
        with xr.open_dataset(path) as ds:
            lat = ds["lat"].values.astype(float)
            lon = ds["lon"].values.astype(float)
            lead = ds["lead_day"].values.astype(int) if "lead_day" in ds else np.arange(ds.sizes["time"])
            v = {k: ds[k].values.astype("float64") for k in READ_VARS}
        if lat[0] > lat[-1]:          # make latitude ascending
            lat = lat[::-1]
            v = {k: a[:, ::-1, :] for k, a in v.items()}
        if not np.array_equal(lead, np.arange(lead.size)):
            return init, mem, None, f"unexpected lead_day {lead[:5]}... in {path}"
        return init, mem, compute_fields(lat, lon, v), None
    except Exception as e:  # report, do not kill the pool
        return init, mem, None, f"{type(e).__name__}: {e} ({path})"


# ----------------------------------------------------------------------------
def write_year(model_key, year, inits, results, n_mem, n_lead):
    m = C.MODELS[model_key]
    lats, lons = np.array(C.INDIA_LATS), np.array(C.INDIA_LONS)
    ni = len(inits)
    tp = np.full((ni, n_mem, n_lead, lats.size, lons.size), np.nan, "float32")
    idx = {k: np.full((ni, n_mem, n_lead), np.nan, "float32") for k in INDEX_VARS}
    pos = {t: i for i, t in enumerate(inits)}
    for (init, mem), out in results.items():
        i = pos[init]
        tp[i, mem] = out["tp"]
        for k in INDEX_VARS:
            idx[k][i, mem] = out[k]
    n_missing = int(np.isnan(tp[:, :, 1:, 0, 0]).all(axis=-1).sum())

    coords = dict(init_time=inits.values, member=np.arange(n_mem), day=np.arange(n_lead),
                  lat=lats, lon=lons)
    ds = xr.Dataset({"tp": (("init_time", "member", "day", "lat", "lon"), tp)}, coords=coords)
    conv = ("day k = 24 h accumulation ending at init + k days 00 UTC; day 0 = 0 (no accumulation yet)"
            if m.get("raw_layout") != "fuxi" else
            "day k = daily total of UTC calendar day init + k (FuXi daily-mean model); day 0 = ERA5 of the "
            "start day (not a forecast; dropped by the loader, which uses effective init = init + 1 d)")
    ds["tp"].attrs = dict(units="mm/day", long_name="Total precipitation (24 h)", convention=conv)
    ds["lat"].attrs = dict(units="degrees_north", standard_name="latitude")
    ds["lon"].attrs = dict(units="degrees_east", standard_name="longitude")
    ds.attrs = dict(model=m["label"], source=str(m["raw_dir"]),
                    regrid="first-order conservative, native 1 deg -> 2 deg India grid",
                    n_missing_member_inits=n_missing,
                    created=pd.Timestamp.now().isoformat(timespec="seconds"),
                    script=os.path.abspath(__file__))
    out_dir = U.ensure_dir(m["tp_dir"])
    enc = {"tp": dict(zlib=True, complevel=1, dtype="float32")}
    fn = out_dir / f"{year}.nc"
    ds.to_netcdf(fn, encoding=enc)

    di = xr.Dataset({k: (("init_time", "member", "day"), idx[k]) for k in INDEX_VARS},
                    coords=dict(init_time=inits.values, member=np.arange(n_mem), day=np.arange(n_lead)))
    for k in INDEX_VARS:
        di[k].attrs = dict(units=C.INDEX_INFO[k]["units"], long_name=C.INDEX_INFO[k]["label"])
    di["wy"].attrs["note"] = "U200 interpolated in ln(p) between 150 and 250 hPa (no 200 hPa level)"
    di["mhi"].attrs["note"] = "V200 interpolated in ln(p) between 150 and 250 hPa (no 200 hPa level)"
    di["olr_cmz"].attrs["note"] = "24 h mean ending at valid time, mean over 2-deg CMZ cells; lead 0 undefined"
    di.attrs = dict(model=m["label"], convention=(
        "day k = valid time init + k days 00 UTC; winds instantaneous, OLR 24 h mean ending at valid time"
        if m.get("raw_layout") != "fuxi" else
        "day k = daily mean of UTC calendar day init + k (winds and OLR); day 0 = ERA5 start day"))
    U.ensure_dir(m["indices_dir"])
    di.to_netcdf(m["indices_dir"] / f"{year}.nc")
    return fn, n_missing


def check_against_cdo(model_key, init, mem=0):
    """Regrid tp24 of one member with CDO remapcon and with our weights; print the difference."""
    path = member_path(model_key, init, mem)
    with xr.open_dataset(path) as ds:
        tp = ds["tp24"].isel(time=slice(1, 4)).load()
        lat, lon = ds["lat"].values.astype(float), ds["lon"].values.astype(float)
    ours = U.india_regridder(lat, lon)(tp.values.astype("float64"))
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", None)) as td:
        src = os.path.join(td, "src.nc")
        dst = os.path.join(td, "dst.nc")
        da = xr.DataArray(tp.values, dims=("time", "lat", "lon"),
                          coords=dict(time=np.arange(3), lat=lat, lon=lon), name="tp24")
        da.lat.attrs = dict(units="degrees_north", standard_name="latitude", axis="Y")
        da.lon.attrs = dict(units="degrees_east", standard_name="longitude", axis="X")
        da.to_netcdf(src)
        subprocess.run(["cdo", "-s", f"remapcon,{C.REPO / 'grid_2deg_india.txt'}", src, dst], check=True)
        cdo = xr.open_dataset(dst)["tp24"].values
    d = np.abs(cdo - ours) * 1000
    print(f"[check-cdo] {path.name}: max |CDO - ours| = {d.max():.2e} mm/day "
          f"(mean tp {np.mean(ours) * 1000:.2f} mm/day)")
    return d.max()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=[k for k, v in C.MODELS.items() if "raw_dir" in v])
    ap.add_argument("--years", nargs="+", default=["ALL"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-cdo", action="store_true", help="compare regridding with CDO on one member and exit")
    a = ap.parse_args()

    inits_all = list_inits(a.model)
    years = sorted(set(inits_all.year)) if a.years == ["ALL"] else [int(y) for y in a.years]
    print(f"{a.model}: {len(inits_all)} inits in {inits_all.min():%Y-%m-%d}..{inits_all.max():%Y-%m-%d}; years {years}")

    if a.check_cdo:
        check_against_cdo(a.model, inits_all[inits_all.year == years[0]][0])
        return

    n_mem = C.MODELS[a.model]["n_members"]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for y in years:
            fn = C.MODELS[a.model]["tp_dir"] / f"{y}.nc"
            if fn.exists() and not a.overwrite:
                print(f"{y}: {fn} exists, skipping (use --overwrite)")
                continue
            inits = inits_all[inits_all.year == y]
            tasks = [(a.model, t, mm) for t in inits for mm in range(n_mem)]
            t0 = time.time()
            results, errors, n_lead = {}, [], None
            for init, mem, out, err in ex.map(process_member, tasks, chunksize=4):
                if err:
                    errors.append(err)
                    continue
                results[(init, mem)] = out
                n_lead = out["tp"].shape[0]
            for e in errors:
                print("  WARNING:", e)
            if not results:
                print(f"{y}: nothing processed")
                continue
            fn, n_missing = write_year(a.model, y, inits, results, n_mem, n_lead)
            print(f"{y}: {len(inits)} inits x {n_mem} members -> {fn} "
                  f"({len(errors)} errors, {n_missing} missing member-inits) in {time.time() - t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
