#!/usr/bin/env python3
"""
Gridpoint RMSE map for S2S ensemble forecasts, verified against ERA5.

Unlike rmse_main.py (which area-averages to a scalar per lead), this keeps the
spatial dimension: RMSE is accumulated at every gridpoint over all init dates,

    rmse(lat, lon, lead) = sqrt( mean_over_inits[ (ens_mean - era5)^2 ] )

Usage:
  python rmse_map.py --model uc_s2s_v2 --var 2m_temperature --years 2019
  python rmse_map.py --model uc_s2s_v2 --var 2m_temperature --max-dates 10

Output:
  {output_dir}/{model_name}/{model_name}_{var_name}_rmse_map[_{tag}].nc
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

DEFAULT_CONFIG = str(Path(__file__).parent / "configuration" / "config.yaml")


def load_era5_year(era5_dir, variable, year, scale=1.0):
    """Open one ERA5 year lazily; returns DataArray with dims (time, lat, lon)."""
    f = os.path.join(era5_dir, variable, f"{year}_180x360.nc")
    da = xr.open_dataset(f)[variable]
    if scale != 1.0:
        da = da * scale
    return da


_DATE_RE_B = re.compile(r"_(\d{10})_ens_0\.nc$")


def find_inits(input_dir, fmt, years, max_dates=None, keep_dates=None):
    """Return [(init_label, handle)] where handle is what the loader needs.

    keep_dates: optional set of YYYYMMDD strings to restrict to, so two models
    can be scored on an identical set of initializations.
    """
    if fmt == "format_b":
        items = [
            (_DATE_RE_B.search(f).group(1), f)
            for f in sorted(glob.glob(f"{input_dir}/*_ens_0.nc"))
            if _DATE_RE_B.search(f)
            and int(_DATE_RE_B.search(f).group(1)[:4]) in years
        ]
    elif fmt == "format_a":
        items = [
            (os.path.basename(d), d)
            for d in sorted(glob.glob(f"{input_dir}/2*"))
            if os.path.isdir(d) and int(os.path.basename(d)[:4]) in years
        ]
    else:
        raise ValueError(f"unknown format {fmt}")
    if keep_dates:
        # format_b labels are YYYYMMDDHH, format_a YYYYMMDD -> match on the day.
        items = [(lbl, h) for lbl, h in items if lbl[:8] in keep_dates]
    return items[:max_dates] if max_dates else items


def load_init_b(handle, model_var, scale, max_members):
    """format_b: one member per file, {prefix}_{YYYYMMDDHH}_ens_{N}.nc."""
    mfiles = sorted(glob.glob(re.sub(r"_ens_0\.nc$", "_ens_*.nc", handle)))
    if max_members:
        mfiles = mfiles[:max_members]
    members, times, lats, lons = [], None, None, None
    for mf in mfiles:
        try:
            ds = xr.open_dataset(mf)
            if times is None:
                times = ds.time.values
                lats, lons = ds.latitude.values, ds.longitude.values
            members.append(ds[model_var].values * scale)
            ds.close()
        except Exception as e:
            print(f"  WARN {os.path.basename(mf)}: {e}")
    if not members:
        return None, None, None, None
    return np.mean(members, axis=0), times, lats, lons


def load_init_a(handle, model_var, scale, max_members):
    """format_a: one dir per init, each seed file holding (member, time, lat, lon)."""
    seed_files = sorted(glob.glob(f"{handle}/*.nc"))
    members, times, lats, lons = [], None, None, None
    for sf in seed_files:
        try:
            ds = xr.open_dataset(sf)
            if times is None:
                times = ds.time.values
                lats, lons = ds.lat.values, ds.lon.values
            members.append(ds[model_var].values * scale)  # (member, time, lat, lon)
            ds.close()
        except Exception as e:
            print(f"  WARN {os.path.basename(sf)}: {e}")
    if not members:
        return None, None, None, None
    ens = np.concatenate(members, axis=0)
    if max_members:
        ens = ens[:max_members]
    return ens.mean(axis=0), times, lats, lons


LOADERS = {"format_a": load_init_a, "format_b": load_init_b}


def main():
    p = argparse.ArgumentParser(description="Gridpoint RMSE maps vs ERA5.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--model", default="uc_s2s_v2")
    p.add_argument("--var", default="2m_temperature")
    p.add_argument("--max-dates", type=int, default=None,
                   help="Limit number of init dates (for quick tests).")
    p.add_argument("--years", type=int, nargs="+", default=None,
                   help="Restrict to these init years (default: config years).")
    p.add_argument("--tag", default=None,
                   help="Suffix for output filename so subsets don't overwrite.")
    p.add_argument("--dates-file", default=None,
                   help="File of YYYYMMDD init dates (one per line) to restrict "
                        "to, for like-for-like comparison across models.")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model_cfg = cfg["models"][args.model]
    var_cfg = model_cfg["variables"][args.var]
    era5_dir = cfg["era5"]["dir"]
    out_parent = cfg["output"]["rmse_dir"]

    model_var = var_cfg["model_var"]
    era5_var = var_cfg["era5_var"]
    model_scale = var_cfg.get("model_scale", 1.0)
    era5_scale = var_cfg.get("era5_scale", 1.0)
    years = args.years if args.years else model_cfg["years"]
    max_members = model_cfg.get("max_members")

    tag = args.tag
    if tag is None and args.years:
        tag = "_".join(str(y) for y in args.years)

    keep_dates = None
    if args.dates_file:
        with open(args.dates_file) as fh:
            keep_dates = {ln.strip()[:8] for ln in fh if ln.strip()}
        print(f"Date list: {args.dates_file} ({len(keep_dates)} dates)")

    input_dir = model_cfg["input_dir"]
    fmt = model_cfg["format"]
    inits = find_inits(input_dir, fmt, years, args.max_dates, keep_dates)
    if not inits:
        sys.exit(f"No init files found under {input_dir} for years {years}")
    if keep_dates and len(inits) != len(keep_dates):
        print(f"  NOTE: {len(inits)} of {len(keep_dates)} requested dates present")
    load_init = LOADERS[fmt]

    print(f"Model    : {args.model}  ({fmt})")
    print(f"Variable : {args.var}  (model={model_var}, era5={era5_var})")
    print(f"Years    : {years}")
    print(f"Inits    : {len(inits)}", flush=True)

    # ERA5 cache: one open handle per year, reused across inits.
    era5_cache = {}

    def era5_at(times):
        """Pull ERA5 fields for a vector of forecast valid times -> (n_lead, lat, lon)."""
        out = []
        for t in times:
            yr = int(str(t)[:4])
            if yr not in era5_cache:
                era5_cache[yr] = load_era5_year(era5_dir, era5_var, yr, era5_scale)
            out.append(era5_cache[yr].sel(time=t, method="nearest").values)
        return np.stack(out, axis=0)

    sq_sum = None   # (n_lead, lat, lon) running sum of squared error
    n_valid = None  # (n_lead, lat, lon) count of contributing inits
    lats = lons = None
    n_lead = None

    for i, (init, handle) in enumerate(inits):
        ens_mean, times, la, lo = load_init(handle, model_var, model_scale, max_members)
        if ens_mean is None:
            print(f"  [{i+1}/{len(inits)}] {init} SKIP (no members)")
            continue
        if lats is None:
            lats, lons = la, lo

        try:
            truth = era5_at(times)
        except Exception as e:
            print(f"  [{i+1}/{len(inits)}] {init} SKIP (ERA5: {e})")
            continue

        if sq_sum is None:
            n_lead = ens_mean.shape[0]
            sq_sum = np.zeros_like(ens_mean, dtype=np.float64)
            n_valid = np.zeros_like(ens_mean, dtype=np.int32)

        err2 = (ens_mean - truth) ** 2
        good = np.isfinite(err2)
        sq_sum += np.where(good, err2, 0.0)
        n_valid += good

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(inits)}] {init} done", flush=True)

    rmse = np.sqrt(sq_sum / np.maximum(n_valid, 1))
    rmse[n_valid == 0] = np.nan

    out_dir = os.path.join(out_parent, args.model)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_nc = os.path.join(out_dir, f"{args.model}_{args.var}_rmse_map{suffix}.nc")

    ds_out = xr.Dataset(
        {
            "rmse": (("lead", "lat", "lon"), rmse.astype(np.float32)),
            "n_valid": (("lead", "lat", "lon"), n_valid),
        },
        coords={"lead": np.arange(n_lead), "lat": lats, "lon": lons},
        attrs={
            "model": args.model,
            "variable": args.var,
            "years": str(years),
            "n_init_dates": len(inits),
            "description": "Gridpoint RMSE of ensemble mean vs ERA5, "
                           "aggregated over all init dates. lead=0 is the "
                           "initial time (analysis).",
        },
    )
    ds_out.to_netcdf(out_nc)
    print(f"\nSaved → {out_nc}")

    # Sanity summary: area-weighted global mean of the map per lead.
    w = np.cos(np.deg2rad(lats))[:, None]
    for L in [0, 1, 5, 10, 20, 30, 45]:
        if L < n_lead:
            m = rmse[L]
            gm = np.nansum(w * m) / np.nansum(w * np.isfinite(m))
            print(f"  lead {L:>2}d  global mean RMSE = {gm:.3f}")


if __name__ == "__main__":
    main()
