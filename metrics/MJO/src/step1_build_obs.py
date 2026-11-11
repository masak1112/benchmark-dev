"""
src/step1_build_obs.py — Build ERA5 observation files for MJOcast.

Reads pipeline_config.yaml and builds the three obs files required by MJOcast:

  ERA5_climo.nc                     — day-of-year mean climatology (dayofyear × lon)
  ERA5_Meridional_Mean_Anomaly.nc   — daily anomaly timeseries (time × lon)
  ERA5_{variable}_anom_filt120.nc   — 120-day high-pass filtered anomalies (for EOF)

All three files are written to cfg.obs_dir.  All variable names inside the files
are kept as MJOcast expects: olr, uwnd850, uwnd200.

Variable dispatch:
  olr    — copies the pre-existing MJOcast ERA5 OLR obs, optionally regridding
            from 1° to 1.5° if cfg.resolution = 1.5
  chi200 — builds from pre-computed vp200_daily.nc + u200_daily.nc + u850_daily.nc
  chi250 — builds from ERA5 h5 files (requires windspharm)

Usage:
  python src/step1_build_obs.py [--config path/to/pipeline_config.yaml]
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PipelineConfig


def build_obs(cfg: PipelineConfig) -> None:
    os.makedirs(cfg.obs_dir, exist_ok=True)
    print(f"\nBuilding obs for variable={cfg.variable}  resolution={cfg.resolution}°")
    print(f"Output dir: {cfg.obs_dir}")

    if cfg.variable == "chi200":
        _build_chi200(cfg)
    elif cfg.variable == "chi250":
        _build_chi250(cfg)
    elif cfg.variable == "olr":
        _build_olr(cfg)
    else:
        raise ValueError(f"Unknown variable: {cfg.variable}")

    print("\nStep 1 complete.")


# ═══════════════════════════════════════════════════════════════════
# chi200 — from pre-computed daily vp200 files
# ═══════════════════════════════════════════════════════════════════

def _build_chi200(cfg: PipelineConfig) -> None:
    import glob as _glob
    vp_dir   = cfg.era5_vp200_dir
    out_dir  = cfg.obs_dir
    lat_s, lat_n = cfg.lat_south, cfg.lat_north
    start    = cfg.anomaly_start_year
    cl_end   = cfg.climo_end_year

    print(f"\nOpening pre-computed chi200 6-hourly yearly files from {vp_dir} ...")
    # Files are per-year 6-hourly: vp200_YYYY_180x360.nc, u200/u200_YYYY_180x360.nc, etc.
    vp_files   = sorted(_glob.glob(os.path.join(vp_dir,       "vp200_*_180x360.nc")))
    u200_files = sorted(_glob.glob(os.path.join(vp_dir, "u200", "u200_*_180x360.nc")))
    u850_files = sorted(_glob.glob(os.path.join(vp_dir, "u850", "u850_*_180x360.nc")))
    if not vp_files:
        raise FileNotFoundError(f"No vp200_*_180x360.nc files found in {vp_dir}")
    print(f"  Found {len(vp_files)} vp200 files, "
          f"{len(u200_files)} u200 files, {len(u850_files)} u850 files")

    # Load yearly files one at a time (no dask), resample to daily, then concat.
    print("  Loading and resampling yearly files to daily means ...")
    def _load_yearly(file_list, var):
        parts = []
        for fp in file_list:
            yr = os.path.basename(fp).split("_")[1]
            ds = xr.open_dataset(fp).load()
            daily = ds[[var]].resample(time="1D").mean()
            ds.close()
            parts.append(daily)
            print(f"    {var} {yr} loaded", flush=True)
        return xr.concat(parts, dim="time")
    ds_chi  = _load_yearly(vp_files,   "vp200")
    ds_u200 = _load_yearly(u200_files, "u_component_of_wind")
    ds_u850 = _load_yearly(u850_files, "u_component_of_wind")

    chi_raw  = ds_chi["vp200"].squeeze("level").sel(lat=slice(lat_s, lat_n)).mean("lat")
    u200_raw = ds_u200["u_component_of_wind"].squeeze("level").sel(lat=slice(lat_s, lat_n)).mean("lat")
    u850_raw = ds_u850["u_component_of_wind"].squeeze("level").sel(lat=slice(lat_s, lat_n)).mean("lat")

    # target lon grid (0.5-offset, 1°)
    lon_fc = _target_lon(cfg.resolution)

    # ── PART A: climatology ────────────────────────────────────────
    print(f"  Climatology: 1979–{cl_end}")
    climo_sel = slice("1979-01-01", f"{cl_end}-12-31")
    chi_cl  = chi_raw.sel(time=climo_sel).groupby("time.dayofyear").mean("time")
    u200_cl = u200_raw.sel(time=climo_sel).groupby("time.dayofyear").mean("time")
    u850_cl = u850_raw.sel(time=climo_sel).groupby("time.dayofyear").mean("time")

    chi_cn  = chi_cl.values.astype(np.float32)
    u200_cn = u200_cl.values.astype(np.float32)
    u850_cn = u850_cl.values.astype(np.float32)
    doy_coord = chi_cl.dayofyear.values

    # regrid from integer lons (0,1,…,359) to target grid via half-step interpolation
    src_lon = ds_chi["vp200"].lon.values   # (360,) — shared across all three datasets
    chi_cn  = _regrid_obs_lon(chi_cn,  src_lon, lon_fc)
    u200_cn = _regrid_obs_lon(u200_cn, src_lon, lon_fc)
    u850_cn = _regrid_obs_lon(u850_cn, src_lon, lon_fc)

    ds_climo = xr.Dataset(
        {"olr":     (["dayofyear","lon"], chi_cn),
         "uwnd850": (["dayofyear","lon"], u850_cn),
         "uwnd200": (["dayofyear","lon"], u200_cn)},
        coords={"dayofyear": doy_coord, "lon": lon_fc})
    _set_attrs(ds_climo, "chi200")
    climo_path = os.path.join(out_dir, "ERA5_climo.nc")
    ds_climo.to_netcdf(climo_path)
    print(f"  Saved: {climo_path}")

    # doy lookup for fast anomaly removal
    chi_doy  = {int(doy_coord[i]): chi_cn[i]  for i in range(len(doy_coord))}
    u200_doy = {int(doy_coord[i]): u200_cn[i] for i in range(len(doy_coord))}
    u850_doy = {int(doy_coord[i]): u850_cn[i] for i in range(len(doy_coord))}

    # ── PART B: anomaly timeseries ─────────────────────────────────
    print(f"  Anomaly: {start}–2024")
    anom_sel  = slice(f"{start}-01-01", "2024-12-31")
    chi_v  = chi_raw.sel(time=anom_sel).values.astype(np.float32)
    u200_v = u200_raw.sel(time=anom_sel).values.astype(np.float32)
    u850_v = u850_raw.sel(time=anom_sel).values.astype(np.float32)
    times  = pd.DatetimeIndex(chi_raw.sel(time=anom_sel).time.values).normalize().values

    # regrid
    chi_v  = _regrid_obs_lon(chi_v,  src_lon, lon_fc)
    u200_v = _regrid_obs_lon(u200_v, src_lon, lon_fc)
    u850_v = _regrid_obs_lon(u850_v, src_lon, lon_fc)

    doy_arr   = np.array([pd.Timestamp(t).day_of_year for t in times])
    chi_anom  = np.zeros_like(chi_v)
    u200_anom = np.zeros_like(u200_v)
    u850_anom = np.zeros_like(u850_v)
    for i, doy in enumerate(doy_arr):
        chi_anom[i]  = chi_v[i]  - chi_doy[doy]
        u200_anom[i] = u200_v[i] - u200_doy[doy]
        u850_anom[i] = u850_v[i] - u850_doy[doy]

    ds_anom = _make_anom_ds(chi_anom, u850_anom, u200_anom, times, doy_arr, lon_fc)
    anom_path = os.path.join(out_dir, "ERA5_Meridional_Mean_Anomaly.nc")
    ds_anom.to_netcdf(anom_path)
    print(f"  Saved: {anom_path}")

    ds_chi.close(); ds_u200.close(); ds_u850.close()

    # ── PART C: 120-day high-pass filter ──────────────────────────
    _apply_120day_filter(ds_anom, "chi200", out_dir)


# ═══════════════════════════════════════════════════════════════════
# chi250 — from ERA5 h5 files (requires windspharm)
# ═══════════════════════════════════════════════════════════════════

def _build_chi250(cfg: PipelineConfig) -> None:
    import h5py
    from windspharm.xarray import VectorWind
    from datetime import datetime, timedelta

    h5_dir  = cfg.era5_h5_dir
    out_dir = cfg.obs_dir
    lat_s, lat_n = cfg.lat_south, cfg.lat_north
    start   = cfg.anomaly_start_year
    cl_end  = cfg.climo_end_year
    END_YEAR = 2018   # h5 files end at 2018

    print(f"\nBuilding chi250 obs from ERA5 h5 files in {h5_dir}")
    climo_file = os.path.join(h5_dir, "1979-2018_mean_climatology.nc")

    lat = np.load(os.path.join(h5_dir, "lat.npy"))
    lon = np.load(os.path.join(h5_dir, "lon.npy"))
    lon_fc  = (lon + 0.5).astype(np.float32)   # integer → half-integer
    lat_mask = (lat >= lat_s) & (lat <= lat_n)
    lon_target = _target_lon(cfg.resolution)

    # ── PART A: climatology ────────────────────────────────────────
    print(f"  Climatology (VectorWind over 366 doy) ...")
    climo_ds = xr.open_dataset(climo_file)
    n_doy = climo_ds.dims["time"]
    u_cl = climo_ds["u_component_of_wind"].sel(plev=250.0).values.astype(np.float32)
    v_cl = climo_ds["v_component_of_wind"].sel(plev=250.0).values.astype(np.float32)
    u_da_cl = xr.DataArray(u_cl, dims=["doy","lat","lon"],
                            coords={"doy": np.arange(n_doy), "lat": lat, "lon": lon})
    v_da_cl = xr.DataArray(v_cl, dims=["doy","lat","lon"],
                            coords={"doy": np.arange(n_doy), "lat": lat, "lon": lon})
    chi_cl_full = VectorWind(u_da_cl, v_da_cl).velocitypotential().values
    chi250_cl = chi_cl_full[:, lat_mask, :].mean(axis=1).astype(np.float32)
    u250_cl   = u_cl[:, lat_mask, :].mean(axis=1).astype(np.float32)
    u850_cl   = (climo_ds["u_component_of_wind"].sel(plev=850.0).values
                 [:, lat_mask, :].mean(axis=1).astype(np.float32))
    climo_ds.close()

    doy_coord = np.arange(1, n_doy + 1)
    # map to target lon
    chi250_cl_r = _regrid_obs_lon(chi250_cl, lon_fc, lon_target)
    u250_cl_r   = _regrid_obs_lon(u250_cl,   lon_fc, lon_target)
    u850_cl_r   = _regrid_obs_lon(u850_cl,   lon_fc, lon_target)

    ds_climo = xr.Dataset(
        {"olr":     (["dayofyear","lon"], chi250_cl_r),
         "uwnd850": (["dayofyear","lon"], u850_cl_r),
         "uwnd200": (["dayofyear","lon"], u250_cl_r)},
        coords={"dayofyear": doy_coord, "lon": lon_target})
    climo_path = os.path.join(out_dir, "ERA5_climo.nc")
    ds_climo.to_netcdf(climo_path)
    print(f"  Saved: {climo_path}")

    chi_doy  = {int(doy_coord[i]): chi250_cl_r[i] for i in range(n_doy)}
    u850_doy = {int(doy_coord[i]): u850_cl_r[i]   for i in range(n_doy)}
    u250_doy = {int(doy_coord[i]): u250_cl_r[i]   for i in range(n_doy)}

    # ── PART B: daily anomaly from h5 files ────────────────────────
    print(f"  Building anomaly timeseries ({start}–{END_YEAR}) from h5 ...")
    years = range(start, END_YEAR + 1)
    all_times, all_chi, all_u850, all_u250 = [], [], [], []

    for year in years:
        is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        n_days = 366 if is_leap else 365
        u_yr   = np.zeros((n_days, 180, 360), dtype=np.float32)
        v_yr   = np.zeros_like(u_yr)
        u85_yr = np.zeros_like(u_yr)
        valid  = np.ones(n_days, dtype=bool)
        for day_idx in range(n_days):
            h5f = os.path.join(h5_dir, f"{year}_{day_idx*4:04d}.h5")
            if not os.path.exists(h5f):
                valid[day_idx] = False
                continue
            with h5py.File(h5f, "r") as f:
                u_yr[day_idx]   = f["input/u_component_of_wind_250.0"][:]
                v_yr[day_idx]   = f["input/v_component_of_wind_250.0"][:]
                u85_yr[day_idx] = f["input/u_component_of_wind_850.0"][:]
        u_da = xr.DataArray(u_yr[valid], dims=["time","lat","lon"],
                            coords={"time": np.arange(valid.sum()), "lat": lat, "lon": lon})
        v_da = xr.DataArray(v_yr[valid], dims=["time","lat","lon"],
                            coords={"time": np.arange(valid.sum()), "lat": lat, "lon": lon})
        chi_yr = VectorWind(u_da, v_da).velocitypotential().values
        vi = 0
        for day_idx in range(n_days):
            if not valid[day_idx]:
                continue
            date = datetime(year, 1, 1) + timedelta(days=day_idx)
            doy  = date.timetuple().tm_yday
            all_times.append(np.datetime64(date, "ns"))
            all_chi.append(
                _regrid_obs_lon(chi_yr[vi][lat_mask,:].mean(0)[np.newaxis],
                                lon_fc, lon_target)[0] - chi_doy[doy])
            all_u850.append(
                _regrid_obs_lon(u85_yr[day_idx][lat_mask,:].mean(0)[np.newaxis],
                                lon_fc, lon_target)[0] - u850_doy[doy])
            all_u250.append(
                _regrid_obs_lon(u_yr[day_idx][lat_mask,:].mean(0)[np.newaxis],
                                lon_fc, lon_target)[0] - u250_doy[doy])
            vi += 1
        print(f"  {year}: {valid.sum()}/{n_days} days", flush=True)

    times_arr = np.array(all_times)
    chi_arr   = np.stack(all_chi)
    u850_arr  = np.stack(all_u850)
    u250_arr  = np.stack(all_u250)
    doy_arr   = np.array([pd.Timestamp(t).day_of_year for t in times_arr])

    # ── 2019 gap fill via scaled OLR anomaly ──────────────────────
    chi_arr, u850_arr, u250_arr, times_arr, doy_arr = _append_2019_via_olr(
        cfg, chi_arr, u850_arr, u250_arr, times_arr, doy_arr, lon_target
    )

    ds_anom = _make_anom_ds(chi_arr, u850_arr, u250_arr, times_arr, doy_arr, lon_target)
    anom_path = os.path.join(out_dir, "ERA5_Meridional_Mean_Anomaly.nc")
    ds_anom.to_netcdf(anom_path)
    print(f"  Saved: {anom_path}")

    _apply_120day_filter(ds_anom, "chi250", out_dir)


# ═══════════════════════════════════════════════════════════════════
# OLR — copy/regrid from MJOcast built-in obs
# ═══════════════════════════════════════════════════════════════════

def _build_olr(cfg: PipelineConfig) -> None:
    mjocast_obs = cfg.era5_mjocast_obs_dir
    out_dir = cfg.obs_dir

    # Files to copy/regrid — keep MJOcast's standard filenames so usr_named_obs
    # matches without any renaming.
    files = {
        "ERA5_climo.nc":                               "ERA5_climo.nc",
        "ERA5_Meridional_Mean_Anomaly.nc":             "ERA5_Meridional_Mean_Anomaly.nc",
        "ERA5_Meridional_Mean_Anomaly_Filtered120.nc": "ERA5_Meridional_Mean_Anomaly_Filtered120.nc",
    }

    if abs(cfg.resolution - 1.5) < 0.01:
        target_lon = np.arange(0.0, 360.0, 1.5)
        print(f"  Regridding OLR obs from 1° → 1.5° ...")
    else:
        target_lon = None  # copy as-is

    for src_name, dst_name in files.items():
        src = os.path.join(mjocast_obs, src_name)
        dst = os.path.join(out_dir, dst_name)
        if os.path.exists(dst):
            print(f"  Already exists: {dst}")
            continue
        if not os.path.exists(src):
            print(f"  WARNING: source not found: {src}  (skipping)")
            continue
        ds = xr.open_dataset(src)
        if target_lon is not None:
            ds = ds.interp(lon=target_lon, method="linear")
            for v in ds.data_vars:
                ds[v] = ds[v].astype(np.float32)
        ds.to_netcdf(dst)
        print(f"  Saved: {dst}")

    # MJO_obs_created.nc is produced by step3, not here
    print("  OLR obs ready.  (MJO_obs_created.nc produced by step3)")


# ═══════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════

def _target_lon(resolution: float) -> np.ndarray:
    if abs(resolution - 1.5) < 0.01:
        return np.arange(0.0, 360.0, 1.5).astype(np.float32)
    # default: 0.5-offset 1° grid matching MJOcast convention
    return np.arange(0.5, 360.0, 1.0).astype(np.float32)


def _regrid_obs_lon(arr: np.ndarray, src_lon: np.ndarray, tgt_lon: np.ndarray) -> np.ndarray:
    """Regrid (..., lon) array from src_lon to tgt_lon via xarray interpolation.

    Appends a periodic wrap-around point at lon=360 (= value at lon=0) so that
    target points near 359.5° are interpolated rather than extrapolated to NaN.
    """
    if np.allclose(src_lon, tgt_lon, atol=0.01) and len(src_lon) == len(tgt_lon):
        return arr
    # extend with periodic wrap-around point at 360°
    lon_ext = np.append(src_lon, 360.0)
    if arr.ndim == 1:
        arr_ext = np.append(arr, arr[..., :1])
        da = xr.DataArray(arr_ext, dims=["lon"], coords={"lon": lon_ext})
        return da.interp(lon=tgt_lon, method="linear").values.astype(np.float32)
    arr_ext = np.concatenate([arr, arr[..., :1]], axis=-1)
    dims = ["d" + str(i) for i in range(arr.ndim - 1)] + ["lon"]
    coords = {dims[-1]: lon_ext}
    da = xr.DataArray(arr_ext, dims=dims, coords=coords)
    return da.interp(lon=tgt_lon, method="linear").values.astype(np.float32)


def _make_anom_ds(chi, u850, u200, times, doy_arr, lon) -> xr.Dataset:
    ds = xr.Dataset(
        {"olr":     (["time","lon"], chi.astype(np.float32)),
         "uwnd850": (["time","lon"], u850.astype(np.float32)),
         "uwnd200": (["time","lon"], u200.astype(np.float32))},
        coords={"time": times, "lon": lon,
                "dayofyear": ("time", doy_arr)})
    return ds


def _set_attrs(ds: xr.Dataset, variable: str) -> None:
    label = "chi200" if variable == "chi200" else \
            "chi250" if variable == "chi250" else "OLR"
    ds["olr"].attrs     = {"long_name": f"{label} 15S-15N climo", "units": "m2 s-1"}
    ds["uwnd850"].attrs = {"long_name": "U850 climo 15S-15N",     "units": "m s-1"}
    ds["uwnd200"].attrs = {"long_name": "U200 climo 15S-15N",     "units": "m s-1"}


def _apply_120day_filter(ds_anom: xr.Dataset, variable: str, out_dir: str) -> None:
    print("  Applying 120-day high-pass filter ...")
    rolled  = ds_anom.rolling(time=120, center=False, min_periods=120).mean()
    ds_filt = (ds_anom - rolled).dropna("time")
    ds_filt = ds_filt.assign_coords(
        dayofyear=("time",
                   np.array([pd.Timestamp(t).day_of_year
                             for t in ds_filt.time.values])))
    for v in ["olr","uwnd850","uwnd200"]:
        ds_filt[v] = ds_filt[v].astype(np.float32)
    filt_path = os.path.join(out_dir, f"ERA5_{variable}_anom_filt120.nc")
    ds_filt.to_netcdf(filt_path)
    print(f"  Saved: {filt_path}")


def _append_2019_via_olr(cfg, chi_arr, u850_arr, u250_arr, times_arr, doy_arr, lon_target):
    """Fill 2019 by scaling the MJOcast ERA5 OLR anomaly (workaround for missing h5 data)."""
    mjocast_obs = cfg.era5_mjocast_obs_dir
    era5_path = os.path.join(mjocast_obs, "ERA5_Meridional_Mean_Anomaly.nc")
    if not os.path.exists(era5_path):
        print("  Skipping 2019 gap fill — MJOcast ERA5 OLR obs not found.")
        return chi_arr, u850_arr, u250_arr, times_arr, doy_arr

    era5_obs  = xr.open_dataset(era5_path)
    era5_2019 = era5_obs.sel(time=slice("2019-01-01", "2019-12-31"))
    if len(era5_2019.time) == 0:
        era5_obs.close()
        return chi_arr, u850_arr, u250_arr, times_arr, doy_arr

    # compute regression scale from overlap
    ov_s = str(times_arr[0])[:10]
    era5_olr_ov = era5_obs.sel(time=slice(ov_s, "2018-12-31"))["olr"].values
    era5_olr_ov_r = _regrid_obs_lon(era5_olr_ov, era5_obs.lon.values, lon_target)
    n_ov = min(len(era5_olr_ov_r), len(chi_arr))
    if n_ov > 0:
        scale = (np.mean(chi_arr[:n_ov] * era5_olr_ov_r[:n_ov], axis=0) /
                 np.maximum(np.var(era5_olr_ov_r[:n_ov], axis=0), 1e-30))
    else:
        scale = np.ones(len(lon_target), dtype=np.float32)

    olr_2019 = _regrid_obs_lon(era5_2019["olr"].values, era5_obs.lon.values, lon_target)
    pad_chi  = (olr_2019 * scale).astype(np.float32)
    pad_u850 = _regrid_obs_lon(era5_2019["uwnd850"].values, era5_obs.lon.values, lon_target)
    pad_u250 = _regrid_obs_lon(era5_2019["uwnd200"].values, era5_obs.lon.values, lon_target)
    pad_times = era5_2019.time.values
    pad_doy  = np.array([pd.Timestamp(t).day_of_year for t in pad_times])
    era5_obs.close()

    print(f"  Appended {len(pad_times)} 2019 days via scaled OLR proxy.")
    return (np.concatenate([chi_arr,  pad_chi],  axis=0),
            np.concatenate([u850_arr, pad_u850], axis=0),
            np.concatenate([u250_arr, pad_u250], axis=0),
            np.concatenate([times_arr, pad_times]),
            np.concatenate([doy_arr,   pad_doy]))


# ── CLI entry point ───────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Build ERA5 obs files for MJOcast.")
    p.add_argument("--config", default="config/pipeline_config.yaml",
                   help="Path to pipeline_config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = PipelineConfig(args.config)
    build_obs(cfg)
