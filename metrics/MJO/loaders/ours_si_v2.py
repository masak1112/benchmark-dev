"""
loaders/ours_si_v2.py — Loader for the SI-S2S postprocess_v2 ensemble dataset.

Source structure (flat directory):
  <src_dir>/
    pangu_plasim_pp_si_infer_v2_24h_45step_YYYYMMDDCC_ens_N.nc
      YYYYMMDDCC : 10-char init datetime, hour always 00 (e.g. 2019050200)
      N          : member index 0…19 (20 members per init date)
  Each NC:  dims (time=46, level=17, lat=180, lon=360)
            level in hPa — 200 hPa absent, interpolated log-pressure from 150/250
            lat: -89.5 → 89.5 (1°), lon: 0.0 → 359.0 (integer degrees)

Grid shift: lon 0.0-offset → 0.5-offset via periodic interpolation to match ERA5 obs.

Public interface:
  list_dates(src_dir) -> [YYYYMMDD, ...]
  process_date(date_str, src_dir, out_dir, variable, resolution,
               max_members, scratch_dir) -> out_file_path
"""

import os
import re
import glob
import numpy as np
import xarray as xr

_LP_150 = np.log(150.0)
_LP_200 = np.log(200.0)
_LP_250 = np.log(250.0)
_W150   = (_LP_250 - _LP_200) / (_LP_250 - _LP_150)
_W250   = 1.0 - _W150

LAT_BOUND = 20.0


def list_dates(src_dir: str) -> list:
    sentinel_files = glob.glob(os.path.join(src_dir, "*_ens_0.nc"))
    dates = sorted({_date_from_filename(f) for f in sentinel_files} - {None})
    return dates


def process_date(date_str: str,
                 src_dir: str,
                 out_dir: str,
                 variable: str,
                 resolution: float,
                 max_members=None,
                 scratch_dir: str = "/tmp") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{variable}_forecast_{date_str}.nc")
    if os.path.exists(out_file):
        print(f"Already exists, skipping: {out_file}")
        return out_file

    print(f"\n{'='*60}")
    print(f"[ours_si_v2] {date_str}  variable={variable}  res={resolution}°")

    date_h = date_str + "00"
    ens_files = sorted(
        glob.glob(os.path.join(src_dir, f"*_{date_h}_ens_*.nc")),
        key=_ens_index
    )
    if not ens_files:
        raise FileNotFoundError(
            f"No ensemble files for {date_str} (pattern: *_{date_h}_ens_*.nc) in {src_dir}")
    if max_members is not None:
        ens_files = ens_files[:max_members]
    print(f"  Members: {len(ens_files)}")

    ds0   = xr.open_dataset(ens_files[0])
    lat   = ds0["latitude"].values
    lon   = ds0["longitude"].values
    times = ds0["time"].values
    ds0.close()
    lat_mask = np.abs(lat) <= LAT_BOUND

    u200_list, v200_list, u850_list = [], [], []
    for ef in ens_files:
        ds = xr.open_dataset(ef)
        u150 = ds["u_component_of_wind"].sel(level=150.0, method="nearest").values
        u250 = ds["u_component_of_wind"].sel(level=250.0, method="nearest").values
        v150 = ds["v_component_of_wind"].sel(level=150.0, method="nearest").values
        v250 = ds["v_component_of_wind"].sel(level=250.0, method="nearest").values
        u850 = ds["u_component_of_wind"].sel(level=850.0, method="nearest").values
        ds.close()
        u200_list.append((_W150 * u150 + _W250 * u250).astype(np.float32))
        v200_list.append((_W150 * v150 + _W250 * v250).astype(np.float32))
        u850_list.append(u850.astype(np.float32))

    u200 = np.stack(u200_list)
    v200 = np.stack(v200_list)
    u850 = np.stack(u850_list)
    n_mem = u200.shape[0]

    if variable in ("chi200", "chi250"):
        print(f"  Computing {variable} via windspharm ({n_mem}×{u200.shape[1]} steps) ...")
        primary = _batch_chi(u200, v200, lat, lon)
    else:
        raise ValueError(f"Variable '{variable}' not supported. Use 'chi200'.")

    primary = primary[:, :, lat_mask, :]
    u850    = u850[:,   :, lat_mask, :]
    u200    = u200[:,   :, lat_mask, :]
    lat_sub = lat[lat_mask]

    lon_tgt = np.arange(0.5, 360.0, 1.0)
    print("  Regridding lon 0°-offset → 0.5°-offset ...")
    primary = _shift_lon(primary, lat_sub, lon, times, lon_tgt)
    u850    = _shift_lon(u850,    lat_sub, lon, times, lon_tgt)
    u200    = _shift_lon(u200,    lat_sub, lon, times, lon_tgt)

    dims   = ["ensemble", "time", "lat", "lon"]
    coords = {"ensemble": np.arange(n_mem), "time": times,
              "lat": lat_sub, "lon": lon_tgt}
    ds_out = xr.Dataset({
        "rlut":   xr.DataArray(primary.astype(np.float32), dims=dims, coords=coords),
        "ua_850": xr.DataArray(u850.astype(np.float32),    dims=dims, coords=coords),
        "ua_200": xr.DataArray(u200.astype(np.float32),    dims=dims, coords=coords),
    })
    ds_out["rlut"].attrs   = {"long_name": "Velocity potential 200 hPa", "units": "m2 s-1"}
    ds_out["ua_850"].attrs = {"long_name": "U850", "units": "m s-1"}
    ds_out["ua_200"].attrs = {"long_name": "U200 (log-p interp 150/250)", "units": "m s-1"}
    ds_out.attrs["init_date"] = date_str
    ds_out.attrs["source"]    = f"ours_si_v2  variable={variable}  lon+0.5deg"
    ds_out.to_netcdf(out_file)
    print(f"  Saved: {out_file}")
    return out_file


def _date_from_filename(fp: str):
    m = re.search(r"_(\d{10})_ens_\d+\.nc$", os.path.basename(fp))
    return m.group(1)[:8] if m else None


def _ens_index(fp: str) -> int:
    m = re.search(r"_ens_(\d+)\.nc$", os.path.basename(fp))
    return int(m.group(1)) if m else 0


def _batch_chi(u, v, lat, lon):
    from windspharm.xarray import VectorWind
    n_mem, n_time, n_lat, n_lon = u.shape
    n_flat = n_mem * n_time
    flat_u = u.reshape(n_flat, n_lat, n_lon)
    flat_v = v.reshape(n_flat, n_lat, n_lon)
    u_da = xr.DataArray(flat_u, dims=["step", "lat", "lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    v_da = xr.DataArray(flat_v, dims=["step", "lat", "lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    chi_flat = VectorWind(u_da, v_da).velocitypotential().values
    return chi_flat.reshape(n_mem, n_time, n_lat, n_lon).astype(np.float32)


def _shift_lon(arr, lat_sub, lon_src, times, lon_tgt):
    n_mem = arr.shape[0]
    lon_ext = np.append(lon_src, 360.0)
    arr_ext = np.concatenate([arr, arr[:, :, :, :1]], axis=-1)
    da = xr.DataArray(
        arr_ext,
        dims=["ensemble", "time", "lat", "lon"],
        coords={"ensemble": np.arange(n_mem), "time": times,
                "lat": lat_sub, "lon": lon_ext}
    )
    return da.interp(lon=lon_tgt, method="linear").values.astype(np.float32)
