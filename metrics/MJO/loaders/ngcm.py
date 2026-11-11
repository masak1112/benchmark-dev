"""
loaders/ngcm.py — Loader for NGCM ensemble forecasts.

Source structure:
  <src_dir>/climatology/YYYY/MEMBER/YYYY-MM-DDTCC.zarr
    MEMBER : 1..51 (one zarr per member per init date)
    dims   : (ensemble=1, time=180, level=23, lon=128, lat=64)
    time   : 6-hourly offsets in hours from init
    level  : hPa — 200 hPa available directly
    lat    : Gaussian grid, -87.86..87.86 (64 pts, ~2.8°)
    lon    : 0..357.19 (128 pts, 2.8125°)

Processing:
  - 200 hPa U/V available directly
  - Resample 6-hourly → daily means (groups of 4)
  - Compute chi200 via windspharm on native Gaussian grid
  - Regrid to 1.0° via scipy RegularGridInterpolator, subset lat ±20°
  - Shift lon from 0.0-offset to 0.5-offset

Public interface:
  list_dates(src_dir) -> [YYYYMMDD, ...]
  process_date(date_str, src_dir, out_dir, variable, resolution,
               max_members, scratch_dir) -> out_file_path
"""

import os
import glob
import re
import numpy as np
import xarray as xr
import pandas as pd

LAT_BOUND = 20.0


def list_dates(src_dir: str) -> list:
    climo_dir = os.path.join(src_dir, "climatology")
    zarr_files = glob.glob(os.path.join(climo_dir, "????", "1", "????-??-??T00.zarr"))
    dates = sorted({re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f)).group(1).replace("-", "")
                    for f in zarr_files
                    if re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))})
    return dates


def process_date(date_str: str,
                 src_dir: str,
                 out_dir: str,
                 variable: str,
                 resolution: float,
                 max_members=None,
                 scratch_dir: str = "/tmp") -> str:
    import zarr

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{variable}_forecast_{date_str}.nc")
    if os.path.exists(out_file):
        print(f"Already exists, skipping: {out_file}")
        return out_file

    yr = date_str[:4]
    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    zarr_name = f"{date_fmt}T00.zarr"
    climo_dir = os.path.join(src_dir, "climatology", yr)

    # find all member dirs that have this date
    member_dirs = sorted(os.listdir(climo_dir), key=lambda x: int(x))
    member_zarrs = []
    for m in member_dirs:
        zp = os.path.join(climo_dir, m, zarr_name)
        if os.path.exists(zp):
            member_zarrs.append((int(m), zp))
    if not member_zarrs:
        raise FileNotFoundError(f"No NGCM zarrs found for {date_str} in {climo_dir}")
    if max_members is not None:
        member_zarrs = member_zarrs[:max_members]

    print(f"\n{'='*60}")
    print(f"[ngcm] {date_str}  variable={variable}  res={resolution}°")
    print(f"  Members: {len(member_zarrs)}")

    # read grid from first zarr
    z0 = zarr.open(member_zarrs[0][1], mode="r")
    lat_gauss = z0["latitude"][:]   # (64,) Gaussian lats
    lon_gauss = z0["longitude"][:]  # (128,) 0..357.19
    level     = z0["level"][:]
    lev200_idx = int(np.where(level == 200)[0][0])
    lev850_idx = int(np.where(level == 850)[0][0])
    n_steps = z0["time"].shape[0]   # 180 (6-hourly)
    n_days  = n_steps // 4          # 45 daily

    # target 1.0° grid
    lat_1deg = np.arange(-89.5, 90.0, 1.0)
    lon_1deg = np.arange(0.0, 360.0, 1.0)
    lat_mask = np.abs(lat_1deg) <= LAT_BOUND
    lat_sub  = lat_1deg[lat_mask]
    lon_tgt  = np.arange(0.5, 360.0, 1.0)

    chi_list, u850_list, u200_list = [], [], []
    for mem_id, zpath in member_zarrs:
        print(f"  member {mem_id}", flush=True)
        z = zarr.open(zpath, mode="r")
        # shape: (1, time, level, lon, lat) — note lon before lat
        u200_raw = z["u_component_of_wind"][0, :, lev200_idx, :, :]  # (180, 128, 64)
        v200_raw = z["v_component_of_wind"][0, :, lev200_idx, :, :]
        u850_raw = z["u_component_of_wind"][0, :, lev850_idx, :, :]

        # transpose to (time, lat, lon) for windspharm
        u200_t = u200_raw.transpose(0, 2, 1).astype(np.float32)  # (180, 64, 128)
        v200_t = v200_raw.transpose(0, 2, 1).astype(np.float32)
        u850_t = u850_raw.transpose(0, 2, 1).astype(np.float32)

        # 6-hourly → daily (groups of 4)
        u200_d = _resample6h(u200_t, n_days)   # (45, 64, 128)
        v200_d = _resample6h(v200_t, n_days)
        u850_d = _resample6h(u850_t, n_days)

        # chi200 on native Gaussian grid
        if variable in ("chi200", "chi250"):
            chi_m = _chi_single(u200_d, v200_d, lat_gauss, lon_gauss)
        else:
            raise ValueError(f"Variable '{variable}' not supported. Use 'chi200'.")

        # regrid Gaussian → 1.0° regular
        chi_1deg  = _regrid_gaussian(chi_m,  lat_gauss, lon_gauss, lat_1deg, lon_1deg)
        u850_1deg = _regrid_gaussian(u850_d, lat_gauss, lon_gauss, lat_1deg, lon_1deg)
        u200_1deg = _regrid_gaussian(u200_d, lat_gauss, lon_gauss, lat_1deg, lon_1deg)

        # subset lat + shift lon
        chi_list.append(_shift_lon(chi_1deg[:,  lat_mask, :], lat_sub, lon_1deg, lon_tgt))
        u850_list.append(_shift_lon(u850_1deg[:, lat_mask, :], lat_sub, lon_1deg, lon_tgt))
        u200_list.append(_shift_lon(u200_1deg[:, lat_mask, :], lat_sub, lon_1deg, lon_tgt))

    primary  = np.stack(chi_list)
    u850_out = np.stack(u850_list)
    u200_out = np.stack(u200_list)
    n_mem    = len(chi_list)

    init_dt = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    times   = np.array([init_dt + pd.Timedelta(days=d+1) for d in range(n_days)],
                       dtype="datetime64[ns]")

    dims   = ["ensemble", "time", "lat", "lon"]
    coords = {"ensemble": np.arange(n_mem), "time": times,
              "lat": lat_sub, "lon": lon_tgt}
    ds_out = xr.Dataset({
        "rlut":   xr.DataArray(primary.astype(np.float32),  dims=dims, coords=coords),
        "ua_850": xr.DataArray(u850_out.astype(np.float32), dims=dims, coords=coords),
        "ua_200": xr.DataArray(u200_out.astype(np.float32), dims=dims, coords=coords),
    })
    ds_out["rlut"].attrs   = {"long_name": "Velocity potential 200 hPa", "units": "m2 s-1"}
    ds_out["ua_850"].attrs = {"long_name": "U850", "units": "m s-1"}
    ds_out["ua_200"].attrs = {"long_name": "U200", "units": "m s-1"}
    ds_out.attrs["init_date"] = date_str
    ds_out.attrs["source"]    = f"NGCM  variable={variable}  Gaussian→1.0deg  lon+0.5"
    ds_out.to_netcdf(out_file)
    print(f"  Saved: {out_file}")
    return out_file


def _resample6h(arr: np.ndarray, n_days: int) -> np.ndarray:
    """(180, lat, lon) 6-hourly → (n_days, lat, lon) daily means."""
    n = n_days * 4
    return arr[:n].reshape(n_days, 4, arr.shape[1], arr.shape[2]).mean(axis=1).astype(np.float32)


def _chi_single(u: np.ndarray, v: np.ndarray,
                lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    from windspharm.xarray import VectorWind
    n_t = u.shape[0]
    u_da = xr.DataArray(u, dims=["time", "lat", "lon"],
                        coords={"time": np.arange(n_t), "lat": lat, "lon": lon})
    v_da = xr.DataArray(v, dims=["time", "lat", "lon"],
                        coords={"time": np.arange(n_t), "lat": lat, "lon": lon})
    return VectorWind(u_da, v_da).velocitypotential().values.astype(np.float32)


def _regrid_gaussian(arr: np.ndarray, lat_src, lon_src, lat_tgt, lon_tgt) -> np.ndarray:
    """Regrid (n_t, lat_src, lon_src) → (n_t, lat_tgt, lon_tgt) via bilinear interp."""
    from scipy.interpolate import RegularGridInterpolator
    n_t = arr.shape[0]
    out = np.zeros((n_t, len(lat_tgt), len(lon_tgt)), dtype=np.float32)
    # periodic extension for lon
    lon_ext = np.append(lon_src, lon_src[0] + 360.0)
    for t in range(n_t):
        slice_ext = np.concatenate([arr[t], arr[t, :, :1]], axis=-1)
        interp = RegularGridInterpolator(
            (lat_src, lon_ext), slice_ext,
            method="linear", bounds_error=False, fill_value=None
        )
        pts = np.array(np.meshgrid(lat_tgt, lon_tgt, indexing="ij")).reshape(2, -1).T
        out[t] = interp(pts).reshape(len(lat_tgt), len(lon_tgt))
    return out


def _shift_lon(arr: np.ndarray, lat_sub, lon_src, lon_tgt) -> np.ndarray:
    lon_ext = np.append(lon_src, 360.0)
    arr_ext = np.concatenate([arr, arr[:, :, :1]], axis=-1)
    da = xr.DataArray(arr_ext, dims=["time", "lat", "lon"],
                      coords={"time": np.arange(arr.shape[0]),
                              "lat": lat_sub, "lon": lon_ext})
    return da.interp(lon=lon_tgt, method="linear").values.astype(np.float32)
