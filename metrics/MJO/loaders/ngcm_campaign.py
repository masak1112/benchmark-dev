"""
loaders/ngcm_campaign.py — Loader for new NGCM data in campaign storage.

Source structure:
  <src_dir>/YYYY/YYYY-MM-DDTCC.zarr
    dims: (1, number=25, time=46, lon=128, lat=64)
    Already daily (prediction_timedelta_daily: 0..45)
    U/V at 200 and 850 hPa available directly
    Gaussian grid: lat 64 pts (-87.86..87.86), lon 128 pts (0..357.19)

Processing:
  - Compute chi200 via windspharm batched over all members×time at once
  - Regrid to 1.0° via scipy RegularGridInterpolator (vectorised, single interp object)
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
    zarr_files = glob.glob(os.path.join(src_dir, "????", "????-??-??T00.zarr"))
    dates = sorted({
        re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f)).group(1).replace("-", "")
        for f in zarr_files
        if re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
    })
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
    zarr_path = os.path.join(src_dir, yr, f"{date_fmt}T00.zarr")

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"No zarr found: {zarr_path}")

    print(f"\n{'='*60}")
    print(f"[ngcm_campaign] {date_str}  variable={variable}  res={resolution}°")

    z = zarr.open(zarr_path, mode="r")
    lat_gauss = z["latitude"][:]   # (64,)
    lon_gauss = z["longitude"][:]  # (128,)
    n_members = z["number"].shape[0]
    if max_members is not None:
        n_members = min(n_members, max_members)
    print(f"  Members: {n_members}")

    # shape: (1, members, time=46, lon=128, lat=64) → (members, time, lat, lon)
    u200 = z["u_component_of_wind_200_avg"][0, :n_members, :, :, :].transpose(0, 1, 3, 2).astype(np.float32)
    v200 = z["v_component_of_wind_200_avg"][0, :n_members, :, :, :].transpose(0, 1, 3, 2).astype(np.float32)
    u850 = z["u_component_of_wind_850_avg"][0, :n_members, :, :, :].transpose(0, 1, 3, 2).astype(np.float32)

    # target 1.0° grid
    lat_1deg = np.arange(-89.5, 90.0, 1.0)
    lon_1deg = np.arange(0.0, 360.0, 1.0)
    lat_mask = np.abs(lat_1deg) <= LAT_BOUND
    lat_sub  = lat_1deg[lat_mask]
    lon_tgt  = np.arange(0.5, 360.0, 1.0)

    # 1. Batch chi200 — stack all members×time, call windspharm once
    if variable in ("chi200", "chi250"):
        print(f"  Computing chi200 (batch {n_members}×{u200.shape[1]} steps) ...", flush=True)
        chi_gauss = _batch_chi(u200, v200, lat_gauss, lon_gauss)  # (mem, t, lat, lon)
    else:
        raise ValueError(f"Variable '{variable}' not supported.")

    # 2. Regrid all three fields at once (vectorised over members×time)
    print("  Regriding to 1.0° ...", flush=True)
    n_mem, n_t = u200.shape[:2]

    chi_1deg  = _regrid_batch(chi_gauss, lat_gauss, lon_gauss, lat_1deg, lon_1deg)
    u850_1deg = _regrid_batch(u850,      lat_gauss, lon_gauss, lat_1deg, lon_1deg)
    u200_1deg = _regrid_batch(u200,      lat_gauss, lon_gauss, lat_1deg, lon_1deg)

    # 3. Subset lat and shift lon
    chi_out  = _shift_lon_batch(chi_1deg[:,  :, lat_mask, :], lat_sub, lon_1deg, lon_tgt)
    u850_out = _shift_lon_batch(u850_1deg[:, :, lat_mask, :], lat_sub, lon_1deg, lon_tgt)
    u200_out = _shift_lon_batch(u200_1deg[:, :, lat_mask, :], lat_sub, lon_1deg, lon_tgt)

    init_dt = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    times   = np.array([init_dt + pd.Timedelta(days=d+1) for d in range(n_t)],
                       dtype="datetime64[ns]")

    dims   = ["ensemble", "time", "lat", "lon"]
    coords = {"ensemble": np.arange(n_mem), "time": times, "lat": lat_sub, "lon": lon_tgt}
    ds_out = xr.Dataset({
        "rlut":   xr.DataArray(chi_out.astype(np.float32),  dims=dims, coords=coords),
        "ua_850": xr.DataArray(u850_out.astype(np.float32), dims=dims, coords=coords),
        "ua_200": xr.DataArray(u200_out.astype(np.float32), dims=dims, coords=coords),
    })
    ds_out["rlut"].attrs   = {"long_name": "Velocity potential 200 hPa", "units": "m2 s-1"}
    ds_out["ua_850"].attrs = {"long_name": "U850", "units": "m s-1"}
    ds_out["ua_200"].attrs = {"long_name": "U200", "units": "m s-1"}
    ds_out.attrs["init_date"] = date_str
    ds_out.attrs["source"]    = f"NGCM_campaign  variable={variable}  Gaussian→1.0deg  lon+0.5"
    ds_out.to_netcdf(out_file)
    print(f"  Saved: {out_file}")
    return out_file


def _batch_chi(u, v, lat, lon):
    """Batch velocity-potential over all member×time steps at once."""
    from windspharm.xarray import VectorWind
    n_mem, n_t, n_lat, n_lon = u.shape
    n_flat = n_mem * n_t
    flat_u = u.reshape(n_flat, n_lat, n_lon)
    flat_v = v.reshape(n_flat, n_lat, n_lon)
    u_da = xr.DataArray(flat_u, dims=["step", "lat", "lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    v_da = xr.DataArray(flat_v, dims=["step", "lat", "lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    chi_flat = VectorWind(u_da, v_da).velocitypotential().values
    return chi_flat.reshape(n_mem, n_t, n_lat, n_lon).astype(np.float32)


def _regrid_batch(arr, lat_src, lon_src, lat_tgt, lon_tgt):
    """Regrid (mem, time, lat_src, lon_src) → (mem, time, lat_tgt, lon_tgt).

    Builds the interpolator once and evaluates all mem×time slices together.
    """
    from scipy.interpolate import RegularGridInterpolator
    n_mem, n_t, n_lat_src, n_lon_src = arr.shape
    n_flat = n_mem * n_t

    # extend lon by one point for periodicity
    lon_ext = np.append(lon_src, lon_src[0] + 360.0)
    arr_ext = np.concatenate([arr, arr[:, :, :, :1]], axis=-1)  # (..., lon+1)

    # build query points once
    pts = np.array(np.meshgrid(lat_tgt, lon_tgt, indexing="ij")).reshape(2, -1).T

    # flatten mem×time, regrid, reshape back
    flat = arr_ext.reshape(n_flat, n_lat_src, len(lon_ext))
    out  = np.zeros((n_flat, len(lat_tgt), len(lon_tgt)), dtype=np.float32)

    # reuse pts across slices; rebuild interp per slice (cheap: same grid axes)
    grid_axes = (lat_src, lon_ext)
    for i in range(n_flat):
        interp = RegularGridInterpolator(
            grid_axes, flat[i],
            method="linear", bounds_error=False, fill_value=None
        )
        out[i] = interp(pts).reshape(len(lat_tgt), len(lon_tgt))

    return out.reshape(n_mem, n_t, len(lat_tgt), len(lon_tgt))


def _shift_lon_batch(arr, lat_sub, lon_src, lon_tgt):
    """Shift lon offset for (mem, time, lat, lon) array."""
    lon_ext = np.append(lon_src, 360.0)
    arr_ext = np.concatenate([arr, arr[:, :, :, :1]], axis=-1)
    n_mem, n_t, n_lat, _ = arr_ext.shape
    da = xr.DataArray(arr_ext.reshape(n_mem * n_t, n_lat, len(lon_ext)),
                      dims=["step", "lat", "lon"],
                      coords={"step": np.arange(n_mem * n_t),
                              "lat": lat_sub, "lon": lon_ext})
    out = da.interp(lon=lon_tgt, method="linear").values.astype(np.float32)
    return out.reshape(n_mem, n_t, n_lat, len(lon_tgt))
