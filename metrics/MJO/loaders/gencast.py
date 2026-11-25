"""
loaders/gencast.py — Loader for GenCast ensemble forecasts.

Source structure:
  /glade/campaign/univ/uchi0018/bing/gencast/
    gencast_YYYY.zarr
      dims: (sample=52, time=26, step=90, [level=10,] lat=721, lon=1440)
      time : days offset from YYYY-05-02; time[0] is corrupt, skip it
      step : hours (12, 24, ..., 1080) — 12-hourly, 45 days
      level: [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
      lat  : -90 to 90 at 0.25°
      lon  : 0 to 359.75 at 0.25°

Processing:
  - 200 hPa available directly (no interpolation needed)
  - Resample 12-hourly steps to daily means (groups of 2)
  - Compute chi200 via windspharm on 0.25° global grid
  - Regrid 0.25° → 1.0° via block averaging, subset lat to ±20°
  - Shift lon from 0.0-offset to 0.5-offset (periodic +0.5°)

Public interface:
  list_dates(src_dir) -> [YYYYMMDD, ...]
  process_date(date_str, src_dir, out_dir, variable, resolution,
               max_members, scratch_dir) -> out_file_path
"""

import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

LAT_BOUND = 20.0


# ── Public API ────────────────────────────────────────────────────

def list_dates(src_dir: str) -> list:
    dates = []
    for yr in range(2019, 2025):
        zpath = os.path.join(src_dir, f"gencast_{yr}.zarr")
        if not os.path.exists(zpath):
            continue
        dates.extend(_dates_in_zarr(zpath, yr))
    return sorted(dates)


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

    yr = int(date_str[:4])
    zpath = os.path.join(src_dir, f"gencast_{yr}.zarr")
    if not os.path.exists(zpath):
        raise FileNotFoundError(f"GenCast zarr not found: {zpath}")

    z = zarr.open(zpath, mode="r")
    all_dates = _dates_in_zarr(zpath, yr)
    if date_str not in all_dates:
        raise ValueError(f"{date_str} not found in {zpath}. Available: {all_dates[:5]}...")

    time_idx = all_dates.index(date_str) + 1   # +1 because time[0] is corrupt

    print(f"\n{'='*60}")
    print(f"[gencast] {date_str}  variable={variable}  res={resolution}°")

    lat  = z["lat"][:]          # (721,) -90..90
    lon  = z["lon"][:]          # (1440,) 0..359.75
    step = z["step"][:]         # (90,) hours
    lev  = z["level"][:]        # (10,)
    lev200_idx = int(np.where(lev == 200)[0][0])

    n_mem_total = z["sample"].shape[0]
    if max_members is not None:
        n_mem_total = min(n_mem_total, max_members)

    lev850_idx = int(np.where(lev == 850)[0][0])
    n_days = len(step) // 2
    # post-coarsen grid (after _coarsen4_2d trims to multiples of 4 then takes [::4])
    n_lat4 = (len(lat) // 4) * 4
    n_lon4 = (len(lon) // 4) * 4
    lat_1deg = lat[:n_lat4:4]    # 180 points
    lon_1deg = lon[:n_lon4:4]    # 360 points
    lat_mask = np.abs(lat_1deg) <= LAT_BOUND
    lat_sub  = lat_1deg[lat_mask]
    lon_tgt  = np.arange(0.5, 360.0, 1.0)

    print(f"  Members: {n_mem_total}, {n_days} daily steps, processing one-by-one ...")

    # Process member-by-member to stay within memory limits
    chi_list, u850_list, u200_list = [], [], []
    for m in range(n_mem_total):
        print(f"  member {m:02d}  ({m+1}/{n_mem_total})", flush=True)
        u200_raw = z["u_component_of_wind"][m, time_idx, :, lev200_idx, :, :]  # (90,721,1440)
        v200_raw = z["v_component_of_wind"][m, time_idx, :, lev200_idx, :, :]
        u850_raw = z["u_component_of_wind"][m, time_idx, :, lev850_idx, :, :]

        # 12-hourly → daily mean (keep 0.25° for windspharm)
        u200_d = _resample1d(u200_raw, n_days)   # (n_days, 721, 1440)
        v200_d = _resample1d(v200_raw, n_days)
        u850_d = _resample1d(u850_raw, n_days)

        # chi200 via windspharm on full 0.25° global grid
        if variable in ("chi200", "chi250"):
            chi_m = _chi_single(u200_d, v200_d, lat, lon)   # (n_days, 721, 1440)
        else:
            raise ValueError(f"Variable '{variable}' not supported. Use 'chi200'.")

        # coarsen 0.25° → 1.0° after chi computation
        chi_m  = _coarsen4_2d(chi_m)    # (n_days, 180, 360)
        u850_c = _coarsen4_2d(u850_d)
        u200_c = _coarsen4_2d(u200_d)

        # subset lat + lon shift
        chi_list.append(_shift_lon_2d(chi_m[:,  lat_mask, :], lat_sub, lon_1deg, lon_tgt))
        u850_list.append(_shift_lon_2d(u850_c[:, lat_mask, :], lat_sub, lon_1deg, lon_tgt))
        u200_list.append(_shift_lon_2d(u200_c[:, lat_mask, :], lat_sub, lon_1deg, lon_tgt))

    primary  = np.stack(chi_list)    # (n_mem, n_days, lat_sub, 360)
    u850_sub = np.stack(u850_list)
    u200_sub = np.stack(u200_list)
    n_mem    = n_mem_total

    # Build time coordinate: init date + 1..n_days days
    init_dt = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    times = np.array([init_dt + pd.Timedelta(days=d+1)
                      for d in range(n_days)], dtype="datetime64[ns]")

    dims   = ["ensemble", "time", "lat", "lon"]
    coords = {"ensemble": np.arange(n_mem), "time": times,
              "lat": lat_sub, "lon": lon_tgt}
    ds_out = xr.Dataset({
        "rlut":   xr.DataArray(primary.astype(np.float32),  dims=dims, coords=coords),
        "ua_850": xr.DataArray(u850_sub.astype(np.float32), dims=dims, coords=coords),
        "ua_200": xr.DataArray(u200_sub.astype(np.float32), dims=dims, coords=coords),
    })
    ds_out["rlut"].attrs   = {"long_name": "Velocity potential 200 hPa", "units": "m2 s-1"}
    ds_out["ua_850"].attrs = {"long_name": "U850", "units": "m s-1"}
    ds_out["ua_200"].attrs = {"long_name": "U200", "units": "m s-1"}
    ds_out.attrs["init_date"] = date_str
    ds_out.attrs["source"]    = f"GenCast  variable={variable}  0.25→1.0deg  lon+0.5"
    ds_out.to_netcdf(out_file)
    print(f"  Saved: {out_file}")
    return out_file


# ── Private helpers ───────────────────────────────────────────────

def _dates_in_zarr(zpath: str, yr: int) -> list:
    import zarr
    z = zarr.open(zpath, mode="r")
    times_raw = z["time"][1:]   # skip corrupt time[0]
    ref = pd.Timestamp(f"{yr}-05-02")
    return [(ref + pd.Timedelta(days=int(d))).strftime("%Y%m%d")
            for d in times_raw]


def _resample1d(arr: np.ndarray, n_days: int) -> np.ndarray:
    """Single member: (90, lat, lon) → daily means → (n_days, lat, lon)."""
    n_steps = n_days * 2
    return arr[:n_steps].reshape(n_days, 2, arr.shape[1], arr.shape[2]).mean(axis=1).astype(np.float32)


def _coarsen4_2d(arr: np.ndarray) -> np.ndarray:
    """(n_days, lat, lon) at 0.25° → 1.0° by block averaging."""
    n_t, n_lat, n_lon = arr.shape
    lat4 = (n_lat // 4) * 4
    lon4 = (n_lon // 4) * 4
    return arr[:, :lat4, :lon4].reshape(
        n_t, lat4 // 4, 4, lon4 // 4, 4
    ).mean(axis=(2, 4)).astype(np.float32)


def _chi_single(u: np.ndarray, v: np.ndarray,
                lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Compute velocity potential for one member: (n_days, lat, lon)."""
    from windspharm.xarray import VectorWind
    n_t, n_lat, n_lon = u.shape
    u_da = xr.DataArray(u, dims=["time", "lat", "lon"],
                        coords={"time": np.arange(n_t), "lat": lat, "lon": lon})
    v_da = xr.DataArray(v, dims=["time", "lat", "lon"],
                        coords={"time": np.arange(n_t), "lat": lat, "lon": lon})
    return VectorWind(u_da, v_da).velocitypotential().values.astype(np.float32)


def _shift_lon_2d(arr: np.ndarray, lat_sub, lon_src, lon_tgt) -> np.ndarray:
    """Shift lon for single member (n_t, lat, lon) with periodic wrap-around."""
    lon_ext = np.append(lon_src, 360.0)
    arr_ext = np.concatenate([arr, arr[:, :, :1]], axis=-1)
    da = xr.DataArray(arr_ext, dims=["time", "lat", "lon"],
                      coords={"time": np.arange(arr.shape[0]),
                              "lat": lat_sub, "lon": lon_ext})
    return da.interp(lon=lon_tgt, method="linear").values.astype(np.float32)
