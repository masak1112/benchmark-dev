"""
loaders/amip_s2s.py — Loader for amip_s2s_train_scratch ensemble dataset.

Source structure:
  <src_dir>/YYYYMMDD/ensemble_YYYYMMDD_seedN.nc
    Each NC: dims (member=4, time=46, plev=26, lat=180, lon=360)
    plev: Pa — 20000 Pa = 200 hPa, 85000 Pa = 850 hPa
    lat: -89.5..89.5 (1°), lon: 0.5..359.5 (already 0.5-offset)
    14 seed files per date × 4 members = 56 members total

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

LAT_BOUND = 20.0


def list_dates(src_dir: str) -> list:
    entries = os.listdir(src_dir)
    return sorted(d for d in entries if re.match(r"^\d{8}$", d)
                  and os.path.isdir(os.path.join(src_dir, d)))


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
    print(f"[amip_s2s] {date_str}  variable={variable}  res={resolution}°")

    # numeric seed order (seed0, seed1, ..., seed13), so max_members takes the lowest seeds;
    # plain sorted() would give seed0, seed1, seed10, seed11, ...
    seed_files = sorted(glob.glob(os.path.join(src_dir, date_str, f"ensemble_{date_str}_seed*.nc")),
                        key=lambda f: int(re.search(r"_seed(\d+)\.nc$", f).group(1)))
    if not seed_files:
        raise FileNotFoundError(f"No seed files for {date_str} in {src_dir}/{date_str}/")

    # read grid from first file
    ds0  = xr.open_dataset(seed_files[0])
    lat  = ds0["lat"].values     # -89.5..89.5
    lon  = ds0["lon"].values     # 0.5..359.5
    times = ds0["time"].values
    plev = ds0["plev"].values    # Pa
    ds0.close()

    lev200_idx = int(np.argmin(np.abs(plev - 20000.0)))
    lev850_idx = int(np.argmin(np.abs(plev - 85000.0)))
    lat_mask = np.abs(lat) <= LAT_BOUND

    u200_list, v200_list, u850_list = [], [], []
    n_collected = 0
    for sf in seed_files:
        if max_members is not None and n_collected >= max_members:
            break
        ds = xr.open_dataset(sf)
        n_mem = ds.dims["member"]
        if max_members is not None:
            n_take = min(n_mem, max_members - n_collected)
        else:
            n_take = n_mem
        for mi in range(n_take):
            u200_list.append(ds["u_component_of_wind"].isel(member=mi, plev=lev200_idx).values.astype(np.float32))
            v200_list.append(ds["v_component_of_wind"].isel(member=mi, plev=lev200_idx).values.astype(np.float32))
            u850_list.append(ds["u_component_of_wind"].isel(member=mi, plev=lev850_idx).values.astype(np.float32))
            n_collected += 1
        ds.close()

    print(f"  Members: {n_collected}")

    u200 = np.stack(u200_list)  # (n_mem, time, lat, lon)
    v200 = np.stack(v200_list)
    u850 = np.stack(u850_list)
    n_mem = u200.shape[0]

    if variable in ("chi200", "chi250"):
        print(f"  Computing {variable} via windspharm ...")
        primary = _batch_chi(u200, v200, lat, lon)
    else:
        raise ValueError(f"Variable '{variable}' not supported. Use 'chi200'.")

    # subset lat (lon already at 0.5-offset, no shift needed)
    primary = primary[:, :, lat_mask, :]
    u850    = u850[:,   :, lat_mask, :]
    u200    = u200[:,   :, lat_mask, :]
    lat_sub = lat[lat_mask]

    dims   = ["ensemble", "time", "lat", "lon"]
    coords = {"ensemble": np.arange(n_mem), "time": times,
              "lat": lat_sub, "lon": lon}
    ds_out = xr.Dataset({
        "rlut":   xr.DataArray(primary.astype(np.float32), dims=dims, coords=coords),
        "ua_850": xr.DataArray(u850.astype(np.float32),    dims=dims, coords=coords),
        "ua_200": xr.DataArray(u200.astype(np.float32),    dims=dims, coords=coords),
    })
    ds_out["rlut"].attrs   = {"long_name": "Velocity potential 200 hPa", "units": "m2 s-1"}
    ds_out["ua_850"].attrs = {"long_name": "U850", "units": "m s-1"}
    ds_out["ua_200"].attrs = {"long_name": "U200", "units": "m s-1"}
    ds_out.attrs["init_date"] = date_str
    ds_out.attrs["source"]    = f"amip_s2s  variable={variable}"
    ds_out.to_netcdf(out_file)
    print(f"  Saved: {out_file}")
    return out_file


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
