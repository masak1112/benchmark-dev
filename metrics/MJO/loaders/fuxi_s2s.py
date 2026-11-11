"""
loaders/fuxi_s2s.py — FuXi S2S dataset loader for the modular MJO pipeline.

Archive structure:
  <src_dir>/YYYYMMDD.7z
    → YYYYMMDD/member/MM/LL.nc   (MM = 00..50, LL = 01..42)
  Each NC:  dims (time=1, lead_time=1, channel=26, lat=121, lon=240)
  Variable: __xarray_dataarray_variable__

FuXi lat: 90 → -90  (121 pts, 1.5°)
FuXi lon:  0 → 358.5 (240 pts, 1.5°)

Public interface (called by src/step2_prepare_forecasts.py)
-----------------------------------------------------------
  list_dates(src_dir) -> [YYYYMMDD, ...]
  process_date(date_str, src_dir, out_dir, variable, resolution,
               max_members, scratch_dir) -> out_file_path
"""

import os
import glob
import subprocess
import shutil
import numpy as np
import xarray as xr


NATIVE_LAT_BOUND = 20.0   # degrees — tropical subsetting band


# ── Public API ────────────────────────────────────────────────────

def list_dates(src_dir: str) -> list[str]:
    """Return sorted list of YYYYMMDD strings with available 7z archives."""
    archives = sorted(glob.glob(os.path.join(src_dir, "????????.7z")))
    return [os.path.basename(a).replace(".7z", "") for a in archives]


def process_date(date_str: str,
                 src_dir: str,
                 out_dir: str,
                 variable: str,
                 resolution: float,
                 max_members: int | None = None,
                 scratch_dir: str = "/tmp") -> str:
    """
    Build a MJOcast-compatible forecast NC for one FuXi init date.

    Output variables:
      rlut    = primary variable (OLR or chi) occupying the OLR slot in MJOcast
      ua_850  = U850
      ua_200  = U250 (proxy for U200)
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{variable}_forecast_{date_str}.nc")
    if os.path.exists(out_file):
        print(f"Already exists, skipping: {out_file}")
        return out_file

    print(f"\n{'='*60}")
    print(f"[fuxi_s2s] {date_str}  variable={variable}  res={resolution}°")

    # 1. Extract archive
    extract_dir = _extract_archive(date_str, src_dir, scratch_dir)
    member_base = _find_member_dir(extract_dir)
    member_ids  = sorted(os.listdir(member_base))
    if max_members is not None:
        member_ids = member_ids[:max_members]
    print(f"  Members: {len(member_ids)}")

    init_dt = np.datetime64(
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    )

    # 2. Read grid metadata from first file
    first = sorted(glob.glob(os.path.join(member_base, member_ids[0], "*.nc")))[0]
    ds0   = xr.open_dataset(first)
    lat_all = ds0.lat.values   # (121,) 90 → -90
    lon_all = ds0.lon.values   # (240,) 0 → 358.5
    ds0.close()
    lat_mask = np.abs(lat_all) <= NATIVE_LAT_BOUND
    lat_sub  = lat_all[lat_mask]

    # 3. Load all members
    u250_members, v250_members, u850_members, olr_members = [], [], [], []
    times = None

    for i, mid in enumerate(member_ids):
        print(f"  member {mid}  ({i+1}/{len(member_ids)})", flush=True)
        u250, v250, u850, olr, t = _load_member(
            os.path.join(member_base, mid), init_dt
        )
        u250_members.append(u250)
        v250_members.append(v250)
        u850_members.append(u850)
        olr_members.append(olr)
        if times is None:
            times = t

    u250_all = np.stack(u250_members)   # (n_mem, n_leads, lat, lon)
    v250_all = np.stack(v250_members)
    u850_all = np.stack(u850_members)
    olr_all  = np.stack(olr_members)
    n_mem, n_leads = u250_all.shape[:2]

    # 4. Compute primary variable
    if variable in ("chi200", "chi250"):
        print(f"  Computing {variable} for {n_mem}×{n_leads} steps ...")
        primary_all = _batch_chi(u250_all, v250_all, lat_all, lon_all)
    else:  # olr
        primary_all = olr_all

    # 5. Subset lat
    prim_sub = primary_all[:, :, lat_mask, :]
    u850_sub = u850_all[:,   :, lat_mask, :]
    u200_sub = u250_all[:,   :, lat_mask, :]   # u250 as U200 proxy

    # 6. Regrid lon to target resolution if needed
    lon_target = _target_lon(resolution)
    if len(lon_all) != len(lon_target) or not np.allclose(lon_all, lon_target, atol=0.01):
        print(f"  Regridding lon: {len(lon_all)} → {len(lon_target)} pts ...")
        prim_sub = _interp_lon(prim_sub, lat_sub, lon_all, times, n_mem, lon_target)
        u850_sub = _interp_lon(u850_sub, lat_sub, lon_all, times, n_mem, lon_target)
        u200_sub = _interp_lon(u200_sub, lat_sub, lon_all, times, n_mem, lon_target)
    else:
        lon_target = lon_all

    # 7. Write NetCDF
    dims   = ["ensemble", "time", "lat", "lon"]
    coords = {"ensemble": np.arange(n_mem), "time": times,
              "lat": lat_sub, "lon": lon_target}
    ds_out = xr.Dataset({
        "rlut":   xr.DataArray(prim_sub.astype(np.float32), dims=dims, coords=coords),
        "ua_850": xr.DataArray(u850_sub.astype(np.float32), dims=dims, coords=coords),
        "ua_200": xr.DataArray(u200_sub.astype(np.float32), dims=dims, coords=coords),
    })
    ds_out["rlut"].attrs   = {"long_name": _long_name(variable), "units": _units(variable)}
    ds_out["ua_850"].attrs = {"long_name": "U850", "units": "m s-1"}
    ds_out["ua_200"].attrs = {"long_name": "U250 (proxy U200)", "units": "m s-1"}
    ds_out.attrs["init_date"] = date_str
    ds_out.attrs["source"]    = f"FuXi-S2S  variable={variable}  resolution={resolution}deg"
    ds_out.to_netcdf(out_file)
    print(f"  Saved: {out_file}")

    # 8. Clean up extraction directory
    try:
        shutil.rmtree(extract_dir)
        print("  Cleaned up extraction directory.")
    except Exception as e:
        print(f"  Warning: cleanup failed ({e})")

    return out_file


# ── Private helpers ───────────────────────────────────────────────

def _load_member(member_dir: str, init_dt: np.datetime64):
    """Load all lead-time NC files for one member.

    Returns
    -------
    u250, v250, u850, olr : each (n_leads, lat, lon) float32
    times                 : (n_leads,) datetime64[ns]
    """
    day_files = sorted(glob.glob(os.path.join(member_dir, "*.nc")))
    if not day_files:
        raise FileNotFoundError(f"No NC files in {member_dir}")

    u250_days, v250_days, u850_days, olr_days, t_list = [], [], [], [], []
    for df in day_files:
        ds   = xr.open_dataset(df)
        data = ds["__xarray_dataarray_variable__"].squeeze(["time", "lead_time"])
        lead = int(ds.lead_time.values[0])
        t_list.append(init_dt + np.timedelta64(lead, "D"))

        u250_days.append(data.sel(channel="u250").values)
        v250_days.append(data.sel(channel="v250").values)
        u850_days.append(data.sel(channel="u850").values)
        # negate ttr: FuXi stores negative of OLR (outgoing)
        olr_days.append(-data.sel(channel="ttr").values)
        ds.close()

    times = np.array(t_list, dtype="datetime64[ns]")
    return (np.stack(u250_days).astype(np.float32),
            np.stack(v250_days).astype(np.float32),
            np.stack(u850_days).astype(np.float32),
            np.stack(olr_days).astype(np.float32),
            times)


def _batch_chi(u_all: np.ndarray, v_all: np.ndarray,
               lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Batch velocity-potential computation across all member×lead steps."""
    from windspharm.xarray import VectorWind
    n_mem, n_leads, n_lat, n_lon = u_all.shape
    n_flat = n_mem * n_leads
    flat_u = u_all.reshape(n_flat, n_lat, n_lon)
    flat_v = v_all.reshape(n_flat, n_lat, n_lon)
    u_da = xr.DataArray(flat_u, dims=["step","lat","lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    v_da = xr.DataArray(flat_v, dims=["step","lat","lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    chi_flat = VectorWind(u_da, v_da).velocitypotential().values
    return chi_flat.reshape(n_mem, n_leads, n_lat, n_lon).astype(np.float32)


def _target_lon(resolution: float) -> np.ndarray:
    if abs(resolution - 1.5) < 0.01:
        return np.arange(0.0, 360.0, 1.5)          # 240 pts
    return np.arange(0.5, 360.0, 1.0).astype(np.float32)  # 360 pts (0.5-offset)


def _interp_lon(arr: np.ndarray, lat_sub, lon_src, times, n_mem, lon_tgt) -> np.ndarray:
    da = xr.DataArray(arr, dims=["ensemble","time","lat","lon"],
                      coords={"ensemble": np.arange(n_mem),
                              "time":     times,
                              "lat":      lat_sub,
                              "lon":      lon_src})
    return da.interp(lon=lon_tgt, method="linear").values.astype(np.float32)


def _extract_archive(date_str: str, src_dir: str, scratch_dir: str) -> str:
    archive     = os.path.join(src_dir, f"{date_str}.7z")
    extract_dir = os.path.join(scratch_dir, date_str)
    if not os.path.exists(archive):
        raise FileNotFoundError(f"Archive not found: {archive}")
    if glob.glob(os.path.join(extract_dir, "**", "member"), recursive=True):
        print(f"  Already extracted: {extract_dir}")
        return extract_dir
    os.makedirs(extract_dir, exist_ok=True)
    print(f"  Extracting {archive} …")
    result = subprocess.run(["7zr","x","-y", archive, f"-o{extract_dir}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"7zr failed:\n{result.stderr}")
    print("  Extraction complete.")
    return extract_dir


def _find_member_dir(extract_dir: str) -> str:
    hits = glob.glob(os.path.join(extract_dir, "**", "member"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"Could not find 'member/' under {extract_dir}")
    return hits[0]


def _long_name(variable: str) -> str:
    return {"olr":   "OLR (negated ttr from FuXi)",
            "chi200": "Velocity potential 200 hPa (OLR proxy)",
            "chi250": "Velocity potential 250 hPa (OLR proxy)"}.get(variable, variable)


def _units(variable: str) -> str:
    return "W m-2" if variable == "olr" else "m2 s-1"
