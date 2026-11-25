"""
loaders/ecmwf_s2s.py — ECMWF S2S loader for the modular MJO pipeline (VP200-based RMM).

Uses wgrib2 (module wgrib2/3.6.0) to extract U/V from GRIB to temporary NetCDF,
computes chi200 (velocity potential at 200 hPa) via windspharm from U200+V200,
then assembles into the standard pipeline format. No cfgrib dependency.

Source structure:
  <src_dir>/ecmwf_uv200_850_cf_{YYYYMMDD}.grib   control forecast (member 0)
  <src_dir>/ecmwf_uv200_850_pf_{YYYYMMDD}.grib   perturbed forecast (members 1-50)

Output (MJOcast-compatible chi200_forecast_{YYYYMMDD}.nc):
  dims: (ensemble, time, lat, lon)
  variables:
    rlut   — chi200 (velocity potential at 200 hPa, m2/s)
    ua_200 — U200 wind (m/s)
    ua_850 — U850 wind (m/s)
  lat: tropical band ±20°, lon: 0.5..359.5 (0.5° offset to match MJOcast obs)
"""

import os
import re
import glob
import subprocess
import tempfile
import numpy as np
import xarray as xr

LAT_BOUND = 20.0
WGRIB2 = "wgrib2"   # must be on PATH (module load wgrib2)


def list_dates(src_dir: str) -> list:
    """Return sorted YYYYMMDD strings that have a cf uv200_850 grib file."""
    files = sorted(glob.glob(os.path.join(src_dir, "ecmwf_uv200_850_cf_????????.grib")))
    return [re.search(r"\d{8}", os.path.basename(f)).group() for f in files]


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
    print(f"[ecmwf_s2s] {date_str}  variable={variable}  res={resolution}°")

    cf_file = os.path.join(src_dir, f"ecmwf_uv200_850_cf_{date_str}.grib")
    pf_file = os.path.join(src_dir, f"ecmwf_uv200_850_pf_{date_str}.grib")

    if not os.path.exists(cf_file):
        raise FileNotFoundError(f"CF grib not found: {cf_file}")

    os.makedirs(scratch_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
        # ── Control forecast (member 0) ───────────────────────────────────────
        u200_cf, v200_cf, u850_cf, times, lat, lon = _extract_uvuv_from_grib(
            cf_file, tmpdir, "cf", date_str)

        # ── Perturbed forecast (members 1-50) ─────────────────────────────────
        u200_pf_list, v200_pf_list, u850_pf_list = [], [], []
        if os.path.exists(pf_file):
            n_members = _count_pf_members(pf_file)
            if max_members is not None:
                n_members = min(n_members, max_members - 1)  # -1 for cf
            for i in range(1, n_members + 1):
                u200_i, v200_i, u850_i, _, _, _ = _extract_uvuv_from_grib(
                    pf_file, tmpdir, f"pf{i}", date_str, member=i)
                u200_pf_list.append(u200_i)
                v200_pf_list.append(v200_i)
                u850_pf_list.append(u850_i)
            print(f"  Members: 1 (cf) + {n_members} (pf) = {1 + n_members} total")
        else:
            print(f"  PF file not found, using control only.")

        # ── Stack into (ensemble, time, lat, lon) ─────────────────────────────
        u200_all = np.stack([u200_cf] + u200_pf_list, axis=0)
        v200_all = np.stack([v200_cf] + v200_pf_list, axis=0)
        u850_all = np.stack([u850_cf] + u850_pf_list, axis=0)
        n_ens = u200_all.shape[0]

        # ── Compute chi200 via windspharm (global, full lat/lon) ──────────────
        if variable in ("chi200", "chi250"):
            print(f"  Computing chi200 via windspharm (global grid) ...")
            chi200_all = _batch_chi(u200_all, v200_all, lat, lon)
        else:
            raise ValueError(f"Variable '{variable}' not supported by ecmwf_s2s loader.")

        # ── Tropical subset then regrid to target resolution ──────────────────
        lat_mask = np.abs(lat) <= LAT_BOUND
        lat_sub  = lat[lat_mask]
        chi200_all = chi200_all[:, :, lat_mask, :]
        u850_all   = u850_all[:,   :, lat_mask, :]
        u200_all   = u200_all[:,   :, lat_mask, :]

        ds_tmp = xr.Dataset(
            {
                "rlut":   (["ensemble", "time", "lat", "lon"], chi200_all.astype(np.float32)),
                "ua_200": (["ensemble", "time", "lat", "lon"], u200_all.astype(np.float32)),
                "ua_850": (["ensemble", "time", "lat", "lon"], u850_all.astype(np.float32)),
            },
            coords={
                "ensemble": np.arange(n_ens),
                "time":     times,
                "lat":      lat_sub,
                "lon":      lon,
            },
        )

        native_res = abs(float(lat[1] - lat[0]))
        if abs(native_res - resolution) > 0.01:
            print(f"  Regridding {native_res}° → {resolution}° (0.5° offset grid)")
            new_lat = np.arange(-LAT_BOUND, LAT_BOUND + resolution / 2, resolution)
            new_lon = np.arange(resolution / 2, 360.0, resolution)
            ds_tmp = ds_tmp.interp(lat=new_lat, lon=new_lon, method="linear")

        ds_tmp.attrs["init_date"] = date_str
        ds_tmp.attrs["source"]    = f"ecmwf_s2s  variable={variable}"
        ds_tmp.to_netcdf(out_file)
        print(f"  Saved: {out_file}")

    return out_file


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_wgrib2(grib_path: str, match: str, out_nc: str):
    """Extract matching fields from grib_path to out_nc via wgrib2."""
    cmd = [WGRIB2, grib_path, "-match", match, "-netcdf", out_nc]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not os.path.exists(out_nc):
        raise RuntimeError(f"wgrib2 failed for {grib_path}: {result.stderr[:500]}")


def _count_pf_members(pf_path: str) -> int:
    """Count distinct ENS= member numbers in pf grib file."""
    result = subprocess.run([WGRIB2, pf_path], capture_output=True, text=True)
    members = set(re.findall(r"ENS=\+(\d+)", result.stdout))
    return len(members)


def _extract_uvuv_from_grib(grib_path: str, tmpdir: str, tag: str,
                             date_str: str, member: int = None):
    """
    Extract U200, V200, U850 from a grib file.
    Returns (u200, v200, u850, times, lat, lon) each (time, lat, lon).
    """
    if member is not None:
        match_u200 = f"UGRD:200 mb.*ENS=\\+{member}"
        match_v200 = f"VGRD:200 mb.*ENS=\\+{member}"
        match_u850 = f"UGRD:850 mb.*ENS=\\+{member}"
    else:
        match_u200 = "UGRD:200 mb"
        match_v200 = "VGRD:200 mb"
        match_u850 = "UGRD:850 mb"

    nc_u200 = os.path.join(tmpdir, f"u200_{tag}.nc")
    nc_v200 = os.path.join(tmpdir, f"v200_{tag}.nc")
    nc_u850 = os.path.join(tmpdir, f"u850_{tag}.nc")

    _run_wgrib2(grib_path, match_u200, nc_u200)
    _run_wgrib2(grib_path, match_v200, nc_v200)
    _run_wgrib2(grib_path, match_u850, nc_u850)

    ds_u200 = xr.open_dataset(nc_u200)
    ds_v200 = xr.open_dataset(nc_v200)
    ds_u850 = xr.open_dataset(nc_u850)

    u200_key = [k for k in ds_u200.data_vars if "UGRD" in k][0]
    v200_key = [k for k in ds_v200.data_vars if "VGRD" in k][0]
    u850_key = [k for k in ds_u850.data_vars if "UGRD" in k][0]

    u200 = ds_u200[u200_key].values   # (time, lat, lon)
    v200 = ds_v200[v200_key].values
    u850 = ds_u850[u850_key].values

    times = ds_u200["time"].values
    lat   = ds_u200["latitude"].values
    lon   = ds_u200["longitude"].values

    ds_u200.close(); ds_v200.close(); ds_u850.close()

    return u200, v200, u850, times, lat, lon


def _batch_chi(u, v, lat, lon):
    """Compute velocity potential at 200 hPa for all ensemble members and times."""
    from windspharm.xarray import VectorWind
    n_ens, n_time, n_lat, n_lon = u.shape
    n_flat = n_ens * n_time
    flat_u = u.reshape(n_flat, n_lat, n_lon)
    flat_v = v.reshape(n_flat, n_lat, n_lon)
    u_da = xr.DataArray(flat_u, dims=["step", "lat", "lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    v_da = xr.DataArray(flat_v, dims=["step", "lat", "lon"],
                        coords={"step": np.arange(n_flat), "lat": lat, "lon": lon})
    chi_flat = VectorWind(u_da, v_da).velocitypotential().values
    return chi_flat.reshape(n_ens, n_time, n_lat, n_lon).astype(np.float32)
