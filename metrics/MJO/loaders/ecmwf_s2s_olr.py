"""
loaders/ecmwf_s2s_olr.py — ECMWF S2S hindcast loader using OLR (ttr) for RMM.

Source files (cfgrib-readable GRIB2):
  <src_dir>/ecmwf_olr_cf_{YYYYMMDD}.grib   — control forecast (1 member)
  <src_dir>/ecmwf_olr_pf_{YYYYMMDD}.grib   — perturbed forecasts (10 members)
  <src_dir>/ecmwf_uv200_850_cf_{YYYYMMDD}.grib  — U/V winds (for ua_200, ua_850)
  <src_dir>/ecmwf_uv200_850_pf_{YYYYMMDD}.grib

TTR (top_net_thermal_radiation) is accumulated J/m². Convert to instantaneous OLR
(W/m²) by differencing consecutive 24-h steps and dividing by 86400 s.
OLR sign: MJOcast expects outgoing (positive = radiating upward) → OLR = -ttr_rate.

Grid: native 1.5° (121 lat × 240 lon). Regrid to pipeline resolution (1.0°) with 0.5° lon offset.
"""

import os
import re
import glob
import subprocess
import tempfile
import numpy as np
import xarray as xr

LAT_BOUND = 20.0
WGRIB2    = "wgrib2"


def list_dates(src_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(src_dir, "ecmwf_olr_cf_????????.grib")))
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
    print(f"[ecmwf_s2s_olr] {date_str}  variable={variable}  res={resolution}°")

    olr_cf  = os.path.join(src_dir, f"ecmwf_olr_cf_{date_str}.grib")
    olr_pf  = os.path.join(src_dir, f"ecmwf_olr_pf_{date_str}.grib")
    uv_cf   = os.path.join(src_dir, f"ecmwf_uv200_850_cf_{date_str}.grib")
    uv_pf   = os.path.join(src_dir, f"ecmwf_uv200_850_pf_{date_str}.grib")

    for fp in [olr_cf, uv_cf]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"File not found: {fp}")

    os.makedirs(scratch_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
        # ── OLR: read with cfgrib via .venv python ────────────────────────────
        olr_cf_arr, times_olr, lat_olr, lon_olr = _load_olr_grib(olr_cf, member=None)
        olr_all = [olr_cf_arr]   # (time, lat, lon)

        if os.path.exists(olr_pf):
            n_pf = _count_pf_olr(olr_pf)
            if max_members is not None:
                n_pf = min(n_pf, max_members - 1)
            for i in range(1, n_pf + 1):
                olr_all.append(_load_olr_grib(olr_pf, member=i)[0])
            print(f"  OLR members: 1 cf + {n_pf} pf = {1+n_pf} total")
        else:
            print("  OLR: control only")

        olr_stack = np.stack(olr_all, axis=0)   # (n_ens, time, lat, lon)

        # ── U/V: extract with wgrib2 ──────────────────────────────────────────
        u200_cf, v200_cf, u850_cf, times_uv, lat_uv, lon_uv = _extract_uvuv(
            uv_cf, tmpdir, "cf", date_str)
        u200_all = [u200_cf]; v200_all = [v200_cf]; u850_all = [u850_cf]

        if os.path.exists(uv_pf):
            n_pf_uv = _count_pf_members(uv_pf)
            if max_members is not None:
                n_pf_uv = min(n_pf_uv, max_members - 1)
            for i in range(1, n_pf_uv + 1):
                u, v, u8, _, _, _ = _extract_uvuv(uv_pf, tmpdir, f"pf{i}", date_str, member=i)
                u200_all.append(u); v200_all.append(v); u850_all.append(u8)

        n_ens_uv = len(u200_all)
        u200_stack = np.stack(u200_all, axis=0)
        v200_stack = np.stack(v200_all, axis=0)
        u850_stack = np.stack(u850_all, axis=0)

        # Use minimum ensemble size across OLR and UV
        n_ens = min(olr_stack.shape[0], n_ens_uv)
        olr_stack  = olr_stack[:n_ens]
        u200_stack = u200_stack[:n_ens]
        v200_stack = v200_stack[:n_ens]
        u850_stack = u850_stack[:n_ens]

        # ── Align times (OLR and UV should match) ────────────────────────────
        n_steps = min(olr_stack.shape[1], u200_stack.shape[1])
        olr_stack  = olr_stack[:,  :n_steps]
        u200_stack = u200_stack[:, :n_steps]
        v200_stack = v200_stack[:, :n_steps]
        u850_stack = u850_stack[:, :n_steps]
        times      = times_olr[:n_steps]

        # ── Regrid OLR from 1.5° to UV grid (1.5°) then to target resolution ─
        # OLR already on same 1.5° grid as UV, just need tropical subset + regrid
        lat = lat_olr
        lon = lon_olr
        lat_mask = np.abs(lat) <= LAT_BOUND
        lat_sub  = lat[lat_mask]

        olr_sub  = olr_stack[:,  :, lat_mask, :]
        u850_sub = u850_stack[:, :, lat_mask, :]
        u200_sub = u200_stack[:, :, lat_mask, :]

        ds_tmp = xr.Dataset(
            {
                "rlut":   (["ensemble", "time", "lat", "lon"], olr_sub.astype(np.float32)),
                "ua_200": (["ensemble", "time", "lat", "lon"], u200_sub.astype(np.float32)),
                "ua_850": (["ensemble", "time", "lat", "lon"], u850_sub.astype(np.float32)),
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
            print(f"  Regridding {native_res:.1f}° → {resolution}° (0.5° offset grid)")
            new_lat = np.arange(-LAT_BOUND, LAT_BOUND + resolution / 2, resolution)
            new_lon = np.arange(resolution / 2, 360.0, resolution)
            ds_tmp = ds_tmp.interp(lat=new_lat, lon=new_lon, method="linear")

        ds_tmp["rlut"].attrs   = {"long_name": "OLR (from ttr, W/m^2, positive up)", "units": "W m-2"}
        ds_tmp["ua_850"].attrs = {"long_name": "U850", "units": "m s-1"}
        ds_tmp["ua_200"].attrs = {"long_name": "U200", "units": "m s-1"}
        ds_tmp.attrs["init_date"] = date_str
        ds_tmp.attrs["source"]    = f"ecmwf_s2s_olr  variable={variable}"
        ds_tmp.to_netcdf(out_file)
        print(f"  Saved: {out_file}")

    return out_file


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_olr_grib(grib_path: str, member=None):
    """
    Load TTR from GRIB, differentiate to get hourly-mean OLR (W/m²), negate.
    Returns (olr, times, lat, lon) where olr shape is (n_steps, nlat, nlon).
    times are the valid_time values at each step (day-1 onward after differencing).
    """
    import subprocess, sys
    # Use .venv python which has cfgrib
    script = f"""
import xarray as xr, numpy as np, sys
kw = {{'indexpath': ''}}
if {member is None}:
    ds = xr.open_dataset('{grib_path}', engine='cfgrib', backend_kwargs=kw)
else:
    ds = xr.open_dataset('{grib_path}', engine='cfgrib', backend_kwargs=kw,
         filter_by_keys={{'dataType': 'pf', 'number': {member}}})
ttr = ds['ttr'].values   # (step, lat, lon) or (1, step, lat, lon)
if ttr.ndim == 4: ttr = ttr[0]
lat = ds.latitude.values
lon = ds.longitude.values
times = ds.valid_time.values
# TTR is accumulated J/m² — diff consecutive steps → J/m² per day → W/m²
# step 0 is 0h (accumulation=0), step k is 24k h
diff = np.diff(ttr, axis=0)       # shape (n_steps-1, lat, lon)
olr = -diff / 86400.0             # W/m², positive upward
times_out = times[1:]             # valid times at end of each 24h window
np.savez('/tmp/olr_out_{member}.npz', olr=olr.astype(np.float32),
         lat=lat, lon=lon, times=times_out.astype('datetime64[ns]').astype(np.int64))
"""
    venv_py = "/glade/work/bgong/.venv/bin/python"
    result = subprocess.run([venv_py, "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"OLR grib load failed: {result.stderr[:500]}")

    npz = np.load(f"/tmp/olr_out_{member}.npz")
    olr   = npz["olr"]
    lat   = npz["lat"]
    lon   = npz["lon"]
    times = npz["times"].astype("datetime64[ns]")
    return olr, times, lat, lon


def _count_pf_olr(pf_path: str) -> int:
    import subprocess
    script = f"""
import xarray as xr
ds = xr.open_dataset('{pf_path}', engine='cfgrib',
     backend_kwargs={{'indexpath': ''}},
     filter_by_keys={{'dataType': 'pf'}})
print(int(ds.dims.get('number', 1)))
"""
    venv_py = "/glade/work/bgong/.venv/bin/python"
    result = subprocess.run([venv_py, "-c", script], capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except Exception:
        return 10  # fallback


def _count_pf_members(pf_path: str) -> int:
    result = subprocess.run([WGRIB2, pf_path], capture_output=True, text=True)
    members = set(re.findall(r"ENS=\+(\d+)", result.stdout))
    return len(members)


def _extract_uvuv(grib_path: str, tmpdir: str, tag: str,
                  date_str: str, member: int = None):
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

    def wgrib2(gp, match, out):
        r = subprocess.run([WGRIB2, gp, "-match", match, "-netcdf", out],
                           capture_output=True, text=True)
        if r.returncode != 0 and not os.path.exists(out):
            raise RuntimeError(f"wgrib2 failed: {r.stderr[:300]}")

    wgrib2(grib_path, match_u200, nc_u200)
    wgrib2(grib_path, match_v200, nc_v200)
    wgrib2(grib_path, match_u850, nc_u850)

    ds_u = xr.open_dataset(nc_u200); ds_v = xr.open_dataset(nc_v200)
    ds_8 = xr.open_dataset(nc_u850)

    u200 = ds_u[[k for k in ds_u.data_vars if "UGRD" in k][0]].values
    v200 = ds_v[[k for k in ds_v.data_vars if "VGRD" in k][0]].values
    u850 = ds_8[[k for k in ds_8.data_vars if "UGRD" in k][0]].values
    times = ds_u["time"].values
    lat   = ds_u["latitude"].values
    lon   = ds_u["longitude"].values
    ds_u.close(); ds_v.close(); ds_8.close()
    return u200, v200, u850, times, lat, lon
