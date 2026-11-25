"""
Format Fuxi loader: fuxi_s2s style.

Archive layout:
    {input_dir}/{YYYYMMDD}.7z
        └── {YYYY}/{YYYYMMDD}/member/{MM}/{LL}.nc

Each .nc file contains one lead-time step for one ensemble member:
    dims: (time=1, lead_time=1, channel=26, lat=121, lon=240)
    variable: '__xarray_dataarray_variable__'
    lead_time: integer 1..42 (6-hourly steps: lead_time*6 hours)
    channel: string coordinate selecting the variable (e.g. 't2m', 'z500')

Members: 00 to 47 (48 members total).
Lead times: 01 to 42 (6-hourly; step 1 = 6 h, step 42 = 252 h).

Returned dataset dimensions:
    (time=init_dates, number=ensemble_members, prediction_timedelta, lat, lon)
Variable is renamed to var_cfg['era5_var'] for downstream compatibility.

Extraction: each 7z archive is extracted to a temporary directory, read,
then cleaned up. Archives are processed one at a time to limit disk usage.

Performance note: each archive is solid-compressed (~3 GB), so it must be
fully decompressed regardless of which variable is requested. If multiple
variables are evaluated sequentially, each archive is re-extracted once per
variable call. For a single-variable run this is fine; for multi-variable
runs consider using a persistent cache_dir (env var FUXI_CACHE_DIR).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr


def _extract_archive_full(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .7z archive preserving internal directory structure."""
    result = subprocess.run(
        ["7zr", "x", str(archive_path), "-o" + str(dest_dir), "-y", "-aoa"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"7zr extraction failed for {archive_path.name}: {result.stderr.strip()}"
        )


def _read_date_from_dir(base: Path, channel: str, max_members: int | None, init_time) -> xr.DataArray | None:
    """
    Read all member/lead nc files from an extracted archive base directory.

    base: path to {YYYY}/{YYYYMMDD}/member/
    Returns DataArray with dims (time=[init], number, prediction_timedelta, lat, lon).
    """
    member_dirs = sorted(base.iterdir())
    if max_members is not None:
        member_dirs = member_dirs[:max_members]

    member_das = []
    for m_dir in member_dirs:
        if not m_dir.is_dir():
            continue
        mem_idx = int(m_dir.name)
        lead_das = []
        for nc_file in sorted(m_dir.glob("*.nc")):
            lead_idx = int(nc_file.stem)
            lead_ns = np.timedelta64(lead_idx * 6, "h").astype("timedelta64[ns]")
            try:
                ds = xr.open_dataset(nc_file, engine="netcdf4")
                da = ds["__xarray_dataarray_variable__"].sel(channel=channel)
                da = da.isel(time=0, lead_time=0, drop=True)
                da = da.expand_dims(prediction_timedelta=[lead_ns])
                lead_das.append(da.load())
                ds.close()
            except Exception as exc:
                print(f"  ERROR reading {nc_file.name}: {exc}")

        if lead_das:
            mem_da = xr.concat(lead_das, dim="prediction_timedelta")
            mem_da = mem_da.expand_dims(number=[mem_idx])
            member_das.append(mem_da)

    if not member_das:
        return None

    date_da = xr.concat(member_das, dim="number")
    # Reassign positional member indices to avoid outer-join NaN fill when
    # different init dates have different member subsets (e.g. missing member dir).
    date_da = date_da.assign_coords(number=np.arange(date_da.sizes["number"]))
    date_da = date_da.expand_dims(time=[init_time])
    return date_da


def load_forecast(
    input_dir: str,
    years: list,
    var_name: str,
    var_cfg: dict,
    max_members: int | None,
) -> xr.Dataset:
    """
    Load Fuxi S2S forecast data from .7z archives.

    Returns xr.Dataset with variable named var_cfg['era5_var'] and dims
    (time, number, prediction_timedelta, lat, lon).

    Set env var FUXI_CACHE_DIR to a writable path to reuse extracted archives
    across multiple variable calls instead of re-extracting each time.
    """
    channel  = var_cfg["model_var"]
    era5_var = var_cfg["era5_var"]
    scale    = var_cfg.get("model_scale", 1.0)

    cache_dir = os.environ.get("FUXI_CACHE_DIR")

    archive_dir = Path(input_dir)
    archives = sorted(
        p for p in archive_dir.glob("*.7z")
        if p.stem.isdigit() and int(p.stem[:4]) in years
    )
    print(f"  Found {len(archives)} archives for years {years}.")

    all_date_das = []
    for archive in archives:
        date_str = archive.stem          # YYYYMMDD
        year     = date_str[:4]

        print(f"  Processing {archive.name} ...", end=" ", flush=True)
        try:
            # Determine extraction target
            if cache_dir:
                extract_root = Path(cache_dir) / date_str
                owned = not extract_root.exists()
                if owned:
                    extract_root.mkdir(parents=True, exist_ok=True)
                    _extract_archive_full(archive, extract_root)
            else:
                tmp_obj = tempfile.TemporaryDirectory(prefix="fuxi_")
                extract_root = Path(tmp_obj.name)
                owned = True
                _extract_archive_full(archive, extract_root)

            base = extract_root / year / date_str / "member"
            if not base.exists():
                print(f"SKIP (member dir not found: {base})")
                continue

            init_time = np.datetime64(
                f"{year}-{date_str[4:6]}-{date_str[6:8]}T00:00:00", "ns"
            )

            da = _read_date_from_dir(base, channel, max_members, init_time)

            # Clean up if we own the temp dir and no cache is configured
            if not cache_dir:
                tmp_obj.cleanup()

            if da is not None:
                all_date_das.append(da)
                print("OK")
            else:
                print("SKIPPED (no data)")

        except Exception as exc:
            print(f"ERROR: {exc}")
            if not cache_dir and 'tmp_obj' in dir():
                try:
                    tmp_obj.cleanup()
                except Exception:
                    pass

    if not all_date_das:
        raise RuntimeError(f"No data loaded for '{var_name}' from {input_dir}.")

    fc_da = xr.concat(all_date_das, dim="time")
    if scale != 1.0:
        fc_da = fc_da * scale

    return fc_da.to_dataset(name=era5_var)
