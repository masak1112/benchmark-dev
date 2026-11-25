"""
Format B loader: pangu_plasim / ours_si style.

Directory layout (flat):
    {input_dir}/{prefix}_{YYYYMMDDHH}_ens_{N}.nc

Each file is a single ensemble member with dims
(time=valid_times, latitude, longitude[, level]) where `time` contains
the 46 valid-time steps.  Init date is encoded in the filename.

Returned dataset dimensions:
    (time=init_dates, number=ensemble_members, prediction_timedelta, lat, lon)
Variable is renamed to var_cfg['era5_var'] for downstream compatibility.
lat/lon coordinate names are normalized to match ERA5 (lat, lon).
"""

from pathlib import Path
import re
import numpy as np
import xarray as xr


_DATE_RE = re.compile(r"_(\d{10})_ens_(\d+)\.nc$")


def _parse_filename(fname: str):
    """Return (init_datetime64, ens_index) from a Format B filename."""
    m = _DATE_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse date/ens from filename: {fname}")
    date_str, ens_str = m.group(1), m.group(2)
    # date_str is YYYYMMDDHH
    dt = np.datetime64(
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{date_str[8:10]}:00:00",
        "ns",
    )
    return dt, int(ens_str)


def _preprocess_member_file(
    file_path: str,
    model_var: str,
    level,
    level_dim: str,
) -> xr.DataArray:
    """Load one ensemble-member file and reshape to (time=[init], prediction_timedelta, number=[ens], lat, lon)."""
    init_time, ens_idx = _parse_filename(Path(file_path).name)

    ds = xr.open_dataset(file_path, chunks={"time": -1})

    # Normalise spatial coord names to match ERA5 (lat, lon)
    rename_map = {}
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    da = ds[model_var]
    if level is not None:
        da = da.sel({level_dim: float(level)}).drop_vars(level_dim, errors="ignore")

    # Convert valid-time axis → prediction_timedelta
    valid_times = da["time"].values
    lead_ns = (valid_times - init_time).astype("timedelta64[ns]")

    da = (
        da
        .assign_coords(time=lead_ns)
        .rename({"time": "prediction_timedelta"})
        .expand_dims(number=[ens_idx])
        .expand_dims(time=[init_time])
    )
    return da


def load_forecast(
    input_dir: str,
    years: list,
    var_name: str,
    var_cfg: dict,
    max_members: int | None,
) -> xr.Dataset:
    """
    Load Format B forecast data.

    Returns xr.Dataset with variable named var_cfg['era5_var'] and dims
    (time, number, prediction_timedelta, lat, lon).
    """
    model_var = var_cfg["model_var"]
    era5_var  = var_cfg["era5_var"]
    level     = var_cfg.get("level")
    level_dim = var_cfg.get("level_dim", "level")
    scale     = var_cfg.get("model_scale", 1.0)

    all_files = sorted(Path(input_dir).glob("*.nc"))
    # Filter to requested years using the date embedded in the filename
    year_files = []
    for f in all_files:
        m = _DATE_RE.search(f.name)
        if m and int(m.group(1)[:4]) in years:
            year_files.append(f)

    if not year_files:
        raise RuntimeError(f"No files found for years {years} in {input_dir}.")

    # Group by init date
    from collections import defaultdict
    by_date: dict[np.datetime64, list] = defaultdict(list)
    for f in year_files:
        init_dt, _ = _parse_filename(f.name)
        by_date[init_dt].append(f)

    print(f"  Found {len(by_date)} init-date groups ({len(year_files)} files total).")

    all_date_das = []
    for init_dt in sorted(by_date):
        member_files = sorted(by_date[init_dt], key=lambda p: _parse_filename(p.name)[1])
        member_das = []
        for mf in member_files:
            try:
                da = _preprocess_member_file(str(mf), model_var, level, level_dim)
                member_das.append(da)
            except Exception as exc:
                print(f"  ERROR loading {mf.name}: {exc}")

        if member_das:
            date_da = xr.concat(member_das, dim="number")
            if max_members is not None:
                date_da = date_da.isel(number=slice(max_members))
            all_date_das.append(date_da)

    if not all_date_das:
        raise RuntimeError(f"No data loaded for '{var_name}' from {input_dir}.")

    fc_da = xr.concat(all_date_das, dim="time")
    if scale != 1.0:
        fc_da = fc_da * scale

    return fc_da.to_dataset(name=era5_var)
