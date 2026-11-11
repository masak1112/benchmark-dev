"""
ERA5 climatological baseline loader.

Treats ERA5 reanalysis years (clim_years, e.g. 1979-2021) as ensemble members.
For each init date in `years` and each lead step, the "forecast" value is the
ERA5 value at the same calendar date in every climatology year.

Directory layout (same as pangu_s2s/newdata):
    {input_dir}/{variable}/{YYYY}_180x360.nc

The dataset has a single variable named var_cfg['era5_var'] with dims:
    (time=init_dates, number=clim_years, prediction_timedelta, lat, lon)

Config keys consumed from var_cfg:
    model_var    : variable name inside the netCDF files (e.g. "2m_temperature")
    era5_var     : output variable name used downstream (same as analysis var)
    level        : pressure level to select (hPa, matches `level` coord), optional
    clim_years   : list of years to use as ensemble members (default 1979-2021)
    lead_days    : number of lead days to build (default 46)
"""

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd


def load_forecast(
    input_dir: str,
    years: list,
    var_name: str,
    var_cfg: dict,
    max_members: int | None,
) -> xr.Dataset:
    model_var   = var_cfg["model_var"]
    era5_var    = var_cfg["era5_var"]
    level       = var_cfg.get("level")
    clim_years  = var_cfg.get("clim_years", list(range(1979, 2022)))
    lead_days   = var_cfg.get("lead_days", 46)

    if max_members is not None:
        clim_years = clim_years[:max_members]

    # ── Load all climatology-year files ──────────────────────────────────────
    var_dir = Path(input_dir) / model_var
    clim_files = []
    for yr in clim_years:
        p = var_dir / f"{yr}_180x360.nc"
        if p.exists():
            clim_files.append((yr, str(p)))
        else:
            print(f"  WARNING: missing clim file {p}, skipping year {yr}.")

    if not clim_files:
        raise FileNotFoundError(f"No climatology files found in {var_dir}.")

    print(f"  ERA5-clim: loading {len(clim_files)} member years from {var_dir}")

    # Open all clim years and concatenate along a new 'number' dim
    member_das = []
    for yr, fp in clim_files:
        ds = xr.open_dataset(fp, chunks={"time": 10})
        da = ds[model_var]
        if level is not None:
            level_dim = var_cfg.get("level_dim", "level")
            da = da.sel({level_dim: float(level)}).drop_vars(level_dim, errors="ignore")
        # Sub-daily data: keep only 00 UTC so each day-of-year appears once
        da = da.sel(time=da["time"].dt.hour == 0)
        # Index by day-of-year (1-based) so we can align across years
        doy = da["time"].dt.dayofyear
        da = da.assign_coords(time=doy).rename({"time": "doy"})
        member_das.append(da.expand_dims(number=[yr]))

    # Shape: (number=n_clim_years, doy, lat, lon)
    clim_da = xr.concat(member_das, dim="number")

    # ── Build init-date list from eval years ─────────────────────────────────
    # Collect all init dates used by other models: any date whose year is in `years`.
    # We discover them from the directory listing so this loader is self-contained
    # even when called standalone. Fall back to every 4th day if no dirs exist.
    init_dates = _get_init_dates(input_dir, years, var_cfg)
    print(f"  ERA5-clim: {len(init_dates)} init dates in eval years {years}")

    # ── Build forecast array (time, number, prediction_timedelta, lat, lon) ──
    lead_nses  = [np.timedelta64(d, "D") for d in range(lead_days)]
    all_init   = []

    for init_ts in init_dates:
        init_dt = pd.Timestamp(init_ts)
        slices  = []
        for ld in range(lead_days):
            valid_dt = init_dt + pd.Timedelta(days=ld)
            doy_val  = valid_dt.dayofyear
            # Wrap doy 366 → 365 for non-leap clim years
            doy_sel = min(doy_val, 365) if doy_val == 366 else doy_val
            try:
                sl = clim_da.sel(doy=doy_sel)  # (number, lat, lon)
            except KeyError:
                sl = clim_da.isel(doy=-1)      # fallback to last doy
            slices.append(sl)

        # Stack lead slices → (lead_days, number, lat, lon)
        lead_da = xr.concat(slices, dim="prediction_timedelta")
        lead_da = lead_da.assign_coords(
            prediction_timedelta=np.array(lead_nses, dtype="timedelta64[ns]")
        )
        # Reorder to (number, prediction_timedelta, lat, lon)
        lead_da = lead_da.transpose("number", "prediction_timedelta", "lat", "lon")
        all_init.append(lead_da.expand_dims(time=[init_ts]))

    # Concatenate over init dates → (time, number, prediction_timedelta, lat, lon)
    fc_da = xr.concat(all_init, dim="time")
    scale = var_cfg.get("model_scale", 1.0)
    if scale != 1.0:
        fc_da = fc_da * scale
    fc_da.name = era5_var

    return fc_da.to_dataset(name=era5_var)


def _get_init_dates(input_dir: str, years: list, var_cfg: dict):
    """
    Return sorted list of init-date timestamps.

    Tries to mirror the init dates used by the amip_s2s_ensembles model by
    looking for date-named subdirectories in the sibling amip_s2s_ensembles dir.
    Falls back to every-4th-day if that directory does not exist.
    """
    # Try to get init dates from a reference model directory
    ref_dirs_to_try = var_cfg.get("init_date_ref_dirs", [])
    for ref_dir in ref_dirs_to_try:
        p = Path(ref_dir)
        if p.exists():
            dates = sorted(
                pd.Timestamp(d.name)
                for d in p.iterdir()
                if d.is_dir() and d.name.isdigit() and int(d.name[:4]) in years
            )
            if dates:
                return [d.to_datetime64() for d in dates]

    # Fallback: every 4th day in each eval year
    init_dates = []
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        end   = pd.Timestamp(f"{yr}-12-31")
        dates = pd.date_range(start, end, freq="4D")
        init_dates.extend(dates.to_list())
    return [pd.Timestamp(d).to_datetime64() for d in init_dates]
