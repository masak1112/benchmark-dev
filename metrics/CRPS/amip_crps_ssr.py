#!/usr/bin/env python3
"""
Compute CRPS and SSR for AMIP S2S ensemble forecasts (2019–2022).

Variables evaluated:
  - 2m_temperature          (AMIP: 2m_temperature       → ERA5: 2m_temperature)
  - geopotential at 500 hPa (AMIP: geopotential/plev=500 → ERA5: geopotential/level=500)
  - sea_surface_temperature (AMIP: skin_temperature      → ERA5: sea_surface_temperature)
  - total_precipitation_24hr (AMIP: PRATEsfc_24h [kg/m²/s] → ERA5: total_precipitation_24hr [m/day])
    Unit conversion: AMIP × 86400 → mm/day; ERA5 × 1000 → mm/day

Input:  /glade/derecho/scratch/bgong/amip_s2s_ensembles/{YYYYMMDD}/ensemble_{YYYYMMDD}_seed{N}.nc
ERA5:   /glade/derecho/scratch/bgong/era5/{variable}/{year}_180x360.nc
Output: /glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/amip_results_2019_2022/
        amip_2019_2022_crps_ssr_{variable_name}.nc
"""

import os
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
warnings.filterwarnings("ignore", category=FutureWarning)

# ── CRPS core (from fast_crps.ipynb) ────────────────────────────────────────

def _detect_member_dim(da, prefer=None):
    if prefer and prefer in da.dims:
        return prefer
    for cand in ("number", "member", "ens", "realization", "ensemble"):
        if cand in da.dims:
            return cand
    raise ValueError("Could not detect ensemble member dim; pass member_dim.")


def _sort_along_dim(da: xr.DataArray, dim: str) -> xr.DataArray:
    return xr.apply_ufunc(
        np.sort, da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        vectorize=False,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def _mean_pair_abs_diff_fast(preds: xr.DataArray, member_dim: str) -> xr.DataArray:
    n = preds.sizes[member_dim]
    if n < 2:
        return xr.zeros_like(preds.isel({member_dim: 0}, drop=True))
    preds_sorted = _sort_along_dim(preds, member_dim)
    w = xr.DataArray(2 * np.arange(1, n + 1) - n - 1, dims=(member_dim,))
    S = (preds_sorted * w).sum(dim=member_dim, skipna=False)
    return (2.0 / (n * (n - 1))) * S


def _crps_ensemble_vec(targets, preds, member_dim):
    mae  = np.abs(preds - targets).mean(dim=member_dim, skipna=False)
    disp = _mean_pair_abs_diff_fast(preds, member_dim)
    return mae - 0.5 * disp


def evaluate_crps_fast(
    forecast: xr.Dataset,
    analysis: xr.Dataset,
    var: str,
    *,
    time_dim: str = "time",
    lead_dim: str = "prediction_timedelta",
    member_dim: str | None = None,
    compute: bool = False,
) -> xr.Dataset:
    """
    Returns Dataset with crps, ssr, spread, error keyed on lead_dim.

    ssr = sqrt(mean(spread²) / mean(error²))  (variance-form, mean over time/lat/lon)
    """
    if var not in forecast or var not in analysis:
        raise KeyError(f"'{var}' must exist in both forecast and analysis.")

    f    = forecast[var]
    mdim = _detect_member_dim(f, member_dim)

    valid_time = forecast[time_dim] + forecast[lead_dim]
    f = f.assign_coords(valid_time=valid_time)
    t = analysis[var].sel({time_dim: f["valid_time"]})

    # ERA5 lon/lat may differ from forecast (e.g. ERA5 lon 0–359 vs AMIP 0.5–359.5).
    # Both grids are the same physical resolution; override so arithmetic aligns.
    for coord in ("lat", "lon"):
        if coord in f.coords and coord in t.coords:
            t = t.assign_coords({coord: f.coords[coord]})

    crps       = _crps_ensemble_vec(t, f, mdim)
    crps.name  = "crps"

    nmem   = f.sizes[mdim]
    spread = f.std(dim=mdim, ddof=1, skipna=True) if nmem > 1 else xr.zeros_like(f.isel({mdim: 0}, drop=True))

    ens_mean = f.mean(dim=mdim, skipna=True)
    err2     = (ens_mean - t) ** 2

    reduce_dims  = [d for d in f.dims if d not in (lead_dim, mdim)]
    spread2_mean = (spread ** 2).mean(dim=reduce_dims, skipna=True) if reduce_dims else (spread ** 2)
    err2_mean    = err2.mean(dim=reduce_dims, skipna=True) if reduce_dims else err2

    ssr      = np.sqrt(spread2_mean / err2_mean)
    ssr.name = "ssr"

    out = xr.Dataset({
        "crps":   crps,
        "ssr":    ssr,
        "spread": np.sqrt(spread2_mean),
        "error":  np.sqrt(err2_mean),
    })
    return out.compute() if compute else out


# ── Configuration ────────────────────────────────────────────────────────────

IN_DIR   = "/glade/derecho/scratch/bgong/amip_s2s_ensembles"
ERA5_DIR = "/glade/derecho/scratch/bgong/era5"
OUT_DIR  = ("/glade/work/bgong/gencast_neuralGCM_crps_ssr"
            "/gencast_neuralGCM_crps_ssr/amip_results_2019_2022")
YEARS     = list(range(2019, 2023))  # restrict to 2019; change to list(range(2019, 2023)) for full run
MAX_MEMBERS = 25      # cap ensemble size; set to None to use all members

# Each entry: output filename stem → loading config
VARS = {
    "2m_temperature": {
        "amip_var":      "2m_temperature",
        "era5_var":      "2m_temperature",
        "level":         None,
        "amip_level_dim": "plev",
        "era5_level_dim": "level",
        "amip_scale":    1.0,
        "era5_scale":    1.0,
    },
    "geopotential_500": {
        "amip_var":      "geopotential",
        "era5_var":      "geopotential",
        "level":         500.0,
        "amip_level_dim": "plev",
        "era5_level_dim": "level",
        "amip_scale":    1.0,
        "era5_scale":    1.0,
    },
    "sea_surface_temperature": {
        "amip_var":      "skin_temperature",   # closest proxy in AMIP output
        "era5_var":      "sea_surface_temperature",
        "level":         None,
        "amip_level_dim": "plev",
        "era5_level_dim": "level",
        "amip_scale":    1.0,
        "era5_scale":    1.0,
    },
    "total_precipitation_24hr": {
        "amip_var":      "PRATEsfc_24h",       # kg/m²/s (daily mean rate)
        "era5_var":      "total_precipitation_24hr",
        "level":         None,
        "amip_level_dim": "plev",
        "era5_level_dim": "level",
        "amip_scale":    86400.0,   # kg/m²/s × 86400 s/day → mm/day
        "era5_scale":    1000.0,    # m/day  × 1000         → mm/day
    },
}


# ── Data loading ─────────────────────────────────────────────────────────────

def preprocess_seed_file(file_path: str, amip_var: str,
                          level=None, amip_level_dim: str = "plev") -> xr.DataArray:
    """
    Load one AMIP seed file and reshape to
    (time=[init_date], prediction_timedelta, number, lat, lon).
    """
    ds   = xr.open_dataset(file_path, chunks={"member": -1, "time": 1})
    seed = int(Path(file_path).stem.split("seed")[-1])

    init_time = ds["time"].values[0]  # datetime64[ns] from file, matches ERA5 time dtype

    da = ds[amip_var]
    if level is not None:
        da = da.sel({amip_level_dim: float(level)}).drop_vars(amip_level_dim, errors="ignore")

    # Convert valid-time coordinate to lead timedelta
    valid_times = da["time"].values
    lead_ns     = (valid_times - init_time).astype("timedelta64[ns]")

    # Offset member IDs so seeds don't collide when concatenated
    n_mem   = da.sizes["member"]
    new_ids = seed * n_mem + np.arange(n_mem)

    da = (
        da
        .assign_coords(member=new_ids, time=lead_ns)
        .rename({"time": "prediction_timedelta", "member": "number"})
        .expand_dims(time=[init_time])
    )
    return da


def load_amip_forecast(var_name: str, cfg: dict) -> xr.Dataset:
    amip_var       = cfg["amip_var"]
    era5_var       = cfg["era5_var"]
    level          = cfg.get("level")
    amip_level_dim = cfg.get("amip_level_dim", "plev")
    amip_scale     = cfg.get("amip_scale", 1.0)

    date_dirs = sorted(
        d for d in Path(IN_DIR).iterdir()
        if d.is_dir() and d.name.isdigit() and int(d.name[:4]) in YEARS
    )
    print(f"  Found {len(date_dirs)} init-date directories.")

    all_date_das = []
    for date_dir in date_dirs:
        date_str   = date_dir.name
        seed_files = sorted(date_dir.glob(f"ensemble_{date_str}_seed*.nc"))
        if not seed_files:
            print(f"  WARNING: no seed files in {date_dir}, skipping.")
            continue

        seed_das = []
        for sf in seed_files:
            try:
                da = preprocess_seed_file(str(sf), amip_var, level, amip_level_dim)
                seed_das.append(da)
            except Exception as exc:
                print(f"  ERROR loading {sf.name}: {exc}")

        if seed_das:
            date_da = xr.concat(seed_das, dim="number")
            if MAX_MEMBERS is not None:
                date_da = date_da.isel(number=slice(MAX_MEMBERS))
            all_date_das.append(date_da)

    if not all_date_das:
        raise RuntimeError(f"No AMIP data loaded for '{var_name}'.")

    fc_da = xr.concat(all_date_das, dim="time")
    if amip_scale != 1.0:
        fc_da = fc_da * amip_scale

    return fc_da.to_dataset(name=era5_var)


def load_era5(cfg: dict) -> xr.Dataset:
    era5_var       = cfg["era5_var"]
    level          = cfg.get("level")
    era5_level_dim = cfg.get("era5_level_dim", "level")
    era5_scale     = cfg.get("era5_scale", 1.0)

    files = [
        f"{ERA5_DIR}/{era5_var}/{y}_180x360.nc"
        for y in YEARS
        if os.path.exists(f"{ERA5_DIR}/{era5_var}/{y}_180x360.nc")
    ]
    if not files:
        raise FileNotFoundError(f"No ERA5 files found for variable '{era5_var}'.")
    print(f"  ERA5: {len(files)} annual files for {era5_var}.")

    ds = xr.open_mfdataset(files, chunks={"time": 4}, engine="netcdf4")
    if level is not None:
        ds = ds.sel({era5_level_dim: float(level)}).drop_vars(era5_level_dim, errors="ignore")
    if era5_scale != 1.0:
        ds[era5_var] = ds[era5_var] * era5_scale
    return ds


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate_variable(var_name: str, cfg: dict) -> None:
    year_tag = f"{min(YEARS)}_{max(YEARS)}" if len(YEARS) > 1 else str(YEARS[0])
    out_path = os.path.join(OUT_DIR, f"amip_{year_tag}_crps_ssr_{var_name}.nc")
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists.")
        return

    era5_var = cfg["era5_var"]
    print(f"\n{'='*65}")
    print(f"  Variable : {var_name}")
    print(f"  AMIP var : {cfg['amip_var']}  ×{cfg.get('amip_scale', 1.0)}")
    print(f"  ERA5 var : {era5_var}  ×{cfg.get('era5_scale', 1.0)}")
    if cfg.get("level"):
        print(f"  Level    : {cfg['level']} hPa")
    print(f"{'='*65}")

    print("Loading AMIP forecast ...")
    forecast = load_amip_forecast(var_name, cfg)
    # Chunk so each init-date is one dask task; member/lead/spatial all in one chunk
    forecast = forecast.chunk({
        "time": 1, "number": -1, "prediction_timedelta": -1, "lat": -1, "lon": -1
    })

    print("Loading ERA5 analysis ...")
    analysis = load_era5(cfg)

    print("Computing CRPS & SSR ...")
    res = evaluate_crps_fast(
        forecast, analysis,
        var=era5_var,
        time_dim="time",
        lead_dim="prediction_timedelta",
        member_dim="number",
        compute=True,
    )

    # CRPS: mean over (time, lat, lon) → shape (prediction_timedelta,)
    reduce_dims  = [d for d in res.crps.dims if d != "prediction_timedelta"]
    by_lead_crps = res.crps.mean(dim=reduce_dims)
    by_lead_ssr  = res.ssr   # already (prediction_timedelta,) from evaluate_crps_fast

    print(f"\n  {'Lead (h)':>8}  {'CRPS':>12}  {'SSR':>8}  {'Spread':>10}  {'Error':>10}")
    for lead_td in by_lead_crps.prediction_timedelta.values:
        hrs      = int(np.round(lead_td / np.timedelta64(1, "h")))
        crps_val = float(by_lead_crps.sel(prediction_timedelta=lead_td).values)
        ssr_val  = float(by_lead_ssr.sel(prediction_timedelta=lead_td).values)
        sprd_val = float(res.spread.sel(prediction_timedelta=lead_td).values)
        err_val  = float(res.error.sel(prediction_timedelta=lead_td).values)
        print(f"  {hrs:>8}  {crps_val:>12.4f}  {ssr_val:>8.3f}  {sprd_val:>10.4f}  {err_val:>10.4f}")

    # Store prediction_timedelta as integer hours to match reference file format
    lead_hours = (
        by_lead_crps.prediction_timedelta.values / np.timedelta64(1, "h")
    ).astype(np.int64)

    out = xr.Dataset(
        {
            "crps": xr.DataArray(by_lead_crps.values.astype(np.float32),
                                  dims=["prediction_timedelta"]),
            "ssr":  xr.DataArray(by_lead_ssr.values.astype(np.float32),
                                  dims=["prediction_timedelta"]),
        },
        coords={"prediction_timedelta": lead_hours},
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_netcdf(out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    import sys

    # Optionally run a single variable: python amip_crps_ssr.py geopotential_500
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        run_vars  = {k: v for k, v in VARS.items() if k in requested}
        if not run_vars:
            print(f"Unknown variable(s): {requested}")
            print(f"Available: {list(VARS.keys())}")
            sys.exit(1)
    else:
        run_vars = VARS

    for var_name, cfg in run_vars.items():
        evaluate_variable(var_name, cfg)

    print("\nAll done.")
