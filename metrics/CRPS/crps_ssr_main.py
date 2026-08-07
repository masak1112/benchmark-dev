#!/usr/bin/env python3
"""
Format-agnostic CRPS/SSR evaluation driver.

Usage:
    python crps_ssr_main.py [--config config.yaml] [--model MODEL] [--var VAR]

    --config   Path to YAML config file (default: config.yaml next to this script)
    --model    Run a single model (must match a key under `models:` in the config)
    --var      Run a single variable (must match a key under the model's `variables:`)
               Can be combined with --model.

Examples:
    python crps_ssr_main.py                                        # all models, all vars
    python crps_ssr_main.py --model ours_si                        # all vars for ours_si
    python crps_ssr_main.py --model amip_s2s_ensembles --var geopotential_500
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import binary_dilation

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Import loader registry ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from loaders import get_loader


# ── CRPS core ────────────────────────────────────────────────────────────────

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

    # Override ERA5 lat/lon with forecast coords to align mismatched grids
    # (e.g. ERA5 lon 0–359 vs model lon 0.5–359.5 at same resolution).
    for coord in ("lat", "lon"):
        if coord in f.coords and coord in t.coords:
            t = t.assign_coords({coord: f.coords[coord]})

    # Build a static open-ocean mask: land (always-NaN in ERA5) dilated by 1
    # pixel to also exclude coastline cells where model fill values contaminate
    # the ensemble spread. Applied to both forecast and analysis so that spread
    # and error share identical spatial support.
    land_2d = np.all(np.isnan(t.values), axis=tuple(
        i for i, d in enumerate(t.dims) if d not in ("lat", "lon")
    ))
    exclude_2d = binary_dilation(land_2d, iterations=1)
    open_ocean = xr.DataArray(~exclude_2d, dims=["lat", "lon"], coords={
        "lat": t["lat"], "lon": t["lon"],
    })
    f = f.where(open_ocean)
    t = t.where(open_ocean)

    crps      = _crps_ensemble_vec(t, f, mdim)
    crps.name = "crps"

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


# ── ERA5 loader ───────────────────────────────────────────────────────────────

def load_era5(era5_cfg: dict, years: list, var_cfg: dict) -> xr.Dataset:
    era5_var       = var_cfg["era5_var"]
    level          = var_cfg.get("level")
    era5_level_dim = era5_cfg.get("level_dim", "level")
    era5_scale     = var_cfg.get("era5_scale", 1.0)
    pattern        = era5_cfg["file_pattern"]
    era5_dir       = era5_cfg["dir"]

    files = [
        os.path.join(era5_dir, pattern.format(variable=era5_var, year=y))
        for y in years
        if os.path.exists(os.path.join(era5_dir, pattern.format(variable=era5_var, year=y)))
    ]
    if not files:
        raise FileNotFoundError(
            f"No ERA5 files found for variable '{era5_var}' in {era5_dir}."
        )
    print(f"  ERA5: {len(files)} annual files for {era5_var}.")

    ds = xr.open_mfdataset(files, chunks={"time": 4}, engine="netcdf4")
    if level is not None:
        ds = ds.sel({era5_level_dim: float(level)}).drop_vars(era5_level_dim, errors="ignore")
    if era5_scale != 1.0:
        ds[era5_var] = ds[era5_var] * era5_scale
    return ds


# ── Per-variable evaluation ───────────────────────────────────────────────────

def evaluate_variable(
    model_name: str,
    var_name: str,
    var_cfg: dict,
    model_cfg: dict,
    era5_cfg: dict,
    output_parent: str,
    max_members: int | None,
) -> None:
    years     = model_cfg["years"]
    input_dir = model_cfg["input_dir"]
    fmt_key   = model_cfg["format"]

    year_tag  = f"{min(years)}_{max(years)}" if len(years) > 1 else str(years[0])
    out_dir   = os.path.join(output_parent, model_name)
    out_path  = os.path.join(out_dir, f"{model_name}_{year_tag}_crps_ssr_{var_name}.nc")

    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists.")
        return

    era5_var = var_cfg["era5_var"]
    print(f"\n{'='*65}")
    print(f"  Model    : {model_name}")
    print(f"  Variable : {var_name}")
    print(f"  Model var: {var_cfg['model_var']}  ×{var_cfg.get('model_scale', 1.0)}")
    print(f"  ERA5 var : {era5_var}  ×{var_cfg.get('era5_scale', 1.0)}")
    if var_cfg.get("level"):
        print(f"  Level    : {var_cfg['level']} hPa")
    print(f"{'='*65}")

    loader = get_loader(fmt_key)

    print("Loading forecast ...")
    forecast = loader.load_forecast(input_dir, years, var_name, var_cfg, max_members)
    forecast = forecast.chunk({
        "time": 1, "number": -1, "prediction_timedelta": -1, "lat": -1, "lon": -1
    })

    print("Loading ERA5 analysis ...")
    analysis = load_era5(era5_cfg, years, var_cfg)

    print("Computing CRPS & SSR ...")
    res = evaluate_crps_fast(
        forecast, analysis,
        var=era5_var,
        time_dim="time",
        lead_dim="prediction_timedelta",
        member_dim="number",
        compute=True,
    )

    reduce_dims  = [d for d in res.crps.dims if d != "prediction_timedelta"]
    by_lead_crps = res.crps.mean(dim=reduce_dims)
    by_lead_ssr  = res.ssr

    print(f"\n  {'Lead (h)':>8}  {'CRPS':>12}  {'SSR':>8}  {'Spread':>10}  {'Error':>10}")
    for lead_td in by_lead_crps.prediction_timedelta.values:
        hrs      = int(np.round(lead_td / np.timedelta64(1, "h")))
        crps_val = float(by_lead_crps.sel(prediction_timedelta=lead_td).values)
        ssr_val  = float(by_lead_ssr.sel(prediction_timedelta=lead_td).values)
        sprd_val = float(res.spread.sel(prediction_timedelta=lead_td).values)
        err_val  = float(res.error.sel(prediction_timedelta=lead_td).values)
        print(f"  {hrs:>8}  {crps_val:>12.4f}  {ssr_val:>8.3f}  {sprd_val:>10.4f}  {err_val:>10.4f}")

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

    os.makedirs(out_dir, exist_ok=True)
    out.to_netcdf(out_path)
    print(f"\nSaved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute CRPS/SSR for S2S ensemble forecasts.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configuration" / "config.yaml"),
                        help="Path to YAML config file.")
    parser.add_argument("--model", default=None,
                        help="Run only this model (key in config models:).")
    parser.add_argument("--var",   default=None,
                        help="Run only this variable (key in model variables:).")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    era5_cfg       = cfg["era5"]
    output_parent  = cfg["output"]["parent_dir"]
    max_members    = cfg["compute"].get("max_members")
    models_cfg     = cfg["models"]

    # Filter models
    if args.model:
        if args.model not in models_cfg:
            print(f"ERROR: model '{args.model}' not in config. Available: {list(models_cfg)}")
            sys.exit(1)
        models_cfg = {args.model: models_cfg[args.model]}

    for model_name, model_cfg in models_cfg.items():
        variables = model_cfg["variables"]

        # Filter variables
        if args.var:
            if args.var not in variables:
                print(f"ERROR: variable '{args.var}' not in model '{model_name}'. "
                      f"Available: {list(variables)}")
                sys.exit(1)
            variables = {args.var: variables[args.var]}

        for var_name, var_cfg in variables.items():
            evaluate_variable(
                model_name, var_name, var_cfg,
                model_cfg, era5_cfg, output_parent, max_members,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
