#!/usr/bin/env python3
"""
Area-weighted RMSE vs lead time for S2S ensemble forecasts, verified against ERA5.

Supported models:
  uc_s2s_v2  — ours_si_postprocess_v2/predictions_0903  (format_b: one member per file)
  si_s2s     — amip_s2s_train_scratch                   (format_a: 4 members per seed file)

Usage:
  python rmse_main.py                        # all models, all variables
  python rmse_main.py --model uc_s2s_v2
  python rmse_main.py --model si_s2s --var 2m_temperature
  python rmse_main.py --config my_config.yaml

Output:
  {output_dir}/{model_name}/{model_name}_{var_name}_rmse.npy   shape (n_init, n_leads)
  {output_dir}/{model_name}/{model_name}_{var_name}_rmse.png   RMSE vs lead time plot
"""

import argparse
import os
import glob
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = str(Path(__file__).parent / "configuration" / "config.yaml")


# ── ERA5 loader ───────────────────────────────────────────────────────────────

def load_era5(era5_dir: str, variable: str, date_str: str, year: str,
              level=None, level_dim="level", scale=1.0) -> np.ndarray:
    f = os.path.join(era5_dir, variable, f"{year}_180x360.nc")
    ds = xr.open_dataset(f)[variable]
    if level is not None:
        ds = ds.sel({level_dim: float(level)})
    day = ds.sel(time=date_str, method="nearest")
    if day.time.ndim > 0:
        day = day.isel(time=0)
    return day.values * scale


# ── Area-weighted RMSE ────────────────────────────────────────────────────────

SST_THRESH = 271.15  # K — exclude sea ice (ERA5 < threshold) and ensemble fill values

def area_rmse(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray,
              ens_min: np.ndarray = None, is_sst: bool = False) -> float:
    """Cos-lat area-weighted RMSE. NaN in truth excluded.
    For SST: also excludes ice cells and ensemble fill-value points."""
    mask = ~np.isnan(truth)
    if is_sst:
        mask &= (truth >= SST_THRESH)
        if ens_min is not None:
            mask &= (ens_min >= SST_THRESH)
    d2 = (pred - truth) ** 2
    d2[~mask] = np.nan
    w2d = weights[:, None] * np.ones((1, pred.shape[1]))
    w2d[~mask] = np.nan
    return float(np.sqrt(np.nansum(w2d * d2) / np.nansum(w2d)))


# ── Format A loader (amip_s2s_train_scratch) ──────────────────────────────────

def load_format_a(input_dir: str, years: list, var_cfg: dict,
                  max_members=None) -> tuple:
    """
    Returns (all_rmse, lead_days, lats) where all_rmse shape: (n_init, n_leads).
    Loads init-date directories; each contains seed files with (member, time, lat, lon).
    """
    model_var = var_cfg["model_var"]
    era5_var  = var_cfg["era5_var"]
    era5_dir  = var_cfg["_era5_dir"]
    level     = var_cfg.get("level")
    era5_level = var_cfg.get("era5_level", level)
    level_dim = var_cfg.get("level_dim", "plev")
    era5_level_dim = var_cfg.get("era5_level_dim", "level")
    model_scale = var_cfg.get("model_scale", 1.0)
    era5_scale  = var_cfg.get("era5_scale", 1.0)

    init_dirs = sorted([
        d for d in glob.glob(f"{input_dir}/2*") if os.path.isdir(d)
        and int(os.path.basename(d)[:4]) in years
    ])
    print(f"  {len(init_dirs)} init dates")

    all_rmse = []
    lats_out = None
    lead_days_out = None

    for i_d, idir in enumerate(init_dirs):
        init_date = os.path.basename(idir)
        seed_files = sorted(glob.glob(f"{idir}/*.nc"))
        if not seed_files:
            continue

        member_arrays = []
        times = None
        lats  = None
        for sf in seed_files:
            try:
                ds = xr.open_dataset(sf)
                if times is None:
                    times = ds.time.values
                    lats  = ds.lat.values
                da = ds[model_var]
                if level is not None:
                    da = da.sel({level_dim: float(level)})
                arr = da.values * model_scale  # (member, time, lat, lon)
                if max_members:
                    arr = arr[:max_members]
                member_arrays.append(arr)
            except Exception as e:
                print(f"  WARN {os.path.basename(sf)}: {e}")

        if not member_arrays or times is None:
            continue

        ens = np.concatenate(member_arrays, axis=0)  # (n_members, n_leads, lat, lon)
        ens_mean = ens.mean(axis=0)
        ens_min  = ens.min(axis=0)
        weights = np.cos(np.deg2rad(lats))
        is_sst = (era5_var == "sea_surface_temperature")
        if lats_out is None:
            lats_out = lats
            lead_days_out = np.arange(1, len(times) + 1)

        rmse_leads = []
        for t_i, t in enumerate(times):
            yr = str(t)[:4]; dt = str(t)[:10]
            try:
                era5 = load_era5(era5_dir, era5_var, dt, yr,
                                 era5_level, era5_level_dim, era5_scale)
                rmse_leads.append(area_rmse(ens_mean[t_i], era5, weights,
                                            ens_min[t_i], is_sst))
            except Exception:
                rmse_leads.append(np.nan)
        all_rmse.append(rmse_leads)
        print(f"  [{i_d+1}/{len(init_dirs)}] {init_date} done", flush=True)

    return np.array(all_rmse), lead_days_out, lats_out


# ── Format B loader (ours_si_postprocess_v2) ──────────────────────────────────

def load_format_b(input_dir: str, years: list, var_cfg: dict,
                  max_members=None) -> tuple:
    """
    Returns (all_rmse, lead_days, lats).
    One member per file: {prefix}_{YYYYMMDDHH}_ens_{N}.nc
    """
    model_var  = var_cfg["model_var"]
    era5_var   = var_cfg["era5_var"]
    era5_dir   = var_cfg["_era5_dir"]
    level      = var_cfg.get("level")
    era5_level = var_cfg.get("era5_level", level)
    level_dim  = var_cfg.get("level_dim", "level")
    era5_level_dim = var_cfg.get("era5_level_dim", "level")
    model_scale = var_cfg.get("model_scale", 1.0)
    era5_scale  = var_cfg.get("era5_scale", 1.0)

    _DATE_RE = re.compile(r"_(\d{10})_ens_0\.nc$")
    files0 = sorted([
        f for f in glob.glob(f"{input_dir}/*_ens_0.nc")
        if _DATE_RE.search(f) and int(_DATE_RE.search(f).group(1)[:4]) in years
    ])
    print(f"  {len(files0)} init dates")

    all_rmse = []
    lats_out = None
    lead_days_out = None

    for i_d, f0 in enumerate(files0):
        init_date = _DATE_RE.search(f0).group(1)[:8]
        base_pattern = re.sub(r'_ens_0\.nc$', '_ens_*.nc', f0)
        mfiles = sorted(glob.glob(base_pattern))
        if max_members:
            mfiles = mfiles[:max_members]

        member_arrays = []
        times = None
        lats  = None
        for mf in mfiles:
            try:
                ds = xr.open_dataset(mf)
                if times is None:
                    times = ds.time.values
                    lats  = ds.latitude.values
                da = ds[model_var]
                if level is not None:
                    da = da.sel({level_dim: float(level)})
                member_arrays.append(da.values * model_scale)
            except Exception as e:
                print(f"  WARN {os.path.basename(mf)}: {e}")

        if not member_arrays or times is None:
            continue

        ens = np.stack(member_arrays, axis=0)
        ens_mean = ens.mean(axis=0)
        ens_min  = ens.min(axis=0)
        weights = np.cos(np.deg2rad(lats))
        is_sst = (era5_var == "sea_surface_temperature")
        if lats_out is None:
            lats_out = lats
            lead_days_out = np.arange(1, len(times) + 1)

        rmse_leads = []
        for t_i, t in enumerate(times):
            yr = str(t)[:4]; dt = str(t)[:10]
            try:
                era5 = load_era5(era5_dir, era5_var, dt, yr,
                                 era5_level, era5_level_dim, era5_scale)
                rmse_leads.append(area_rmse(ens_mean[t_i], era5, weights,
                                            ens_min[t_i], is_sst))
            except Exception:
                rmse_leads.append(np.nan)
        all_rmse.append(rmse_leads)
        print(f"  [{i_d+1}/{len(files0)}] {init_date} done", flush=True)

    return np.array(all_rmse), lead_days_out, lats_out


LOADERS = {"format_a": load_format_a, "format_b": load_format_b}


# ── Per-variable evaluation ───────────────────────────────────────────────────

def evaluate_variable(model_name, var_name, var_cfg, model_cfg, era5_cfg, output_parent):
    years     = model_cfg["years"]
    input_dir = model_cfg["input_dir"]
    fmt_key   = model_cfg["format"]

    out_dir  = os.path.join(output_parent, model_name)
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"{model_name}_{var_name}_rmse.npy")
    png_path = os.path.join(out_dir, f"{model_name}_{var_name}_rmse.png")

    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Variable: {var_name}")
    print(f"  Format  : {fmt_key}")
    print(f"{'='*60}")

    # inject era5 dir into var_cfg for loader
    var_cfg = dict(var_cfg)
    var_cfg["_era5_dir"] = era5_cfg["dir"]

    loader = LOADERS[fmt_key]
    rmse_arr, lead_days, lats = loader(
        input_dir, years, var_cfg,
        max_members=model_cfg.get("max_members")
    )

    np.save(npy_path, rmse_arr)
    print(f"\nSaved RMSE array → {npy_path}  shape={rmse_arr.shape}")

    # summary
    mean_rmse = np.nanmean(rmse_arr, axis=0)
    std_rmse  = np.nanstd(rmse_arr, axis=0)
    print(f"\n  {'Lead(day)':>10}  {'Mean RMSE':>12}  {'Std':>8}")
    for d, (m, s) in enumerate(zip(mean_rmse, std_rmse)):
        if (d + 1) in [1, 5, 10, 15, 20, 30, 45]:
            print(f"  {d+1:>10}  {m:>12.4f}  {s:>8.4f}")

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lead_days, mean_rmse, linewidth=2.2, label=model_name)
    ax.fill_between(lead_days, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2)
    ax.set_xlabel("Lead time (days)", fontsize=12)
    ax.set_ylabel(f"RMSE", fontsize=12)
    ax.set_title(f"{var_name} RMSE vs Lead Time — {model_name}\n"
                 f"n={rmse_arr.shape[0]} init dates | Area-weighted global ensemble-mean", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lead_days[0], lead_days[-1])
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot        → {png_path}")

    return rmse_arr, lead_days


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute RMSE for S2S ensemble forecasts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None)
    parser.add_argument("--var",   default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    era5_cfg      = cfg["era5"]
    output_parent = cfg["output"]["rmse_dir"]
    models_cfg    = cfg["models"]

    if args.model:
        if args.model not in models_cfg:
            print(f"ERROR: model '{args.model}' not in config. Available: {list(models_cfg)}")
            sys.exit(1)
        models_cfg = {args.model: models_cfg[args.model]}

    results = {}
    for model_name, model_cfg in models_cfg.items():
        variables = model_cfg["variables"]
        if args.var:
            if args.var not in variables:
                print(f"ERROR: var '{args.var}' not in model '{model_name}'.")
                sys.exit(1)
            variables = {args.var: variables[args.var]}
        for var_name, var_cfg in variables.items():
            rmse_arr, lead_days = evaluate_variable(
                model_name, var_name, var_cfg, model_cfg, era5_cfg, output_parent
            )
            results[(model_name, var_name)] = (rmse_arr, lead_days)

    # multi-model comparison plot if >1 model
    if len(results) > 1:
        var_names = list({v for _, v in results})
        for var_name in var_names:
            entries = [(m, r, ld) for (m, v), (r, ld) in results.items() if v == var_name]
            if len(entries) < 2:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.tab10.colors
            for ci, (mname, rmse_arr, lead_days) in enumerate(entries):
                mean_r = np.nanmean(rmse_arr, axis=0)
                std_r  = np.nanstd(rmse_arr, axis=0)
                ax.plot(lead_days, mean_r, linewidth=2.2, color=colors[ci],
                        label=f"{mname} (n={rmse_arr.shape[0]})")
                ax.fill_between(lead_days, mean_r - std_r, mean_r + std_r,
                                color=colors[ci], alpha=0.15)
            ax.set_xlabel("Lead time (days)", fontsize=12)
            ax.set_ylabel("RMSE", fontsize=12)
            ax.set_title(f"{var_name} RMSE vs Lead Time — Model Comparison", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            cmp_path = os.path.join(output_parent, f"comparison_{var_name}_rmse.png")
            plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nComparison plot → {cmp_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
