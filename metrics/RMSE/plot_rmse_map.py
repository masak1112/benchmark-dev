#!/usr/bin/env python3
"""
Plot gridpoint RMSE maps produced by rmse_map.py.

Usage:
  python plot_rmse_map.py --model uc_s2s_v2 --var 2m_temperature --tag 2019
  python plot_rmse_map.py --model uc_s2s_v2 --var 2m_temperature --tag 2019 --leads 1 7 14 21 30 45
"""

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DEFAULT_CONFIG = str(Path(__file__).parent / "configuration" / "config.yaml")
PLOT_DIR = "/glade/work/bgong/plots"

# Sequential single-hue ramp: RMSE is a positive magnitude with a true zero.
# (Never a rainbow/jet ramp - it invents banding that isn't in the data.)
CMAP = "magma_r"

UNITS = {
    "2m_temperature": "K",
    "sea_surface_temperature": "K",
}

TITLES = {
    "uc_s2s_v2": "UC-S2S v2",
    "si_s2s": "SI-S2S",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--model", default="uc_s2s_v2")
    p.add_argument("--var", default="2m_temperature")
    p.add_argument("--tag", default=None)
    p.add_argument("--leads", type=int, nargs="+", default=[1, 7, 14, 21, 30, 45])
    p.add_argument("--vmax", type=float, default=None,
                   help="Fix the colorbar maximum. Pass the SAME value for every "
                        "model so figures are visually comparable "
                        "(default: this file's own 99th percentile).")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    out_parent = cfg["output"]["rmse_dir"]

    suffix = f"_{args.tag}" if args.tag else ""
    nc = os.path.join(out_parent, args.model,
                      f"{args.model}_{args.var}_rmse_map{suffix}.nc")
    ds = xr.open_dataset(nc)
    rmse = ds["rmse"]
    n_init = ds.attrs.get("n_init_dates", "?")
    years = ds.attrs.get("years", "")

    unit = UNITS.get(args.var, "")
    model_label = TITLES.get(args.model, args.model)
    leads = [L for L in args.leads if L in ds.lead.values]

    # Shared color scale across panels so leads are directly comparable.
    # --vmax additionally makes separate figures comparable to each other.
    vmax = args.vmax if args.vmax else float(
        np.nanpercentile(rmse.sel(lead=leads).values, 99))
    vmin = 0.0
    levels = np.linspace(vmin, vmax, 21)

    ncol = 3
    nrow = int(np.ceil(len(leads) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.4 * ncol, 3.0 * nrow),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=180)},
    )
    axes = np.atleast_1d(axes).ravel()

    lon = ds.lon.values
    lat = ds.lat.values

    for ax, L in zip(axes, leads):
        field = rmse.sel(lead=L).values
        # Wrap in longitude so no seam appears at the prime meridian.
        f_c, lon_c = np.concatenate([field, field[:, :1]], axis=1), np.append(lon, 360.0)
        m = ax.contourf(
            lon_c, lat, f_c, levels=levels, cmap=CMAP,
            transform=ccrs.PlateCarree(), extend="max",
        )
        ax.coastlines(linewidth=0.4, color="0.25")
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="0.5")
        ax.set_global()
        gm = _area_mean(field, lat)
        ax.set_title(f"Day {L}   (global {gm:.2f} {unit})", fontsize=11, pad=5)

    for ax in axes[len(leads):]:
        ax.axis("off")

    cbar = fig.colorbar(
        m, ax=axes[:len(leads)].tolist(), orientation="horizontal",
        fraction=0.04, pad=0.05, aspect=45,
    )
    cbar.set_label(f"RMSE ({unit})", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    var_label = args.var.replace("_", " ")
    scale_note = f"  |  common scale 0–{vmax:g} {unit}" if args.vmax else ""
    fig.suptitle(
        f"{model_label} — {var_label} RMSE vs ERA5\n"
        f"ensemble mean, {n_init} initializations {years}{scale_note}",
        fontsize=14, y=0.98,
    )

    os.makedirs(PLOT_DIR, exist_ok=True)
    out_png = os.path.join(PLOT_DIR,
                           f"{args.model}_{args.var}_rmse_map{suffix}.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_png}")


def _area_mean(field, lat):
    w = np.cos(np.deg2rad(lat))[:, None]
    good = np.isfinite(field)
    return float(np.nansum(w * np.where(good, field, 0)) / np.nansum(w * good))


if __name__ == "__main__":
    main()
