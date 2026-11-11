#!/usr/bin/env python3
"""
Publication-quality rank-histogram (Talagrand) panel grid.

Rows = models, columns = lead times. Each panel shows the relative frequency of
the observation's rank among the ensemble members, with a dashed line at the
flat (perfectly reliable) reference 1/(N+1).

Usage:
    python plot_hist_paper.py --var 2m_temperature --days 15 20 25 30 40 \
        --years 2019_2021
"""
import argparse, glob, os, sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_color
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_histogram_main import _reliability_diagnostics

RESULTS = "/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr"
PLOTDIR = "/glade/work/bgong/plots"

MODELS = [("ours_si_postprocess_v2_0903", "UC-S2S V2"),
          ("amip_s2s_train_scratch",      "SI-S2S"),
          ("ngcm",                        "NeuralGCM"),
          ("gencast",                     "GenCast"),
          ("fuxi_s2s",                    "FuXi"),
          ("ecmwf_s2s",                   "ECMWF"),
          ("ecmwf_hindcast",              "ECMWF hindcast"),
          ("era5_climatology",            "Climatology")]

PRETTY = {
    "2m_temperature":          "2-m temperature",
    "temperature_850":         "850-hPa temperature",
    "geopotential_500":        "500-hPa geopotential",
    "geopotential_850":        "850-hPa geopotential",
    "specific_humidity_850":   "850-hPa specific humidity",
    "u_component_of_wind_500": "500-hPa zonal wind",
}


def _find(model, var, years):
    hits = [h for h in glob.glob(f"{RESULTS}/{model}/*rank_hist_{var}.nc")
            if not any(t in h for t in ("_weekly", "_land", "_ocean", "_common"))]
    if years:
        hits = [h for h in hits if f"_{years}_rank_hist_" in h]
    return sorted(hits)[-1] if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--var", default="2m_temperature")
    p.add_argument("--days", type=int, nargs="+", default=[15, 20, 25, 30, 40])
    p.add_argument("--years", default="2019_2021")
    p.add_argument("--no-clim", action="store_true")
    p.add_argument("--annotate", action="store_true",
                   help="Print RI in each panel.")
    p.add_argument("--stem", default=None)
    a = p.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelsize": 11, "legend.fontsize": 10,
        "savefig.bbox": "tight",
    })

    # gather the models that actually have this variable
    rows = []
    for key, label in MODELS:
        if a.no_clim and label == "Climatology":
            continue
        f = _find(key, a.var, a.years)
        if f is not None:
            rows.append((label, xr.open_dataset(f)))
    if not rows:
        raise SystemExit(f"no data for {a.var}")

    nrow, ncol = len(rows), len(a.days)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             figsize=(1.52 * ncol + 0.9, 1.34 * nrow + 0.7))

    for r, (label, ds) in enumerate(rows):
        lead = ds["prediction_timedelta"].values / 24.0
        col = model_color(label)
        # one y scale per row so a model's lead-to-lead evolution is comparable
        fr = []
        for d in a.days:
            c = ds["rank_counts"].values[int(np.argmin(np.abs(lead - d)))]
            fr.append(c / c.sum())
        ymax = max(f.max() for f in fr) * 1.18

        for c_i, d in enumerate(a.days):
            ax = axes[r][c_i]
            k = int(np.argmin(np.abs(lead - d)))
            counts = ds["rank_counts"].values[k]
            freq = counts / counts.sum()
            nb = len(counts)

            # Plot on a normalised rank axis: climatology has 44 bins and the
            # models 21, so an absolute x axis would leave the model rows
            # filling only half the panel.
            x = (np.arange(nb) + 0.5) / nb
            ax.axhline(1.0 / nb, color="0.35", ls=(0, (3.5, 2.5)), lw=0.8, zorder=3)
            ax.bar(x, freq, width=0.92 / nb, color=col,
                   edgecolor="none", zorder=2)
            ax.set_ylim(0, ymax)
            ax.set_xlim(0, 1)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["low", "high"], fontsize=8.5)
            ax.tick_params(length=2.5)

            if a.annotate:
                ri = _reliability_diagnostics(counts)[0]
                ax.text(0.5, 0.97, f"RI={ri:.2f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8.5)
            if r == 0:
                ax.set_title(f"day {d}", fontsize=11, pad=4)
            if c_i == 0:
                ax.set_ylabel(label, fontsize=10.5)
            else:
                ax.tick_params(labelleft=False)

    fig.supxlabel("Ensemble member rank of the observation", fontsize=11, y=0.012)
    fig.supylabel("Relative frequency", fontsize=11, x=0.002)
    fig.tight_layout(rect=[0.035, 0.035, 1, 1])

    stem = a.stem or f"fig_rank_histogram_{a.var}"
    for ext in ("pdf", "png"):
        out = os.path.join(PLOTDIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
