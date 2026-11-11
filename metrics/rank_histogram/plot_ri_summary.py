#!/usr/bin/env python3
"""
Reliability-index summary: one panel per variable, RI vs lead, all models.

Only the RI curve is drawn (no histograms, no bias panel) so several variables
can be compared on one page. Colors/linestyles come from the shared scheme.

Usage:
    python plot_ri_summary.py                       # daily, day 1-45
    python plot_ri_summary.py --weekly              # weekly means, weeks 1-6
    python plot_ri_summary.py --region land
"""
import argparse, os, sys, glob
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_histogram_main import _reliability_diagnostics

RESULTS = "/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr"
PLOTDIR = "/glade/work/bgong/plots"
MODELS = [("ours_si_postprocess_v2_0903", "UC-S2S V2"),
          ("amip_s2s_train_scratch",      "SI-S2S"),
          ("ngcm",                        "NeuralGCM"),
          ("gencast",                     "GenCast"),
          ("fuxi_s2s",                    "FuXi"),
          ("era5_climatology",            "Climatology")]
TEXT, GRID = "#0b0b0b", "#d8d7d2"

def find(model, var, weekly, region, years=""):
    tag = (f"_{region}" if region != "all" else "") + ("_weekly" if weekly else "")
    hits = glob.glob(f"{RESULTS}/{model}/*rank_hist_{var}{tag}.nc")
    if not weekly and region == "all":
        hits = [h for h in hits if "_weekly" not in h and "_land" not in h
                and "_ocean" not in h and "_common" not in h]
    if years:
        hits = [h for h in hits if f"_{years}_rank_hist_" in h]
    return sorted(hits)[-1] if hits else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+",
                   default=["2m_temperature", "temperature_850",
                            "geopotential_500", "total_precipitation_24hr"])
    p.add_argument("--weekly", action="store_true")
    p.add_argument("--region", choices=["all", "land", "ocean"], default="all")
    p.add_argument("--years", default="", help="e.g. 2019_2021")
    p.add_argument("--metric", choices=["ri","bias"], default="ri",
                   help="ri = reliability index (0 = flat); bias = normalised "
                        "mean-rank bias (0 = unbiased, >0 forecast too low).")
    p.add_argument("--no-clim", action="store_true",
                   help="Omit the Climatology reference curve.")
    p.add_argument("--lead-range", type=float, nargs=2, default=None,
                   metavar=("LO","HI"),
                   help="Restrict the x axis to this lead range (days, or weeks "
                        "with --weekly), e.g. --lead-range 14 40.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    drawn = {}
    n = len(a.vars)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.8 * ncol, 4.2 * nrow),
                             squeeze=False)

    for i, var in enumerate(a.vars):
        ax = axes[i // ncol][i % ncol]
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.grid(color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
        ax.tick_params(colors="#52514e", labelsize=13)

        plotted = 0
        for key, label in MODELS:
            if a.no_clim and label == "Climatology":
                continue
            f = find(key, var, a.weekly, a.region, a.years)
            if f is None: continue
            ds = xr.open_dataset(f)
            lead = ds["prediction_timedelta"].values
            if a.metric == "bias":
                ri = (ds["bias"].values if "bias" in ds else
                      np.array([_reliability_diagnostics(c)[2]
                                for c in ds["rank_counts"].values]))
            else:
                ri = (ds["ri"].values if "ri" in ds else
                      np.array([_reliability_diagnostics(c)[0]
                                for c in ds["rank_counts"].values]))
            x = lead if a.weekly else lead / 24.0
            if not a.weekly:                 # drop lead 0 (alignment check)
                m = x >= 1; x, ri = x[m], ri[m]
            if a.lead_range:
                lo, hi = a.lead_range
                m = (x >= lo) & (x <= hi)
                x, ri = x[m], ri[m]
            ax.plot(x, ri, label=label, **model_style(label))
            drawn[label] = ax.lines[-1]
            plotted += 1

        ax.set_title(var, fontsize=15, color=TEXT)
        ax.set_xlabel("lead week" if a.weekly else "lead time (days)",
                      fontsize=14, color=TEXT)
        if a.metric == "bias":
            ax.set_ylabel("mean-rank bias  (>0: forecast low)", fontsize=14,
                          color=TEXT)
            ax.axhline(0.0, color="#52514e", ls="--", lw=1.2, zorder=1)
        else:
            ax.set_ylabel("reliability index  (0 = flat)", fontsize=14,
                          color=TEXT)
            ax.set_ylim(bottom=0)
        if plotted == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#52514e", fontsize=13)


    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    if drawn:
        order = [lab for _, lab in MODELS if lab in drawn]
        fig.legend([drawn[l] for l in order], order, frameon=False, fontsize=14,
                   loc="lower center", ncol=min(len(order), 5),
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        fig.tight_layout()
    tag = (f"_{a.region}" if a.region != "all" else "") + \
          ("_weekly" if a.weekly else "")
    stem = "bias_summary" if a.metric == "bias" else "ri_summary"
    out = a.out or os.path.join(PLOTDIR, f"{stem}{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
