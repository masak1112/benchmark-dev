#!/usr/bin/env python3
"""
Plot rank histograms produced by rank_histogram_main.py.

Produces two figures:
  1. Talagrand diagrams at selected lead times, UC-S2S vs SI-S2S side by side.
  2. Flatness (delta) and mean-rank bias as a function of lead time.

Usage:
    python plot_rank_histogram.py --var geopotential_500 \
        --leads 24 168 336 672 1080
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_ROOT = "/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr"
PLOT_DIR     = "/glade/work/bgong/plots"

# Colors/labels come from the shared scheme so a model looks the same in every
# figure (ACC, CRPS/SSR, rank histogram, ...). Labels follow the dataset registry.
sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_color, model_style

MODELS = [
    ("ours_si_postprocess_v2_0903", "UC-S2S V2"),
    ("amip_s2s_train_scratch",      "SI-S2S"),
    ("ngcm",                        "NeuralGCM"),
    ("gencast",                     "GenCast"),
    ("fuxi_s2s",                    "FuXi"),
    ("era5_climatology",            "Climatology"),
]

TEXT_PRIMARY   = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID           = "#d8d7d2"


def _find(model_name: str, var_name: str, date_tag: str = "",
          weekly: bool = False, region: str = "all",
          years: str = "") -> str | None:
    """
    Locate a model's rank-histogram output for `var_name`.

    `date_tag` selects a date-subset run (e.g. "common104" -> files ending
    `rank_hist_{var}_common104.nc`). With no tag, only untagged full-period
    files match, so a tagged run never silently stands in for the full period
    (or vice versa).
    """
    d = os.path.join(RESULTS_ROOT, model_name)
    if not os.path.isdir(d):
        return None
    tags = (f"_{region}" if region != "all" else "") \
        + ("_weekly" if weekly else "") + (f"_{date_tag}" if date_tag else "")
    want = f"rank_hist_{var_name}{tags}.nc"
    hits = [f for f in os.listdir(d) if f.endswith(want)]
    if years:
        # Several year ranges can coexist (e.g. _2019_2024_ and _2019_2021_);
        # sorted()[-1] would silently prefer the widest, so select explicitly.
        hits = [f for f in hits if f"_{years}_rank_hist_" in f]
    if not weekly:
        hits = [f for f in hits if "_weekly" not in f]
    return os.path.join(d, sorted(hits)[-1]) if hits else None


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_talagrand(datasets, var_name, leads, date_tag="", subtitle="", weekly=False, region="all"):
    n_row, n_col = len(datasets), len(leads)
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(2.5 * n_col, 2.6 * n_row),
                             squeeze=False)

    for i, (label, color, ds) in enumerate(datasets):
        avail = ds["prediction_timedelta"].values
        for j, lead in enumerate(leads):
            ax = axes[i][j]
            _style(ax)
            k = int(np.argmin(np.abs(avail - lead)))
            counts = ds["rank_counts"].values[k]
            n = counts.sum()
            if n == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color=TEXT_SECONDARY)
                continue
            freq = counts / n
            nb = len(counts)
            # Flat reference: a perfectly reliable ensemble puts 1/(N+1) in each bin.
            ax.axhline(1.0 / nb, color=TEXT_SECONDARY, lw=1.2, ls="--",
                       zorder=3, alpha=0.8)
            ax.bar(np.arange(nb), freq, width=0.82, color=color,
                   edgecolor="#fcfcfb", linewidth=0.5, zorder=2)
            r = float(ds["ri"].values[k])
            lbl = (f"week {int(avail[k])}" if weekly
                   else f"day {int(round(avail[k]/24))}")
            ax.set_title(f"{lbl}\nRI={r:.3f}", fontsize=9, color=TEXT_PRIMARY)
            ax.set_xticks([0, nb - 1])
            ax.set_xticklabels(["low", "high"], fontsize=8)
            ax.set_ylim(0, max(freq.max() * 1.25, 2.0 / nb))
            if j == 0:
                ax.set_ylabel(f"{label}\nrelative frequency",
                              fontsize=9, color=TEXT_PRIMARY)

    fig.suptitle(f"Rank histograms — {var_name}{subtitle}   "
                 f"(dashed = flat / perfectly reliable)",
                 fontsize=11, color=TEXT_PRIMARY, y=1.0)
    fig.tight_layout()
    suffix = ((f"_{region}" if region != "all" else "")
              + ("_weekly" if weekly else "") + (f"_{date_tag}" if date_tag else ""))
    out = os.path.join(PLOT_DIR, f"rank_histogram_{var_name}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close(fig)


def plot_vs_lead(datasets, var_name, date_tag="", subtitle="", weekly=False, region="all"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        _style(ax)

    for label, color, ds in datasets:
        days = (ds["prediction_timedelta"].values if weekly
                else ds["prediction_timedelta"].values / 24.0)
        if not weekly:
            # Drop lead 0: it is a pipeline alignment check (forecast == analysis),
            # not a forecast result, and for models initialised off their own
            # analysis (NeuralGCM) it is a large artifact that dominates the axis.
            m = days >= 1
            days = days[m]
            ds = ds.isel(prediction_timedelta=np.flatnonzero(m))
        st = model_style(label)
        axes[0].plot(days, ds["ri"].values, label=label, **st)
        axes[1].plot(days, ds["bias"].values, label=label, **st)
        # Direct label at the line end (<=4 series, so label them).
        axes[0].annotate(label, (days[-1], ds["ri"].values[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         color=color, fontsize=9, va="center")

    axes[0].axhline(0.0, color=TEXT_SECONDARY, ls="--", lw=1.2)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("reliability index  (0 = flat)", fontsize=10,
                       color=TEXT_PRIMARY)
    axes[0].set_title("Departure from flatness\n(fraction of mass in wrong bin)",
                      fontsize=11, color=TEXT_PRIMARY)

    axes[1].axhline(0.0, color=TEXT_SECONDARY, ls="--", lw=1.2)
    axes[1].set_ylabel("normalised mean-rank bias", fontsize=10,
                       color=TEXT_PRIMARY)
    axes[1].set_title("Ensemble bias  (>0: forecast too low)", fontsize=11,
                      color=TEXT_PRIMARY)

    for ax in axes:
        ax.set_xlabel("lead week" if weekly else "lead time (days)",
                      fontsize=10, color=TEXT_PRIMARY)
    # Left panel carries end-of-line direct labels, so it needs no legend box;
    # the right panel has no room for direct labels and keeps the legend.
    axes[0].margins(x=0.12)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(f"Ensemble reliability vs lead time — {var_name}{subtitle}",
                 fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    suffix = ((f"_{region}" if region != "all" else "")
              + ("_weekly" if weekly else "") + (f"_{date_tag}" if date_tag else ""))
    out = os.path.join(PLOT_DIR,
                       f"rank_reliability_vs_lead_{var_name}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--var", default="geopotential_500")
    p.add_argument("--leads", type=int, nargs="+",
                   default=[24, 168, 336, 672, 1080],   # day 1,7,14,28,45
                   help="Lead times in hours to show as Talagrand diagrams.")
    p.add_argument("--years", default="",
                   help="Select a specific year range in the filename, e.g. 2019_2021. "
                        "Models whose only file already has that range match anyway.")
    p.add_argument("--region", choices=["all","land","ocean"], default="all",
                   help="Plot the land-only or ocean-only outputs.")
    p.add_argument("--weeks", type=int, nargs="+", default=None,
                   help="With --weekly: which week numbers to show (e.g. 3 4 5 6). "
                        "Default 1-6.")
    p.add_argument("--weekly", action="store_true",
                   help="Plot the weekly-mean (_weekly) outputs.")
    p.add_argument("--date-tag", default="",
                   help="Plot a date-subset run (e.g. 'common104'). Figures are "
                        "written with the tag appended so full-period plots are "
                        "not overwritten.")
    args = p.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    subtitle = f"  ({args.date_tag} common init dates)" if args.date_tag else ""
    if args.weekly:
        subtitle = "  (weekly means)" + subtitle
    if args.region != "all":
        subtitle = f"  ({args.region} only)" + subtitle

    datasets = []
    for model_name, label in MODELS:
        color = model_color(label)
        path = _find(model_name, args.var, args.date_tag, args.weekly, args.region, args.years)
        # A model whose full record already equals the common set has no tagged
        # file (re-running it would be identical work), so fall back to its
        # untagged output rather than dropping it from the comparison.
        if path is None and args.date_tag:
            path = _find(model_name, args.var, "", args.weekly, args.region, args.years)
            if path is not None:
                print(f"  [note] {label}: no _{args.date_tag} file; using "
                      f"full-period output (its record matches the common set)")
        if path is None:
            print(f"[WARN] no rank-histogram output for {model_name} / {args.var}")
            continue
        print(f"  {label}: {path}")
        ds = xr.open_dataset(path)
        # Older output files predate the `ri` variable; derive it on the fly so
        # they stay plottable without recomputing the whole histogram.
        if "ri" not in ds:
            from rank_histogram_main import _reliability_diagnostics
            ri = np.array([_reliability_diagnostics(r)[0]
                           for r in ds["rank_counts"].values], dtype=np.float32)
            ds["ri"] = xr.DataArray(ri, dims=["prediction_timedelta"])
        datasets.append((label, color, ds))

    if not datasets:
        raise SystemExit("No rank histogram files found — run rank_histogram_main.py first.")

    # With --weekly, --leads is interpreted as WEEK NUMBERS (default 1-6).
    if args.weekly:
        leads = args.weeks if args.weeks else [1, 2, 3, 4, 5, 6]
    else:
        leads = args.leads
    plot_talagrand(datasets, args.var, leads, args.date_tag, subtitle, args.weekly, args.region)
    plot_vs_lead(datasets, args.var, args.date_tag, subtitle, args.weekly, args.region)


if __name__ == "__main__":
    main()
