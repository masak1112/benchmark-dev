#!/usr/bin/env python3
"""
Publication-quality reliability-index figure.

Differences from plot_ri_summary.py (which is for quick looks):
  - serif typography, vector PDF + 300 dpi PNG
  - (a)-(d) panel labels, LaTeX-style variable names with units
  - shared y-label per row, outer ticks, no redundant axis text
  - n (init dates) reported per model in the legend
  - no figure title (caption carries it in a paper)

Usage:
    python plot_ri_paper.py --years 2019_2021 --lead-range 14 42
"""
import argparse, glob, os, sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_histogram_main import _reliability_diagnostics

RESULTS = "/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr"
PLOTDIR = "/glade/work/bgong/plots"

MODELS = [("ours_si_postprocess_v2_0903", "UC-S2S V2"),
          ("ours_si_sstanom",             "UC-S2S SSTanom"),
          ("amip_s2s_train_scratch",      "SI-S2S"),
          ("ngcm",                        "NeuralGCM"),
          ("gencast",                     "GenCast"),
          ("fuxi_s2s",                    "FuXi"),
          ("ecmwf_s2s",                   "ECMWF"),
          ("ecmwf_hindcast",              "ECMWF hindcast"),
          ("era5_climatology",            "Climatology")]

# display name (with units) for each variable key
MARKERS = {"UC-S2S V2": "o", "UC-S2S SSTanom": "*", "SI-S2S": "s", "NeuralGCM": "^",
           "GenCast": "D", "FuXi": "v", "ECMWF": "P", "ECMWF hindcast": "X", "Climatology": None}

PRETTY = {
    "2m_temperature":          r"2-m temperature",
    "temperature_850":         r"850-hPa temperature",
    "geopotential_500":        r"500-hPa geopotential",
    "geopotential_850":        r"850-hPa geopotential",
    "specific_humidity_850":   r"850-hPa specific humidity",
    "u_component_of_wind_500": r"500-hPa zonal wind",
    "total_precipitation_24hr": r"24-h precipitation",
    "sea_surface_temperature": r"sea surface temperature",
}


def _find(model, var, years, bc=False):
    tail = "_bc.nc" if bc else ".nc"
    hits = [h for h in glob.glob(f"{RESULTS}/{model}/*rank_hist_{var}{tail}")
            if not any(t in h for t in ("_weekly", "_land", "_ocean", "_common"))]
    if years:
        hits = [h for h in hits if f"_{years}_rank_hist_" in h]
    return sorted(hits)[-1] if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+",
                   default=["2m_temperature", "geopotential_500",
                            "specific_humidity_850", "u_component_of_wind_500"])
    p.add_argument("--years", default="2019_2021")
    p.add_argument("--lead-range", type=float, nargs=2, default=[14, 42])
    p.add_argument("--ylim", type=float, nargs=2, default=None,
                   metavar=("LO","HI"),
                   help="Fix the same y range on every panel, e.g. --ylim -0.35 0.35. "
                        "Values outside the range would be clipped, so check the data first.")
    p.add_argument("--metric", choices=["ri","bias"], default="ri",
                   help="ri = reliability index; bias = normalised mean-rank bias.")
    p.add_argument("--no-clim", action="store_true")
    p.add_argument("--bc", action="store_true",
                   help="Also overlay the bias-corrected (_bc) curve, dotted, "
                        "for every model that has one.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to these model labels (e.g. --only 'UC-S2S V2' "
                        "'UC-S2S SSTanom').")
    p.add_argument("--no-ecmwf", action="store_true",
                   help="Omit the ECMWF S2S curve.")
    p.add_argument("--ncol", type=int, default=None,
                   help="Panel columns (default 2 for <=4 vars, else 3).")
    p.add_argument("--stem", default=None)
    a = p.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 11, "savefig.bbox": "tight",
    })

    n = len(a.vars)
    ncol = a.ncol or (3 if n > 4 else 2)
    nrow = int(np.ceil(n / ncol))
    # ~2.4 in per column keeps 3-wide panels legible at journal double-column
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.45 * ncol + 0.9, 2.35 * nrow),
                             squeeze=False, sharex=True)

    drawn, counts = {}, {}
    for i, var in enumerate(a.vars):
        ax = axes[i // ncol][i % ncol]
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", lw=0.4, color="0.85", zorder=0)
        ax.set_axisbelow(True)

        for key, label in MODELS:
            if a.no_clim and label == "Climatology":
                continue
            if a.no_ecmwf and label == "ECMWF":
                continue
            if a.only and label not in a.only:
                continue
            # raw first, then the bias-corrected overlay if requested
            for bc in ((False, True) if a.bc else (False,)):
                f = _find(key, var, a.years, bc)
                if f is None:
                    continue
                label_b = label + " (BC)" if bc else label
                ds = xr.open_dataset(f)
                lead = ds["prediction_timedelta"].values / 24.0
                if a.metric == "bias":
                    ri = (ds["bias"].values if "bias" in ds else
                          np.array([_reliability_diagnostics(c)[2]
                                    for c in ds["rank_counts"].values]))
                else:
                    ri = (ds["ri"].values if "ri" in ds else
                          np.array([_reliability_diagnostics(c)[0]
                                    for c in ds["rank_counts"].values]))
                m = (lead >= a.lead_range[0]) & (lead <= a.lead_range[1])
                st = dict(model_style(label))
                st["linewidth"] = max(st["linewidth"] * 0.65, 1.0)
                if bc:
                    st["linestyle"] = ":"
                    st["linewidth"] = max(st["linewidth"], 1.6)
                # Sparse open markers so curves that coincide (e.g. UC-S2S V2 and
                # SI-S2S for u500, which differ by <0.005) remain distinguishable.
                mk = MARKERS.get(label)
                if mk and not bc:
                    st.update(marker=mk, markersize=4.5, markevery=4,
                              markerfacecolor="white", markeredgewidth=1.0)
                ln, = ax.plot(lead[m], ri[m], solid_capstyle="round", **st)
                drawn.setdefault(label_b, ln)
                # record sample size (init dates) for the legend
                nd = int(ds["n_valid"].values[0] // (180 * 360 * 1)) if "n_valid" in ds else None
                counts.setdefault(label_b, nd)

        # Panel label above the axes, left-aligned; the title sits centred
        # below it (title pad reserves the room) so the two never overlap.
        ax.text(0.0, 1.14, f"({chr(97+i)})", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=12, fontweight="bold")
        ax.set_title(PRETTY.get(var, var), pad=4)
        if a.metric == "bias":
            ax.axhline(0.0, color="0.35", ls=(0, (4, 3)), lw=0.8, zorder=1)
        else:
            ax.set_ylim(bottom=0)
        if a.ylim:
            ax.set_ylim(*a.ylim)
        ax.xaxis.set_major_locator(MultipleLocator(7))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        if i >= n - ncol:
            ax.set_xlabel("Lead time (days)")
        if i % ncol == 0:
            ax.set_ylabel("Mean-rank bias" if a.metric == "bias"
                          else "Reliability index")
        elif a.ylim:
            # shared scale -> inner panels need no repeated tick labels
            ax.tick_params(labelleft=False)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    order = [lab for _, lab in MODELS if lab in drawn]
    ncol_leg = min(len(order), 4 if len(order) > 5 else 5)
    nleg = int(np.ceil(len(order) / ncol_leg))
    # Reserve space per legend row. A 1-row panel grid with many entries
    # otherwise puts the legend on top of the x tick labels.
    pad = 0.05 * nleg + (0.10 if nrow == 1 else 0.0)
    fig.legend([drawn[l] for l in order], order, frameon=False,
               loc="lower center", ncol=ncol_leg,
               bbox_to_anchor=(0.5, -0.055 - 0.07 * (nrow == 1)),
               handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=[0, pad, 1, 1])

    stem = a.stem or ("fig_ensemble_bias" if a.metric == "bias"
                      else "fig_reliability_index")
    for ext in ("pdf", "png"):
        out = os.path.join(PLOTDIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
