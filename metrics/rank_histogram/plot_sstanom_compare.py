#!/usr/bin/env python3
"""
SST-anomaly run (UC-S2S sstanom) vs the other models: rank-histogram reliability.

Panel (a) reliability index vs lead, (b) normalised mean-rank bias vs lead,
(c)-(f) Talagrand histograms at selected leads for raw vs bias-corrected.

The "_bc" curves use a leave-one-year-out, lead-dependent, grid-point-wise
model climatology correction (see rank_histogram_main.compute_rank_histogram).

Usage:
    python plot_sstanom_compare.py --years 2019_2021
"""
import argparse, glob, os, sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_histogram_main import _reliability_diagnostics

RESULTS = "/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr"
PLOTDIR = "/glade/work/bgong/plots"

# (model dir, bias-corrected?, legend label, style key)
SERIES = [
    ("ours_si_sstanom",             False, "UC-S2S SSTanom",      "UC-S2S V2"),
    ("ours_si_sstanom",             True,  "UC-S2S SSTanom (BC)", "UC-S2S V2"),
    ("ours_si_postprocess_v2_0903", False, "UC-S2S V2",           "__v2"),
    ("ours_si_postprocess_v2_0903", True,  "UC-S2S V2 (BC)",      "__v2"),
    ("amip_s2s_train_scratch",      False, "SI-S2S",              "SI-S2S"),
    ("fuxi_s2s",                    False, "FuXi",                "FuXi"),
    ("ecmwf_s2s",                   False, "ECMWF",               "ECMWF"),
]


def _find(model, var, years, bc):
    pat = f"{RESULTS}/{model}/*_{years}_rank_hist_{var}"
    pat += "_bc.nc" if bc else ".nc"
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def _curve(path):
    ds = xr.open_dataset(path)
    leads = ds["prediction_timedelta"].values / 24.0   # hours -> days
    ri, bi = [], []
    for i in range(len(leads)):
        r, _, b = _reliability_diagnostics(ds["rank_counts"].values[i])
        ri.append(r); bi.append(b)
    return leads, np.array(ri), np.array(bi), ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--var", default="sea_surface_temperature")
    p.add_argument("--years", default="2019_2021")
    p.add_argument("--lead-range", type=float, nargs=2, default=[1, 45])
    p.add_argument("--days", type=int, nargs="+", default=[15, 30, 45])
    p.add_argument("--stem", default=None)
    a = p.parse_args()

    data = []
    for model, bc, label, skey in SERIES:
        f = _find(model, a.var, a.years, bc)
        if not f:
            print(f"  [missing] {label}")
            continue
        leads, ri, bi, ds = _curve(f)
        data.append(dict(label=label, skey=skey, bc=bc, leads=leads,
                         ri=ri, bias=bi, ds=ds))
        print(f"  [ok] {label:22s} {os.path.basename(f)}")
    if not data:
        sys.exit("No input files found.")

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.linewidth": 0.8, "mathtext.fontset": "dejavuserif"})
    nd = len(a.days)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.42, top=0.88)
    gs_top = gs[0].subgridspec(1, 2, wspace=0.30)
    gs_bot = gs[1].subgridspec(1, nd, wspace=0.34)
    ax_ri = fig.add_subplot(gs_top[0])
    ax_bi = fig.add_subplot(gs_top[1])

    lo, hi = a.lead_range
    for d in data:
        if d["skey"] == "__v2":          # overlaps SSTanom almost exactly
            col, base_ls = "0.35", "--"
        else:
            st = model_style(d["skey"])
            col, base_ls = st.get("color", "k"), st.get("linestyle", "-")
        ls = ":" if d["bc"] else base_ls
        lw = 1.8 if d["bc"] else 1.3
        m = (d["leads"] >= lo) & (d["leads"] <= hi)
        for ax, key in ((ax_ri, "ri"), (ax_bi, "bias")):
            ax.plot(d["leads"][m], d[key][m], color=col, ls=ls, lw=lw,
                    label=d["label"], zorder=3 if d["bc"] else 2)

    ax_ri.set_ylabel("reliability index")
    ax_bi.set_ylabel("ensemble bias")
    ax_bi.axhline(0, color="0.6", lw=0.6, zorder=1)
    for ax, lab in ((ax_ri, "(a)"), (ax_bi, "(b)")):
        ax.set_xlabel("lead time (days)")
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.0, 1.04, lab, transform=ax.transAxes, fontweight="bold", va="bottom")
    h, lb = ax_ri.get_legend_handles_labels()
    fig.legend(h, lb, fontsize=7.5, frameon=False, ncol=len(lb),
               loc="upper center", bbox_to_anchor=(0.5, 0.955))

    # Talagrand panels: sstanom raw vs BC at the requested leads
    tal = [d for d in data if d["label"].startswith("UC-S2S SSTanom")]
    for j, day in enumerate(a.days):
        ax = fig.add_subplot(gs_bot[j])
        for d in tal:
            ds = d["ds"]
            i = int(np.argmin(np.abs(d["leads"] - day)))
            c = ds["rank_counts"].values[i].astype(float)
            freq = c / c.sum()
            k = len(freq)
            x = (np.arange(k) + 0.5) / k
            ax.step(x, freq * k, where="mid", lw=1.3,
                    ls=":" if d["bc"] else "-",
                    color=model_style("UC-S2S V2").get("color", "k"), zorder=3,
                    label="BC" if d["bc"] else "raw")
        ax.axhline(1.0, color="0.6", lw=0.6)
        ax.set_title(f"day {day}", fontsize=8.5)
        ax.set_xlabel("normalised rank")
        if j == 0:
            ax.set_ylabel("freq. / uniform")
            ax.legend(fontsize=7, frameon=False)
        ax.text(0.0, 1.14, f"({chr(99+j)})", transform=ax.transAxes,
                fontweight="bold", va="bottom")
        ax.set_xlim(0, 1)

    stem = a.stem or f"fig_sstanom_{a.var}"
    os.makedirs(PLOTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"{PLOTDIR}/{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
