"""
Reliability diagrams of P(MJO amplitude >= 1), stratified by the ERA5 MJO state
at initialization (active: init_amp_obs >= 1, inactive: < 1).

Per init date and lead:
  forecast probability p = fraction of ensemble members with amp >= 1
  outcome o              = ERA5 amp_obs >= 1 on the valid date
One sample per init date (not per member). Use --lead-window to pool
neighbouring leads (e.g. 2 -> lead-2..lead+2) and several datasets can be pooled.

Panel annotations:
  n     number of (init, lead) samples
  slope count-weighted LS slope of observed freq on forecast prob (binned)
  REL   Murphy reliability term
  BSS   1 - BS / BS_clim, clim = stratum base rate (init-state conditional)
Shading: 5-95% consistency band around the diagonal (Broecker & Smith 2007),
i.e. range of binned observed freq expected if the forecast were reliable.
Dotted lines: mean forecast probability (vertical) and base rate (horizontal).

Input: <root>/<dataset>/chi200_1p0deg/mjo_phases_<dataset>.nc
       (from run_scripts/assign_mjo_phases.py)

Usage:
  python run_scripts/plot_reliability_by_init_state.py ours_si_v3 \
      --leads 14 20 25 40 --lead-window 0 --out reliability_ours_si_v3.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = "/glade/campaign/univ/uchi0018/bing/mjo"
PLOT_DIR = "/glade/work/bgong/MJO_results/plots"

ROWS = [  # (label, init_active_obs value, colour, band colour)
    ("MJO active at init", True, "#2a78d6", "#cde2fb"),
    ("MJO inactive at init", False, "#6b6f78", "#dcdcd8"),
]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#b5b4ae"


def load_samples(datasets, root, leads, window):
    """Return {lead: (p, o, init_active)} pooled over datasets and lead window."""
    out = {L: ([], [], []) for L in leads}
    for name in datasets:
        path = os.path.join(root, name, "chi200_1p0deg", f"mjo_phases_{name}.nc")
        ds = xr.open_dataset(path)
        p_all = (ds.amp >= 1.0).where(ds.amp.notnull()).mean("number")  # (init, lead)
        o_all = (ds.amp_obs >= 1.0).where(ds.amp_obs.notnull())
        act = (ds.init_amp_obs >= 1.0).values
        has0 = np.isfinite(ds.init_amp_obs.values)
        for L in leads:
            for l in range(L - window, L + window + 1):
                if l not in ds.lead.values:
                    continue
                p = p_all.sel(lead=l).values
                o = o_all.sel(lead=l).values
                ok = has0 & np.isfinite(p) & np.isfinite(o)
                out[L][0].append(p[ok]); out[L][1].append(o[ok]); out[L][2].append(act[ok])
        ds.close()
    return {L: tuple(np.concatenate(v) for v in out[L]) for L in leads}


def reliability_stats(p, o, bins, min_count, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    nb = len(bins) - 1
    cnt = np.bincount(idx, minlength=nb)
    pbar = np.bincount(idx, weights=p, minlength=nb) / np.maximum(cnt, 1)
    obar = np.bincount(idx, weights=o, minlength=nb) / np.maximum(cnt, 1)

    n, base = len(p), o.mean()
    rel = np.sum(cnt * (pbar - obar) ** 2) / n
    bs = np.mean((p - o) ** 2)
    unc = base * (1 - base)
    bss = 1 - bs / unc if unc > 0 else np.nan

    keep = cnt >= min_count
    slope = np.nan
    if keep.sum() >= 2:
        w = cnt[keep]
        x, y = pbar[keep], obar[keep]
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        slope = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
        icpt = ym - slope * xm
    else:
        icpt = np.nan

    # consistency band: resample outcomes assuming perfect reliability
    boot = np.empty((n_boot, nb))
    for b in range(n_boot):
        ob = rng.random(n) < p
        boot[b] = np.bincount(idx, weights=ob, minlength=nb) / np.maximum(cnt, 1)
    lo, hi = np.percentile(boot, [5, 95], axis=0)

    return dict(n=n, cnt=cnt, pbar=pbar, obar=obar, keep=keep, rel=rel, bss=bss,
                slope=slope, icpt=icpt, base=base, pmean=p.mean(), lo=lo, hi=hi)


def draw_panel(ax, s, bins, colour, band):
    k = s["keep"]
    ax.plot([0, 1], [0, 1], color=INK, lw=1.2, zorder=2)
    ax.fill_between(s["pbar"][k], s["lo"][k], s["hi"][k], color=band, alpha=0.8, lw=0, zorder=1)
    if np.isfinite(s["slope"]):
        xx = np.array([0, 1])
        ax.plot(xx, s["icpt"] + s["slope"] * xx, ls="--", color=GRID, lw=1.2, zorder=2)
    ax.axvline(s["pmean"], color=GRID, ls=":", lw=1, zorder=1)
    ax.axhline(s["base"], color=GRID, ls=":", lw=1, zorder=1)
    ax.plot(s["pbar"][k], s["obar"][k], color=colour, lw=2, marker="o", ms=5,
            mec="white", mew=1, zorder=3)

    txt = (f"n     {s['n']:,}\nslope {s['slope']:.2f}\n"
           f"REL   {s['rel']:.4f}\nBSS   {s['bss']:+.3f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=8.5, color=INK)

    ins = ax.inset_axes([0.62, 0.07, 0.34, 0.23])
    centres = 0.5 * (bins[:-1] + bins[1:])
    ins.bar(centres, s["cnt"], width=(bins[1] - bins[0]) * 0.85, color=colour)
    ins.set_xlim(0, 1); ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(GRID)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--leads", nargs="+", type=int, default=[14, 20, 25, 40])
    ap.add_argument("--lead-window", type=int, default=0,
                    help="pool leads L-w..L+w into each panel")
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=10,
                    help="hide bins with fewer samples than this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bins = np.linspace(0, 1, args.nbins + 1)
    samples = load_samples(args.datasets, args.root, args.leads, args.lead_window)

    nrow, ncol = len(ROWS), len(args.leads)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 1.0, 3.5 * nrow + 0.9),
                             squeeze=False)
    for i, (label, flag, colour, band) in enumerate(ROWS):
        for j, L in enumerate(args.leads):
            p, o, act = samples[L]
            m = act == flag
            ax = axes[i, j]
            if m.sum() == 0:
                ax.set_axis_off(); continue
            s = reliability_stats(p[m], o[m].astype(float), bins, args.min_count)
            draw_panel(ax, s, bins, colour, band)
            if i == 0:
                win = f" (±{args.lead_window})" if args.lead_window else ""
                ax.set_title(f"lead day {L}{win}", fontsize=12, fontweight="bold", color=INK)
            if i == nrow - 1:
                ax.set_xlabel("forecast probability", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("observed frequency", fontsize=10, color=INK)
                ax.annotate(label, xy=(-0.33, 0.5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=11, fontweight="bold",
                            color=colour)

    fig.suptitle("Reliability of  P(MJO amplitude ≥ 1)  — stratified by the ERA5 state at initialization",
                 fontsize=13, fontweight="bold", color=INK, y=0.99)
    per = (f"one sample per init date × lead, leads pooled ±{args.lead_window} d"
           if args.lead_window else "one sample per init date")
    fig.text(0.5, 0.945, f"{', '.join(args.datasets)}  ·  {per}"
             "  ·  conditioning on init state only, so verification is not circular",
             ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])

    out = args.out or f"reliability_init_state_{'_'.join(args.datasets)}.png"
    if not os.path.isabs(out):
        os.makedirs(PLOT_DIR, exist_ok=True)
        out = os.path.join(PLOT_DIR, out)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"saved {out}")
    for L in args.leads:
        p, o, act = samples[L]
        print(f"lead {L}: active n={int(act.sum())}, inactive n={int((~act).sum())}")


if __name__ == "__main__":
    main()
