"""
Rank histograms in RMM space: where the ensemble is under-dispersed, and in which direction.

Rows:
  amplitude      rank of ERA5 amp_obs among member amplitudes
                 right-leaning -> obs stronger than members -> model damps amplitude
  phase deviation rank of the ERA5 phase angle among member angles, both measured
                 relative to the ensemble-mean RMM angle (wrapped to [-180, 180)).
                 MJO propagates counter-clockwise, so
                 right-leaning -> obs ahead of members -> model propagates too slowly
Shape: U = under-dispersive, dome = over-dispersive, slope = systematic bias.

Ranks are computed as randomized PIT values u = (#members < obs + U*(#ties+1)) / (M+1),
so inits with different ensemble sizes (e.g. NGCM 25/51) can be pooled; bins default
to M+1 when M is constant (the exact rank histogram).

Stratification (--stratify):
  init        (default) ERA5 amp >= 1 at INIT: active vs inactive. Not circular.
  transition  ERA5 state at init vs at VALID date: sustained (active->active) vs
              onset/decay (state changed). Conditions on the verifying obs, so the
              AMPLITUDE row is biased even for a perfect ensemble; use with care.

Input: <root>/<dataset>/chi200_1p0deg/mjo_phases_<dataset>.nc (assign_mjo_phases.py)

Usage:
  python run_scripts/plot_rank_hist_rmm.py ours_si_v3 --leads 14 20 25 40 --lead-window 2
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = "/glade/campaign/univ/uchi0018/bing/mjo"
PLOT_DIR = "/glade/work/bgong/MJO_results/plots"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#b5b4ae"
C1, C2 = "#2a78d6", "#eb6834"  # categorical slots 1, 2

GROUPS = {
    "init": [("MJO active at init", C1), ("MJO inactive at init", C2)],
    "transition": [("MJO sustained", C1), ("MJO onset / decay", C2)],
}
ROWS = [
    ("amp", "Amplitude rank\nof the observation",
     "right-leaning  →  obs stronger than members  →  model damps amplitude"),
    ("phase", "Phase-deviation rank\nof the observation",
     "right-leaning  →  obs ahead of members  →  model propagates too slowly"),
]


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def pit(members, obs, rng):
    """members (N, M) with NaN padding, obs (N,) -> randomized PIT in [0, 1]."""
    valid = np.isfinite(members)
    m = valid.sum(1)
    less = np.sum(valid & (members < obs[:, None]), 1)
    ties = np.sum(valid & (members == obs[:, None]), 1)
    return (less + rng.random(len(obs)) * (ties + 1)) / (m + 1), m


def load(datasets, root, leads, window, stratify, rng):
    """Return {lead: {'amp': u, 'phase': u, 'group': g, 'M': m}} pooled."""
    acc = {L: {"amp": [], "phase": [], "group": [], "M": []} for L in leads}
    for name in datasets:
        ds = xr.open_dataset(os.path.join(root, name, "chi200_1p0deg", f"mjo_phases_{name}.nc"))
        amp = ds.amp.transpose("init", "lead", "number").values
        ang = np.degrees(np.arctan2(ds.RMM2, ds.RMM1)).transpose("init", "lead", "number").values
        amp_o = ds.amp_obs.transpose("init", "lead").values
        ang_o = np.degrees(np.arctan2(ds.RMM2_obs, ds.RMM1_obs)).transpose("init", "lead").values
        ang_em = np.degrees(np.arctan2(ds.RMM2_emean, ds.RMM1_emean)).transpose("init", "lead").values
        act0 = ds.init_amp_obs.values >= 1.0
        has0 = np.isfinite(ds.init_amp_obs.values)
        lead_idx = {int(v): k for k, v in enumerate(ds.lead.values)}  # label -> position
        for L in leads:
            for lab in range(L - window, L + window + 1):
                if lab not in lead_idx:
                    continue
                l = lead_idx[lab]
                mem_a, obs_a = amp[:, l, :], amp_o[:, l]
                mem_p = wrap(ang[:, l, :] - ang_em[:, l, None])
                obs_p = wrap(ang_o[:, l] - ang_em[:, l])
                ok = has0 & np.isfinite(obs_a) & np.isfinite(obs_p) & np.isfinite(ang_em[:, l]) \
                    & (np.isfinite(mem_a).sum(1) > 1)
                ua, m = pit(mem_a[ok], obs_a[ok], rng)
                up, _ = pit(mem_p[ok], obs_p[ok], rng)
                if stratify == "init":
                    g = np.where(act0[ok], 0, 1)
                else:
                    act_l = obs_a[ok] >= 1.0
                    g = np.where(act0[ok] & act_l, 0, np.where(act0[ok] != act_l, 1, -1))
                for k, v in (("amp", ua), ("phase", up), ("group", g), ("M", m)):
                    acc[L][k].append(v)
        ds.close()
    return {L: {k: np.concatenate(v) for k, v in d.items()} for L, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--leads", nargs="+", type=int, default=[14, 20, 25, 40])
    ap.add_argument("--lead-window", type=int, default=0)
    ap.add_argument("--stratify", choices=["init", "transition"], default="init")
    ap.add_argument("--nbins", type=int, default=None, help="default: M+1 if ensemble size constant, else 20")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    data = load(args.datasets, args.root, args.leads, args.lead_window, args.stratify, rng)
    Ms = np.unique(np.concatenate([d["M"] for d in data.values()]))
    if args.nbins:
        nbins = args.nbins
    elif len(Ms) == 1:
        # exact rank histogram; for big ensembles (FuXi 51) merge adjacent ranks:
        # largest divisor of M+1 that is <= 30
        nbins = max(k for k in range(1, 31) if (int(Ms[0]) + 1) % k == 0)
    else:
        nbins = 20
    edges = np.linspace(0, 1, nbins + 1)
    flat = 1.0 / nbins
    groups = GROUPS[args.stratify]

    hists, ymax = {}, 0
    for L, d in data.items():
        for r, *_ in ROWS:
            for gi in range(len(groups)):
                u = d[r][d["group"] == gi]
                h = np.histogram(u, edges)[0] / max(len(u), 1)
                hists[(r, L, gi)] = (h, len(u))
                ymax = max(ymax, h.max())
    ymax = args.ymax or min(np.ceil(ymax * 20) / 20, 0.5)

    nrow, ncol = len(ROWS), len(args.leads)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol + 1.0, 3.0 * nrow + 1.6),
                             squeeze=False, sharey=True)
    for i, (r, rlabel, note) in enumerate(ROWS):
        for j, L in enumerate(args.leads):
            ax = axes[i, j]
            ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.5)
            for gi, (glabel, col) in enumerate(groups):
                h, n = hists[(r, L, gi)]
                if n == 0:
                    continue
                x = np.r_[edges[:-1], edges[-1]]
                y = np.r_[h, h[-1]]
                ax.fill_between(x, y, step="post", color=col, alpha=0.10, lw=0)
                ax.step(x, y, where="post", color=col, lw=2,
                        label=(f"{glabel}  (n={n:,}" + (" per lead)" if args.stratify == "init"
                               else f" at lead {L}, varies)")) if (i, j) == (0, 0) else None)
            ax.axhline(flat, color=INK, lw=1.2, label="flat = calibrated" if (i, j) == (0, 0) else None)
            ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
            ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["lowest", "", "highest"])
            ax.tick_params(colors=INK2, labelsize=9)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if i == 0:
                win = f" (±{args.lead_window})" if args.lead_window else ""
                ax.set_title(f"lead day {L}{win}", fontsize=12, fontweight="bold", color=INK)
            if j == 0:
                ax.set_ylabel("relative frequency", fontsize=10, color=INK)
                ax.annotate(rlabel, xy=(-0.28, 0.5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)

    fig.suptitle("Rank histograms in RMM space — where the ensemble is under-dispersed and in which direction",
                 fontsize=13, fontweight="bold", color=INK, y=0.99)
    sub = f"{', '.join(args.datasets)}  ·  U-shape = under-dispersive · dome = over-dispersive · slope = systematic bias"
    if args.stratify == "transition":
        sub += "\n⚠ groups use the verifying obs → amplitude row is biased by construction"
    fig.text(0.5, 0.93, sub, ha="center", fontsize=9.5, color=INK2)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0.03, 0.10, 1, 0.91], h_pad=3.0)
    # row notes centred under the whole row (placed after layout is final)
    for i, (_, _, note) in enumerate(ROWS):
        left, right = axes[i, 0].get_position(), axes[i, -1].get_position()
        fig.text(0.5 * (left.x0 + right.x1), left.y0 - 0.055, note,
                 ha="center", va="top", fontsize=9.5, color=INK2)

    out = args.out or f"rank_hist_rmm_{args.stratify}_{'_'.join(args.datasets)}.png"
    if not os.path.isabs(out):
        os.makedirs(PLOT_DIR, exist_ok=True)
        out = os.path.join(PLOT_DIR, out)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"saved {out}  (nbins={nbins}, ensemble sizes {Ms.tolist()})")


if __name__ == "__main__":
    main()
