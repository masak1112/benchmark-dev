"""
src/step5_phase_space.py — RMM phase space plots comparing all enabled datasets.

For each month listed in config plots.phase_space_months (e.g. "202005" for May 2020),
this script:
  1. Finds the first available init date in that month across any dataset
  2. Loads observation and all enabled dataset ensemble-mean RMM trajectories
  3. Produces one figure per month showing all three trajectories on the same phase diagram

Output: cfg.plot_dir/phase_space_YYYYMM.png (one per configured month)

A separate figure per dataset is also saved to each dataset's own plot dir.

Usage:
  python src/step5_phase_space.py [--config CFG]
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PipelineConfig
from src.step5_acc_plot import DS_LABELS
sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_style, model_color


# ── Phase diagram drawing ─────────────────────────────────────────

def draw_phase_diagram(ax):
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    for ang in [45, 135]:
        r = 4.5
        ax.plot([-r*np.cos(np.radians(ang)),  r*np.cos(np.radians(ang))],
                [-r*np.sin(np.radians(ang)),  r*np.sin(np.radians(ang))],
                color="k", lw=0.5)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="k", lw=0.8))
    labels = [
        (1, "W. Hem\n& Africa",    (-3.0, -1.8)),
        (2, "Indian\nOcean",       (-1.8, -3.0)),
        (3, "Indian\nOcean",       ( 1.8, -3.0)),
        (4, "Maritime\nContinent", ( 3.0, -1.8)),
        (5, "Maritime\nContinent", ( 3.0,  1.8)),
        (6, "W. Pacific",          ( 1.8,  3.0)),
        (7, "W. Pacific",          (-1.8,  3.0)),
        (8, "W. Hem\n& Africa",    (-3.0,  1.8)),
    ]
    for ph, reg, (rx, ry) in labels:
        ax.text(rx*0.85, ry*0.85, str(ph), fontsize=13, color="0.65",
                ha="center", va="center", fontweight="bold")
        ax.text(rx*1.38, ry*1.38, reg, fontsize=6.5, color="0.55",
                ha="center", va="center")
    ax.text( 0,  4.3, "Pacific Ocean",       ha="center", va="bottom",  fontsize=9, color="0.5")
    ax.text( 0, -4.3, "Indian Ocean",        ha="center", va="top",     fontsize=9, color="0.5")
    ax.text(-4.3, 0,  "W. Hem.\n& Africa",   ha="center", va="center",  fontsize=9, color="0.5", rotation=90)
    ax.text( 4.3, 0,  "Maritime\nContinent", ha="center", va="center",  fontsize=9, color="0.5", rotation=-90)
    ax.set_xlim(-4.5, 4.5);  ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("RMM1", fontsize=11);  ax.set_ylabel("RMM2", fontsize=11)
    ax.set_aspect("equal");  ax.tick_params(labelsize=9)


# ── RMM file lookup ───────────────────────────────────────────────

def find_rmm_file(rmm_dir: str, date_str: str, variable: str) -> str | None:
    """Return path to RMM NC for date_str (YYYYMMDD) in rmm_dir, or None."""
    direct = os.path.join(rmm_dir, f"MJO_{variable}_Forecast_Init_{date_str}.nc")
    if os.path.exists(direct):
        return direct
    # fallback: glob with any matching prefix
    hits = sorted(glob.glob(os.path.join(rmm_dir, f"MJO_*Forecast_Init_{date_str}.nc")))
    return hits[0] if hits else None


def first_date_in_month(rmm_dir: str, yyyymm: str, variable: str) -> str | None:
    """Return the first YYYYMMDD init date that has an RMM file in yyyymm."""
    pattern = os.path.join(rmm_dir, f"MJO_{variable}_Forecast_Init_{yyyymm}??.nc")
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    m = re.search(r"(\d{8})\.nc$", os.path.basename(hits[0]))
    return m.group(1) if m else None


# ── Plotting ──────────────────────────────────────────────────────

# Styling per dataset: colour / line style / width from model_colors via DS_LABELS;
# _DEFAULT_STYLES is only a fallback for keys model_colors doesn't know.
_MARKERS = ["o", "s", "^", "D", "v"]
_DEFAULT_STYLES = [
    dict(color="seagreen",    ls="-.",  marker="^"),
    dict(color="darkorange",  ls=":",   marker="D"),
    dict(color="purple",      ls="-",   marker="v"),
]


def _get_style(ds_name: str, idx: int) -> dict:
    try:
        label = DS_LABELS[ds_name]
        st = model_style(label)
        return dict(color=st["color"], ls=st["linestyle"], lw=st["linewidth"],
                    marker=_MARKERS[idx % len(_MARKERS)], label=f"{label} [{ds_name}]")
    except KeyError:
        s = dict(_DEFAULT_STYLES[idx % len(_DEFAULT_STYLES)])
        s["label"] = ds_name
        return s


def make_phase_space_plots(cfg: PipelineConfig) -> None:
    os.makedirs(cfg.plot_dir, exist_ok=True)

    for yyyymm in cfg.phase_space_months:
        # Accept either exact YYYYMMDD or YYYYMM month prefix
        if len(yyyymm) == 8:
            date_str = yyyymm
            print(f"\nPhase space — init date: {date_str}")
        else:
            print(f"\nPhase space — month: {yyyymm}")
            date_str = None
            for ds_name in cfg.datasets:
                d = first_date_in_month(cfg.rmm_dir(ds_name), yyyymm, cfg.variable)
                if d is not None:
                    date_str = d
                    print(f"  Using init date {date_str} (from {ds_name})")
                    break

        if date_str is None:
            print(f"  WARNING: No RMM files found for {yyyymm} — skipping.")
            continue

        init_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # ── collect trajectories ──────────────────────────────────
        trajectories = {}   # ds_name → {obs, emean} arrays
        obs_rmm1 = obs_rmm2 = None

        for ds_name, ds_cfg in cfg.datasets.items():
            # try config resolution first, then fall back to other resolution
            rmm_fp = find_rmm_file(cfg.rmm_dir(ds_name), date_str, cfg.variable)
            if rmm_fp is None:
                for res in ["1p0deg", "1p5deg"]:
                    alt = os.path.join(ds_cfg.out_dir, f"{cfg.variable}_{res}", "rmm_output")
                    rmm_fp = find_rmm_file(alt, date_str, cfg.variable)
                    if rmm_fp:
                        break
            if rmm_fp is None:
                print(f"  {ds_name}: RMM file not found for {date_str} — skipping.")
                continue
            ds = xr.open_dataset(rmm_fp)
            n_leads = len(ds.time)
            r1e = ds["RMM1_emean"].values[:n_leads]
            r2e = ds["RMM2_emean"].values[:n_leads]
            r1o = ds["RMM1_obs"].values[:n_leads]
            r2o = ds["RMM2_obs"].values[:n_leads]
            ds.close()

            trajectories[ds_name] = {"r1e": r1e, "r2e": r2e}
            if obs_rmm1 is None:
                obs_rmm1, obs_rmm2 = r1o, r2o
                n_total = n_leads

        if not trajectories:
            print(f"  No data found for {yyyymm} — skipping.")
            continue

        # common lead count
        n_leads = n_total

        # ── single combined figure ────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        draw_phase_diagram(ax)

        # observation (black)
        ax.plot(obs_rmm1, obs_rmm2, color="k", lw=2.0, zorder=5)
        ax.plot(obs_rmm1[0], obs_rmm2[0], "k*", ms=14, zorder=7, label="Observation")

        # each dataset — solid color, no colormap
        for idx, (ds_name, traj) in enumerate(trajectories.items()):
            st = _get_style(ds_name, idx)
            r1e, r2e = traj["r1e"][:n_leads], traj["r2e"][:n_leads]
            ax.plot(r1e, r2e, color=st["color"], lw=st.get("lw", 2.0), ls=st["ls"], zorder=4)
            ax.plot(r1e[0], r2e[0], st["marker"], ms=10, color=st["color"],
                    markeredgecolor="k", zorder=6, label=st.get("label", ds_name))

        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
        ax.set_title(f"MJO RMM Phase Space ({cfg.variable})\nInit: {init_str}",
                     fontsize=12, pad=10)
        fig.tight_layout()

        out_png = os.path.join(cfg.plot_dir, f"phase_space_{date_str}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_png}")

    print("\nStep 5 (phase space) complete.")


def _parse_args():
    p = argparse.ArgumentParser(description="Generate RMM phase space comparison plots.")
    p.add_argument("--config", default="config/pipeline_config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = PipelineConfig(args.config)
    make_phase_space_plots(cfg)
