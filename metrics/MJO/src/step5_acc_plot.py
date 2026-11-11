"""
src/step5_acc_plot.py — Bivariate ACC skill curve for all enabled datasets.

Loads RMM output files produced by step4 and computes:
  - Bivariate Anomaly Correlation Coefficient (ACC) vs lead time
  - Bivariate RMSE vs lead time

Produces a single comparison figure with one ACC curve per enabled dataset,
saved to cfg.plot_dir/bivariate_acc_comparison.png.

Usage:
  python src/step5_acc_plot.py [--config CFG]
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PipelineConfig
sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_style, model_color


# ── Skill metrics ─────────────────────────────────────────────────

def bivariate_acc(r1f, r2f, r1o, r2o):
    """(n_inits, n_leads) arrays → (n_leads,) ACC."""
    n_leads = r1f.shape[1]
    acc = np.full(n_leads, np.nan)
    for lead in range(n_leads):
        f1, f2 = r1f[:, lead], r2f[:, lead]
        o1, o2 = r1o[:, lead], r2o[:, lead]
        mask = ~(np.isnan(f1)|np.isnan(f2)|np.isnan(o1)|np.isnan(o2))
        if mask.sum() < 2:
            continue
        f1, f2, o1, o2 = f1[mask], f2[mask], o1[mask], o2[mask]
        num = np.sum(f1*o1 + f2*o2)
        den = np.sqrt(np.sum(f1**2 + f2**2) * np.sum(o1**2 + o2**2))
        acc[lead] = num / den if den > 0 else np.nan
    return acc


def bivariate_rmse(r1f, r2f, r1o, r2o):
    n_leads = r1f.shape[1]
    rmse = np.full(n_leads, np.nan)
    for lead in range(n_leads):
        f1, f2 = r1f[:, lead], r2f[:, lead]
        o1, o2 = r1o[:, lead], r2o[:, lead]
        mask = ~(np.isnan(f1)|np.isnan(f2)|np.isnan(o1)|np.isnan(o2))
        if mask.sum() < 1:
            continue
        rmse[lead] = np.sqrt(np.mean((f1[mask]-o1[mask])**2 + (f2[mask]-o2[mask])**2))
    return rmse


def _skilful_day(acc, leads):
    for l, a in zip(leads, acc):
        if not np.isnan(a) and a < 0.5:
            return l
    return None


# ── Data loading ──────────────────────────────────────────────────

def load_rmm_dir(rmm_dir: str, variable: str, dates: set = None):
    """Load RMM files from a directory, optionally filtered to a set of YYYYMMDD dates."""
    pattern   = os.path.join(rmm_dir, f"MJO_{variable}_Forecast_Init_????????.nc")
    rmm_files = sorted(glob.glob(pattern))
    if dates is not None:
        rmm_files = [f for f in rmm_files
                     if re.search(r"Init_(\d{8})", f) and
                        re.search(r"Init_(\d{8})", f).group(1) in dates]
    if not rmm_files:
        return None, None, None, None, 0

    max_lead = None
    r1_em, r2_em, r1_ob, r2_ob = [], [], [], []
    for fp in rmm_files:
        ds = xr.open_dataset(fp)
        r1e = ds["RMM1_emean"].values
        r2e = ds["RMM2_emean"].values
        r1o = ds["RMM1_obs"].values
        r2o = ds["RMM2_obs"].values
        if max_lead is None:
            max_lead = len(r1e)
        n = max_lead
        pad = lambda a: a[:n] if len(a) >= n else np.concatenate([a, np.full(n-len(a), np.nan)])
        r1_em.append(pad(r1e)); r2_em.append(pad(r2e))
        r1_ob.append(pad(r1o)); r2_ob.append(pad(r2o))
        ds.close()

    return (np.stack(r1_em), np.stack(r2_em),
            np.stack(r1_ob), np.stack(r2_ob), max_lead)


# ── Main plotting function ────────────────────────────────────────

COLORS = ["steelblue", "firebrick", "seagreen", "darkorange", "purple"]  # fallback only
MARKERS = ["o", "s", "^", "D", "v"]

# Config dataset key -> canonical model label (model_colors), mapped by src_dir per the
# dataset registry. ours_si_v3 reads predictions_0903, which is UC-S2S V2 (not V3);
# ours_si_v2 RMM output is byte-identical to ours_si_v3 on overlapping dates.
# ours_si_s2s (sampling_temperature_15_tg_01) is UC-S2S V1; ours_si_slow_v1
# (slow_si_infer_v1, exp16_nvidia_v3) is the same run as postprocess_slow = UC-S2S V3.
# Keys not listed here fall back to COLORS.
DS_LABELS = {
    "ours_si_s2s":   "UC-S2S V1",
    "fuxi_s2s":      "FuXi",
    "fuxi_s2s_15":   "FuXi",
    "amip_s2s":      "SI-S2S",
    "ours_si_v2":    "UC-S2S V2",
    "ours_si_v3":    "UC-S2S V2",
    "ours_si_slow_v1": "UC-S2S V3",
    "gencast":       "GenCast",
    "ngcm":          "NeuralGCM",
    "ngcm_campaign": "NeuralGCM",
}


def ds_style(ds_name: str, idx: int):
    """(legend label, ax.plot kwargs) for a config dataset key."""
    try:
        label = DS_LABELS[ds_name]
        return label, model_style(label)
    except KeyError:
        return ds_name, dict(color=COLORS[idx % len(COLORS)], linewidth=2)


def _find_rmm_dir(ds_name: str, ds_cfg, variable: str) -> str:
    """Return the RMM output dir for a dataset, trying 1.5° then 1.0°."""
    for res in ["1p5deg", "1p0deg"]:
        d = os.path.join(ds_cfg.out_dir, f"{variable}_{res}", "rmm_output")
        if glob.glob(os.path.join(d, f"MJO_{variable}_Forecast_Init_????????.nc")):
            return d
    return ""


def make_acc_plot(cfg: PipelineConfig) -> None:
    os.makedirs(cfg.plot_dir, exist_ok=True)

    # Collect all dates per dataset for reference (no mandatory fuxi dependency)
    all_dataset_dates = {}
    for ds_name, ds_cfg in cfg.datasets.items():
        d = _find_rmm_dir(ds_name, ds_cfg, cfg.variable)
        all_dataset_dates[ds_name] = {
            re.search(r"Init_(\d{8})", f).group(1)
            for f in glob.glob(os.path.join(d, f"MJO_{cfg.variable}_Forecast_Init_????????.nc"))
            if re.search(r"Init_(\d{8})", f)
        }
        print(f"{ds_name}: {len(all_dataset_dates[ds_name])} dates")

    fig_acc,  ax_acc  = plt.subplots(figsize=(11, 5))
    fig_rmse, ax_rmse = plt.subplots(figsize=(11, 5))

    # several config keys can share a model label; append the key to keep legends distinct
    label_count = {}
    for i, n in enumerate(cfg.datasets):
        lab = ds_style(n, i)[0]
        label_count[lab] = label_count.get(lab, 0) + 1

    any_data = False
    for idx, (ds_name, ds_cfg) in enumerate(cfg.datasets.items()):
        name, style = ds_style(ds_name, idx)
        if label_count[name] > 1:
            name = f"{name} [{ds_name}]"
        marker = MARKERS[idx % len(MARKERS)]
        rmm_dir = _find_rmm_dir(ds_name, ds_cfg, cfg.variable)
        if not rmm_dir:
            print(f"  No RMM files found for {ds_name} — skipping.")
            continue

        if ds_name == "fuxi_s2s":
            r1e, r2e, r1o, r2o, max_lead = load_rmm_dir(rmm_dir, cfg.variable, dates=None)
            if r1e is None:
                continue
            n = r1e.shape[0]
            leads = np.arange(1, max_lead + 1)
            acc  = bivariate_acc(r1e, r2e, r1o, r2o)
            rmse = bivariate_rmse(r1e, r2e, r1o, r2o)
            skill_day = _skilful_day(acc, leads)
            label = f"{name}  (n={n})" + (f"  →day {skill_day}" if skill_day else "")
            ax_acc.plot(leads, acc,  marker=marker, ms=4, label=label, **style)
            ax_rmse.plot(leads, rmse, marker=marker, ms=4, label=label, **style)
            print(f"  fuxi_s2s: {n} inits, skill horizon day {skill_day}")
        else:
            r1e, r2e, r1o, r2o, max_lead = load_rmm_dir(rmm_dir, cfg.variable, dates=None)
            if r1e is None:
                print(f"  No RMM files found for {ds_name} — skipping.")
                continue
            n_inits = r1e.shape[0]
            leads   = np.arange(1, max_lead + 1)
            acc  = bivariate_acc(r1e, r2e, r1o, r2o)
            rmse = bivariate_rmse(r1e, r2e, r1o, r2o)
            skill_day = _skilful_day(acc, leads)
            label = f"{name}  (n={n_inits})" + (f"  →day {skill_day}" if skill_day else "")
            ax_acc.plot(leads, acc,  marker=marker, ms=4, label=label, **style)
            ax_rmse.plot(leads, rmse, marker=marker, ms=4, label=label, **style)
            print(f"  {ds_name}: {n_inits} inits, skill horizon day {skill_day}")
        any_data = True

    if not any_data:
        print("No RMM data found — run step4 first.")
        return

    # ── ACC figure ────────────────────────────────────────────────
    ax_acc.axhline(0.5, color="grey", ls=":", lw=1.5, label="ACC = 0.5")
    ax_acc.axhline(0.0, color="k",    ls=":", lw=0.8)
    ax_acc.set(xlabel="Lead time (days)", ylabel="Bivariate ACC",
               title=f"MJO Bivariate ACC — {cfg.variable}",
               xlim=(1, None), ylim=(-0.15, 1.05))
    ax_acc.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax_acc.grid(alpha=0.3)
    fig_acc.tight_layout()
    acc_png = os.path.join(cfg.plot_dir, cfg.acc_plot_filename)
    fig_acc.savefig(acc_png, dpi=150, bbox_inches="tight")
    plt.close(fig_acc)
    print(f"Saved ACC plot: {acc_png}")

    # ── RMSE figure ───────────────────────────────────────────────
    ax_rmse.axhline(np.sqrt(2), color="grey", ls=":", lw=1.5,
                    label=r"RMSE=$\sqrt{2}$ (random)")
    ax_rmse.set(xlabel="Lead time (days)", ylabel="Bivariate RMSE",
                title=f"MJO Bivariate RMSE — {cfg.variable}",
                xlim=(1, None))
    ax_rmse.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax_rmse.grid(alpha=0.3)
    fig_rmse.tight_layout()
    rmse_stem = os.path.splitext(cfg.acc_plot_filename)[0]
    rmse_png  = os.path.join(cfg.plot_dir, f"{rmse_stem}_rmse.png")
    fig_rmse.savefig(rmse_png, dpi=150, bbox_inches="tight")
    plt.close(fig_rmse)
    print(f"Saved RMSE plot: {rmse_png}")

    print("\nStep 5 (ACC) complete.")


def _parse_args():
    p = argparse.ArgumentParser(description="Generate bivariate ACC comparison plot.")
    p.add_argument("--config", default="config/pipeline_config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = PipelineConfig(args.config)
    make_acc_plot(cfg)
