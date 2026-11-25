"""
Bivariate MJO ACC vs lead, grouped by the ERA5 MJO amplitude on the init day:
strong (amp >= 1.5), moderate (1.0 <= amp < 1.5), inactive (amp < 1.0).
Shading = 95% bootstrap CI over init dates; legend gives the first lead with ACC < 0.5.

Lead alignment: lead = valid date - init date (init from the filename). UC-S2S and
SI-S2S files start at lead 0, FuXi and NeuralGCM at lead 1, so array position is
NOT lead; everything is placed on leads 1..MAX_LEAD here.

Grouping: one common definition for all models, the ERA5 chi200-RMM amplitude on the
init day (looked up by date from the obs stored in the chi200 RMM files), so FuXi and
FuXi (OLR) use identical groups. Each model is verified against its own ERA5 RMM
(chi200-based, or OLR-based for FuXi (OLR): chi250_eval 1.5 deg, 25 members).

Output: /glade/work/bgong/MJO_results/plots/acc_by_amplitude_group.png
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
import pandas as pd
import xarray as xr

sys.path.insert(0, "/glade/work/bgong/MJO_modular")
sys.path.insert(0, "/glade/work/bgong")
from src.step5_acc_plot import bivariate_acc, _skilful_day
from model_colors import model_style, model_color

R = "/glade/campaign/univ/uchi0018/bing/mjo"
N_BOOT, CI = 1000, 95
AMP_STRONG, AMP_INACTIVE = 1.5, 1.0
MAX_LEAD = 40
OUT = "/glade/work/bgong/MJO_results/plots/acc_by_amplitude_group.png"
OBS_RMM = "/glade/work/bgong/MJO_results/obs/chi200_1p0deg/MJO_obs_created.nc"
rng = np.random.default_rng(42)


def rmm_glob(ds, var="chi200"):
    return f"{R}/{ds}/{var}_1p0deg/rmm_output/MJO_{var}_Forecast_Init_????????.nc"


def date_of(f):
    return re.search(r"\d{8}", os.path.basename(f)).group()


def get_dates(pattern):
    return {date_of(f) for f in glob.glob(pattern)}


common_uc_ngcm = get_dates(rmm_glob("ours_si_v3")) & get_dates(rmm_glob("ngcm_campaign"))

# FuXi with standard OLR-based RMM: earlier chi250_eval run on FuXi's native 1.5 deg grid,
# 25 members, files start at lead 1 (sign checked: day-1 bivariate corr with ERA5 = +0.96).
# Restricted to the FuXi chi200 init dates so the two FuXi curves are like-for-like.
FUXI_OLR = "/glade/work/bgong/MJO/chi250_eval/rmm_output_olr_1p5deg/MJO_olr_1p5deg_Forecast_Init_????????.nc"
fuxi_dates = get_dates(rmm_glob("fuxi_s2s"))

# (label, pattern, date filter, filter always applied (else only with --common-dates),
#  chi200 index -> align sign to shared ERA5)
# Pairs always share init dates: FuXi (OLR) uses the FuXi chi200 dates, and UC-S2S V2 / NeuralGCM
# use their common dates. SI-S2S uses all its dates.
MODELS = [
    ("FuXi",       rmm_glob("fuxi_s2s"),          None,           False, True),
    ("FuXi (OLR)", FUXI_OLR,                      fuxi_dates,     True,  False),
    ("FuXi (200 hPa extrap.)", rmm_glob("fuxi_s2s_ext200"), fuxi_dates, True, True),
    ("SI-S2S",     rmm_glob("amip_s2s"),          None,           False, True),
    ("UC-S2S V2",  rmm_glob("ours_si_v3"),        common_uc_ngcm, True,  True),
    ("NeuralGCM",  rmm_glob("ngcm_campaign"),     common_uc_ngcm, True,  True),
]


def load_arrays(pattern, date_filter=None, ref=None):
    """-> inits (list of str), r1e, r2e, r1o, r2o each (n_init, MAX_LEAD) on leads 1..MAX_LEAD.
    If ref (ERA5 RMM table by date) is given, a file whose stored ERA5 RMM anti-correlates
    with ref was projected on the sign-flipped EOF set (ngcm_campaign reruns); all its
    RMM1/RMM2 are multiplied by -1 so every file shares one sign convention."""
    files = sorted(glob.glob(pattern))
    if date_filter is not None:
        files = [f for f in files if date_of(f) in date_filter]
    inits, arrs = [], {k: [] for k in ("r1e", "r2e", "r1o", "r2o")}
    n_flip = [0]
    for fp in files:
        d = date_of(fp)
        init = pd.Timestamp(d)
        with xr.open_dataset(fp) as ds:
            if ds.time.dtype.kind == "M":
                lead = ((pd.to_datetime(ds.time.values) - init) / pd.Timedelta("1D")).astype(int).values
            else:  # raw lead index, no units (one ours_si_v2-era file)
                lead = np.arange(ds.sizes["time"])
            src = dict(r1e=ds.RMM1_emean.values, r2e=ds.RMM2_emean.values,
                       r1o=ds.RMM1_obs.values, r2o=ds.RMM2_obs.values)
            if ref is not None and ds.time.dtype.kind == "M":
                rr = ref.reindex(pd.to_datetime(ds.time.values))
                dot = np.nansum(src["r1o"] * rr.r1.values + src["r2o"] * rr.r2.values)
                if dot < 0:
                    src = {k: -v for k, v in src.items()}
                    n_flip[0] += 1
        keep = (lead >= 1) & (lead <= MAX_LEAD)
        for k, v in src.items():
            row = np.full(MAX_LEAD, np.nan)
            row[lead[keep] - 1] = v[keep]
            arrs[k].append(row)
        inits.append(d)
    if n_flip[0]:
        print(f"  sign-aligned {n_flip[0]} file(s) to the shared ERA5 RMM convention")
    return inits, {k: np.stack(v) for k, v in arrs.items()}


def remove_mean_bias(inits, a):
    """Subtract the lead-dependent mean error of the ensemble mean, estimated
    leave-one-year-out (the bias applied to a year never uses that year)."""
    yrs = np.array([d[:4] for d in inits])
    out = dict(a)
    for k, o in (("r1e", "r1o"), ("r2e", "r2o")):
        c = a[k].copy()
        for y in np.unique(yrs):
            m = yrs == y
            if (~m).sum() == 0:
                continue
            c[m] -= np.nanmean(a[k][~m] - a[o][~m], 0)
        out[k] = c
    return out


def era5_rmm_table():
    """ERA5 chi200 RMM (r1, r2) by date, pooled from the obs in the chi200 RMM files."""
    recs = []
    # Only runs projected on the original EOF set. Excluded (sign-flipped EOFs from the
    # 2026-09-26 obs rewrite): ngcm_campaign reruns, fuxi_s2s_ext200, and amip_s2s
    # (20-member rerun 2026-10-03). Pooling mixed signs cancels the amplitude.
    for ds_name in ("ours_si_v3", "fuxi_s2s"):
        for fp in glob.glob(rmm_glob(ds_name)):
            with xr.open_dataset(fp) as ds:
                if ds.time.dtype.kind != "M":
                    continue
                recs.append(pd.DataFrame({"t": pd.to_datetime(ds.time.values),
                                          "r1": ds.RMM1_obs.values, "r2": ds.RMM2_obs.values,
                                          "src": ds_name}))
    df = pd.concat(recs).dropna()
    spread = df.groupby("t")[["r1", "r2"]].agg(lambda x: x.max() - x.min()).max().max()
    print(f"ERA5 chi200 RMM: max disagreement between files on the same date = {spread:.2e}")
    if spread > 1e-3:
        raise RuntimeError("ERA5 RMM differs between source files (mixed EOF sign conventions?)")
    return df.groupby("t")[["r1", "r2"]].mean()


def era5_init_amplitude():
    g = era5_rmm_table()
    return np.hypot(g.r1, g.r2)


def bootstrap_acc(r1e, r2e, r1o, r2o):
    n = r1e.shape[0]
    leads = np.arange(1, MAX_LEAD + 1)
    acc = bivariate_acc(r1e, r2e, r1o, r2o)
    sd = _skilful_day(acc, leads)
    boot_acc = np.zeros((N_BOOT, MAX_LEAD))
    boot_sd = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a = bivariate_acc(r1e[idx], r2e[idx], r1o[idx], r2o[idx])
        boot_acc[i] = a
        s = _skilful_day(a, leads)
        boot_sd[i] = s if s is not None else leads[-1]
    lo, hi = (100 - CI) / 2, 100 - (100 - CI) / 2
    return (acc, np.nanpercentile(boot_acc, lo, axis=0), np.nanpercentile(boot_acc, hi, axis=0),
            sd, np.percentile(boot_sd, lo), np.percentile(boot_sd, hi), leads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-dates", action="store_true",
                    help="old behaviour: UC-S2S V2 & NeuralGCM on their common dates, "
                         "FuXi (OLR) on the FuXi chi200 dates (default: every model uses all its dates)")
    ap.add_argument("--bias-correct", action="store_true",
                    help="remove lead-dependent mean RMM error (leave-one-year-out)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="subset of model labels to plot (default: all in MODELS)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = era5_rmm_table()
    amp_by_date = np.hypot(ref.r1, ref.r2)
    # fallback for dates no chi200 forecast file covers (e.g. FuXi (OLR) 2017-2018): the
    # full ERA5 record from step3. It uses the sign-flipped EOF set (rewritten 2026-09-26),
    # but amplitude is sign-independent: median |diff| 0.03, same group on 98% of days.
    with xr.open_dataset(OBS_RMM) as o:
        amp_full = pd.Series(np.hypot(o.RMM1_obs.values, o.RMM2_obs.values),
                             index=pd.to_datetime(o.time.values))
    amp_by_date = amp_by_date.combine_first(amp_full)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    titles = [f"Strong MJO (amp ≥ {AMP_STRONG})",
              f"Moderate MJO ({AMP_INACTIVE} ≤ amp < {AMP_STRONG})",
              f"Inactive MJO (amp < {AMP_INACTIVE})"]
    for ax, title in zip(axes, titles):
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xlabel("Lead time (days)", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(1, MAX_LEAD)
        ax.set_ylim(0.1, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Bivariate ACC", fontsize=12)

    for label, pattern, date_filter, always, chi in MODELS:
        if args.models and label not in args.models:
            continue
        if not glob.glob(pattern):
            print(f"{label}: no files at {pattern}, skipping")
            continue
        inits, a = load_arrays(pattern, date_filter if (args.common_dates or always) else None,
                               ref if chi else None)
        if args.bias_correct:
            a = remove_mean_bias(inits, a)
        init_amp = amp_by_date.reindex(pd.to_datetime(inits)).values
        ok = np.isfinite(init_amp)
        groups = [ok & (init_amp >= AMP_STRONG),
                  ok & (init_amp >= AMP_INACTIVE) & (init_amp < AMP_STRONG),
                  ok & (init_amp < AMP_INACTIVE)]
        print(f"{label}: total={len(inits)} (no init-day ERA5: {(~ok).sum()}), "
              f"strong={groups[0].sum()}, moderate={groups[1].sum()}, inactive={groups[2].sum()}")
        for ax, m in zip(axes, groups):
            if m.sum() < 5:
                continue
            acc, lo, hi, sd, sdlo, sdhi, leads = bootstrap_acc(a["r1e"][m], a["r2e"][m],
                                                                a["r1o"][m], a["r2o"][m])
            ax.fill_between(leads, lo, hi, color=model_color(label), alpha=0.15)
            sd_txt = f"day {sd}" if sd is not None else f">{MAX_LEAD}"
            ax.plot(leads, acc, label=f"{label} (n={m.sum()})  →{sd_txt} [{sdlo:.0f}–{sdhi:.0f}]",
                    **model_style(label))

    for ax in axes:
        ax.legend(fontsize=8, title="skill day [95% CI]")
    shown = [m[0] for m in MODELS if not args.models or m[0] in args.models]
    fuxi_set = [m for m in ("FuXi", "FuXi (OLR)", "FuXi (200 hPa extrap.)") if m in shown]
    parts = []
    if len(fuxi_set) > 1:
        parts.append(" & ".join(fuxi_set) + " on the same init dates")
    if {"UC-S2S V2", "NeuralGCM"} <= set(shown):
        parts.append("UC-S2S V2 & NeuralGCM on shared init dates")
    rest = [m for m in shown if (m not in fuxi_set or len(fuxi_set) == 1)
            and not ({"UC-S2S V2", "NeuralGCM"} <= set(shown) and m in ("UC-S2S V2", "NeuralGCM"))]
    if rest:
        parts.append(", ".join(rest) + " all their dates")
    sub = "common init dates" if args.common_dates else "; ".join(parts)
    if args.bias_correct:
        sub += ", lead-dependent mean RMM bias removed (leave-one-year-out)"
    fig.suptitle("MJO ACC by Initial Amplitude — Strong / Moderate / Inactive\n"
                 f"groups: ERA5 amplitude on the init day; {sub}", fontsize=12)
    fig.tight_layout()
    out = args.out or OUT
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(OUT), out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
