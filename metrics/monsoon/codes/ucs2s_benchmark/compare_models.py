#!/usr/bin/env python
"""
Multi-model comparison tables and figures (UC-S2S plain / SST-anomaly and FuXi-S2S, raw and
drift/bias-corrected, plus NGCM).

Reads the outputs of run_onset_core.py and raw_forecast_metrics.py and writes to
<OUT_ROOT>/comparison/ :

  onset_summary_<mode>.csv        CMZ MAE (mean +- SE over years), FAR, MR for days
                                  1-15 / 16-30, overall AUC, fair BSS, fair RPSS
  onset_binned_<mode>.csv         per-bin fair BSS and AUC (model and climatology)
  fig_onset_<mode>.png/.pdf       (a) MAE (b) FAR (c) MR (d) BSS per bin (e) AUC per bin
                                  (f) reliability (15 d)
  fig_scorecard_onset_weekly_<mode>  onset fair BSS (%) and AUC per WEEKLY bin, rows = climatology +
                                  models (style of Masiwal et al. 2026, Fig. 3a,b); _5day_30 = 5-day bins
  fig_scorecard_<target>_<tag>    weekly CRPSS (%) / ACC scorecards of the raw-forecast metrics
  raw_summary_<tag>.csv           weekly ACC / CRPSS / bias / spread ratio, all targets
  fig_rain_cmz_<tag>              CMZ rainfall bias, ACC, CRPSS, spread ratio vs lead
  fig_rain_maps_<tag>             weekly bias (%) and ACC maps, weeks 1-4
  fig_rain_stats_<tag>            wet-day / onset-criterion frequency, daily quantiles
  fig_indices_<tag>               circulation + OLR index bias, ACC, CRPSS vs lead
  fig_mok_<tag>                   LLJ onset-over-Kerala error vs lead (native only)
  lead_alignment_<tag>.csv        lead-alignment sanity check for every model

The CMZ aggregation of the MAE/FAR/MR maps reproduces plot_spatial_metrics() in
the core script (MAE: CMZ mean per year, then mean +- SE across years).

Usage: python compare_models.py --init-mode common native
"""
import argparse
import json

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

import config as C
import common as U

# Colour = model (validated palette, fixed slot order: blue, orange, aqua, yellow, magenta);
# line style / marker fill = raw (solid, filled) vs drift/bias-corrected (dashed, open; hatched bars)
_BASE = {"ucs2s_plain": ("#2a78d6", "o"), "ucs2s_sstanom": ("#eb6834", "s"), "ngcm": ("#1baf7a", "^"),
         "fuxi": ("#eda100", "D"), "ngcm_legacy": ("#e87ba4", "v")}
STYLE = {}
for _k, (_c, _mk) in _BASE.items():
    STYLE[_k] = dict(color=_c, marker=_mk, ls=":" if _k == "ngcm_legacy" else "-")
    STYLE[f"{_k}_bc"] = dict(color=_c, marker=_mk, ls="--", mfc="white")
OBS_STYLE = dict(color="#0b0b0b", ls="-", lw=1.2)
INK2 = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "savefig.dpi": 300, "axes.axisbelow": True})


def lab(k):
    return C.MODELS[k]["label"]


def letter(n):
    """panel letter: a..z, then aa, ab, ..."""
    a = "abcdefghijklmnopqrstuvwxyz"
    return a[n] if n < 26 else a[n // 26 - 1] + a[n % 26]


def save(fig, stem):
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{stem}.png")


_LAST_LEGEND = {}


def footnote(fig, text, y=-0.02):
    """grey one-line footnote; placed below the figure legend when there is one (avoids overlap)."""
    leg = _LAST_LEGEND.pop(id(fig), None)
    if leg is not None:
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        y = min(y, bb.y0 - 0.01)
    fig.text(0.5, y, text, ha="center", va="top", fontsize=8, color="0.35")


def model_legend(fig, keys, extra=(), y=-0.005, ncol=None):
    h = [Line2D([], [], label=lab(k), **{kk: v for kk, v in STYLE[k].items()}, markersize=6) for k in keys]
    h += list(extra)
    _LAST_LEGEND[id(fig)] = fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y),
                                       ncol=ncol or min(len(h), 4), frameon=False)


# ============================================================================
# Onset core
# ============================================================================
def cmz_onset_stats(nc):
    ds = xr.open_dataset(nc)
    cmz = U.cmz_mask(ds.lat.values, ds.lon.values).values
    far = float(np.nanmean(ds["false_alarm_rate"].values[cmz] * 100))
    mr = float(np.nanmean(ds["miss_rate"].values[cmz] * 100))
    yearly = [float(np.nanmean(ds[v].values[cmz])) for v in ds.data_vars
              if v.startswith("mae_") and v[4:].isdigit()]
    yearly = [x for x in yearly if np.isfinite(x)]
    mae = np.mean(yearly) if yearly else np.nan
    se = np.std(yearly, ddof=1) / np.sqrt(len(yearly)) if len(yearly) > 1 else np.nan
    return dict(mae=mae, mae_se=se, far=far, mr=mr, n_years=len(yearly))


def onset_tables(mode, keys):
    rows, binned, rel = [], [], {}
    for k in keys:
        d = C.OUT_ROOT / "onset" / mode / k
        if not d.exists():
            continue
        name = f"{k}_{mode}"
        info = json.load(open(d / "run_info.json"))
        r = dict(model=k, label=lab(k), init_mode=mode, years=" ".join(map(str, info["years"])),
                 n_inits=sum(info["n_inits"].values()))
        for tag, short in (("1_15_day", "d1_15"), ("16_30_day", "d16_30")):
            f = d / f"results_{name}_{tag}_MOK.nc"
            if f.exists():
                for kk, v in cmz_onset_stats(f).items():
                    r[f"{kk}_{short}"] = v
        wk_day = C.ONSET_WEEK_BINS[-1][1]
        for mfd, sfx, col in ((15, "", "15"), (30, "", "30"), (wk_day, "_weekly", "wk")):
            f = d / f"overall_skill_scores_{name}{sfx}_{mfd}day.csv"
            if f.exists():
                o = pd.read_csv(f).set_index("Metric")["Score"]
                r[f"auc_{col}"] = o.get("AUC")
                r[f"auc_clim_{col}"] = o.get("AUC_Climatology")
                r[f"bss_{col}"] = o.get("Fair Brier Skill Score")
                r[f"rpss_{col}"] = o.get("Fair RPS Skill Score")
            f = d / f"binned_skill_scores_{name}{sfx}_{mfd}day.csv"
            if f.exists():
                b = pd.read_csv(f)
                b.insert(0, "bins", "weekly" if sfx else f"5day_{mfd}")
                b.insert(0, "model", k)
                binned.append(b)
            f = d / f"reliability_results_{mfd}day_mok.csv"
            if f.exists():
                rel[(k, mfd)] = pd.read_csv(f)
        rows.append(r)
    return pd.DataFrame(rows), (pd.concat(binned) if binned else pd.DataFrame()), rel


def fig_onset(mode, summ, binned, rel, out):
    keys = list(summ.model)
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.0))
    ax = ax.ravel()
    width = 0.8 / len(keys)
    x = np.arange(2)
    for p, (col, ylab, letter) in enumerate((("mae", "CMZ MAE (days)", "a"),
                                             ("far", "CMZ false alarm rate (%)", "b"),
                                             ("mr", "CMZ miss rate (%)", "c"))):
        a = ax[p]
        for i, k in enumerate(keys):
            s = summ.set_index("model").loc[k]
            vals = [s.get(f"{col}_d1_15", np.nan), s.get(f"{col}_d16_30", np.nan)]
            err = [s.get("mae_se_d1_15", np.nan), s.get("mae_se_d16_30", np.nan)] if col == "mae" else None
            a.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width * 0.92, yerr=err,
                  color=STYLE[k]["color"], edgecolor="white", linewidth=1.0,
                  hatch="////" if k.endswith("_bc") else None,
                  error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2))
        a.set_xticks(x, ["days 1–15", "days 16–30"])
        a.set_ylabel(ylab)
        a.set_title(f"({letter})", loc="left")
        a.grid(axis="x", visible=False)
    for p, (col, clim, ylab, letter) in enumerate((("Fair_Brier_Skill_Score", None, "Fair BSS", "d"),
                                                   ("AUC", "AUC_Climatology", "AUC", "e"))):
        a = ax[3 + p]
        if len(binned):
            bw = binned[binned.bins == "weekly"]
            for k in keys:
                bk = bw[bw.model == k]
                if not len(bk):
                    continue
                xl = [f"wk{i + 1}" for i in range(len(bk))]
                a.plot(xl, bk[col], **STYLE[k], markersize=5)
                if clim and k == keys[0]:
                    a.plot(xl, bk[clim], color="0.55", ls=":", marker="x", markersize=5, label="IMD climatology")
        if col.startswith("Fair"):
            a.axhline(0, color="0.4", lw=0.8)
        a.set_xlabel("Onset lead week (days 1–7 … 22–28)")
        a.set_ylabel(ylab)
        a.set_title(f"({letter})", loc="left")
    a = ax[5]
    a.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--")
    for k in keys:
        r = rel.get((k, 15))
        if r is not None:
            a.plot(r["Mean_Forecast_Prob"], r["Observed_Frequency"], **STYLE[k], markersize=4)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_xlabel("Forecast probability")
    a.set_ylabel("Observed frequency")
    a.set_title("(f) reliability, days 1–15", loc="left")
    fig.tight_layout()
    extra = [Line2D([], [], color="0.55", ls=":", marker="x", label="IMD climatology")]
    model_legend(fig, keys, extra)
    yrs = summ.years.iloc[0]
    footnote(fig, f"CMZ (2°) · inits 1 May–31 Jul · {mode} init dates · {C.MEM_NUM} members · "
                  f"years {yrs} · MAE bars ±1 SE across years · hatched / dashed = bias-corrected · (d,e) weekly bins", y=-0.05)
    save(fig, out / f"fig_onset_{mode}")


# ============================================================================
# Scorecards (style of Masiwal et al. 2026, arXiv:2602.03767, Fig. 3a,b)
# ============================================================================
BSS_CMAP = matplotlib.colors.BoundaryNorm(np.linspace(-40, 40, 17), 256, clip=True)
AUC_CMAP = matplotlib.colors.BoundaryNorm(np.linspace(0.5, 1.0, 11), 256, clip=True)


def _scorecard(ax, M, rows, cols, cmap, norm, fmt, title):
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                r, g, bl, _ = plt.get_cmap(cmap)(norm(v))
                dark = 0.2126 * r + 0.7152 * g + 0.0722 * bl < 0.5
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left")
    return im


def fig_onset_scorecard(mode, summ, binned, out, bins="weekly"):
    """(a) fair BSS (%) and (b) AUC per onset bin; rows = climatology + models."""
    if not len(binned):
        return
    b = binned[binned.bins == bins]
    keys = [k for k in summ.model if k in set(b.model)]
    if not keys:
        return
    ref = b[b.model == keys[0]]
    cols = (["Week %d\n(%s)" % (i + 1, x.replace("Days ", "")) for i, x in enumerate(ref.Bin)]
            if bins == "weekly" else [x.replace("Days ", "") for x in ref.Bin])
    rows = ["Climatology"] + [lab(k) for k in keys]
    bss = [np.zeros(len(cols))] + [b[b.model == k].Fair_Brier_Skill_Score.values * 100 for k in keys]
    auc = [ref.AUC_Climatology.values] + [b[b.model == k].AUC.values for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(4.4 + 1.25 * len(cols) * 2, 0.42 * len(rows) + 1.9))
    im0 = _scorecard(ax[0], np.array(bss, float), rows, cols, "RdBu", BSS_CMAP, "{:.1f}", "(a) Fair Brier skill score (%)")
    im1 = _scorecard(ax[1], np.array(auc, float), rows, cols, "Blues", AUC_CMAP, "{:.2f}", "(b) AUC")
    ax[1].set_yticklabels([])
    for a, im in zip(ax, (im0, im1)):
        cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.06, pad=0.16 if bins == "weekly" else 0.1)
        cb.ax.tick_params(labelsize=8)
        a.set_xlabel("Onset forecast window (days)")
    fig.tight_layout()
    yrs = summ.years.iloc[0]
    footnote(fig, f"CMZ (2°) · inits 1 May–31 Jul, {mode} init dates · {C.MEM_NUM} members · years {yrs} · "
                  f"BSS vs IMD climatological onset ensemble (climatology row = {lab(keys[0])} init dates)", y=0.0)
    save(fig, out / f"fig_scorecard_onset_{bins}_{mode}")


def fig_raw_scorecard(tag, sc, keys, out, target="rain_cmz", weeks=(1, 2, 3, 4, 5, 6)):
    w = sc[(sc["window"] == "weekly") & (sc.target == target) & sc.lead.isin(weeks)]
    keys = [k for k in keys if k in set(w.model)]
    if not keys:
        return
    cols = [f"Week {i}" for i in weeks]
    rows = [lab(k) for k in keys]
    crpss = np.array([[w[(w.model == k) & (w.lead == i)].crpss.mean() * 100 for i in weeks] for k in keys])
    acc = np.array([[w[(w.model == k) & (w.lead == i)].acc.mean() for i in weeks] for k in keys])
    fig, ax = plt.subplots(1, 2, figsize=(4.4 + 1.0 * len(cols) * 2, 0.42 * len(rows) + 1.8))
    im0 = _scorecard(ax[0], crpss, rows, cols, "RdBu", BSS_CMAP, "{:.0f}", "(a) CRPSS (%)")
    accn = matplotlib.colors.BoundaryNorm(np.linspace(0, 1, 11), 256, clip=True)
    im1 = _scorecard(ax[1], acc, rows, cols, "Blues", accn, "{:.2f}", "(b) anomaly correlation")
    ax[1].set_yticklabels([])
    for a, im in zip(ax, (im0, im1)):
        cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.06, pad=0.12)
        cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    what = "IMD 2° CMZ-mean rainfall" if target == "rain_cmz" else f"{C.INDEX_INFO[target]['label']} vs ERA5"
    footnote(fig, f"{what} · weekly means (week w = lead days 7w−6 … 7w) · {tag} · "
                  f"CRPSS vs 1991–2018 climatological ensemble", y=0.0)
    save(fig, out / f"fig_scorecard_{target}_{tag}")


# ============================================================================
# Raw metrics
# ============================================================================
def load_scores(tag, keys):
    dfs = []
    for k in keys:
        f = C.OUT_ROOT / "raw_metrics" / tag / k / "scores.csv"
        if f.exists():
            dfs.append(pd.read_csv(f))
    return pd.concat(dfs) if dfs else pd.DataFrame()


def line_ci(a, x, y, lo, hi, k, **kw):
    a.plot(x, y, color=STYLE[k]["color"], ls=STYLE[k]["ls"], **kw)
    if lo is not None:
        a.fill_between(x, lo, hi, color=STYLE[k]["color"], alpha=0.12, lw=0)


def fig_rain_cmz(tag, sc, keys, out):
    r = sc[sc.target == "rain_cmz"]
    if not len(r):
        return
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.6))
    ax = ax.ravel()
    for k in keys:
        d = r[(r.model == k) & (r["window"] == "daily")].sort_values("lead")
        w = r[(r.model == k) & (r["window"] == "weekly")].sort_values("lead")
        if not len(d):
            continue
        line_ci(ax[0], d.lead, d.bias, None, None, k)
        line_ci(ax[1], d.lead, d.acc, d.acc_lo, d.acc_hi, k)
        off = (keys.index(k) - (len(keys) - 1) / 2) * 0.12
        ax[2].errorbar(w.lead + off, w.crpss, yerr=[w.crpss - w.crpss_lo, w.crpss_hi - w.crpss],
                       color=STYLE[k]["color"], marker=STYLE[k]["marker"], ls=STYLE[k]["ls"],
                       markersize=5, capsize=2, elinewidth=0.8)
        ax[3].plot(w.lead + off, w.spread_ratio, color=STYLE[k]["color"], marker=STYLE[k]["marker"],
                   ls=STYLE[k]["ls"], markersize=5)
    ax[0].axhline(0, color="0.4", lw=0.8)
    ax[0].set(xlabel="Lead (days)", ylabel="Bias (mm day$^{-1}$)")
    ax[0].set_title("(a) daily CMZ rainfall bias", loc="left")
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set(xlabel="Lead (days)", ylabel="ACC")
    ax[1].set_title("(b) daily anomaly correlation", loc="left")
    ax[2].axhline(0, color="0.4", lw=0.8)
    ax[2].set(xlabel="Lead week", ylabel="CRPSS")
    ax[2].set_title("(c) weekly-mean CRPSS vs IMD climatology", loc="left")
    ax[3].axhline(1, color="0.4", lw=0.8)
    ax[3].set(xlabel="Lead week", ylabel="Spread / RMSE")
    ax[3].set_title("(d) weekly-mean spread–error ratio", loc="left")
    for a in ax[2:]:
        a.set_xticks(range(1, 7))
    fig.tight_layout()
    model_legend(fig, [k for k in keys if k in set(r.model)])
    footnote(fig, f"IMD 2° CMZ mean (cells valid in all years) · {tag.replace('_', ' inits, months ')} · "
                  f"ACC vs 1991–2018 IMD day-of-year climatology · shading / bars: 90% year-block bootstrap",
             y=-0.05)
    save(fig, out / f"fig_rain_cmz_{tag}")


def india_outline():
    g = gpd.read_file(C.SHAPEFILE)
    return g.boundary


def fig_rain_maps(tag, keys, out):
    files = {k: C.OUT_ROOT / "raw_metrics" / tag / k / "rain_maps_weekly.nc" for k in keys}
    files = {k: f for k, f in files.items() if f.exists()}
    if not files:
        return
    outline = india_outline()
    weeks = [1, 2, 3, 4]
    for var, cmap, vmin, vmax, clab in (("bias_pct", "BrBG", -100, 100, "Bias (% of IMD)"),
                                        ("acc", "RdBu_r", -0.8, 0.8, "ACC")):
        fig, ax = plt.subplots(len(files), len(weeks), figsize=(2.55 * len(weeks), 2.55 * len(files)),
                               squeeze=False, sharex=True, sharey=True)
        n = 0
        for i, (k, f) in enumerate(files.items()):
            ds = xr.open_dataset(f)
            lon, lat = ds.lon.values, ds.lat.values
            le = np.r_[lon - 1, lon[-1] + 1]
            ae = np.r_[lat - 1, lat[-1] + 1]
            for j, w in enumerate(weeks):
                a = ax[i, j]
                pm = a.pcolormesh(le, ae, ds[var].sel(week=w).values, cmap=cmap, vmin=vmin, vmax=vmax)
                outline.plot(ax=a, color="0.15", lw=0.5)
                a.plot(C.CMZ_POLY_LON, C.CMZ_POLY_LAT, color="0.15", lw=0.9)
                a.set_xlim(66, 100)
                a.set_ylim(6, 38)
                a.set_aspect("equal")
                a.grid(False)
                a.set_title(f"({letter(n)}) week {w}" if i == 0 else
                            f"({letter(n)})", loc="left")
                n += 1
                if j == 0:
                    a.set_ylabel(lab(k))
        H = fig.get_figheight()
        fig.tight_layout(rect=(0, 0.95 / H, 1, 1))
        cax = fig.add_axes([0.32, 0.42 / H, 0.36, 0.12 / H])
        cb = fig.colorbar(pm, cax=cax, orientation="horizontal")
        cb.set_label(clab, fontsize=9)
        footnote(fig, f"weekly-mean rainfall vs IMD 2° · {tag} · week w = lead days 7w−6 … 7w · "
                      f"outline: CMZ", y=0.0)
        save(fig, out / f"fig_rain_maps_{var}_{tag}")


def fig_rain_stats(tag, keys, out):
    st = {k: C.OUT_ROOT / "raw_metrics" / tag / k / "rain_daily_stats.csv" for k in keys}
    st = {k: pd.read_csv(f) for k, f in st.items() if f.exists()}
    if not st:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    first = True
    for k, d in st.items():
        ax[0].plot(d.lead, d.wet_freq_f * 100, color=STYLE[k]["color"], ls=STYLE[k]["ls"])
        ax[1].plot(d.lead, d.onset_crit_freq_f * 100, color=STYLE[k]["color"], ls=STYLE[k]["ls"])
        q = d.dropna(subset=["q90_f"])
        ax[2].plot(q.lead, q.q90_f, color=STYLE[k]["color"], ls=STYLE[k]["ls"], marker=STYLE[k]["marker"], ms=4)
        ax[2].plot(q.lead, q.q99_f, color=STYLE[k]["color"], ls=STYLE[k]["ls"], marker=STYLE[k]["marker"],
                   ms=4, mfc="white")
        # IMD at the same valid dates (identical across models in common mode)
        ax[0].plot(d.lead, d.wet_freq_o * 100, color=STYLE[k]["color"], lw=0.7, alpha=0.6, ls=(0, (1, 1)))
        ax[1].plot(d.lead, d.onset_crit_freq_o * 100, color=STYLE[k]["color"], lw=0.7, alpha=0.6, ls=(0, (1, 1)))
        if first:
            ax[2].plot(q.lead, q.q90_o, **OBS_STYLE, marker="o", ms=3)
            ax[2].plot(q.lead, q.q99_o, **OBS_STYLE, marker="o", ms=3, mfc="white")
            first = False
    ax[0].set(xlabel="Lead (days)", ylabel="Days with > 1 mm (%)")
    ax[0].set_title("(a) wet-day frequency, CMZ cells", loc="left")
    ax[1].set(xlabel="Lead (days)", ylabel="Days meeting onset test (%)")
    ax[1].set_title("(b) onset-criterion exceedance", loc="left")
    ax[2].set(xlabel="Lead (days)", ylabel="Daily rainfall (mm day$^{-1}$)")
    ax[2].set_title("(c) 90th (filled) / 99th (open) percentile", loc="left")
    fig.tight_layout()
    extra = [Line2D([], [], color="0.3", lw=0.7, ls=(0, (1, 1)), label="IMD, same valid dates (thin)"),
             Line2D([], [], **OBS_STYLE, marker="o", ms=3, label="IMD")]
    model_legend(fig, list(st), extra, y=-0.02)
    footnote(fig, f"2° CMZ cells pooled · onset test = day 1 > 1 mm and 5-day total > MWmean threshold · {tag}",
             y=-0.11)
    save(fig, out / f"fig_rain_stats_{tag}")


def fig_indices(tag, sc, keys, out):
    targets = [t for t in ("wy", "imi", "llj", "mhi", "olr_cmz") if t in set(sc.target)]
    if not targets:
        return
    fig, ax = plt.subplots(len(targets), 3, figsize=(11.5, 2.35 * len(targets)), squeeze=False)
    n = 0
    for i, t in enumerate(targets):
        r = sc[sc.target == t]
        for k in keys:
            d = r[(r.model == k) & (r["window"] == "daily")].sort_values("lead")
            w = r[(r.model == k) & (r["window"] == "weekly")].sort_values("lead")
            if not len(d):
                continue
            ax[i, 0].plot(d.lead, d.bias, color=STYLE[k]["color"], ls=STYLE[k]["ls"])
            line_ci(ax[i, 1], d.lead, d.acc, d.acc_lo, d.acc_hi, k)
            off = (keys.index(k) - (len(keys) - 1) / 2) * 0.12
            ax[i, 2].errorbar(w.lead + off, w.crpss, yerr=[w.crpss - w.crpss_lo, w.crpss_hi - w.crpss],
                              color=STYLE[k]["color"], marker=STYLE[k]["marker"], ls=STYLE[k]["ls"],
                              markersize=4, capsize=2, elinewidth=0.8)
        units = C.INDEX_INFO[t]["units"].replace("m s-1", "m s$^{-1}$").replace("W m-2", "W m$^{-2}$")
        ax[i, 0].set_ylabel(f"{t}\nbias ({units})")
        ax[i, 1].set_ylabel("ACC")
        ax[i, 2].set_ylabel("CRPSS (weekly)")
        ax[i, 2].set_xticks(range(1, 7))
        for j in range(3):
            ax[i, j].axhline(0, color="0.4", lw=0.8)
            ttl = ["bias", "ACC", "CRPSS"][j]
            ax[i, j].set_title(f"({letter(n)}) {C.INDEX_INFO[t]['label'] if j == 0 else ttl}",
                               loc="left", fontsize=9.5)
            n += 1
    ax[-1, 0].set_xlabel("Lead (days)")
    ax[-1, 1].set_xlabel("Lead (days)")
    ax[-1, 2].set_xlabel("Lead week")
    fig.tight_layout()
    model_legend(fig, [k for k in keys if k in set(sc[sc.target.isin(targets)].model)], y=-0.003)
    footnote(fig, f"vs ERA5 1° (same operators) · U/V200 from ln-p interpolation of 150/250 hPa for UC-S2S "
                  f"and its ERA5 reference; NGCM WY uses the true 200 hPa · {tag}", y=-0.025)
    save(fig, out / f"fig_indices_{tag}")


def fig_mok(tag, keys, out, variant=C.LLJ_ONSET_DEFAULT):
    par = C.LLJ_ONSET_VARIANTS[variant]
    dfs = {k: C.OUT_ROOT / "raw_metrics" / tag / k / f"mok_onset_{variant}.csv" for k in keys}
    dfs = {k: pd.read_csv(f) for k, f in dfs.items() if f.exists()}
    dfs = {k: d for k, d in dfs.items() if len(d)}
    if not dfs:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    rows = []
    bins = [0, 7, 14, 21, 45 - max(par["persist"], par["mean_window"]) + 1]
    for k, d in dfs.items():
        d = d[d.lead_to_onset <= bins[-1]]
        ax[0].scatter(d.lead_to_onset, d.error_days, color=STYLE[k]["color"], marker=STYLE[k]["marker"],
                      s=22, edgecolor="white", linewidth=0.6, zorder=3)
        d = d.assign(lbin=pd.cut(d.lead_to_onset, bins))
        g = d.groupby("lbin", observed=True)
        mae = g.error_days.apply(lambda s: s.abs().mean())
        hit = g.frac_within3.mean()
        cmae = g.clim_error_days.apply(lambda s: s.abs().mean())
        mids = [iv.mid for iv in mae.index]
        ax[1].plot(mids, mae.values, color=STYLE[k]["color"], marker=STYLE[k]["marker"], ls=STYLE[k]["ls"], ms=5)
        if k == list(dfs)[0]:
            ax[1].plot(mids, cmae.values, color="0.55", ls=":", marker="x", ms=5)
        for iv in mae.index:
            rows.append(dict(model=k, variant=variant, lead_to_onset_bin=str(iv), n=int(g.size()[iv]),
                             mae_days=mae[iv], clim_mae_days=cmae[iv], frac_members_within3=hit[iv],
                             frac_inits_no_ens_onset=float(g.error_days.apply(lambda s: s.isna().mean())[iv])))
    ax[0].axhline(0, color="0.4", lw=0.8)
    ax[0].set(xlabel="Days from init to observed onset", ylabel="Ensemble onset error (days)")
    ax[0].set_title("(a) per initialisation", loc="left")
    ax[1].set(xlabel="Days from init to observed onset", ylabel="|error| (days)")
    ax[1].set_title("(b) mean absolute error by lead", loc="left")
    fig.tight_layout()
    extra = [Line2D([], [], color="0.55", ls=":", marker="x", label="ERA5 climatological onset (1991–2018)")]
    model_legend(fig, list(dfs), extra, y=-0.02)
    cond = f" and {par['mean_window']}-day mean above it" if par["mean_window"] else ""
    footnote(fig, f"onset = first day ≥ 1 May with LLJ (U850, 5–15°N 40–80°E) > {par['thr']} m s⁻¹ for "
                  f"{par['persist']} days{cond} · ensemble = rounded mean if ≥ 50% of members onset · vs ERA5",
             y=-0.11)
    save(fig, out / f"fig_mok_{variant}_{tag}")
    return pd.DataFrame(rows)


def raw_tables(tag, sc, keys, out):
    w = sc[sc["window"] == "weekly"]
    cols = ["model", "target", "lead", "lead_label", "n", "n_years", "bias", "ratio", "acc", "acc_lo",
            "acc_hi", "rmse", "spread_ratio", "crpss", "crpss_lo", "crpss_hi"]
    w[[c for c in cols if c in w]].round(3).to_csv(out / f"raw_summary_{tag}.csv", index=False)
    al = []
    for k in keys:
        f = C.OUT_ROOT / "raw_metrics" / tag / k / "lead_alignment.csv"
        if f.exists():
            al.append(pd.read_csv(f).assign(model=k))
    if al:
        a = pd.concat(al).pivot_table(index=["model", "step"], columns="offset", values="corr").round(3)
        a.to_csv(out / f"lead_alignment_{tag}.csv")
        print(f"lead alignment ({tag}): corr(step k, IMD[init+k+offset])\n{a}")


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-mode", nargs="+", default=["common", "native"])
    ap.add_argument("--raw-tags", nargs="+", default=None,
                    help="raw-metric folders (default: <mode>_m567 for each mode + native_m56789)")
    a = ap.parse_args()
    out = U.ensure_dir(C.OUT_ROOT / "comparison")

    for mode in a.init_mode:
        keys = C.MODEL_ORDER + (["ngcm_legacy"] if mode == "native" else [])
        summ, binned, rel = onset_tables(mode, keys)
        if len(summ):
            summ.round(4).to_csv(out / f"onset_summary_{mode}.csv", index=False)
            if len(binned):
                binned.round(4).to_csv(out / f"onset_binned_{mode}.csv", index=False)
            fig_onset(mode, summ, binned, rel, out)
            fig_onset_scorecard(mode, summ, binned, out, "weekly")
            fig_onset_scorecard(mode, summ, binned, out, "5day_30")
            show = ["label", "n_inits", "mae_d1_15", "mae_se_d1_15", "far_d1_15", "mr_d1_15", "mae_d16_30",
                    "far_d16_30", "mr_d16_30", "auc_15", "bss_15", "rpss_15", "bss_30", "rpss_30",
                    "bss_wk", "rpss_wk", "auc_wk"]
            print(f"\nONSET CORE [{mode}]\n", summ[[c for c in show if c in summ]].round(3).to_string(index=False))

    tags = a.raw_tags or [f"{m}_m567" for m in a.init_mode] + ["native_m56789"]
    for tag in tags:
        keys = list(C.MODEL_ORDER)
        sc = load_scores(tag, keys)
        if not len(sc):
            continue
        raw_tables(tag, sc, keys, out)
        for tgt in ("rain_cmz", "wy", "llj", "olr_cmz"):
            fig_raw_scorecard(tag, sc, keys, out, tgt)
        fig_rain_cmz(tag, sc, keys, out)
        fig_rain_maps(tag, keys, out)
        fig_rain_stats(tag, keys, out)
        fig_indices(tag, sc, keys, out)
        if tag.startswith("native"):
            moks = [fig_mok(tag, keys, out, v) for v in C.LLJ_ONSET_VARIANTS]
            moks = [m for m in moks if m is not None]
            if moks:
                mok = pd.concat(moks)
                mok.round(3).to_csv(out / f"mok_summary_{tag}.csv", index=False)
                print(f"\nMOK (LLJ) [{tag}]\n", mok.round(2).to_string(index=False))
        w = sc[(sc["window"] == "weekly")].pivot_table(index=["target", "lead"], columns="model",
                                                 values=["acc", "crpss"]).round(2)
        print(f"\nRAW weekly scores [{tag}]\n{w}")


if __name__ == "__main__":
    main()
