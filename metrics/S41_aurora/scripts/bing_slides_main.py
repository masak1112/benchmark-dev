"""Main-session slide figures for the 2026-11-13 Bing evaluation set (IDs M01...). Style constants copied from s2storm
``s2storm/sci/slide_style.py`` (branch aurora-port, 2026-11-12; not imported across branches): white, 16:9, Open Sans/DejaVu 16 pt, no
green, no figure title, grey footer, fixed model colours (UC-S2S SST-anomaly #D55E00, plain #0072B2, FuXi #000000, IFS #CC79A7,
ERA5 #7a7a7a).

    python scripts/bing_slides_main.py m01      # W7.2 response test
    python scripts/bing_slides_main.py m02      # W12 ablations (when scored)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

R = Path("/lus/flare/projects/S2Storm/s2storm")
OUT = R / "runs/bing/eval/v1.5/slides"
W, H = 13.33, 7.5
C = {"sstanom": "#D55E00", "plain": "#0072B2", "fuxi": "#000000", "ifs": "#CC79A7", "era5": "#7a7a7a"}
GREY = "#7a7a7a"
VAR = {"tp24": "precipitation", "olr": "OLR", "t2m": "2-m temperature", "z500": "z500", "sst": "SST", "u850": "u850"}


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    p.rcParams.update({"font.family": ["Open Sans", "DejaVu Sans"], "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 16,
                       "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14, "figure.facecolor": "white",
                       "axes.facecolor": "white", "savefig.facecolor": "white", "axes.spines.top": False,
                       "axes.spines.right": False})
    return p


def save(f, fid, name, footer):
    import matplotlib.pyplot as p
    OUT.mkdir(parents=True, exist_ok=True)
    f.text(0.01, 0.01, footer, fontsize=11, color=GREY, ha="left", va="bottom")
    path = OUT / f"{fid}_{name}.png"
    for dpi in (150, 130, 110, 90):
        f.savefig(path, dpi=dpi)
        if path.stat().st_size < 1_000_000:
            break
    p.close(f)
    return path


def caption(fid, text):
    d = OUT / "captions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{fid}.md").write_text(text.strip() + "\n")


def m01():
    src = R / "runs/bing/eval/v1.5/w7/response_test_sstanom_vs_plain.csv"
    rows = list(csv.DictReader(open(src)))
    p = plt()
    f, axs = p.subplots(1, 2, figsize=(W, H), sharey=True)
    f.subplots_adjust(left=0.15, right=0.9, bottom=0.2, top=0.86, wspace=0.28)
    fields = ["t2m", "olr", "tp24", "z500"]
    for ax, rg, ttl in zip(axs, ("tropics", "global60"), ("tropics (20°S–20°N), ocean", "60°S–60°N, ocean")):
        for j, w in enumerate(("w12", "w34")):
            for i, fld in enumerate(fields):
                r = [x for x in rows if x["field"] == fld and x["window"] == w and x["region"] == rg][0]
                y = i + (0.18 if j == 0 else -0.18)
                m, lo, hi = float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"])
                col = C["sstanom"]
                ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o" if j else "s", color=col, mfc=col if j else "white",
                            ms=9, capsize=4, lw=2, label=("weeks 3–4" if j else "weeks 1–2") if (i == 0) else None)
                ax.plot(float(r["era5_r_context"]), y, marker="D", color=C["era5"], ms=7, ls="none",
                        label="ERA5's own relation" if (i == 0 and j == 0) else None)
                g = r["grade"]
                ax.text(1.02, y, g, transform=ax.get_yaxis_transform(), ha="left", va="center", clip_on=False,
                        fontsize=12, color="k" if g == "robust" else GREY)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(fields)), [VAR[x] for x in fields])
        ax.set_xlim(-0.5, 0.85)
        ax.set_xlabel("pattern corr. with start-day SST anomaly", fontsize=15)
        ax.set_title(ttl, fontsize=18)
        ax.invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.52, 1.0))
    foot = ("SST-anomaly minus plain UC-S2S, drift-corrected 30-member ensemble means, 403 starts 2019–26, 1.5° ocean cells. Mean of per-start "
            "r, 95 % block-bootstrap CI.\nPre-registered 2026-11-12 00:50 UTC. Grades: robust = CI excludes 0 and same sign in 2019–24 and "
            "2025–26. The two models also differ in base checkpoint (confound; ablations test it).")
    path = save(f, "M01", "sst_conditioning_imprint", foot)
    t = {(x["field"], x["window"], x["region"]): x for x in rows}
    caption("M01", f"""## M01 — `M01_sst_conditioning_imprint.png`
- **Shows:** the start-dependent imprint of the SST-anomaly conditioning: pattern correlation between (SST-anomaly − plain) ensemble-mean anomalies and the start-day SST anomaly (W7.2, pre-registered).
- **Key numbers:** tropical t2m r = {float(t[('t2m','w34','tropics')]['mean']):+.2f} (weeks 3–4, robust; ERA5's own relation {float(t[('t2m','w34','tropics')]['era5_r_context']):+.2f}) → the SST-anomaly model damps the initial anomaly relative to plain; tropical OLR r = {float(t[('olr','w34','tropics')]['mean']):+.3f} (robust, same sign as ERA5 {float(t[('olr','w34','tropics')]['era5_r_context']):+.2f}) → slightly more convection over warm water; precipitation weeks 3–4 {float(t[('tp24','w34','tropics')]['mean']):+.3f} (suggestive, opposite sign).
- **Grade:** pre-registered; 12/16 tests pass BH q = 0.1. Confounded with the base checkpoint until the ablations.
- **Source:** `{src}`
""")
    print(path)


def m02():
    src = R / "runs/bing/eval/v1.5/w7/w7_4_ssta_persistence.csv"
    rows = list(csv.DictReader(open(src)))
    p = plt()
    f, axs = p.subplots(1, 2, figsize=(W, H), sharey=True)
    f.subplots_adjust(left=0.09, right=0.97, bottom=0.22, top=0.86, wspace=0.12)
    lab = {"ucs2s_sstanom": ("UC-S2S SST-anomaly", C["sstanom"]), "ucs2s_plain": ("UC-S2S plain", C["plain"]), "era5": ("ERA5", C["era5"])}
    for ax, rg, ttl in zip(axs, ("tropics", "global"), ("tropics (20°S–20°N)", "60°S–60°N")):
        for src_k, (nm, col) in lab.items():
            rr = [r for r in rows if r["region"] == rg and r["source"] == src_k]
            x = np.arange(1, len(rr) + 1)
            y = np.array([float(r["projection"]) for r in rr])
            lo = np.array([float(r["ci_lo"]) for r in rr]); hi = np.array([float(r["ci_hi"]) for r in rr])
            ax.plot(x, y, "-o", color=col, lw=2.5, ms=7, label=nm)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.set_xticks(range(1, 7), [f"week {k}" for k in range(1, 7)])
        ax.set_ylim(0.4, 1.0)
        ax.set_title(ttl, fontsize=18)
        ax.grid(axis="y", color="#e0e0e0")
    axs[0].set_ylabel("initial SST anomaly retained\n(projection on start-day pattern)")
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=15, bbox_to_anchor=(0.53, 1.0))
    t = {(r["region"], r["source"], r["window"]): float(r["projection"]) for r in rows}
    foot = ("Projection of the weekly SST anomaly (drift-corrected ensemble mean − 1991–2020 climatology) on the start-day ERA5 SST anomaly, ocean "
            "cells, cos-lat;\n403 starts 2019–26, 30 members, 95 % block-bootstrap band. Exploratory (W7.4). UC-S2S land/sea mask fix applied (v1.5).")
    path = save(f, "M02", "sst_anomaly_persistence", foot)
    caption("M02", f"""## M02 — `M02_sst_anomaly_persistence.png`
- **Shows:** how much of the initial SST anomaly each model keeps, week by week, vs ERA5.
- **Key numbers (tropics, week 4):** SST-anomaly {t[('tropics','ucs2s_sstanom','w4')]:.2f}, plain {t[('tropics','ucs2s_plain','w4')]:.2f}, ERA5 {t[('tropics','era5','w4')]:.2f}; week 6: {t[('tropics','ucs2s_sstanom','w6')]:.2f} / {t[('tropics','ucs2s_plain','w6')]:.2f} / {t[('tropics','era5','w6')]:.2f}. Both models lose the anomaly too fast; the SST-anomaly model faster than plain (CIs separate from week 2).
- **Grade:** exploratory (not pre-registered).
- **Source:** `{src}`
""")
    print(path)


def m03():
    src = R / "runs/bing/eval/v1.5/w7/w7_3_dskill_vs_ssta.csv"
    rows = list(csv.DictReader(open(src)))
    p = plt()
    f, axs = p.subplots(1, 3, figsize=(W, H), sharey=True)
    f.subplots_adjust(left=0.24, right=0.98, bottom=0.2, top=0.88, wspace=0.12)
    subs = [("all", "all starts (403)"), ("la_nina_2020_22", "La Niña 2020–22 (130)"), ("el_nino_2023_24", "El Niño 2023–24 (48)"),
            ("abs_nino34_ge_1", "|Niño-3.4| ≥ 1 K (97)"), ("top_decile_regional_ssta", "top-10 % tropical SSTA (41)")]
    for ax, fld in zip(axs, ("sst", "t2m", "tp24")):
        for i, (k, nm) in enumerate(subs):
            r = [x for x in rows if x["field"] == fld and x["region"] == "tropics" and x["name"] == k][0]
            m, lo, hi = float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"])
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=C["sstanom"], ms=9, capsize=4, lw=2)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{VAR[fld]}, tropics", fontsize=18)
        ax.set_xlabel("Δ CRPSS (SST-anom. − plain)", fontsize=15)
    axs[0].set_yticks(range(len(subs)), [s[1] for s in subs])
    axs[0].invert_yaxis()
    foot = ("Weeks 3–4, drift-corrected, trend-adjusted reference; per-start ΔCRPSS = (CRPS plain − CRPS SST-anomaly)/CRPS reference, mean with 95 % "
            "block-bootstrap CI.\nExploratory (W7.3); the all-starts values are close to the pre-registered headline. Subsets overlap.")
    path = save(f, "M03", "sst_effect_vs_anomaly_strength", foot)
    t = {(r["field"], r["region"], r["name"]): r for r in rows}
    def v(fl, nm):
        r = t[(fl, "tropics", nm)]
        return f"{float(r['value']):+.4f} [{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]"
    caption("M03", f"""## M03 — `M03_sst_effect_vs_anomaly_strength.png`
- **Shows:** whether the SST-anomaly model gains (or loses) more when the initial SST anomaly is large.
- **Key numbers (tropics, weeks 3–4 ΔCRPSS):** SST all {v('sst','all')}, El Niño 2023–24 {v('sst','el_nino_2023_24')}; t2m all {v('t2m','all')}, El Niño {v('t2m','el_nino_2023_24')}; precipitation all {v('tp24','all')}, top-10 % SSTA {v('tp24','top_decile_regional_ssta')}. Slope of SST ΔCRPSS on |Niño-3.4|: {float(t[('sst','tropics','slope_vs_abs_nino34')]['value']):+.4f} per K.
- **Grade:** exploratory.
- **Source:** `{src}`
""")
    print(path)


def m04():
    d = R / "runs/bing/eval/ablation/reseed_compare"
    diag = list(csv.DictReader(open(d / "reseed_member_diag.csv")))
    pr = list(csv.DictReader(open(d / "reseed_vs_production_paired.csv")))
    p = plt()
    f, axs = p.subplots(1, 2, figsize=(W, H))
    f.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.86, wspace=0.25)
    fl = ["sst", "t2m", "tp24", "olr", "u850", "z500"]
    x = np.arange(len(fl))
    for k, (sy, nm, col) in enumerate((("ab_control", "production seeds (member m = seed m at every start)", C["sstanom"]),
                                       ("rs_reseed", "per-start seeds (Bing's scheme)", GREY))):
        v = [float([r for r in diag if r["field"] == fl_ and r["system"] == sy and r["region"] == "tropics"][0]["offset_frac"]) for fl_ in fl]
        axs[0].bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label=nm)
    axs[0].axhline(1 / 82, color="k", lw=1, ls="--")
    axs[0].text(len(fl) - 0.5, 1 / 82 + 0.025, "independent members (1/S)", ha="right", fontsize=12)
    axs[0].set_xticks(x, ["SST", "t2m", "precip.", "OLR", "u850", "z500"], fontsize=14)
    axs[0].set_ylabel("fixed member offsets / ensemble variance\n(tropics, weeks 3–4)")
    axs[0].set_title("member offsets", fontsize=18)
    for k, ref in enumerate(("fixed", "trend")):
        for i, fl_ in enumerate(fl):
            r = [q for q in pr if q["field"] == fl_ and q["region"] == "global" and q["metric"] == "crpss" and q["reference"] == ref][0]
            m, lo, hi = float(r["diff"]), float(r["ci_lo"]), float(r["ci_hi"])
            y = i + (k - 0.5) * 0.3
            axs[1].errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o" if k else "s", color="k", mfc="k" if k else "white", ms=8, capsize=4,
                            label=(f"vs {'1991–2020' if ref == 'fixed' else 'trend-adjusted'} climatology" if i == 0 else None))
            if r["grade"] != "noise":
                axs[1].text(hi + 0.002, y, r["grade"], va="center", fontsize=11)
    axs[1].axvline(0, color="k", lw=1)
    axs[1].set_yticks(range(len(fl)), [VAR.get(v, v) for v in fl])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Δ CRPSS, per-start − production seeds (global)")
    axs[1].set_title("skill", fontsize=18)
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    f.legend(h + h2, l + l2, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    foot = ("UC-S2S SST-anomaly, 82 Mondays 2025–26 with a drift estimate, members 0–9, weeks 3–4, drift-corrected, paired block bootstrap "
            "(halves 2025 / 2026).\nPer-start seeds remove the offsets; spread/error unchanged (±3 %); skill unchanged except SST "
            "(production slightly optimistic). PI-approved test, 23.8 node-h.")
    path = save(f, "M04", "seed_test", foot)
    g = {(r["field"], r["reference"]): r for r in pr if r["region"] == "global" and r["metric"] == "crpss"}
    caption("M04", f"""## M04 — `M04_seed_test.png`
- **Shows:** our hindcasts reused member seeds at every start (one start per run); a rerun with per-start seeds (Bing's scheme) removes the fixed member offsets and leaves spread and skill unchanged except SST.
- **Key numbers:** tropical SST offset share 0.62 → 0.011 (1/S = 0.012); spread/error SST global 0.845 → 0.822, t2m 0.775 → 0.776; ΔCRPSS global SST {float(g[('sst','fixed')]['diff']):+.3f} (fixed, {g[('sst','fixed')]['grade']}) / {float(g[('sst','trend')]['diff']):+.3f} (trend); t2m {float(g[('t2m','fixed')]['diff']):+.3f}, tp24 {float(g[('tp24','fixed')]['diff']):+.3f}, z500 {float(g[('z500','fixed')]['diff']):+.3f} (noise).
- **Grade:** test approved by the PI; paired grades as in the package (halves 2025/2026).
- **Source:** `{d}`
""")
    print(path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("fig", choices=["m01", "m02", "m03", "m04"])
    a = p.parse_args(argv)
    {"m01": m01, "m02": m02, "m03": m03, "m04": m04}[a.fig]()


if __name__ == "__main__":
    main()
