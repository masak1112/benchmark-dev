"""W7.2 response test (pre-registered in reports/2026-11-11_aurora_bing-evaluation.md §v2.1, 2026-11-12 00:50 UTC, commit 8d8ec5a):
does the SST-anomaly conditioning leave a start-dependent imprint on the forecast?

Per start s: D_s = em_dc(A) - em_dc(B) (package drift-corrected 30-member ensemble means, window weeks 1-2 = mean of w1, w2; weeks 3-4 =
w34); P_s = ERA5 SST anomaly of the start day (00 UTC) vs the 1991-2020 mean of the same calendar day (00 UTC, Feb 29 -> Feb 28), 1.5 deg.
r_s = centred cos-lat-weighted pattern correlation of D_s with P_s over ocean cells (P_s finite) of the region; statistic = mean r_s.
Moving-block bootstrap over starts in date order (block 4, 1,000 draws, 95 % CI), grade robust / suggestive / noise (halves 2019-24 /
2025-26), Benjamini-Hochberg q = 0.1 over the family. Context (not graded): r with the ERA5 anomaly (O - Cfixed) in place of D_s; slope of
D_s on P_s (units per K).

    python scripts/bing_response_test.py --eval runs/bing/eval/v1 --a ucs2s_sstanom --b ucs2s_plain --out <csv>
    python scripts/bing_response_test.py ... --d_npy <dir>   # ablations: D_s and P_s given as files (see --help)
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

R = Path("/lus/flare/projects/S2Storm/s2storm")
STAGE = R / "data/clim_trend_bing/stage"
ERA = R / "data/era5_bing_1p5/sst"
LAT = np.arange(-90.0, 90.1, 1.5)
WK = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w34', 'w56']
REG = {'tropics': (-20, 20), 'global60': (-60, 60)}
FIELDS = ['tp24', 'olr', 't2m', 'z500']


def doy365(d):
    if d.month == 2 and d.day == 29:
        d = date(d.year, 2, 28)
    return (date(2001, d.month, d.day) - date(2001, 1, 1)).days


def doy366(d):
    return (d - date(d.year, 1, 1)).days


def ssta_start(starts):
    """[S, 121, 240] ERA5 00 UTC SST anomaly of each start day vs the 1991-2020 same-calendar-day mean"""
    need = sorted({doy365(s) for s in starts})
    clim = {}
    acc = np.zeros((len(need), 240, 121))
    for y in range(1991, 2021):
        a = np.load(STAGE / f"sst_{y}.npy", mmap_mode="r")
        acc += np.asarray(a[need, 0], np.float64)
    for k, dd in enumerate(need):
        clim[dd] = (acc[k] / 30.0).T                                  # [121 lat asc, 240]
    out = np.full((len(starts), 121, 240), np.nan, np.float32)
    cache = {}
    for i, s in enumerate(starts):
        if s.year not in cache:
            cache[s.year] = np.load(ERA / f"{s.year}.npy", mmap_mode="r")
        out[i] = cache[s.year][doy366(s), 0] - clim[doy365(s)]
    return out


def pattern_r(D, P, lat_lo, lat_hi):
    """D, P [S, y, x] -> r [S], slope [S]"""
    band = (LAT >= lat_lo) & (LAT <= lat_hi)
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, 240))
    r, b = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for i in range(len(D)):
        m = band[:, None] & np.isfinite(P[i]) & np.isfinite(D[i])
        if m.sum() < 50:
            continue
        ww, d, p = w[m], D[i][m].astype(np.float64), P[i][m].astype(np.float64)
        d = d - (ww * d).sum() / ww.sum()
        p = p - (ww * p).sum() / ww.sum()
        cov, vd, vp = (ww * d * p).sum(), (ww * d * d).sum(), (ww * p * p).sum()
        if vd > 0 and vp > 0:
            r[i] = cov / np.sqrt(vd * vp)
            b[i] = cov / vp
    return r, b


def block_idx(n, block, rng):
    st = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in st])[:n]


def grade_stat(r, years, n_boot=1000, block=4, seed=20261112, halves=((None, 2024), (2025, None))):
    ok = np.isfinite(r)
    r, years = r[ok], years[ok]
    mean = float(r.mean())
    rng = np.random.default_rng(seed)
    bs = np.array([r[block_idx(len(r), block, rng)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    h = []
    for a, b in halves:
        m = np.ones(len(r), bool)
        if a is not None:
            m &= years >= a
        if b is not None:
            m &= years <= b
        h.append(float(r[m].mean()) if m.any() else np.nan)
    excl = lo > 0 or hi < 0
    same = np.sign(h[0]) == np.sign(h[1]) == np.sign(mean)
    return dict(mean=mean, ci_lo=float(lo), ci_hi=float(hi), p_boot=p, half1=h[0], half2=h[1], n=int(len(r)),
                grade="robust" if excl and same else ("suggestive" if excl else "noise"))


def bh(ps, q=0.1):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    n = len(ps)
    thr = q * (np.arange(1, n + 1) / n)
    passed = ps[o] <= thr
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[o[:k]] = True
    return out


def load_em(ev, sysname, field):
    sd = Path(ev) / "score" / sysname / f"{field}_g15"
    st = [date.fromisoformat(s) for s in json.loads((sd / "done_dc.json").read_text())["starts"]]
    return st, np.load(sd / "em_dc.npy", mmap_mode="r")


def win(x, w):
    if w == "w12":
        return 0.5 * (np.asarray(x[:, WK.index("w1")]) + np.asarray(x[:, WK.index("w2")]))
    return np.asarray(x[:, WK.index(w)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default=str(R / "runs/bing/eval/v1"))
    p.add_argument("--a", default="ucs2s_sstanom")
    p.add_argument("--b", default="ucs2s_plain")
    p.add_argument("--out", required=True)
    p.add_argument("--fields", nargs="+", default=FIELDS)
    a = p.parse_args(argv)
    rows = []
    P_cache = {}
    for f in a.fields:
        sa, A = load_em(a.eval, a.a, f)
        sb, B = load_em(a.eval, a.b, f)
        common = sorted(set(sa) & set(sb))
        ia = [sa.index(s) for s in common]
        ib = [sb.index(s) for s in common]
        key = tuple(common)
        if key not in P_cache:
            P_cache[key] = ssta_start(common)
        P = P_cache[key]
        years = np.array([s.year for s in common])
        vd = json.loads((Path(a.eval) / "score" / a.a / f"{f}_g15" / "done_dc.json").read_text())["verif"]
        vst = [date.fromisoformat(s) for s in json.loads((Path(vd) / "meta.json").read_text())["starts"]]
        iv = [vst.index(s) for s in common]
        O = np.load(Path(vd) / "O.npy", mmap_mode="r")
        C = np.load(Path(vd) / "Cfixed.npy", mmap_mode="r")
        for w in ("w12", "w34"):
            D = win(A[ia], w) - win(B[ib], w)
            Eo = win(O[iv], w) - win(C[iv], w)
            for rg, (lo, hi) in REG.items():
                r, slope = pattern_r(D, P, lo, hi)
                g = grade_stat(r, years)
                re, _ = pattern_r(Eo, P, lo, hi)
                rows.append(dict(field=f, window=w, region=rg, **g, slope_per_K=float(np.nanmean(slope)),
                                 era5_r_context=float(np.nanmean(re)), a=a.a, b=a.b))
                print(rows[-1], flush=True)
    sig = bh([r["p_boot"] for r in rows])
    for r, s in zip(rows, sig):
        r["bh_q10"] = bool(s)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
