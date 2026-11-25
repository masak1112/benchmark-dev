"""Pre-registered headline, MJO part (report 2026-11-11_aurora_bing-evaluation.md §2; metric fixed §4 at 20:05 UTC):
paired A - B of the weeks-3-4 MJO bivariate COR = mean over leads 15..28 of the daily COR of the drift-corrected 30-member ensemble-mean
RMM vs ERA5 RMM, on the starts both systems have. Moving-block bootstrap over starts in date order (block 4, 1,000 draws, 95 %
percentile CI); grade robust / suggestive / noise as plan §2 (robust = pooled CI excludes 0 and same sign in 2019-24 and 2025-26).
Secondary: winds-only RMM, and the COR of the 15-28-day mean RMM.

    python scripts/bing_headline_mjo.py --modes runs/bing/eval/modes --a ucs2s_sstanom --b ucs2s_plain --out <csv>
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import xarray as xr


def cor(f, o):
    ok = np.isfinite(f).all(-1) & np.isfinite(o).all(-1)
    ff = np.where(ok, (f ** 2).sum(-1), 0).sum(0)
    oo = np.where(ok, (o ** 2).sum(-1), 0).sum(0)
    num = np.where(ok, (f * o).sum(-1), 0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.sqrt(ff * oo)


def metric(f, o, kind):
    if kind == "mean_daily_cor":                 # primary
        return float(np.nanmean(cor(f[:, 15:29], o[:, 15:29])))
    fm, om = np.nanmean(f[:, 15:29], 1, keepdims=True), np.nanmean(o[:, 15:29], 1, keepdims=True)
    return float(cor(fm, om)[0])                 # COR of the 15-28-day mean RMM


def block_idx(n, block, rng):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", required=True)
    p.add_argument("--a", default="ucs2s_sstanom")
    p.add_argument("--b", default="ucs2s_plain")
    p.add_argument("--out", required=True)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261111)
    a = p.parse_args(argv)
    rows = []
    for index in ("full", "winds"):
        A = xr.open_dataset(f"{a.modes}/mjo/perstart_mjo_{index}_{a.a}.nc")
        B = xr.open_dataset(f"{a.modes}/mjo/perstart_mjo_{index}_{a.b}.nc")
        common = np.intersect1d(A.start.values, B.start.values)
        A, B = A.sel(start=common), B.sel(start=common)
        o = A.rmm_obs.values
        assert np.allclose(o, B.rmm_obs.values, equal_nan=True), "ERA5 RMM differs between files"
        fa, fb = A.rmm_em30_dc.values, B.rmm_em30_dc.values
        yrs = common.astype("datetime64[Y]").astype(int) + 1970
        for kind in ("mean_daily_cor", "cor_of_w34_mean"):
            ma, mb = metric(fa, o, kind), metric(fb, o, kind)
            sub = {}
            for nm, m in (("2019-24", yrs <= 2024), ("2025-26", yrs >= 2025), ("no2019", yrs >= 2020)):
                sub[nm] = metric(fa[m], o[m], kind) - metric(fb[m], o[m], kind)
            rng = np.random.default_rng(a.seed)
            n = len(common)
            bs = []
            for _ in range(a.n_boot):
                ix = block_idx(n, a.block, rng)
                bs.append(metric(fa[ix], o[ix], kind) - metric(fb[ix], o[ix], kind))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            d = ma - mb
            excl = lo > 0 or hi < 0
            same = np.sign(sub["2019-24"]) == np.sign(sub["2025-26"]) == np.sign(d)
            grade = "robust" if excl and same else ("suggestive" if excl else "noise")
            rows.append(dict(field=f"mjo_{index}", metric=kind, a=ma, b=mb, diff=d, ci_lo=lo, ci_hi=hi, n_starts=n,
                             diff_2019_24=sub["2019-24"], diff_2025_26=sub["2025-26"], diff_no2019=sub["no2019"], grade=grade,
                             primary=(index == "full" and kind == "mean_daily_cor")))
            print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
