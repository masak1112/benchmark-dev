"""W12 ablations, MJO part (pre-registered report §v2.2): bivariate COR of the 10-member drift-corrected ensemble-mean RMM, mean of the
daily COR over leads 15-28 (same metric, EOFs and fixed rotation as the headline MJO test, ``scripts/bing_headline_mjo.py``), variant -
control paired by start, moving-block bootstrap (block 4, 1,000 draws), halves 2025 / 2026. Full and winds-only RMM.

Needs the bing-modes code on the path (``s41eval.bing_modes``; run with PYTHONPATH=<bing-modes worktree>):
    python bing_ablation_mjo.py extract --workers 24          # compute node: member files -> RMM per (system, start)
    python bing_ablation_mjo.py score --out <csv>             # login node (light)
Drift correction: LOYO within +-30 days (bing_modes.loyo_bias) -> with 2025-26 only, each start's bias comes from the other year.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import pickle
from datetime import date
from pathlib import Path

import numpy as np

R = Path("/lus/flare/projects/S2Storm/s2storm")
OUT = R / "runs/bing/eval/ablation/mjo"
MODES = R / "runs/bing/eval/modes"
SYS = {
    "ab_control": "runs/bing/hc/sstanom/S2S_SSTANOM/S41u_sstanom_{d}/predictions/pangu_plasim_S41u_sstanom_{d}_24h_45step_{d}00_ens_{m}_h25.nc",
}
for _v in ("zeroed", "oracle", "damped", "evolving"):
    SYS[f"ab_{_v}"] = (f"runs/bing/ablation/{_v}/S2S_SSTANOM/S41a_{_v}_{{d}}/predictions/"
                       f"pangu_plasim_S41a_{_v}_{{d}}_24h_45step_{{d}}00_ens_{{m}}_h25.nc")
M = 10


def starts():
    import csv as _c
    out = []
    with open(R / "runs/bing/starts/starts.csv") as f:
        for r in _c.DictReader(f):
            d = date.fromisoformat(r["start"])
            if date(2025, 1, 6) <= d <= date(2026, 9, 14):
                out.append(d)
    return sorted(set(out))


_IX = {}


def ix():
    if not _IX:
        for nm in ("full", "winds"):
            _IX[nm] = pickle.load(open(MODES / "era5" / f"mjo_{nm}.pkl", "rb"))
    return _IX


def one(arg):
    from s41eval import bing_modes as BM
    sysn, d = arg
    f = OUT / "extract" / sysn / f"{d:%Y%m%d}.npz"
    if f.exists():
        return f"{sysn} {d} cached"
    paths = [R / SYS[sysn].format(d=f"{d:%Y%m%d}", m=m) for m in range(M)]
    if not all(p.exists() for p in paths):
        return f"{sysn} {d} missing {sum(not p.exists() for p in paths)}"
    mem = [BM.read_member(str(p), d) for p in paths]
    band = {v: np.stack([x[f"band_{v}"] for x in mem]) for v in ("olr", "u850", "u200")}
    out = {}
    for nm, I in ix().items():
        out[f"F_{nm}"] = I.forecast(BM.state_samp("00z"), d, {v: band[v] for v in I.fields}).astype(np.float32)
        out[f"O_{nm}"] = I.obs(BM.state_samp("00z"), BM.valid_dates(d)).astype(np.float32)
    out["nino34"] = np.stack([x["nino34"] for x in mem]).astype(np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".partial.npz")
    np.savez(tmp, **out)
    os.replace(tmp, f)
    return f"{sysn} {d} ok"


def cmd_extract(a):
    jobs = [(s, d) for s in (a.systems or list(SYS)) for d in starts()]
    with mp.get_context("fork").Pool(a.workers) as p:
        for r in p.imap_unordered(one, jobs):
            print(r, flush=True)


def cor(f, o):
    ok = np.isfinite(f).all(-1) & np.isfinite(o).all(-1)
    num = np.where(ok, (f * o).sum(-1), 0).sum(0)
    ff = np.where(ok, (f ** 2).sum(-1), 0).sum(0)
    oo = np.where(ok, (o ** 2).sum(-1), 0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.sqrt(ff * oo)


def metric(f, o):
    return float(np.nanmean(cor(f[:, 15:29], o[:, 15:29])))


def block_idx(n, block, rng):
    st = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in st])[:n]


def load(sysn, nm, st):
    F, O, ok = [], [], []
    for d in st:
        f = OUT / "extract" / sysn / f"{d:%Y%m%d}.npz"
        if not f.exists():
            ok.append(False)
            F.append(None)
            O.append(None)
            continue
        z = np.load(f)
        F.append(z[f"F_{nm}"])
        O.append(z[f"O_{nm}"])
        ok.append(True)
    return F, O, np.array(ok)


def cmd_score(a):
    from s41eval import bing_modes as BM
    st = starts()
    rows = []
    for nm in ("full", "winds"):
        Fc, Oc, okc = load("ab_control", nm, st)
        for v in ("zeroed", "oracle", "damped", "evolving"):
            Fv, Ov, okv = load(f"ab_{v}", nm, st)
            keep = np.flatnonzero(okc & okv)
            if len(keep) < 20:
                print(f"{v} {nm}: {len(keep)} common starts, skipped")
                continue
            ss = [st[i] for i in keep]
            o = np.stack([Oc[i] for i in keep])
            assert np.allclose(o, np.stack([Ov[i] for i in keep]), equal_nan=True)
            ems = {}
            for k, F in (("a", Fv), ("b", Fc)):
                em = np.stack([F[i].mean(0) for i in keep])               # [S, 46, 2]
                ems[k] = em - BM.loyo_bias(em - o, ss)
            fa, fb = ems["a"], ems["b"]
            yrs = np.array([d.year for d in ss])
            d0 = metric(fa, o) - metric(fb, o)
            halves = {y: metric(fa[yrs == y], o[yrs == y]) - metric(fb[yrs == y], o[yrs == y]) for y in (2025, 2026)}
            rng = np.random.default_rng(20261112)
            bs = []
            for _ in range(1000):
                b = block_idx(len(ss), 4, rng)
                bs.append(metric(fa[b], o[b]) - metric(fb[b], o[b]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            excl = lo > 0 or hi < 0
            same = np.sign(halves[2025]) == np.sign(halves[2026]) == np.sign(d0)
            rows.append(dict(variant=v, index=nm, metric="mean_daily_cor_15_28", variant_value=metric(fa, o),
                             control_value=metric(fb, o), diff=d0, ci_lo=lo, ci_hi=hi, diff_2025=halves[2025], diff_2026=halves[2026],
                             n_starts=len(ss), members=M, grade="robust" if excl and same else ("suggestive" if excl else "noise"),
                             primary=(nm == "full")))
            print(rows[-1], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--workers", type=int, default=24)
    e.add_argument("--systems", nargs="*", default=None)
    s = sp.add_parser("score")
    s.add_argument("--out", default=str(OUT / "ablation_mjo_vs_control.csv"))
    a = p.parse_args(argv)
    {"extract": cmd_extract, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
