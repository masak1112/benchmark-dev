"""W7 items 3 and 4 (plan v2; EXPLORATORY, not pre-registered): SST-anomaly (A) vs plain (B).

3. Does the A - B skill difference grow with the strength of the initial SST anomaly?
   Per start, dCRPSS_s = (CRPS_B - CRPS_A) / CRPS_ref (drift-corrected, weeks 3-4, trend-adjusted reference; package perstart files).
   - regression of dCRPSS_s on |Nino-3.4 initial anomaly| (modes/enso/perstart_nino34.csv) and on the RMS start-day SSTA of the
     region (ocean cells; same SSTA as the response test);
   - subsets: El Nino 2023-06..2024-04, La Nina 2020-09..2023-02, top-decile regional SSTA starts; mean dCRPSS with block bootstrap.
4. SST-anomaly persistence: projection of the drift-corrected forecast SST anomaly (em_dc - Cfixed) of week k onto the start-day ERA5
   SSTA pattern, a_k = <F_k, P> / <P, P> (ocean, cos-lat, region), for A, B and ERA5 (O - Cfixed); mean over starts by week.

    python scripts/bing_w7_extras.py --eval runs/bing/eval/v1.5 --out runs/bing/eval/v1.5/w7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bing_response_test as RT  # noqa: E402

R = Path("/lus/flare/projects/S2Storm/s2storm")
WK = RT.WK
REGB = {"global": (-90, 90), "tropics": (-20, 20), "nh_extratropics": (20, 90), "conus": (25, 50)}
PREG = {"global": (-60, 60), "tropics": (-20, 20), "nh_extratropics": (20, 60), "conus": (-60, 60)}   # conus: global SSTA magnitude (remote forcing)


def perstart(ev, sysn, field):
    import xarray as xr
    d = xr.open_dataset(Path(ev) / "score" / sysn / f"{field}_g15" / "perstart_dc.nc")
    st = [date.fromisoformat(str(s)[:10]) for s in d.start.values]
    return st, d


def rms_ssta(P, lo, hi):
    lat = RT.LAT
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 240))
    band = (lat >= lo) & (lat <= hi)
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        m = band[:, None] & np.isfinite(P[i])
        out[i] = np.sqrt((w[m] * P[i][m] ** 2).sum() / w[m].sum())
    return out


def boot_mean(x, n=1000, block=4, seed=7):
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = [x[RT.block_idx(len(x), min(block, len(x)), rng)].mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def boot_slope(x, y, n=1000, block=4, seed=8):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rng = np.random.default_rng(seed)
    def sl(ix):
        xx, yy = x[ix], y[ix]
        return np.polyfit(xx, yy, 1)[0]
    b = sl(np.arange(len(x)))
    bs = [sl(RT.block_idx(len(x), block, rng)) for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(b), float(lo), float(hi), int(len(x))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default=str(R / "runs/bing/eval/v1.5"))
    p.add_argument("--a", default="ucs2s_sstanom")
    p.add_argument("--b", default="ucs2s_plain")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nino = {}
    with open(R / "runs/bing/eval/modes/enso/perstart_nino34.csv") as f:
        for r in csv.DictReader(f):
            if r["system_key"] == "ucs2s_sstanom" and r["correction"] == "raw":
                nino[date.fromisoformat(r["start"])] = float(r["init_anom"])
    rows3 = []
    P_all = None
    for field in ("t2m", "tp24", "sst", "olr", "z500"):
        sa, A = perstart(a.eval, a.a, field)
        sb, B = perstart(a.eval, a.b, field)
        common = sorted(set(sa) & set(sb))
        ia = [sa.index(s) for s in common]
        ib = [sb.index(s) for s in common]
        if P_all is None or len(P_all[0]) != len(common):
            P_all = (common, RT.ssta_start(common))
        P = P_all[1]
        wi = WK.index("w34")
        n34 = np.array([abs(nino.get(s, np.nan)) for s in common])
        for rg, (plo, phi) in PREG.items():
            ri = list(A.region.values).index(rg)
            ca = A["crps_dc"].values[ia, wi, ri]
            cb = B["crps_dc"].values[ib, wi, ri]
            cr = A["cc_trend_dc"].values[ia, wi, ri]
            d = (cb - ca) / cr
            mag = rms_ssta(P, plo, phi)
            subsets = {
                "all": np.ones(len(common), bool),
                "el_nino_2023_24": np.array([date(2023, 6, 1) <= s <= date(2024, 4, 30) for s in common]),
                "la_nina_2020_22": np.array([date(2020, 9, 1) <= s <= date(2023, 2, 28) for s in common]),
                "top_decile_regional_ssta": mag >= np.nanpercentile(mag, 90),
                "abs_nino34_ge_1": n34 >= 1.0,
            }
            for nm, m in subsets.items():
                mu, lo, hi = boot_mean(d[m])
                rows3.append(dict(field=field, region=rg, kind="subset", name=nm, n=int(np.isfinite(d[m]).sum()), value=mu,
                                  ci_lo=lo, ci_hi=hi))
            for nm, x in (("slope_vs_abs_nino34", n34), ("slope_vs_rms_regional_ssta", mag)):
                b, lo, hi, n = boot_slope(x, d)
                rows3.append(dict(field=field, region=rg, kind="regression", name=nm, n=n, value=b, ci_lo=lo, ci_hi=hi))
    with open(out / "w7_3_dskill_vs_ssta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows3[0]))
        w.writeheader()
        w.writerows(rows3)
    # 4. persistence of the SST anomaly
    rows4 = []
    sa, EA = RT.load_em(a.eval, a.a, "sst")
    sb, EB = RT.load_em(a.eval, a.b, "sst")
    common = sorted(set(sa) & set(sb))
    ia = [sa.index(s) for s in common]
    ib = [sb.index(s) for s in common]
    vd = json.loads((Path(a.eval) / "score" / a.a / "sst_g15" / "done_dc.json").read_text())["verif"]
    vst = [date.fromisoformat(s) for s in json.loads((Path(vd) / "meta.json").read_text())["starts"]]
    iv = [vst.index(s) for s in common]
    O = np.load(Path(vd) / "O.npy", mmap_mode="r")
    C = np.load(Path(vd) / "Cfixed.npy", mmap_mode="r")
    P = RT.ssta_start(common)
    lat = RT.LAT
    wl = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 240))
    for rg, (plo, phi) in PREG.items():
        band = (lat >= plo) & (lat <= phi)
        for src, X, ix in (("ucs2s_sstanom", EA, ia), ("ucs2s_plain", EB, ib), ("era5", O, iv)):
            for k, wk in enumerate(WK[:6]):
                vals = []
                for j, s in enumerate(common):
                    F = np.asarray(X[ix[j], k]) - np.asarray(C[iv[j], k])
                    m = band[:, None] & np.isfinite(F) & np.isfinite(P[j])
                    pp = P[j][m]
                    den = (wl[m] * pp * pp).sum()
                    if den > 0:
                        vals.append((wl[m] * F[m] * pp).sum() / den)
                v = np.array(vals)
                mu, lo, hi = boot_mean(v)
                rows4.append(dict(region=rg, source=src, window=wk, projection=mu, ci_lo=lo, ci_hi=hi, n=len(v)))
    with open(out / "w7_4_ssta_persistence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows4[0]))
        w.writeheader()
        w.writerows(rows4)
    print("wrote", out)


if __name__ == "__main__":
    main()
