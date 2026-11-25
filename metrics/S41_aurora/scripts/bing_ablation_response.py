"""W12 ablations, response part (pre-registered report §v2.2): the W7.2 statistic for each variant.

D_s = em_dc(variant) - em_dc(control) (10 members, package ablation tag), P_s = the conditioning difference of the window:
  zeroed   -SSTA(start)
  oracle   window-mean ERA5 SSTA(t) - SSTA(start)         (ERA5 O - Cfixed of the window, sst)
  damped   SSTA(start) * (window-mean exp(-t/tau) - 1)    (tau on 1.5 deg: runs/bing/ablation/tau/tau_1p5.npy)
  evolving window-mean forecast SSTA of the variant - SSTA(start)   (em_dc sst - Cfixed)
SSTA(start) as in W7.2 (ERA5 00 UTC vs the 1991-2020 same-day mean). r_s = centred cos-lat pattern correlation over ocean cells; mean over
starts; block bootstrap (4, 1,000); grade with halves 2025 / 2026. Fields tp24, olr, t2m, z500; windows w12 (if scored) and w34;
regions tropics and 60S-60N.

    BING_EXTRA_WINDOWS=w12:1-14 python scripts/bing_ablation_response.py --eval runs/bing/eval/ablation --out <csv>
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bing_response_test as RT  # noqa: E402
from s41eval.bing_eval import core as K  # noqa: E402

R = Path("/lus/flare/projects/S2Storm/s2storm")
TAU15 = R / "runs/bing/ablation/tau/tau_1p5.npy"


def load(ev, sysn, field):
    sd = Path(ev) / "score" / sysn / f"{field}_g15"
    dj = json.loads((sd / "done_dc.json").read_text())
    return [date.fromisoformat(s) for s in dj["starts"]], np.load(sd / "em_dc.npy", mmap_mode="r"), Path(dj["verif"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", default=["zeroed", "oracle", "damped", "evolving"])
    a = p.parse_args(argv)
    windows = [w for w in ("w12", "w34") if w in K.WINDOWS]
    tau = np.load(TAU15)
    rows = []
    sc, EC_sst, vd_sst = load(a.eval, "ab_control", "sst")
    for v in a.variants:
        try:
            sv, EV_sst, _ = load(a.eval, f"ab_{v}", "sst")
        except FileNotFoundError:
            print(v, "not scored: skipped")
            continue
        for fld in ("tp24", "olr", "t2m", "z500"):
            s1, EC, _ = load(a.eval, "ab_control", fld)
            s2, EV, _ = load(a.eval, f"ab_{v}", fld)
            common = sorted(set(s1) & set(s2) & set(sc) & set(sv))
            P0 = RT.ssta_start(common)
            vst = [date.fromisoformat(s) for s in json.loads((vd_sst / "meta.json").read_text())["starts"]]
            iv = [vst.index(s) for s in common]
            Osst = np.load(vd_sst / "O.npy", mmap_mode="r")
            Csst = np.load(vd_sst / "Cfixed.npy", mmap_mode="r")
            years = np.array([s.year for s in common])
            for w in windows:
                k = K.WK.index(w)
                a0, b0 = K.WINDOWS[w]
                D = np.stack([np.asarray(EV[s2.index(s), k]) - np.asarray(EC[s1.index(s), k]) for s in common])
                if v == "zeroed":
                    P = -P0
                elif v == "oracle":
                    P = np.stack([np.asarray(Osst[iv[j], k]) - np.asarray(Csst[iv[j], k]) for j in range(len(common))]) - P0
                elif v == "damped":
                    t = np.arange(a0, b0 + 1, dtype=np.float64)
                    fac = np.exp(-t[:, None, None] / tau[None]).mean(0)
                    P = P0 * (fac[None] - 1.0)
                else:
                    P = np.stack([np.asarray(EV_sst[sv.index(s), k]) - np.asarray(Csst[iv[j], k])
                                  for j, s in enumerate(common)]) - P0
                for rg, (lo, hi) in RT.REG.items():
                    r, slope = RT.pattern_r(D, P, lo, hi)
                    g = RT.grade_stat(r, years, halves=((2025, 2025), (2026, 2026)))
                    g["half_2025"], g["half_2026"] = g.pop("half1"), g.pop("half2")
                    rows.append(dict(variant=v, field=fld, window=w, region=rg, **g, slope=float(np.nanmean(slope))))
                    print(rows[-1], flush=True)
    sig = RT.bh([r["p_boot"] for r in rows])
    for r, s in zip(rows, sig):
        r["bh_q10"] = bool(s)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader()
        wri.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
