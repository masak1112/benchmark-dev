"""S41 reseed test (PI OK 2026-11-12): production seeds (member m = seed m at every start) vs per-start seeds (100000 * (1 + start index) + m),
SST-anomaly model, 89 Mondays 2025-26, members 0-9, scored in the ablation tag (systems ab_control, rs_reseed).

Per field (sst, t2m, tp24, z500, olr, u850), weeks 3-4 raw member window means (work/reduce):
  * member-offset diagnostic, copy of helper B's ``bing_weak.member_offsets`` (branch bing-eval-helperB b744caf): odd/even-start pattern
    correlation of each member's mean deviation from the ensemble mean (exchangeable -> ~0), and offset variance / ensemble variance
    (exchangeable -> ~1/S);
  * ensemble variance (area mean) and fair CRPSS: drift-corrected, from the package perstart files (both references);
  * paired Delta (reseed - production) with the headline statistics: ``bing_ablation_compare.py``-style run_headline.

    python scripts/bing_reseed_compare.py --eval runs/bing/eval/ablation --out runs/bing/eval/ablation/reseed_compare
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval.bing_eval import core as K  # noqa: E402
from s41eval.bing_eval import stats as S  # noqa: E402

R = Path("/lus/flare/projects/S2Storm/s2storm")
FIELDS = ["sst", "t2m", "tp24", "z500", "olr", "u850"]


def member_offsets(F, m, grid="g15"):
    S_, M = F.shape[:2]
    dev = F - np.nanmean(F, 1, keepdims=True)
    d_all = np.nanmean(dev, 0)
    d_o, d_e = np.nanmean(dev[0::2], 0), np.nanmean(dev[1::2], 0)
    r = np.array([float(K.acc_pattern(d_o[i] - K.amean(d_o[i], m, grid), d_e[i] - K.amean(d_e[i], m, grid), m, grid)) for i in range(M)])
    off = float(K.amean(np.nanmean(d_all ** 2, 0) * M / (M - 1), m, grid))
    ens = float(K.amean(np.nanmean(np.nanvar(F, 1, ddof=1), 0), m, grid))
    return r, off / ens, ens


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    ev, out = Path(a.eval), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    K.set_lsm(np.load(R / "data/era5_bing_1p5/lsm.npy"))
    mk = K.masks("g15")
    wi = K.WK.index("w34")
    rows = []
    for f in FIELDS:
        for sy in ("ab_control", "rs_reseed"):
            dj = json.loads((ev / "score" / sy / f"{f}_g15" / "done_dc.json").read_text())
            st = [date.fromisoformat(s) for s in dj["starts"]]
            F = np.stack([np.load(ev / "work" / "reduce" / sy / f / f"{s:%Y%m%d}.npy", mmap_mode="r")[:10, wi] for s in st])
            import xarray as xr
            ps = xr.open_dataset(ev / "score" / sy / f"{f}_g15" / "perstart_dc.nc")
            for rn in ("global", "tropics", "nh_extratropics"):
                r, frac, ens = member_offsets(F, mk[rn])
                j = list(ps.region.values).index(rn)
                c = ps["crps_dc"].values[:, wi, j]
                se = ps["se_dc"].values[:, wi, j]
                var = ps["var_dc"].values[:, wi, j]
                row = dict(field=f, system=sy, region=rn, n_starts=len(st), r_odd_even_mean=float(np.nanmean(r)),
                           offset_frac=frac, offset_frac_expected=1.0 / len(st), ens_var=ens,
                           spread_error=float(np.sqrt(np.nanmean(var) * 11 / 10 / np.nanmean(se))))
                for ref in ("fixed", "trend"):
                    cr = ps[f"cc_{ref}_dc"].values[:, wi, j]
                    ok = np.isfinite(c) & np.isfinite(cr)
                    row[f"crpss_{ref}"] = float(1 - c[ok].sum() / cr[ok].sum())
                rows.append(row)
                print(row, flush=True)
    with open(out / "reseed_member_diag.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    K.PERIODS.clear()
    K.PERIODS.update({"2019-24": (2025, 2025, ()), "2025-26": (2026, 2026, ()), "pooled": (2025, 2026, ()),
                      "pooled_no2019": (2025, 2026, ())})
    S.HEADLINE["window"] = "w34"
    hr = S.run_headline(ev, "rs_reseed", "ab_control", out_csv=None)
    for r in hr:
        r["diff_2025"] = r.pop("diff_2019-24")
        r["diff_2026"] = r.pop("diff_2025-26")
    with open(out / "reseed_vs_production_paired.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(hr[0]))
        w.writeheader()
        w.writerows(hr)
    print("wrote", out)


if __name__ == "__main__":
    main()
