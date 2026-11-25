"""W12 ablation comparisons, pre-registered (report 2026-11-11_aurora_bing-evaluation.md §v2.2, 2026-11-12 00:50 UTC, commit 8d8ec5a).

variant - control (members 0-9, paired by start), for zeroed / oracle / damped / evolving: weeks 3-4 (primary) and weeks 1-2 (``w12``
window, needs the ablation tag scored with BING_EXTRA_WINDOWS=w12:1-14) drift-corrected fair CRPSS vs both references and centred
ACC for t2m, tp24, z500, sst, olr, u850 x global / tropics / NH extratropics / CONUS, plus Nino-3.4 ACC: the package's
``stats.run_headline`` with its two halves set to 2025 and 2026 (the grade's sign-agreement halves; only 2025-26 exist).

    BING_EXTRA_WINDOWS=w12:1-14 python scripts/bing_ablation_compare.py --eval runs/bing/eval/ablation --out runs/bing/eval/ablation/compare
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval.bing_eval import core as K  # noqa: E402
from s41eval.bing_eval import stats as S  # noqa: E402

VARIANTS = ["zeroed", "oracle", "damped", "evolving"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--windows", nargs="+", default=["w34", "w12"])
    a = p.parse_args(argv)
    # halves for the grade = the two years (keys kept so run_headline's grade() call is unchanged; renamed in the output)
    K.PERIODS.clear()
    K.PERIODS.update({"2019-24": (2025, 2025, ()), "2025-26": (2026, 2026, ()), "pooled": (2025, 2026, ()),
                      "pooled_no2019": (2025, 2026, ())})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for w in a.windows:
        if w not in K.WINDOWS:
            print(f"window {w} not available (set BING_EXTRA_WINDOWS=w12:1-14 for the scoring and this script): skipped")
            continue
        S.HEADLINE["window"] = w
        for v in a.variants:
            rows = S.run_headline(a.eval, f"ab_{v}", "ab_control", out_csv=None)
            for r in rows:
                r = dict(r)
                r["diff_2025"] = r.pop("diff_2019-24")
                r["diff_2026"] = r.pop("diff_2025-26")
                r.pop("diff_pooled_no2019", None)
                r.update(variant=v, window=w)
                allrows.append(r)
            print(f"{v} {w}: {len(rows)} rows", flush=True)
    keys = ["variant", "window"] + [k for k in allrows[0] if k not in ("variant", "window")]
    with open(out / "ablation_vs_control.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=keys)
        wri.writeheader()
        wri.writerows(allrows)
    print("wrote", out / "ablation_vs_control.csv")


if __name__ == "__main__":
    main()
