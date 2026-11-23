"""S41 production task list for the UC-S2S hindcasts (s2storm.aurora.taskfarm): one task = one (model, start, member) on one tile.

    python scripts/bing_hc_tasks.py --models sstanom plain --years 2025 2026 --members 30 --out runs/bing/hc/tasks_2025-26.jsonl \
        [--exclude 2026-09-21 2026-09-28] [--pin]

Order: model by model (sstanom first), starts in date order, members 0..M-1 (so a start's members finish together).
Run dirs: <hc_root>/<model>/<SECTION>/S41u_<model>_<YYYYMMDD>/predictions (one upstream run_name per start, so every start has its own
predictions dir; ``--pin`` pins each start's dir tree to its own healthy OST with ``lfs setstripe -c 1 -i k`` (k cycles over 0..159
without --bad_osts), the flare OST-outage workaround of 2026-11-11).
Checks before writing: every input h5 day S..S+45 exists in --data_dir and none is on a bad OST (lfs getstripe -i).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

R = Path("/lus/flare/projects/S2Storm/s2storm")
B = R / "aurora_bundle_v5"
MODELS = {
    "sstanom": dict(yaml=B / "checkpoints_sstanom/config/exp16_midway_sstanom.yaml", section="S2S_SSTANOM",
                    ckpt=B / "checkpoints_sstanom", base="loaded/3_base_full_state_rollout_finetuned_sstanom/rollout_ckpt_650.tar",
                    pp="loaded/4_postprocess_si/pp_si_ckpt_11400.tar", extra="--sst_cond_mode persistent"),
    "plain": dict(yaml=B / "checkpoints_plain/config/exp16_nvidia_v3.yaml", section="S2S", ckpt=B / "checkpoints_plain",
                  base="loaded/3_base_full_state_rollout_finetuned/rollout_ckpt_1000.tar",
                  pp="loaded/4_postprocess_si/pp_si_ckpt_10800.tar", extra=""),
}


def h5name(t):
    return f"{t.year}_{((t - datetime(t.year, 1, 1)).days * 4 + t.hour // 6):04d}.h5"


def ost(p):
    r = subprocess.run(["lfs", "getstripe", "-i", str(p)], stdout=subprocess.PIPE, universal_newlines=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["sstanom"])
    p.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    p.add_argument("--members", type=int, default=30)
    p.add_argument("--exclude", nargs="*", default=["2026-09-21", "2026-09-28"])
    p.add_argument("--starts_csv", default=str(R / "runs/bing/starts/starts.csv"))
    p.add_argument("--data_dir", default=str(B / "h5data"))
    p.add_argument("--hc_root", default=str(R / "runs/bing/hc"))
    p.add_argument("--out", required=True)
    p.add_argument("--pin", action="store_true")
    p.add_argument("--bad_osts", type=int, nargs="*", default=[9, 13])
    p.add_argument("--est_s", type=float, default=1100.0, help="one 45-day member with PP + writes on one tile")
    p.add_argument("--ckpt_root", default=None, help="dir holding checkpoints_<model>/ (e.g. the per-node /dev/shm copy); "
                                                     "default = the aurora_bundle_v5 bundles")
    a = p.parse_args(argv)
    starts = []
    with open(a.starts_csv) as f:
        for row in csv.DictReader(f):
            s = datetime.strptime(row["start"], "%Y-%m-%d")
            if s.year in a.years and row["start"] not in set(a.exclude):
                starts.append(s)
    starts.sort()
    bad = set(a.bad_osts)
    problems = []
    for s in starts:
        for k in range(0, 46):
            f = Path(a.data_dir) / h5name(s + timedelta(days=k))
            if not f.exists():
                problems.append(f"missing {f.name} (start {s:%Y-%m-%d})")
            elif ost(f) in bad:
                problems.append(f"bad OST {f.name} (start {s:%Y-%m-%d})")
    if problems:
        print("\n".join(sorted(set(problems))))
        raise SystemExit(f"{len(set(problems))} input problems; nothing written")
    good = [k for k in range(160) if k not in bad]
    tasks, ki = [], 0
    for mk in a.models:
        m = dict(MODELS[mk])
        if a.ckpt_root:
            ck = Path(a.ckpt_root) / f"checkpoints_{mk}"
            m["ckpt"] = ck
            m["yaml"] = ck / "config" / Path(m["yaml"]).name
        for s in starts:
            run = f"S41u_{mk}_{s:%Y%m%d}"
            pred = Path(a.hc_root) / mk / m["section"] / run / "predictions"
            if a.pin and not pred.exists():
                for d in (Path(a.hc_root) / mk / m["section"] / run, pred):
                    d.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["lfs", "setstripe", "-c", "1", "-i", str(good[ki % len(good)]), str(d)], check=True)
                ki += 7
            for mem in range(a.members):
                tid = f"{run}_m{mem:02d}"
                cmd = (f"S2STORM_MEMBER_SHARD={mem}/{a.members} python -m s2storm.models.ucs2s_xpu --mode dsi "
                       f"--out $S2STORM_STATE/{tid}.json -- --run_name {run} --init_dates {s:%Y-%m-%d} --save_mode hindcast "
                       f"--full_members {' '.join(str(i) for i in range(a.members))} --yaml {m['yaml']} --section {m['section']} "
                       f"--base_ckpt_rel {m['base']} --pp_ckpt_rel {m['pp']} --ckpt_dir {m['ckpt']} --data_dir {a.data_dir} "
                       f"--out_root {Path(a.hc_root) / mk} --timing_json $S2STORM_STATE/timing/{tid} -- --num_members {a.members} "
                       f"--inference_steps 45 --blend 1.0 --feedback base --base_latent si {m['extra']}").strip()
                tasks.append(dict(id=tid, model=mk, start=f"{s:%Y-%m-%d}", member=mem, cmd=cmd, est_s=a.est_s))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"{len(tasks)} tasks, {len(starts)} starts x {len(a.models)} models x {a.members} members -> {a.out}")


if __name__ == "__main__":
    main()
