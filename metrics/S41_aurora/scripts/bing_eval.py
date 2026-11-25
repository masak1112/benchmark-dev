"""CLI of the S41 evaluation package ``s41eval.bing_eval`` (brief 2026-11-11 tasks 3-6).

    PY=<python with numpy, netCDF4, xarray, pyyaml>   # Aurora: frameworks python + envs/bing_pylib on PYTHONPATH
    $PY scripts/bing_eval.py tasks   --config configs/bing_eval/aurora.yaml --tag v1 --phase 1   # reduce + verification task list
    $PY scripts/bing_eval.py tasks   --config ... --tag v1 --phase 2                              # score task list
    $PY scripts/bing_eval.py run     --config ... --tag v1 --task reduce:ucs2s_sstanom:20250106   # one task (what the farm runs)
    $PY scripts/bing_eval.py local   --config ... --tag v1 --phase 1 [--match fuxi]               # run a phase serially (small tests)
    $PY scripts/bing_eval.py status  --config ... --tag v1
    $PY scripts/bing_eval.py collect --config ... --tag v1      # scores.csv/.nc, fail_summary.csv, provenance.json, summary.md, figs
    $PY scripts/bing_eval.py regress --config configs/bing_eval/v0reg.yaml --tag v0reg   # compare with v0_ucs2s/scores.csv
    $PY scripts/bing_eval.py headline --config ... --tag v1 --a ucs2s_sstanom --b ucs2s_plain   # the pre-registered headline

Task farm (compute nodes): ``python -m s2storm.aurora.taskfarm --tasks <out>/<tag>/tasks_phase1.jsonl --state <out>/<tag>/state1``
(``runs/bing/eval/jobs/s41e_bulk.pbs``). Each task writes a done marker; finished tasks are skipped on resubmission.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k, _v in (('OMP_NUM_THREADS', '4'), ('MKL_NUM_THREADS', '4'), ('OPENBLAS_NUM_THREADS', '4'), ('HDF5_USE_FILE_LOCKING', 'FALSE')):
    os.environ.setdefault(_k, _v)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s41eval.bing_eval import pipeline as P  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['tasks', 'run', 'local', 'status', 'collect', 'headline', 'summary', 'regress'])
    p.add_argument('--config', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--grid', default='g15', choices=['g15', 'g30'], help='headline: common grid (g30 = 3 deg sensitivity)')
    p.add_argument('--phase', default='1', help='1 (reduce + verif), reduce, verif, 2 (score + checks)')
    p.add_argument('--task', default=None)
    p.add_argument('--match', default=None, help='local: only tasks whose id contains this')
    p.add_argument('--a', default='ucs2s_sstanom')
    p.add_argument('--b', default='ucs2s_plain')
    p.add_argument('--v0', default='/lus/flare/projects/S2Storm/s2storm/runs/bing/scores/v0_ucs2s/scores.csv')
    p.add_argument('--set', action='append', default=[], help='override a top-level config key: key=json')
    a = p.parse_args(argv)
    ov = {k: json.loads(v) for k, v in (s.split('=', 1) for s in a.set)}
    cfg = P.load_config(a.config, ov)
    od = P.out_dir(cfg, a.tag)
    script = str(Path(__file__).resolve())
    if a.cmd == 'tasks':
        tf, n = P.make_tasks(cfg, a.tag, a.phase, script, Path(a.config).resolve(), cfg.get('est_s'))
        print(tf, n, 'tasks')
    elif a.cmd == 'run':
        t0 = time.time()
        r = P.run_task(cfg, a.tag, a.task)
        print(f'{a.task}: {r} ({time.time() - t0:.0f}s)', flush=True)
    elif a.cmd == 'local':
        tf, n = P.make_tasks(cfg, a.tag, a.phase, script, Path(a.config).resolve())
        for line in tf.read_text().splitlines():
            t = json.loads(line)
            tid = t['cmd'].split('--task ')[1]
            if a.match and a.match not in tid:
                continue
            t0 = time.time()
            print(tid, P.run_task(cfg, a.tag, tid), f'{time.time() - t0:.0f}s', flush=True)
    elif a.cmd == 'status':
        for ph in ('1', 'reduce', 'verif', '2'):
            tf = od / f'tasks_phase{ph}.jsonl'
            if not tf.exists():
                continue
            ids = [json.loads(x)['id'] for x in tf.read_text().splitlines()]
            st = od / f'state{ph}'
            done = sum((st / f'{i}.done').exists() for i in ids)
            failed = [i for i in ids if (st / f'{i}.failed').exists() and not (st / f'{i}.done').exists()]
            print(f'phase {ph}: {done}/{len(ids)} done, {len(failed)} failed', failed[:10])
    elif a.cmd == 'collect':
        P.collect(cfg, a.tag)
        from s41eval.bing_eval import report
        report.write_summary(cfg, a.tag)
    elif a.cmd == 'summary':
        from s41eval.bing_eval import report
        report.write_summary(cfg, a.tag)
    elif a.cmd == 'regress':
        from s41eval.bing_eval import report
        r = report.compare_v0(cfg, a.tag, a.v0)
        print(r['summary'])
        for w in r['worst']:
            print(w)
    elif a.cmd == 'headline':
        from s41eval.bing_eval import stats
        g = getattr(a, 'grid', 'g15') or 'g15'
        suf = '' if g == 'g15' else f'_{g}'
        rows = stats.run_headline(od, a.a, a.b, out_csv=od / f'headline_{a.a}_vs_{a.b}{suf}.csv', grid=g)
        print(len(rows), 'rows ->', od / f'headline_{a.a}_vs_{a.b}{suf}.csv')


if __name__ == '__main__':
    main()
