#!/usr/bin/env python
"""S41 W5b equatorial waves (helper C, exploratory): extraction of the 15S-15N strip, ERA5 strip, WK spectra, filtered-wave skill.
    python scripts/bing_waves.py era5
    python scripts/bing_waves.py tasks [--chunk 8] [--out F]
    python scripts/bing_waves.py extract --system S --starts YYYY-MM-DD ...
    python scripts/bing_waves.py spectra --system S [--workers N]
    python scripts/bing_waves.py filtered --system S [--workers N]"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval import bing_modes as BM  # noqa: E402
from s41eval import bing_scores as BS  # noqa: E402
from s41eval import bing_waves as W  # noqa: E402

SYSTEMS = ['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']


def log(*a):
    print(f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}', *a, flush=True)


def cmd_era5(a):
    m = W.build_era5_eq(first_year=a.first_year)
    log(json.dumps({k: (v['n'], v['n_missing']) for k, v in m.items()}))


def cmd_tasks(a):
    st = BM.list_starts()
    lines = []
    for s in a.systems:
        todo = [d for d in st if not W.eq_path(s, d).exists() and BM.start_dir(s, d).is_dir()]
        for i in range(0, len(todo), a.chunk):
            ch = todo[i:i + a.chunk]
            lines.append(json.dumps(dict(id=f'S41e_C_eq_{s}_{ch[0]:%Y%m%d}_{len(ch)}', est_s=45 * len(ch),
                                         cmd=f'python $C/scripts/bing_waves.py extract --system {s} --starts '
                                             + ' '.join(f'{d}' for d in ch))))
    if a.era5:
        lines.insert(0, json.dumps(dict(id='S41e_C_era5eq', est_s=900, cmd='python $C/scripts/bing_waves.py era5')))
    Path(a.out).write_text('\n'.join(lines) + '\n')
    log(len(lines), 'tasks ->', a.out)


def cmd_extract(a):
    bad = []
    for st in a.starts:
        d = date.fromisoformat(st)
        out = W.eq_path(a.system, d)
        if out.exists() and not a.force:
            log('exists', out)
            continue
        try:
            W.extract_eq_start(a.system, d, out, log=log)
        except Exception:  # noqa: BLE001
            import traceback
            log(f'ERROR {a.system} {d}: {traceback.format_exc()[-1500:]}')
            bad.append(st)
    if bad:
        raise SystemExit(f'failed starts {bad}')


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('era5')
    e.add_argument('--first_year', type=int, default=2017)
    t = sub.add_parser('tasks')
    t.add_argument('--systems', nargs='+', default=SYSTEMS)
    t.add_argument('--chunk', type=int, default=8)
    t.add_argument('--era5', action='store_true')
    t.add_argument('--out', default=str(W.OUT / 'tasks_eq.jsonl'))
    x = sub.add_parser('extract')
    x.add_argument('--system', required=True)
    x.add_argument('--starts', nargs='+', required=True)
    x.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    {'era5': cmd_era5, 'tasks': cmd_tasks, 'extract': cmd_extract}[a.cmd](a)


if __name__ == '__main__':
    main()
