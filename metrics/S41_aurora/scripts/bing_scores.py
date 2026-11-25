"""CLI for s41eval.bing_scores (scores of the 2025-26 Monday starts: UC-S2S sstanom/plain, FuXi-S2S, IFS ext-range).

Environment (login node; the aifs-xpu venv's xarray has files on the hung flare OST 9 since 2026-11-11, so use the frameworks python
plus a pip --target lib that lives on a healthy OST):
    export PYTHONPATH=/lus/flare/projects/S2Storm/s2storm/envs/bing_pylib:/lus/flare/projects/S2Storm/s2storm/code/wt-bing
    export HDF5_USE_FILE_LOCKING=FALSE
    PY=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python3

    nice -n 10 $PY scripts/bing_scores.py --tag v1 --systems ucs2s_sstanom ucs2s_plain fuxi ifs --procs 8 [--common]
        [--vars t2m tp24 z500 msl u850 olr t850 u250 v250 sst] [--first 2025-01-01 --last 2026-12-31] [--root fuxi=<dir>]
        [--max_members N] [--no_mjo] [--no_figs] [--keep_cache]

Output: runs/bing/scores/<tag>/{scores.csv,scores.json,summary.md,maps_<system>.nc,figs/*.png,mjo.csv,mjo.json,mjo_skill.png,
meta.json,log.txt}. The window-mean member cache (cache/<system>_<var>.npy, ~2.5 GB per var for 30 members, ~8.4 GB for IFS 101) is
deleted at the end unless --keep_cache.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

for _k, _v in (('OMP_NUM_THREADS', '4'), ('MKL_NUM_THREADS', '4'), ('OPENBLAS_NUM_THREADS', '4'), ('HDF5_USE_FILE_LOCKING', 'FALSE')):
    os.environ.setdefault(_k, _v)            # shared login node: no 208-thread BLAS per worker
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s41eval import bing_scores as B  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True)
    p.add_argument('--systems', nargs='+', default=list(B.SYSTEMS), choices=list(B.SYSTEMS))
    p.add_argument('--vars', nargs='+', default=B.DEFAULT_VARS, choices=list(B.VARS))
    p.add_argument('--first', default='2025-01-01')
    p.add_argument('--last', default='2026-12-31')
    p.add_argument('--starts', nargs='+', default=None, help='explicit start dates (default: runs/bing/starts/starts.csv in range)')
    p.add_argument('--root', action='append', default=[], help='system=<dir> (h25: dir with <YYYYMMDD>/m<NN>_h25.nc; ifs: dir '
                   'with <YYYYMMDD>/p1sfc.grib ...). Defaults: runs/bing/hc/{sstanom,plain,fuxi}, runs/bing/ifs/data')
    p.add_argument('--procs', type=int, default=8)
    p.add_argument('--max_members', type=int, default=None)
    p.add_argument('--det_members_ifs', type=int, default=11, help='IFS members (0..N-1) for ACC, RMSE, spread/error and MJO')
    p.add_argument('--common', action='store_true', help='score every system on the starts all systems have')
    p.add_argument('--out_root', default=str(B.OUT))
    p.add_argument('--no_mjo', action='store_true')
    p.add_argument('--no_figs', action='store_true')
    p.add_argument('--keep_cache', action='store_true')
    a = p.parse_args(argv)
    starts = [date.fromisoformat(s) for s in a.starts] if a.starts else B.list_starts(first=a.first, last=a.last)
    roots = dict(r.split('=', 1) for r in a.root)
    B.DET_MEMBERS['ifs'] = a.det_members_ifs
    out = B.run(a.systems, a.vars, starts, a.tag, roots=roots, procs=a.procs, max_members=a.max_members, common=a.common,
                mjo=not a.no_mjo, figures=not a.no_figs, out_root=a.out_root)
    if not a.keep_cache:
        shutil.rmtree(out / 'cache', ignore_errors=True)
    print('output', out)


if __name__ == '__main__':
    main()
