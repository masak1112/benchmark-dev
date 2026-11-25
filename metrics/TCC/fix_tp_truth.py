"""
One-off fix of the cached precipitation TRUTH for FuXi and NeuralGCM (2026-10-12).

Their tp for lead day L is the total over day L (00->24 UTC), which is ERA5 total_precipitation_24hr at
00 UTC of day L+1 (pattern correlation 0.84 vs 0.71 for the daily-mean truth of day L that was used before).
The forecasts in the cache are unchanged; only the tp entry of 'obs' is recomputed with the new truth
(tcc_weekly.truth_spec). The previous values are kept in 'obs_tp_before_fix' so the change is reversible.
Idempotent: files already carrying tp_truth='00z+1' are skipped.
"""

import os
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np

import tcc_weekly as t

CACHE_SUBDIRS = ['fuxi', 'fuxi_m20', 'ngcm', 'ngcm_m20']
I_TP = t.LABELS.index('tp')
TP_SUBDIR, TP_LEVEL = next((sub, lev) for lbl, sub, lev, *_ in t.VARIABLES if lbl == 'tp')


def fix(path):
    z = dict(np.load(path))
    if str(z.get('tp_truth', '')) == '00z+1':
        return path, 'already fixed'
    model = 'fuxi' if os.path.basename(os.path.dirname(path)).startswith('fuxi') else 'ngcm'
    daily, offset = t.truth_spec(model, 'tp')
    assert (daily, offset) == (False, 1), (model, daily, offset)
    init = os.path.basename(path)[:8]
    t0 = datetime.strptime(init, '%Y%m%d')
    era5 = t.ERA5(daily)
    try:
        obs = []
        for L in range(1, 7 * t.N_WEEKS + 1):
            o = era5.get(TP_SUBDIR, TP_LEVEL, t0 + timedelta(days=L + offset))
            if o is None:
                break
            obs.append(o)
    finally:
        era5.close()
    new = np.full(z['obs'][I_TP].shape, np.nan, dtype=np.float32)
    for k in range(t.N_WEEKS):
        if len(obs) >= 7 * (k + 1):
            new[k] = np.mean(obs[7 * k:7 * (k + 1)], axis=0)
    z['obs_tp_before_fix'] = z['obs'][I_TP].copy()
    z['obs'][I_TP] = new
    z['tp_truth'] = np.array('00z+1')
    np.savez(path + '.tmp.npz', **z)
    os.replace(path + '.tmp.npz', path)
    return path, 'fixed'


if __name__ == '__main__':
    files = sorted(os.path.join(t.CACHE_DIR, d, f) for d in CACHE_SUBDIRS
                   if os.path.isdir(os.path.join(t.CACHE_DIR, d))
                   for f in os.listdir(os.path.join(t.CACHE_DIR, d)) if f.endswith('.npz') and f[:8].isdigit())
    print(f'{len(files)} cache files', flush=True)
    counts = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 32) as p:
        for k, (path, msg) in enumerate(p.imap_unordered(fix, files), 1):
            counts[msg] = counts.get(msg, 0) + 1
            if k % 200 == 0:
                print(f'  {k}/{len(files)} {counts}', flush=True)
    print('done', counts, flush=True)
