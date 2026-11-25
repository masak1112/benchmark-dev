"""MJO rows of helper A's scorecards: the pre-registered headline MJO metric (``bing_headline_mjo.py``: mean daily bivariate COR of
the drift-corrected 30-member ensemble-mean RMM vs ERA5 RMM over the window's leads) for every lead window (w1 ... w6, w34, w56)
and for every pair of systems (each system against its own ERA5 RMM if they differ). Block bootstrap over starts in date order
(block 4, 1,000 draws, 95 % CI), grade as plan §2. Only w34 full-RMM SST-anomaly vs plain is pre-registered.

    python scripts/bing_helperA_mjo.py --modes runs/bing/eval/modes --out <csv>
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import xarray as xr

WINDOWS = {'w1': (1, 7), 'w2': (8, 14), 'w3': (15, 21), 'w4': (22, 28), 'w5': (29, 35), 'w6': (36, 42), 'w34': (15, 28), 'w56': (29, 42)}
PAIRS = [('ucs2s_sstanom', 'ucs2s_plain'), ('ucs2s_sstanom', 'fuxi'), ('ucs2s_plain', 'fuxi')]


def cor(f, o):
    ok = np.isfinite(f).all(-1) & np.isfinite(o).all(-1)
    ff = np.where(ok, (f ** 2).sum(-1), 0).sum(0)
    oo = np.where(ok, (o ** 2).sum(-1), 0).sum(0)
    num = np.where(ok, (f * o).sum(-1), 0).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / np.sqrt(ff * oo)


def metric(f, o, a, b):
    return float(np.nanmean(cor(f[:, a:b + 1], o[:, a:b + 1])))


def block_idx(n, block, rng):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modes', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261111)
    a = p.parse_args(argv)
    rows = []
    for index in ('full', 'winds'):
        for sa, sb in PAIRS:
            A = xr.open_dataset(f'{a.modes}/mjo/perstart_mjo_{index}_{sa}.nc')
            B = xr.open_dataset(f'{a.modes}/mjo/perstart_mjo_{index}_{sb}.nc')
            common = np.intersect1d(A.start.values, B.start.values)
            A, B = A.sel(start=common), B.sel(start=common)
            oa, ob = A.rmm_obs.values, B.rmm_obs.values
            same = bool(np.allclose(oa, ob, equal_nan=True))
            fa, fb = A.rmm_em30_dc.values, B.rmm_em30_dc.values
            yrs = common.astype('datetime64[Y]').astype(int) + 1970
            n = len(common)
            for wn, (l0, l1) in WINDOWS.items():
                def st(ix):
                    return metric(fa[ix], oa[ix], l0, l1) - metric(fb[ix], ob[ix], l0, l1)
                ma, mb = metric(fa, oa, l0, l1), metric(fb, ob, l0, l1)
                sub = {nm: st(np.flatnonzero(m)) for nm, m in (('2019-24', yrs <= 2024), ('2025-26', yrs >= 2025), ('no2019', yrs >= 2020))}
                rng = np.random.default_rng(a.seed)
                bs = [st(block_idx(n, 4, rng)) for _ in range(a.n_boot)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                d = ma - mb
                excl = lo > 0 or hi < 0
                same_sign = np.sign(sub['2019-24']) == np.sign(sub['2025-26']) == np.sign(d)
                grade = 'robust' if excl and same_sign else ('suggestive' if excl else 'noise')
                pre = index == 'full' and wn == 'w34' and (sa, sb) == PAIRS[0]
                rows.append(dict(field=f'mjo_{index}', window=wn, metric='mean_daily_cor', a_sys=sa, b_sys=sb, a=ma, b=mb, diff=d,
                                 ci_lo=lo, ci_hi=hi, n_starts=n, diff_2019_24=sub['2019-24'], diff_2025_26=sub['2025-26'],
                                 diff_no2019=sub['no2019'], grade=grade, same_obs=same,
                                 status='pre-registered' if pre else 'exploratory'))
                print(rows[-1], flush=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
