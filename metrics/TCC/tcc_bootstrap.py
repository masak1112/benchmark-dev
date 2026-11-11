"""
Block-bootstrap confidence intervals for the weekly TCC (regional, cos-lat area means).

Each resample draws init dates with replacement, recomputes the grid-point TCC from the
resampled sample, and area-averages it per region. Inits are resampled in blocks
(default: calendar month) because inits a few days apart verify over overlapping weeks,
so they are not independent; an i.i.d. bootstrap would give intervals that are too narrow.

With several models on the same dates, every model uses the SAME resampled dates, so the
bootstrap distribution of (model A - model B) is a paired comparison. Differences are
reported as first model minus each other model.

Usage:
  python tcc_bootstrap.py --models ucs2s_v2 fuxi --tag common86           # paired, 86 common inits
  python tcc_bootstrap.py --models fuxi --dates-file dates_fuxi_2017_2021.txt --tag y2017_2021
Output: results/tcc_bootstrap_{tag}_{block}.pkl
"""

import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np

import tcc_weekly
from tcc_weekly import (HERE, LABELS, MASKS, MODELS, REGIONS, RES_DIR, W2D, WEEK_NAMES,
                        list_dates, load_anomalies)


def block_ids(dates, block):
    """Integer block label per init date."""
    if block == 'none':
        return np.arange(len(dates))
    key = {'month': lambda d: d[:6], 'season': lambda d: d[:4] + str((int(d[4:6]) % 12) // 3)}[block]
    labels = OrderedDict((key(d), None) for d in dates)
    index = {k: i for i, k in enumerate(labels)}
    return np.array([index[key(d)] for d in dates])


def resample_weights(bid, n_boot, rng):
    """(n_boot + 1, n_init) multiplicity of each init per resample; row 0 = original sample."""
    nb = bid.max() + 1
    members = [np.where(bid == b)[0] for b in range(nb)]
    W = np.zeros((n_boot + 1, len(bid)))
    W[0] = 1.0
    for r in range(1, n_boot + 1):
        for b in rng.integers(0, nb, nb):          # draw nb blocks with replacement
            W[r, members[b]] += 1.0
    return W


def boot_regional_tcc(fa, oa, W):
    """fa, oa: (N, P) anomalies at P grid points; W: (R, N) weights.
    Returns (R, n_regions) cos-lat weighted area mean of the resampled grid-point TCC."""
    n = W.sum(axis=1, keepdims=True)
    sx, sy = W @ fa, W @ oa
    sxx, syy, sxy = W @ fa ** 2, W @ oa ** 2, W @ (fa * oa)
    cov = sxy / n - sx * sy / n ** 2
    vx = sxx / n - (sx / n) ** 2
    vy = syy / n - (sy / n) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where((vx > 0) & (vy > 0), cov / np.sqrt(vx * vy), np.nan)
    out = np.empty((W.shape[0], len(MASKS)))
    for j, m in enumerate(MASKS.values()):
        rm = r[:, m.ravel()]
        w = np.broadcast_to(W2D[m], rm.shape)
        ok = ~np.isnan(rm)
        out[:, j] = np.where(ok, rm * w, 0).sum(axis=1) / np.where(ok, w, 0).sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['ucs2s_v2', 'fuxi'], choices=list(MODELS))
    ap.add_argument('--dates-file', default=os.path.join(HERE, 'common_dates_ucs2sv2_fuxi.txt'))
    ap.add_argument('--tag', default='common86')
    ap.add_argument('--block', default='month', choices=['month', 'season', 'none'])
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-members', type=int, default=None,
                    help='read the truncated-ensemble caches written with tcc_weekly.py --n-members N')
    ap.add_argument('--detrend', action='store_true', help='use the 2002-2016 linear-trend climatology')
    ap.add_argument('--monthly-mean', action='store_true', help="subtract the sample's own mean per init month")
    ap.add_argument('--yearly-mean', action='store_true', help="subtract the sample's own mean per init year")
    a = ap.parse_args()
    tcc_weekly.N_MEMBERS = a.n_members
    tcc_weekly.DETREND = a.detrend
    tcc_weekly.MONTHLY_MEAN = a.monthly_mean
    tcc_weekly.YEARLY_MEAN = a.yearly_mean

    want = sorted({l.strip() for l in open(a.dates_file) if l.strip()})
    dates = sorted(set(want).intersection(*[set(list_dates(m)) for m in a.models]))
    anoms = {}
    for m in a.models:
        used, fa, oa = load_anomalies(m, dates)
        anoms[m] = (used, fa, oa)
    # every model must end up on the identical init list, or the pairing is meaningless
    common = sorted(set.intersection(*[set(v[0]) for v in anoms.values()]))
    for m, (used, fa, oa) in anoms.items():
        keep = [used.index(d) for d in common]
        anoms[m] = (fa[keep], oa[keep])
    n = len(common)

    bid = block_ids(common, a.block)
    rng = np.random.default_rng(a.seed)
    W = resample_weights(bid, a.n_boot, rng)
    print(f'{n} common inits, {bid.max() + 1} blocks ({a.block}), {a.n_boot} resamples', flush=True)

    # boot[model][label] -> (R+1, n_weeks, n_regions); row 0 = full-sample estimate
    boot = {m: {} for m in a.models}
    for m in a.models:
        fa, oa = anoms[m]
        for i, lbl in enumerate(LABELS):
            if lbl in MODELS[m].get('missing', ()):
                boot[m][lbl] = np.full((W.shape[0], len(WEEK_NAMES), len(MASKS)), np.nan)
                continue
            boot[m][lbl] = np.stack([boot_regional_tcc(fa[:, i, k].reshape(n, -1), oa[:, i, k].reshape(n, -1), W)
                                     for k in range(len(WEEK_NAMES))], axis=1)
        print(f'  {MODELS[m]["name"]} done', flush=True)
        del anoms[m]

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f'tcc_bootstrap_{a.tag}_{a.block}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'boot': boot, 'models': {m: MODELS[m]['name'] for m in a.models},
                     'weeks': WEEK_NAMES, 'regions': list(MASKS), 'init_dates': common,
                     'block': a.block, 'n_blocks': int(bid.max() + 1), 'n_boot': a.n_boot,
                     'seed': a.seed}, f)
    print(f'Saved {out}\n')

    def ci(x):
        return np.nanpercentile(x[1:], [2.5, 97.5], axis=0)

    g = list(MASKS).index('global')
    for m in a.models:
        print(f'=== {MODELS[m]["name"]}: global TCC  [95% CI] ===')
        for lbl in LABELS:
            b = boot[m][lbl][:, :, g]
            if np.isnan(b).all():
                continue
            lo, hi = ci(b)
            print(f'{lbl:5s} ' + '  '.join(f'{w}:{b[0, k]:.3f}[{lo[k]:.3f},{hi[k]:.3f}]'
                                           for k, w in enumerate(WEEK_NAMES)))
    m1 = a.models[0]
    for m2 in a.models[1:]:
        print(f'\n=== {MODELS[m1]["name"]} - {MODELS[m2]["name"]}: global TCC diff [95% CI] '
              f'(* = CI excludes 0) ===')
        for lbl in LABELS:
            d = boot[m1][lbl][:, :, g] - boot[m2][lbl][:, :, g]
            if np.isnan(d).all():
                continue
            lo, hi = ci(d)
            print(f'{lbl:5s} ' + '  '.join(
                f'{w}:{d[0, k]:+.3f}[{lo[k]:+.3f},{hi[k]:+.3f}]{"*" if lo[k] > 0 or hi[k] < 0 else " "}'
                for k, w in enumerate(WEEK_NAMES)))


if __name__ == '__main__':
    main()
