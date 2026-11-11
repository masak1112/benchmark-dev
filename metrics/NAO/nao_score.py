"""
Weekly DJF NAO skill of the ensemble mean (Phase A: from the TCC weekly caches).

For every DJF init (Dec/Jan/Feb) in tcc_cache/{model}[_m20]/, the cached weekly-mean
ensemble-mean Z500 (and the matching ERA5 truth) minus the model's ERA5 DOY climatology
(same rules as tcc_weekly.py: 00Z models vs clim_00z, daily-mean models vs clim_dailymean)
is projected onto the ERA5 DJF NAO EOF (nao_eof.py). Index unit = observed weekly NAO std.

Scores per lead week (week1..6, week3-4, week5-6):
  cor   correlation of forecast vs ERA5 index across inits
  rmse  RMSE (climatology forecast ~ 1)
  amp   std(forecast index) / std(ERA5 index)
  bias  mean(forecast - ERA5)
  pers_cor  persistence baseline: ERA5 index of days -6..0 correlated with ERA5 week k
95% CIs: block bootstrap (calendar-month blocks of inits, 1000 resamples).
Paired differences (A - B, same resampled inits) for every model pair sharing >= MIN_COMMON inits.

Usage:  python nao_score.py [--models ...] [--years 2019 2021 --tag y1921]   (default: every DJF init, tag alldjf)
Output: results/nao_weekly_{model}_{tag}.csv, results/nao_scores_{tag}.pkl
"""

import argparse
import csv
import itertools
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TCC'))
import tcc_weekly  # noqa: E402
from tcc_weekly import CLIM_ROOT, LABELS, MODELS, N_WEEKS, WEEK_NAMES, cache_dir, weekly_clim  # noqa: E402
from tcc_bootstrap import block_ids, resample_weights  # noqa: E402

RES_DIR = os.path.join(HERE, 'results')
EOF_FILE = os.path.join(RES_DIR, 'nao_eof_djf.npz')
DJF = ('12', '01', '02')
N_MEMBERS = 20
# extra caches (dir name, members) read for inits missing from the 20-member cache;
# SI-S2S Dec 2020 inits only have 16 members (seed 3 missing)
EXTRA_CACHES = {'si_s2s': [('si_s2s_m16', 16)]}
N_BOOT = 1000
MIN_COMMON = 15
SEED = 0


def load_eof():
    e = np.load(EOF_FILE)
    return {k: e[k] for k in e.files}


def project(field, e):
    """(..., 180, 360) anomaly -> raw NAO index (...)."""
    box = field[..., e['lat_idx'], :][..., e['lon_idx']]
    return np.einsum('...ij,ij->...', box * e['wgt'], e['eof'])


def model_index(model, e, years=None):
    """Weekly forecast and ERA5 NAO index for the model's cached DJF inits (init year in
    `years` if given). Returns dates, fc (N, 8), ob (N, 8) in ERA5 weekly-std units, n_mem (N,)."""
    cdir, cpat = MODELS[model]['clim']
    clim = np.load(os.path.join(CLIM_ROOT, cdir, cpat.format('z500')))
    iz = LABELS.index('z500')
    caches = [(cache_dir(model), N_MEMBERS)] + [(os.path.join(tcc_weekly.CACHE_DIR, c), n)
                                                 for c, n in EXTRA_CACHES.get(model, [])]
    files = {}
    for d, nmem in reversed(caches):            # earlier caches win
        if os.path.isdir(d):
            files.update({f[:8]: (os.path.join(d, f), nmem) for f in os.listdir(d) if f.endswith('.npz')})
    dates, fc, ob, n_mem = [], [], [], []
    for init in sorted(files):
        if init[4:6] not in DJF or (years and not years[0] <= int(init[:4]) <= years[1]):
            continue
        path, nmem = files[init]
        z = np.load(path)
        assert list(z['labels']) == LABELS
        assert int(z['n_members']) == nmem, f'{model} {init}: {int(z["n_members"])} members'
        fz, oz = z['fc'][iz].astype(np.float64), z['obs'][iz].astype(np.float64)
        if np.isnan(fz).any() or np.isnan(oz).any():
            print(f'  {model} {init}: incomplete weeks, skipped')
            continue
        c = weekly_clim(clim, init)
        dates.append(init)
        n_mem.append(nmem)
        fc.append(project(fz - c, e))
        ob.append(project(oz - c, e))
    fc, ob = np.array(fc) / e['sd_weekly'], np.array(ob) / e['sd_weekly']
    # weeks 3-4 / 5-6 = mean of the weekly indices
    fc = np.concatenate([fc, fc[:, 2:4].mean(1, keepdims=True), fc[:, 4:6].mean(1, keepdims=True)], axis=1)
    ob = np.concatenate([ob, ob[:, 2:4].mean(1, keepdims=True), ob[:, 4:6].mean(1, keepdims=True)], axis=1)
    return dates, fc, ob, np.array(n_mem)


def persistence(dates, e):
    """ERA5 daily-mean NAO index averaged over days -6..0 of each init (weekly-std units)."""
    pos = {d: i for i, d in enumerate(e['dates'])}
    out = []
    for init in dates:
        i = pos[init]
        out.append(e['pc_daily'][i - 6:i + 1].mean())
    return np.array(out) / e['sd_weekly']


def era5_week_index(dates, e):
    """ERA5 daily-mean NAO index averaged over lead days 7k-6..7k, k=1..6 (for the cache check)."""
    pos = {d: i for i, d in enumerate(e['dates'])}
    return np.array([[e['pc_daily'][pos[init] + 7 * k + 1:pos[init] + 7 * k + 8].mean()
                      for k in range(N_WEEKS)] for init in dates]) / e['sd_weekly']


def wstats(f, o, W):
    """Weighted scores for every resample. f, o: (N, K); W: (R, N). Returns dict of (R, K)."""
    n = W.sum(1, keepdims=True)
    mf, mo = W @ f / n, W @ o / n
    vf, vo = W @ f ** 2 / n - mf ** 2, W @ o ** 2 / n - mo ** 2
    cov = W @ (f * o) / n - mf * mo
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'cor': cov / np.sqrt(vf * vo),
                'rmse': np.sqrt(W @ (f - o) ** 2 / n),
                'amp': np.sqrt(vf / vo),
                'bias': mf - mo}


def summarize(stats):
    """{metric: (R, K)} -> {metric: {'val', 'lo', 'hi'}} (row 0 = original sample)."""
    return {m: {'val': s[0], 'lo': np.nanpercentile(s[1:], 2.5, axis=0),
                'hi': np.nanpercentile(s[1:], 97.5, axis=0)} for m, s in stats.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['ucs2s_v2', 'fuxi', 'si_s2s', 'ngcm'])
    p.add_argument('--years', nargs=2, type=int, help='first and last init year (default: every DJF init)')
    p.add_argument('--tag', default='alldjf', help='output suffix')
    a = p.parse_args()
    tcc_weekly.N_MEMBERS = N_MEMBERS
    e = load_eof()
    rng = np.random.default_rng(SEED)
    print(f'ERA5 DJF NAO EOF1: {e["evr"][0]:.1%} variance; scores in weekly-std units\n')

    data = {}
    for m in a.models:
        dates, fc, ob, n_mem = model_index(m, e, a.years)
        pers = persistence(dates, e)
        # check: cached ERA5 truth projected == ERA5 daily index averaged over the same days
        chk = era5_week_index(dates, e)
        r = np.corrcoef(chk.ravel(), ob[:, :N_WEEKS].ravel())[0, 1]
        winters = sorted({int(d[:4]) + (d[4:6] == '12') for d in dates})
        print(f'{MODELS[m]["name"]}: {len(dates)} DJF inits, {len(winters)} winters {winters}, '
              f'members {sorted(set(n_mem))}; cache-truth vs ERA5 daily index: '
              f'r={r:.4f}, max|diff|={np.abs(chk - ob[:, :N_WEEKS]).max():.3f}')
        data[m] = dict(dates=dates, fc=fc, ob=ob, pers=pers, n_mem=n_mem)
        os.makedirs(RES_DIR, exist_ok=True)
        with open(os.path.join(RES_DIR, f'nao_weekly_{m}_{a.tag}.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['init'] + [f'fc_{k}' for k in WEEK_NAMES] + [f'era5_{k}' for k in WEEK_NAMES] + ['era5_days-6..0'])
            for i, d in enumerate(dates):
                w.writerow([d] + [f'{x:.4f}' for x in fc[i]] + [f'{x:.4f}' for x in ob[i]] + [f'{pers[i]:.4f}'])

    own = {}
    for m, D in data.items():
        W = resample_weights(block_ids(D['dates'], 'month'), N_BOOT, rng)
        s = wstats(D['fc'], D['ob'], W)
        s['pers_cor'] = wstats(np.repeat(D['pers'][:, None], len(WEEK_NAMES), 1), D['ob'], W)['cor']
        own[m] = dict(n=len(D['dates']), dates=D['dates'], n_mem=D['n_mem'], **summarize(s))

    pairs = {}
    for ma, mb in itertools.combinations(data, 2):
        common = sorted(set(data[ma]['dates']) & set(data[mb]['dates']))
        if len(common) < MIN_COMMON:
            continue
        ia = [data[ma]['dates'].index(d) for d in common]
        ib = [data[mb]['dates'].index(d) for d in common]
        # each model keeps its own truth convention (00Z vs daily mean); they differ by <~0.2 std
        assert np.allclose(data[ma]['ob'][ia], data[mb]['ob'][ib], atol=0.3), 'ERA5 truth differs a lot'
        W = resample_weights(block_ids(common, 'month'), N_BOOT, rng)
        sa = wstats(data[ma]['fc'][ia], data[ma]['ob'][ia], W)
        sb = wstats(data[mb]['fc'][ib], data[mb]['ob'][ib], W)
        res = {'n': len(common), 'dates': common, 'a': summarize(sa), 'b': summarize(sb),
               'diff': summarize({k: sa[k] - sb[k] for k in ('cor', 'rmse', 'amp')})}
        for k in ('cor', 'rmse'):
            d = sa[k][1:] - sb[k][1:]
            res['diff'][k]['p_gt0'] = (d > 0).mean(axis=0)
        pairs[(ma, mb)] = res

    with open(os.path.join(RES_DIR, f'nao_scores_{a.tag}.pkl'), 'wb') as fh:
        pickle.dump({'week_names': WEEK_NAMES, 'own': own, 'pairs': pairs,
                     'names': {m: MODELS[m]['name'] for m in data}, 'n_boot': N_BOOT,
                     'evr': e['evr'], 'members': N_MEMBERS, 'tag': a.tag, 'years': a.years}, fh)

    cols = ['week1', 'week2', 'week3', 'week4', 'week3-4', 'week5-6']
    ci = [WEEK_NAMES.index(c) for c in cols]
    print('\nOwn DJF record — correlation [95% CI]  (persistence)')
    for m, r in own.items():
        print(f'  {MODELS[m]["name"]:<10} n={r["n"]:3d} ' + '  '.join(
            f'{c}:{r["cor"]["val"][k]:.2f}[{r["cor"]["lo"][k]:.2f},{r["cor"]["hi"][k]:.2f}]({r["pers_cor"]["val"][k]:.2f})'
            for c, k in zip(cols, ci)))
    print('\nOwn DJF record — RMSE / amplitude ratio / bias')
    for m, r in own.items():
        print(f'  {MODELS[m]["name"]:<10} ' + '  '.join(
            f'{c}:{r["rmse"]["val"][k]:.2f}/{r["amp"]["val"][k]:.2f}/{r["bias"]["val"][k]:+.2f}' for c, k in zip(cols, ci)))
    print('\nPaired differences (A - B) on common inits — correlation [95% CI]')
    for (ma, mb), r in pairs.items():
        print(f'  {MODELS[ma]["name"]} - {MODELS[mb]["name"]} (n={r["n"]}): ' + '  '.join(
            f'{c}:{r["diff"]["cor"]["val"][k]:+.2f}[{r["diff"]["cor"]["lo"][k]:+.2f},{r["diff"]["cor"]["hi"][k]:+.2f}]'
            for c, k in zip(cols, ci)))
    print(f'\nwrote {RES_DIR}/nao_scores_{a.tag}.pkl and nao_weekly_*_{a.tag}.csv')


if __name__ == '__main__':
    main()
