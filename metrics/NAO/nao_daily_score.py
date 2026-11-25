"""
Daily DJF NAO skill by lead day (1..42) from the per-member index cache (nao_extract.py).

Index unit = std of the ERA5 DJF daily index (1979-2018). Truth = ERA5 index of the valid day:
00Z ERA5 (pc_00z) for the 00Z models (UC-S2S, SI-S2S), daily-mean ERA5 (pc_daily) for FuXi / NeuralGCM.

Scores per lead day:
  acc    anomaly correlation (uncentered, like the MJO ACC): sum(f*o) / sqrt(sum f^2 * sum o^2)
         over inits, f / o = ensemble-mean / ERA5 index anomaly; unlike cor it is hurt by mean bias
  cor    correlation of the ensemble mean with ERA5 across inits (centered)
  rmse   RMSE of the ensemble mean (climatology ~ 1)
  amp    std(ensemble mean) / std(ERA5)
  bias   mean(ensemble mean - ERA5)
  ssr    spread / RMSE, spread = sqrt(mean ensemble variance * (M+1)/M)  (1 = reliable)
  crpss  1 - fair CRPS / CRPS of the climatological N(0, 1)
  pers_cor / pers_acc  persistence baseline: ERA5 index on the init day vs ERA5 at lead L
95% CIs: block bootstrap by calendar month of init (1000 resamples). Paired differences (A - B,
same resampled inits) for model pairs sharing >= MIN_COMMON inits.

Usage:  python nao_daily_score.py [--models ...] [--years 2019 2021 --tag y1921]
Output: results/nao_daily_scores_{tag}.pkl
"""

import argparse
import itertools
import os
import pickle
import sys
from datetime import datetime

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TCC'))
from tcc_weekly import MODELS  # noqa: E402
# ECMWF S2S reforecasts (nao_extract.py --model ecmwf_rf2018): 00 UTC instantaneous -> 00Z truth
MODELS['ecmwf_rf2018'] = {'name': 'ECMWF', 'daily_mean': False}
# ECMWF S2S real-time forecasts 2019- (nao_extract.py --model ecmwf_rt), 20 members, 00 UTC
MODELS['ecmwf_rt'] = {'name': 'ECMWF', 'daily_mean': False}
# ECMWF 2025-model hindcasts of 2019-2021 (nao_extract.py --model ecmwf_rf2025), 11 members, 00 UTC
MODELS['ecmwf_rf2025'] = {'name': 'ECMWF hindcast', 'daily_mean': False}
# UC-S2S V2 with feedback correction, all variables (pp_si_fbcorr_allvars), 20 members, 00 UTC
MODELS['ucs2s_v2_fbcorr'] = {'name': 'UC-S2S V2-fbcorr', 'daily_mean': False}
# UC-S2S multi-condition run (pp_si_infer_multicond_2019/predictions), 20 members, 00 UTC
MODELS['ucs2s_multicond'] = {'name': 'UC-S2S multicond', 'daily_mean': False}
# UC-S2S multi-condition pp12800 run (pp_si_infer_multicond_pp12800/predictions), 20 members, 00 UTC
MODELS['ucs2s_mc12800'] = {'name': 'UC-S2S multicond pp12800', 'daily_mean': False}
# UC-S2S SST-anomaly conditioning run (pp_si_infer_sstanom/predictions), 20 members, 00 UTC
MODELS['ucs2s_sstanom'] = {'name': 'UC-S2S SSTanom', 'daily_mean': False}
from tcc_bootstrap import block_ids, resample_weights  # noqa: E402

RES_DIR = os.path.join(HERE, 'results')
SEASON = os.environ.get('NAO_SEASON', 'djf')          # ndjfm: Nov-Mar pattern + cache (see nao_extract.py)
EOF_FILE = os.path.join(RES_DIR, f'nao_eof_{SEASON}.npz')
CACHE = '/glade/derecho/scratch/bgong/nao_cache' + ('' if SEASON == 'djf' else f'_{SEASON}')
N_LEAD = 42
N_BOOT = 1000
MIN_COMMON = 15
SEED = 0
CRPS_CLIM = 2 * norm.pdf(0) - 1 / np.sqrt(np.pi)       # mean CRPS of N(0,1) vs N(0,1) truth (reference only)


def crps_normal(y):
    """CRPS of the N(0, 1) climatological forecast for observation(s) y."""
    return y * (2 * norm.cdf(y) - 1) + 2 * norm.pdf(y) - 1 / np.sqrt(np.pi)


def fair_crps(ens, y):
    """ens (M, L), y (L) -> fair CRPS (L)."""
    m = ens.shape[0]
    t1 = np.abs(ens - y).mean(axis=0)
    t2 = np.abs(ens[:, None] - ens[None]).sum(axis=(0, 1)) / (2 * m * (m - 1))
    return t1 - t2


def load_model(model, e, years):
    truth = e['pc_daily' if MODELS[model]['daily_mean'] else 'pc_00z'] / e['sd_daily']
    pos = {d: i for i, d in enumerate(e['dates'])}
    d = os.path.join(CACHE, model)
    dates, em, ob, var, crps, cref, pers, nmem = [], [], [], [], [], [], [], []
    for f in sorted(os.listdir(d)):
        init = f[:8]
        if not f.endswith('.npz') or (years and not years[0] <= int(init[:4]) <= years[1]):
            continue
        z = np.load(os.path.join(d, f))
        x = z['idx'][:, 1:N_LEAD + 1] / e['sd_daily']            # (M, 42), leads 1..42
        m = x.shape[0]
        i0 = pos[init]
        y = truth[i0 + 1:i0 + N_LEAD + 1]
        if len(y) < N_LEAD:                                       # verification beyond the ERA5 record
            continue
        dates.append(init)
        em.append(x.mean(axis=0))
        ob.append(y)
        var.append(x.var(axis=0, ddof=1) * (m + 1) / m)
        crps.append(fair_crps(x, y))
        cref.append(crps_normal(y))
        pers.append(truth[i0])
        nmem.append(m)
    return dict(dates=dates, em=np.array(em), ob=np.array(ob), var=np.array(var),
                crps=np.array(crps), cref=np.array(cref), pers=np.array(pers), n_mem=np.array(nmem))


def subset(D, idx):
    return {k: ([v[i] for i in idx] if isinstance(v, list) else v[idx]) for k, v in D.items()}


def match_dates(D, ref_dates, tol):
    """Re-key D onto ref_dates: for each reference init take D's nearest init within tol days.
    The model keeps its own truth (its own valid days); 'init' holds its actual init dates."""
    t = np.array([np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:]}') for d in D['dates']])
    idx, keys, off = [], [], []
    for r in ref_dates:
        dd = (t - np.datetime64(f'{r[:4]}-{r[4:6]}-{r[6:]}')).astype(int)
        k = int(np.argmin(np.abs(dd)))
        if abs(dd[k]) <= tol:
            idx.append(k); keys.append(r); off.append(int(dd[k]))
    out = subset(D, np.array(idx))
    out['init'] = out['dates']
    out['dates'] = keys
    out['offset'] = np.array(off)
    return out


def wstats(D, idx, W):
    """Scores for every resample. idx selects inits; W (R, n). Returns {metric: (R, N_LEAD)}."""
    f, o = D['em'][idx], D['ob'][idx]
    n = W.sum(1, keepdims=True)
    mf, mo = W @ f / n, W @ o / n
    vf, vo = W @ f ** 2 / n - mf ** 2, W @ o ** 2 / n - mo ** 2
    cov = W @ (f * o) / n - mf * mo
    mse = W @ (f - o) ** 2 / n
    p = np.repeat(D['pers'][idx][:, None], f.shape[1], 1)
    mp = W @ p / n
    covp = W @ (p * o) / n - mp * mo
    vp = W @ p ** 2 / n - mp ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'acc': (W @ (f * o)) / np.sqrt((W @ f ** 2) * (W @ o ** 2)),
                'pers_acc': (W @ (p * o)) / np.sqrt((W @ p ** 2) * (W @ o ** 2)),
                'cor': cov / np.sqrt(vf * vo), 'rmse': np.sqrt(mse), 'amp': np.sqrt(vf / vo),
                'bias': mf - mo, 'ssr': np.sqrt(W @ D['var'][idx] / n / mse),
                'crpss': 1 - (W @ D['crps'][idx]) / (W @ D['cref'][idx]),
                'pers_cor': covp / np.sqrt(vp * vo)}


def summarize(stats):
    return {m: {'val': s[0], 'lo': np.nanpercentile(s[1:], 2.5, axis=0),
                'hi': np.nanpercentile(s[1:], 97.5, axis=0)} for m, s in stats.items()}


def first_below(x, thr):
    """First lead day (1-based) where x drops below thr; None if never."""
    k = np.where(x < thr)[0]
    return int(k[0]) + 1 if len(k) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['ucs2s_v2', 'fuxi', 'si_s2s', 'ngcm'])
    p.add_argument('--years', nargs=2, type=int, help='first and last init year (default: all DJF inits)')
    p.add_argument('--months', nargs='+', help='init months to keep (MM), default: all cached')
    p.add_argument('--start', help='first init date YYYYMMDD (applied after --years)')
    p.add_argument('--end', help='last init date YYYYMMDD')
    p.add_argument('--match-to', help='reference model: every other model uses its nearest init within '
                                      '--match-tol days of each reference init, so all share the same date keys')
    p.add_argument('--match-tol', type=int, default=3)
    p.add_argument('--ref-dates-file', help='restrict the --match-to model to these inits (YYYYMMDD per line)')
    p.add_argument('--monthu', nargs='+', default=[], metavar='MODEL',
                   help='keep only Monday/Thursday inits for these models (e.g. ecmwf_rf2025, whose hindcasts start every other day)')
    p.add_argument('--tag', default='alldjf')
    a = p.parse_args()
    e = {k: v for k, v in np.load(EOF_FILE).items()}
    rng = np.random.default_rng(SEED)

    data = {m: load_model(m, e, a.years) for m in a.models}
    for m, D in data.items():
        keep = np.array([(not a.start or d >= a.start) and (not a.end or d <= a.end)
                         and (not a.months or d[4:6] in a.months)
                         and (m not in a.monthu or datetime.strptime(d, '%Y%m%d').weekday() in (0, 3))
                         for d in D['dates']])
        data[m] = subset(D, np.where(keep)[0])
    if a.match_to:
        if a.ref_dates_file:
            want = {l.strip() for l in open(a.ref_dates_file) if l.strip()}
            D = data[a.match_to]
            data[a.match_to] = subset(D, np.array([i for i, d in enumerate(D['dates']) if d in want], dtype=int))
        ref = data[a.match_to]['dates']
        for m in data:
            if m != a.match_to:
                data[m] = match_dates(data[m], ref, a.match_tol)
                print(f'{MODELS[m]["name"]}: {len(data[m]["dates"])}/{len(ref)} {MODELS[a.match_to]["name"]} inits matched '
                      f'within {a.match_tol} days (mean |offset| {np.mean(np.abs(data[m]["offset"])):.1f} d)')
    own = {}
    for m, D in data.items():
        n = len(D['dates'])
        W = resample_weights(block_ids(D['dates'], 'month'), N_BOOT, rng)
        own[m] = dict(n=n, dates=D['dates'], n_mem=D['n_mem'], **summarize(wstats(D, np.arange(n), W)))

    pairs = {}
    for ma, mb in itertools.combinations(data, 2):
        common = sorted(set(data[ma]['dates']) & set(data[mb]['dates']))
        if len(common) < MIN_COMMON:
            continue
        ia = np.array([data[ma]['dates'].index(d) for d in common])
        ib = np.array([data[mb]['dates'].index(d) for d in common])
        W = resample_weights(block_ids(common, 'month'), N_BOOT, rng)
        sa, sb = wstats(data[ma], ia, W), wstats(data[mb], ib, W)
        pairs[(ma, mb)] = {'n': len(common), 'dates': common, 'a': summarize(sa), 'b': summarize(sb),
                           'diff': summarize({k: sa[k] - sb[k] for k in ('acc', 'cor', 'rmse', 'crpss', 'ssr')})}

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f'nao_daily_scores_{a.tag}.pkl')
    with open(out, 'wb') as fh:
        pickle.dump({'leads': np.arange(1, N_LEAD + 1), 'own': own, 'pairs': pairs,
                     'names': {m: MODELS[m]['name'] for m in data}, 'n_boot': N_BOOT,
                     'tag': a.tag, 'years': a.years, 'start': a.start, 'end': a.end, 'months': a.months,
                     'match_to': a.match_to, 'match_tol': a.match_tol}, fh)

    show = [1, 5, 7, 10, 14, 21, 28, 35]
    print('Own DJF record (daily index):')
    for m, r in own.items():
        winters = sorted({int(d[:4]) + (d[4:6] == '12') for d in r['dates']})
        print(f'\n  {data[m] and MODELS[m]["name"]}: n={r["n"]}, winters {winters}, members {sorted(set(r["n_mem"]))}')
        print(f'    ACC<0.6 at day {first_below(r["acc"]["val"], 0.6)}, ACC<0.5 at day {first_below(r["acc"]["val"], 0.5)} '
              f'[CI: {first_below(r["acc"]["lo"], 0.5)}-{first_below(r["acc"]["hi"], 0.5)}]; '
              f'persistence ACC<0.5 at day {first_below(r["pers_acc"]["val"], 0.5)}')
        print(f'    cor<0.6 at day {first_below(r["cor"]["val"], 0.6)}, cor<0.5 at day {first_below(r["cor"]["val"], 0.5)}, '
              f'CRPSS<0 at day {first_below(r["crpss"]["val"], 0.0)}; persistence cor<0.5 at day {first_below(r["pers_cor"]["val"], 0.5)}')
        for k in ('acc', 'cor', 'rmse', 'crpss', 'ssr', 'amp', 'bias', 'pers_cor'):
            v = r[k]
            print(f'    {k:8s} ' + ' '.join(f'd{L}:{v["val"][L - 1]:+.2f}' for L in show))
    print('\nPaired differences A - B (ACC; * = 95% CI excludes 0):')
    for (ma, mb), r in pairs.items():
        d = r['diff']['acc']
        print(f'  {MODELS[ma]["name"]} - {MODELS[mb]["name"]} (n={r["n"]}): ' + ' '.join(
            f'd{L}:{d["val"][L - 1]:+.2f}{"*" if d["lo"][L - 1] > 0 or d["hi"][L - 1] < 0 else ""}' for L in show))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
