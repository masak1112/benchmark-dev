"""Helper-C utilities (S41 W5/W5b/W6, exploratory): moving-block bootstrap and grades (copied from ``bing_eval.stats`` on branch
``bing-eval`` 5cb98fa, same rules: block 4 consecutive starts in date order, 1,000 draws, 95 % percentile CI; robust = CI excludes 0
and same sign in 2019-24 and 2025-26; suggestive = CI only; noise otherwise), loaders of the modes per-start files."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

R = Path('/lus/flare/projects/S2Storm/s2storm')
MODES = R / 'runs' / 'bing' / 'eval' / 'modes'
V15 = R / 'runs' / 'bing' / 'eval' / 'v1.5'
OUTC = V15 / 'helperC'
SYSTEMS = ['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']
LABEL = {'ucs2s_sstanom': 'UC-S2S SST-anomaly', 'ucs2s_plain': 'UC-S2S plain', 'fuxi': 'FuXi-S2S', 'ifs': 'IFS'}
SEASONS = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}


def mbb_indices(n, block=4, n_draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    o = rng.integers(0, max(n - block, 0) + 1, size=(n_draws, nb))
    return (o[:, :, None] + np.arange(block)[None, None]).reshape(n_draws, -1)[:, :n]


def ci_of(draws, ci=0.95):
    a = (1 - ci) / 2
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    return (float(np.quantile(d, a)), float(np.quantile(d, 1 - a))) if d.size else (float('nan'), float('nan'))


def grade(ci_pooled, mean_a, mean_b):
    lo, hi = ci_pooled
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 'noise'
    if not (lo > 0 or hi < 0):
        return 'noise'
    s = np.sign(lo) if lo > 0 else -1.0
    if np.isfinite(mean_a) and np.isfinite(mean_b) and np.sign(mean_a) == s and np.sign(mean_b) == s:
        return 'robust'
    return 'suggestive'


def halves(starts):
    y = np.array([d.year for d in starts])
    return y <= 2024, y >= 2025


def boot_stat(stat, n, seed=0, n_draws=1000):
    """stat(idx) -> scalar or array; returns (full, draws [n_draws, ...])."""
    idx = mbb_indices(n, 4, n_draws, seed)
    full = np.asarray(stat(np.arange(n)), float)
    return full, np.array([stat(i) for i in idx], float)


def season_of(d):
    for k, ms in SEASONS.items():
        if d.month in ms:
            return k


def to_dates(a):
    return [date.fromisoformat(str(x)[:10]) for x in np.asarray(a)]


def load_mjo(system, index='full'):
    import xarray as xr
    ds = xr.open_dataset(MODES / 'mjo' / f'perstart_mjo_{index}_{system}.nc')
    out = dict(starts=to_dates(ds.start.values), obs=ds.rmm_obs.values, init=ds.rmm_init_obs.values, dc=ds.rmm_em30_dc.values,
               raw=ds.rmm_em30_raw.values, members=ds.rmm_members_raw.values)
    ds.close()
    return out


def common_starts(lists):
    s = set(lists[0])
    for x in lists[1:]:
        s &= set(x)
    return sorted(s)


def subset(d, starts):
    ix = {s: i for i, s in enumerate(d['starts'])}
    j = np.array([ix[s] for s in starts])
    return {k: (v[j] if isinstance(v, np.ndarray) and v.shape[:1] == (len(d['starts']),) else v) for k, v in d.items()} | {'starts': starts}


def rmm_phase(r1, r2):
    ang = np.arctan2(r2, r1)
    return (np.floor((ang + np.pi) / (np.pi / 4)).astype(int) % 8) + 1


def cor_curve(f, o):
    """f, o [S, L, 2] -> bivariate COR per lead [L]."""
    ok = np.isfinite(f).all(-1) & np.isfinite(o).all(-1)
    num = np.where(ok, (f * o).sum(-1), 0).sum(0)
    ff = np.where(ok, (f ** 2).sum(-1), 0).sum(0)
    oo = np.where(ok, (o ** 2).sum(-1), 0).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / np.sqrt(ff * oo)


def skill_day(cor, leads, thr=0.5):
    cor = np.asarray(cor, float)
    leads = np.asarray(leads, float)
    for j in range(len(cor)):
        if not np.isfinite(cor[j]):
            continue
        if cor[j] < thr:
            if j == 0 or not np.isfinite(cor[j - 1]):
                return float(leads[j]) if j else float('nan')
            a, b = cor[j - 1], cor[j]
            return float(leads[j - 1] + (a - thr) / (a - b) * (leads[j] - leads[j - 1]))
    return float(leads[np.isfinite(cor)][-1]) if np.isfinite(cor).any() else float('nan')


def write_csv(rows, path):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
