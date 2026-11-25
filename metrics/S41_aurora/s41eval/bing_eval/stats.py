"""Uncertainty and grading (plan §2; pre-registered headline spec, evaluation report §2, 2026-11-11 18:30 UTC).

* Moving-block bootstrap over starts in date order: block = 4 consecutive starts, 1,000 draws, 95 % percentile CI.
* Paired differences A - B per start; CRPSS difference of pooled sums; centred ACC difference recomputed per draw from per-point
  sufficient statistics (resample multiplicities), so the per-point correlation is redone on every bootstrap sample.
* Grades: robust = pooled 2019-26 CI excludes 0 and the mean difference has the same sign in 2019-24 and in 2025-26;
  suggestive = pooled CI excludes 0 only; noise = otherwise.

The parent session runs the headline (``run_headline``); this module only provides the machinery.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from s41eval.bing_eval import core as K

HEADLINE = dict(fields=['t2m', 'tp24', 'z500', 'sst', 'olr', 'u850'], regions=['global', 'tropics', 'nh_extratropics', 'conus'],
                window='w34', correction='dc', references=['fixed', 'trend'], block=4, n_draws=1000, ci=0.95,
                nino34=dict(field='sst', box=(-5, 5, 190, 240)))


def mbb_indices(n, block=4, n_draws=1000, seed=0):
    """moving-block bootstrap: [n_draws, n] start indices (blocks of ``block`` consecutive starts, uniform block origins in
    0..n-block, concatenated and truncated to n)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    o = rng.integers(0, max(n - block, 0) + 1, size=(n_draws, nb))
    idx = (o[:, :, None] + np.arange(block)[None, None]).reshape(n_draws, -1)[:, :n]
    return idx


def counts_from_idx(idx, n):
    """[n_draws, n] indices -> multiplicities [n_draws, n]."""
    c = np.zeros((idx.shape[0], n))
    for i in range(idx.shape[0]):
        c[i] = np.bincount(idx[i], minlength=n)
    return c


def ci_of(draws, ci=0.95):
    a = (1 - ci) / 2
    d = np.asarray(draws)[np.isfinite(draws)]
    return (float(np.quantile(d, a)), float(np.quantile(d, 1 - a))) if d.size else (float('nan'), float('nan'))


def grade(ci_pooled, mean_a, mean_b):
    """grading rule of plan §2: ci_pooled = (lo, hi) of the pooled mean difference; mean_a / mean_b = mean differences in
    2019-24 and 2025-26."""
    lo, hi = ci_pooled
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 'noise'
    excl = lo > 0 or hi < 0
    if not excl:
        return 'noise'
    s = np.sign(lo)
    if np.isfinite(mean_a) and np.isfinite(mean_b) and np.sign(mean_a) == s and np.sign(mean_b) == s:
        return 'robust'
    return 'suggestive'


# ------------------------------------------------------------------ paired statistics
def crpss_diff_stat(ca, cb, cr):
    """CRPSS_A - CRPSS_B on a set of starts = (sum cB - sum cA) / sum cRef (same reference)."""
    return lambda ix: (np.nansum(cb[ix]) - np.nansum(ca[ix])) / np.nansum(cr[ix])


def boot_scalar(stat, n, block=4, n_draws=1000, seed=0, ci=0.95):
    idx = mbb_indices(n, block, n_draws, seed)
    full = stat(np.arange(n))
    d = np.array([stat(i) for i in idx])
    return float(full), ci_of(d, ci), d


def acc_from_sums(c, f, o, ff, oo, fo):
    """weighted Pearson per point from multiplicity-weighted sums: c [B, S] @ per-start arrays [S, P] -> [B, P]."""
    n = c.sum(1, keepdims=True)
    sf, so, sff, soo, sfo = c @ f, c @ o, c @ ff, c @ oo, c @ fo
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = sfo - sf * so / n
        vf = sff - sf * sf / n
        vo = soo - so * so / n
        return cov / np.sqrt(vf * vo)


def paired_acc_diff(fa_a, fa_b, oa, mask, grid='g15', block=4, n_draws=1000, seed=0, ci=0.95, batch=100):
    """centred per-point ACC (across starts), area mean over mask, A - B, block bootstrap.
    fa_a, fa_b, oa: [S, ny, nx] anomalies (forecast A, forecast B, ERA5) on the same starts in date order; NaN points dropped
    (points NaN in any start are excluded)."""
    S = fa_a.shape[0]
    w = (K.cosw(grid) * mask).ravel()
    ok = (w > 0) & np.isfinite(fa_a).reshape(S, -1).all(0) & np.isfinite(fa_b).reshape(S, -1).all(0) & \
        np.isfinite(oa).reshape(S, -1).all(0)
    A, B, O = (x.reshape(S, -1)[:, ok].astype(np.float64) for x in (fa_a, fa_b, oa))
    ww = w[ok] / w[ok].sum()
    pre = dict(o=O, oo=O * O, fa=A, ffa=A * A, foa=A * O, fb=B, ffb=B * B, fob=B * O)

    def stat_c(c):
        ra = acc_from_sums(c, pre['fa'], pre['o'], pre['ffa'], pre['oo'], pre['foa'])
        rb = acc_from_sums(c, pre['fb'], pre['o'], pre['ffb'], pre['oo'], pre['fob'])
        return (ra @ ww), (rb @ ww)
    a0, b0 = stat_c(np.ones((1, S)))
    idx = mbb_indices(S, block, n_draws, seed)
    d = []
    for i in range(0, n_draws, batch):
        c = counts_from_idx(idx[i:i + batch], S)
        ra, rb = stat_c(c)
        d.append(ra - rb)
    d = np.concatenate(d)
    return dict(a=float(a0[0]), b=float(b0[0]), diff=float(a0[0] - b0[0]), ci=ci_of(d, ci), draws=d)


def paired_index_acc_diff(ia, ib, io, block=4, n_draws=1000, seed=0, ci=0.95):
    """centred correlation of scalar indices (e.g. Nino-3.4 anomaly) per start, A - B."""
    def st(ix):
        return np.corrcoef(ia[ix], io[ix])[0, 1] - np.corrcoef(ib[ix], io[ix])[0, 1]
    full, c, d = boot_scalar(st, len(io), block, n_draws, seed, ci)
    return dict(a=float(np.corrcoef(ia, io)[0, 1]), b=float(np.corrcoef(ib, io)[0, 1]), diff=full, ci=c, draws=d)


def box_mask(box, grid='g15'):
    s, n, w, e = box
    lat, lon = K.LAT[grid], K.LON[grid]
    return ((lat[:, None] >= s) & (lat[:, None] <= n) & (lon[None] >= w) & (lon[None] <= e)).astype(float)


# ------------------------------------------------------------------ headline (the parent runs this)
def _load_system(od, system, field, grid='g15'):
    import xarray as xr
    sd = Path(od) / 'score' / system / f'{field}_{grid}'
    dj = json.loads((sd / 'done_dc.json').read_text())
    ps = xr.open_dataset(sd / 'perstart_dc.nc')
    return sd, dj, ps


def run_headline(od, sys_a, sys_b, out_csv=None, fields=None, regions=None, seed=20261111, log=print, grid='g15'):
    """pre-registered headline: paired A - B, drift-corrected weeks 3-4, per (field, region): CRPSS vs both references and centred
    ACC (anomalies vs each reference climatology), plus Nino-3.4 ACC; pooled / 2019-24 / 2025-26 / no-2019 means, block-bootstrap CI
    (pooled) and grade. Starts = those both systems scored (date order). -> list of dict rows (also written to out_csv)."""
    od = Path(od)
    H = HEADLINE
    fields = fields or H['fields']
    regions = regions or H['regions']
    w = K.WK.index(H['window'])
    rows = []
    for f in fields:
        sda, dja, psa = _load_system(od, sys_a, f, grid)
        sdb, djb, psb = _load_system(od, sys_b, f, grid)
        common = sorted(set(dja['starts']) & set(djb['starts']))
        ia = np.array([dja['starts'].index(s) for s in common])
        ib = np.array([djb['starts'].index(s) for s in common])
        st = [date.fromisoformat(s) for s in common]
        ua = psa['used_dc'].values[ia, w].astype(bool) & psb['used_dc'].values[ib, w].astype(bool)
        keep = np.flatnonzero(ua)
        st = [st[i] for i in keep]
        ia, ib = ia[keep], ib[keep]
        vmeta = json.loads((Path(dja['verif']) / 'meta.json').read_text())
        vpos = {s: i for i, s in enumerate(vmeta['starts'])}
        iv = np.array([vpos[str(s)] for s in st])
        vd = Path(dja['verif'])
        if dja['verif'] != djb['verif']:
            log(f'warning {f}: systems use different verification ({dja["verif"]} vs {djb["verif"]})')
        sfx = '' if grid == 'g15' else '_g30'
        O = np.load(vd / f'O{sfx}.npy', mmap_mode='r')[iv, w]
        if grid == 'g15':
            ema = np.load(sda / 'em_dc.npy', mmap_mode='r')[ia, w]
            emb = np.load(sdb / 'em_dc.npy', mmap_mode='r')[ib, w]
        else:   # ensemble means are only stored at 1.5 deg; the conservative regrid is linear, so regrid(mean) = mean(regrid)
            ga = Path(str(sda).replace(f'_{grid}', '_g15'))
            gb = Path(str(sdb).replace(f'_{grid}', '_g15'))
            ema = K.regrid(np.asarray(np.load(ga / 'em_dc.npy', mmap_mode='r')[ia, w]), 'g15', grid)
            emb = K.regrid(np.asarray(np.load(gb / 'em_dc.npy', mmap_mode='r')[ib, w]), 'g15', grid)
        per = {p: np.array([K.in_period(s, p) for s in st]) for p in K.PERIODS}
        for ref in H['references']:
            C = np.load(vd / f'C{ref}{sfx}.npy', mmap_mode='r')[iv, w]
            for rn in regions:
                j = K.RK.index(rn)
                ca = psa['crps_dc'].values[ia, w, j]
                cb = psb['crps_dc'].values[ib, w, j]
                cr = psa[f'cc_{ref}_dc'].values[ia, w, j]
                means = {}
                for p, m in per.items():
                    ix = np.flatnonzero(m)
                    means[p] = crpss_diff_stat(ca, cb, cr)(ix) if ix.size else float('nan')
                full, c, _ = boot_scalar(crpss_diff_stat(ca, cb, cr), len(st), H['block'], H['n_draws'], seed)
                rows.append(dict(field=f, region=rn, metric='crpss', reference=ref, a=1 - np.nansum(ca) / np.nansum(cr),
                                 b=1 - np.nansum(cb) / np.nansum(cr), diff=full, ci_lo=c[0], ci_hi=c[1], n_starts=len(st),
                                 **{f'diff_{p}': v for p, v in means.items()}, grade=grade(c, means['2019-24'], means['2025-26'])))
                mk = K.masks(grid)[rn]
                r = paired_acc_diff(np.asarray(ema) - C, np.asarray(emb) - C, np.asarray(O) - C, mk, grid, H['block'], H['n_draws'],
                                    seed)
                means = {}
                for p, m in per.items():
                    ix = np.flatnonzero(m)
                    if ix.size >= 3:
                        rr = paired_acc_diff((np.asarray(ema) - C)[ix], (np.asarray(emb) - C)[ix], (np.asarray(O) - C)[ix], mk, grid,
                                             n_draws=1)
                        means[p] = rr['diff']
                    else:
                        means[p] = float('nan')
                rows.append(dict(field=f, region=rn, metric='acc', reference=ref, a=r['a'], b=r['b'], diff=r['diff'], ci_lo=r['ci'][0],
                                 ci_hi=r['ci'][1], n_starts=len(st), **{f'diff_{p}': v for p, v in means.items()},
                                 grade=grade(r['ci'], means['2019-24'], means['2025-26'])))
            if f == H['nino34']['field']:
                mk = box_mask(H['nino34']['box'], grid)
                xa, xb, xo = (K.amean(np.asarray(x) - C, mk, grid) for x in (ema, emb, O))
                r = paired_index_acc_diff(xa, xb, xo, H['block'], H['n_draws'], seed)
                means = {p: (np.corrcoef(xa[m], xo[m])[0, 1] - np.corrcoef(xb[m], xo[m])[0, 1]) if m.sum() >= 3 else float('nan')
                         for p, m in per.items()}
                rows.append(dict(field='nino34', region='nino34', metric='acc', reference=ref, a=r['a'], b=r['b'], diff=r['diff'],
                                 ci_lo=r['ci'][0], ci_hi=r['ci'][1], n_starts=len(st), **{f'diff_{p}': v for p, v in means.items()},
                                 grade=grade(r['ci'], means['2019-24'], means['2025-26'])))
        log(f'headline {f}: {len(st)} common starts')
    if out_csv:
        with open(out_csv, 'w', newline='') as fo:
            wri = csv.DictWriter(fo, fieldnames=list(rows[0]))
            wri.writeheader()
            wri.writerows(rows)
    return rows
