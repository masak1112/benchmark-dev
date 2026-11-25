"""W1 scorecards (plan v2 §2): the pre-registered headline method (``stats.run_headline``) extended to every lead window and to
pairs of systems that use different verification samplings (UC-S2S 00 UTC vs FuXi daily means).

Method per (field, region, window), identical to the headline at w34 for two systems on the same verification:
* paired A - B on the common starts both systems scored and used (``used_dc``), in date order;
* CRPSS difference = CRPSS_A - CRPSS_B of pooled sums, each system against **its own** reference CRPS (same reference type,
  ``cc_<ref>_dc``); for a shared verification this is exactly (sum cB - sum cA) / sum cRef, the headline statistic;
* centred per-point ACC (anomalies vs each reference climatology C, ERA5 anomaly O - C of each system's own verification),
  area mean over the region, A - B, per-point correlation recomputed on every bootstrap draw;
* tercile RPSS difference (Weigel 2007 debiased, as the package: 1 - mean RPS / (mean RPS_clim + D(M)));
* moving-block bootstrap over starts (block 4, 1,000 draws, 95 % percentile CI); grade robust / suggestive / noise.
Only w34 of the SST-anomaly vs plain pair is pre-registered; everything else is exploratory (column ``status``).
Per-system rows (A03): CRPSS / RPSS / ACC of each system with a block-bootstrap CI, per period.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from s41eval.bing_eval import core as K
from s41eval.bing_eval import stats as ST

FIELDS = ['t2m', 'tp24', 'z500', 'sst', 'olr', 'u850']
REGIONS = ['global', 'tropics', 'nh_extratropics', 'conus']
REFS = ['fixed', 'trend']


def paired_acc_diff2(fa_a, oa_a, fa_b, oa_b, mask, grid='g15', block=4, n_draws=1000, seed=0, ci=0.95, batch=100, idx=None):
    """``stats.paired_acc_diff`` with a separate ERA5 anomaly per system (oa_a for A, oa_b for B). With oa_b is oa_a it gives the
    same numbers (same bootstrap indices for the same seed). Points NaN in any start of any array are excluded."""
    S = fa_a.shape[0]
    w = (K.cosw(grid) * mask).ravel()
    ok = (w > 0)
    for x in (fa_a, fa_b, oa_a, oa_b):
        ok &= np.isfinite(x).reshape(S, -1).all(0)
    A, B, OA, OB = (x.reshape(S, -1)[:, ok].astype(np.float64) for x in (fa_a, fa_b, oa_a, oa_b))
    ww = w[ok] / w[ok].sum() if ok.any() else w[ok]

    def stat_c(c):
        ra = ST.acc_from_sums(c, A, OA, A * A, OA * OA, A * OA)
        rb = ST.acc_from_sums(c, B, OB, B * B, OB * OB, B * OB)
        return ra @ ww, rb @ ww
    a0, b0 = stat_c(np.ones((1, S)))
    if n_draws <= 1:
        return dict(a=float(a0[0]), b=float(b0[0]), diff=float(a0[0] - b0[0]), ci=(np.nan, np.nan), draws=np.array([]))
    idx = ST.mbb_indices(S, block, n_draws, seed) if idx is None else idx
    d = []
    for i in range(0, len(idx), batch):
        c = ST.counts_from_idx(idx[i:i + batch], S)
        ra, rb = stat_c(c)
        d.append(ra - rb)
    d = np.concatenate(d)
    return dict(a=float(a0[0]), b=float(b0[0]), diff=float(a0[0] - b0[0]), ci=ST.ci_of(d, ci), draws=d)


def _sys(od, system, field, grid='g15'):
    import xarray as xr
    sd = Path(od) / 'score' / system / f'{field}_{grid}'
    dj = json.loads((sd / 'done_dc.json').read_text())
    ps = xr.open_dataset(sd / 'perstart_dc.nc')
    vd = Path(dj['verif'])
    vm = json.loads((vd / 'meta.json').read_text())
    return dict(sd=sd, dj=dj, ps=ps, vd=vd, vpos={s: i for i, s in enumerate(vm['starts'])}, M=int(dj['members']))


def _common(sa, sb, w):
    common = sorted(set(sa['dj']['starts']) & set(sb['dj']['starts']))
    ia = np.array([sa['dj']['starts'].index(s) for s in common])
    ib = np.array([sb['dj']['starts'].index(s) for s in common])
    u = sa['ps']['used_dc'].values[ia, w].astype(bool) & sb['ps']['used_dc'].values[ib, w].astype(bool)
    keep = np.flatnonzero(u)
    st = [date.fromisoformat(common[i]) for i in keep]
    return st, ia[keep], ib[keep]


def _status(sys_a, sys_b, window, metric):
    pre = sys_a == 'ucs2s_sstanom' and sys_b == 'ucs2s_plain' and window == 'w34' and metric in ('crpss', 'acc')
    return 'pre-registered' if pre else 'exploratory'


def paired_rows(od, sys_a, sys_b, field, windows=None, regions=None, seed=20261111, grid='g15', log=print, n_draws=1000,
                nino34=True, rpss=True):
    """rows for one field: every window x region x reference x metric (crpss, acc, rpss) [+ nino34 acc]."""
    windows = windows or K.WK
    regions = regions or REGIONS
    H = ST.HEADLINE
    sa, sb = _sys(od, sys_a, field, grid), _sys(od, sys_b, field, grid)
    same_verif = sa['dj']['verif'] == sb['dj']['verif']
    rows = []
    for wn in windows:
        w = K.WK.index(wn)
        st, ia, ib = _common(sa, sb, w)
        n = len(st)
        if n < 8:
            log(f'{field} {wn}: only {n} common starts, skipped')
            continue
        iva = np.array([sa['vpos'][str(s)] for s in st])
        ivb = np.array([sb['vpos'][str(s)] for s in st])
        Oa = np.asarray(np.load(sa['vd'] / 'O.npy', mmap_mode='r')[iva, w])
        Ob = Oa if same_verif else np.asarray(np.load(sb['vd'] / 'O.npy', mmap_mode='r')[ivb, w])
        ema = np.asarray(np.load(sa['sd'] / 'em_dc.npy', mmap_mode='r')[ia, w])
        emb = np.asarray(np.load(sb['sd'] / 'em_dc.npy', mmap_mode='r')[ib, w])
        per = {p: np.array([K.in_period(s, p) for s in st]) for p in K.PERIODS}
        base = dict(field=field, window=wn, a_sys=sys_a, b_sys=sys_b, n_starts=n, same_verif=same_verif)
        for ref in H['references']:
            Ca = np.asarray(np.load(sa['vd'] / f'C{ref}.npy', mmap_mode='r')[iva, w])
            Cb = Ca if same_verif else np.asarray(np.load(sb['vd'] / f'C{ref}.npy', mmap_mode='r')[ivb, w])
            for rn in regions:
                j = K.RK.index(rn)
                ca = sa['ps']['crps_dc'].values[ia, w, j].astype(np.float64)
                cb = sb['ps']['crps_dc'].values[ib, w, j].astype(np.float64)
                cra = sa['ps'][f'cc_{ref}_dc'].values[ia, w, j].astype(np.float64)
                crb = cra if same_verif else sb['ps'][f'cc_{ref}_dc'].values[ib, w, j].astype(np.float64)
                if same_verif:
                    stat = ST.crpss_diff_stat(ca, cb, cra)
                else:
                    def stat(ix, ca=ca, cb=cb, cra=cra, crb=crb):
                        return (1 - np.nansum(ca[ix]) / np.nansum(cra[ix])) - (1 - np.nansum(cb[ix]) / np.nansum(crb[ix]))
                means = {p: stat(np.flatnonzero(m)) if m.any() else np.nan for p, m in per.items()}
                full, c, _ = ST.boot_scalar(stat, n, H['block'], n_draws, seed)
                rows.append(dict(base, region=rn, metric='crpss', reference=ref, a=1 - np.nansum(ca) / np.nansum(cra),
                                 b=1 - np.nansum(cb) / np.nansum(crb), diff=full, ci_lo=c[0], ci_hi=c[1],
                                 **{f'diff_{p}': v for p, v in means.items()}, grade=ST.grade(c, means['2019-24'], means['2025-26']),
                                 status=_status(sys_a, sys_b, wn, 'crpss')))
                if rpss and f'rps_{ref}_dc' in sa['ps'] and f'rps_{ref}_dc' in sb['ps']:
                    ra_ = sa['ps'][f'rps_{ref}_dc'].values[ia, w, j].astype(np.float64)
                    rb_ = sb['ps'][f'rps_{ref}_dc'].values[ib, w, j].astype(np.float64)
                    rca = sa['ps'][f'rpscl_{ref}_dc'].values[ia, w, j].astype(np.float64)
                    rcb = rca if same_verif else sb['ps'][f'rpscl_{ref}_dc'].values[ib, w, j].astype(np.float64)
                    Da, Db = K.rpss_debias_D(sa['M']), K.rpss_debias_D(sb['M'])

                    def rst(ix, ra_=ra_, rb_=rb_, rca=rca, rcb=rcb, Da=Da, Db=Db):
                        return (1 - np.nanmean(ra_[ix]) / (np.nanmean(rca[ix]) + Da)) - (1 - np.nanmean(rb_[ix]) / (np.nanmean(rcb[ix]) + Db))
                    means = {p: rst(np.flatnonzero(m)) if m.any() else np.nan for p, m in per.items()}
                    full, c, _ = ST.boot_scalar(rst, n, H['block'], n_draws, seed)
                    rows.append(dict(base, region=rn, metric='rpss', reference=ref,
                                     a=1 - np.nanmean(ra_) / (np.nanmean(rca) + Da), b=1 - np.nanmean(rb_) / (np.nanmean(rcb) + Db),
                                     diff=full, ci_lo=c[0], ci_hi=c[1], **{f'diff_{p}': v for p, v in means.items()},
                                     grade=ST.grade(c, means['2019-24'], means['2025-26']), status=_status(sys_a, sys_b, wn, 'rpss')))
                mk = K.masks(grid)[rn]
                fa, fb, oa, ob = ema - Ca, emb - Cb, Oa - Ca, Ob - Cb
                r = paired_acc_diff2(fa, oa, fb, ob, mk, grid, H['block'], n_draws, seed)
                means = {}
                for p, m in per.items():
                    ix = np.flatnonzero(m)
                    means[p] = paired_acc_diff2(fa[ix], oa[ix], fb[ix], ob[ix], mk, grid, n_draws=1)['diff'] if ix.size >= 3 else np.nan
                rows.append(dict(base, region=rn, metric='acc', reference=ref, a=r['a'], b=r['b'], diff=r['diff'], ci_lo=r['ci'][0],
                                 ci_hi=r['ci'][1], **{f'diff_{p}': v for p, v in means.items()},
                                 grade=ST.grade(r['ci'], means['2019-24'], means['2025-26']), status=_status(sys_a, sys_b, wn, 'acc')))
            if nino34 and field == H['nino34']['field']:
                mk = ST.box_mask(H['nino34']['box'], grid)
                xa, xb = K.amean(ema - Ca, mk, grid), K.amean(emb - Cb, mk, grid)
                xoa, xob = K.amean(Oa - Ca, mk, grid), K.amean(Ob - Cb, mk, grid)

                def nst(ix):
                    return np.corrcoef(xa[ix], xoa[ix])[0, 1] - np.corrcoef(xb[ix], xob[ix])[0, 1]
                full, c, _ = ST.boot_scalar(nst, n, H['block'], n_draws, seed)
                means = {p: nst(np.flatnonzero(m)) if m.sum() >= 3 else np.nan for p, m in per.items()}
                rows.append(dict(base, field='nino34', region='nino34', metric='acc', reference=ref,
                                 a=float(np.corrcoef(xa, xoa)[0, 1]), b=float(np.corrcoef(xb, xob)[0, 1]), diff=full, ci_lo=c[0],
                                 ci_hi=c[1], **{f'diff_{p}': v for p, v in means.items()},
                                 grade=ST.grade(c, means['2019-24'], means['2025-26']),
                                 status='pre-registered' if _status(sys_a, sys_b, wn, 'acc') == 'pre-registered' else 'exploratory'))
        log(f'{sys_a} vs {sys_b} {field} {wn}: {n} starts')
    return rows


def system_rows(od, system, field, windows=None, regions=None, seed=20261111, grid='g15', n_draws=1000):
    """per-system CRPSS (dc, both references) and RPSS with block-bootstrap CI (pooled) and the values per period (A03)."""
    windows = windows or K.WK
    regions = regions or REGIONS
    s = _sys(od, system, field, grid)
    starts = [date.fromisoformat(x) for x in s['dj']['starts']]
    rows = []
    for wn in windows:
        w = K.WK.index(wn)
        u = np.flatnonzero(s['ps']['used_dc'].values[:, w].astype(bool))
        st = [starts[i] for i in u]
        per = {p: np.array([K.in_period(x, p) for x in st]) for p in K.PERIODS}
        for ref in REFS:
            for rn in regions:
                j = K.RK.index(rn)
                c = s['ps']['crps_dc'].values[u, w, j].astype(np.float64)
                cr = s['ps'][f'cc_{ref}_dc'].values[u, w, j].astype(np.float64)

                def stat(ix, c=c, cr=cr):
                    return 1 - np.nansum(c[ix]) / np.nansum(cr[ix])
                full, ci, _ = ST.boot_scalar(stat, len(u), 4, n_draws, seed)
                rows.append(dict(system=system, field=field, window=wn, region=rn, metric='crpss', reference=ref, value=full,
                                 ci_lo=ci[0], ci_hi=ci[1], n_starts=len(u),
                                 **{f'v_{p}': stat(np.flatnonzero(m)) if m.any() else np.nan for p, m in per.items()}))
                if f'rps_{ref}_dc' in s['ps']:
                    a_ = s['ps'][f'rps_{ref}_dc'].values[u, w, j].astype(np.float64)
                    cl = s['ps'][f'rpscl_{ref}_dc'].values[u, w, j].astype(np.float64)
                    D = K.rpss_debias_D(s['M'])

                    def rst(ix, a_=a_, cl=cl, D=D):
                        return 1 - np.nanmean(a_[ix]) / (np.nanmean(cl[ix]) + D)
                    full, ci, _ = ST.boot_scalar(rst, len(u), 4, n_draws, seed)
                    rows.append(dict(system=system, field=field, window=wn, region=rn, metric='rpss', reference=ref, value=full,
                                     ci_lo=ci[0], ci_hi=ci[1], n_starts=len(u),
                                     **{f'v_{p}': rst(np.flatnonzero(m)) if m.any() else np.nan for p, m in per.items()}))
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as fo:
        wri = csv.DictWriter(fo, fieldnames=keys)
        wri.writeheader()
        wri.writerows(rows)
    return path
