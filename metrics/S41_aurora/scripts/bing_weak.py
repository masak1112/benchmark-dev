#!/usr/bin/env python
"""S41 helper B (W3 weaknesses / W4 ensemble quality) bulk driver. See ``s41eval.bing_weak``.

    python scripts/bing_weak.py <phase> [--nproc N] [--rank r --nranks n] [--limit k]
phases: era5 | start | profile | members (run in this order; every task skips when its output exists).
Rank split: task i goes to rank i % nranks (default: PALS_RANKID and NRANKS from the environment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)
from s41eval import bing_weak as B                      # noqa: E402
from s41eval.bing_eval import core as K                 # noqa: E402

R = B.R
V15 = R / 'runs/bing/eval/v1.5'
OUT = Path(os.environ.get('BW_OUT', V15 / 'helperB'))


def starts_all():
    return [date.fromisoformat(s) for s in json.loads((V15 / 'work/starts.json').read_text())]


def save_atomic(path, **arrs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial.npz')
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def save_npy(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial.npy')
    np.save(tmp, x)
    os.replace(tmp, path)


# ------------------------------------------------------------------ phase era5
def era5_days():
    ds = set()
    for s in starts_all():
        for k in range(K.NLEAD):
            ds.add(s + timedelta(days=k))
    return sorted(ds)


def task_era5(d):
    f = OUT / 'era5_pl' / f'{d:%Y%m%d}.npy'
    g = OUT / 'era5_spec' / f'{d:%Y%m%d}.npz'
    if f.exists() and g.exists():
        return 'cached'
    x15, x1 = B.era5_day_pl(d)
    if x15 is None:
        return 'missing h5'
    sp = B.era5_spec(d, x1)
    save_npy(f, x15)
    save_atomic(g, **sp)
    return 'ok'


# ------------------------------------------------------------------ phase start
_ST = {}


def store():
    if 'E' not in _ST:
        _ST['E'] = B.Store()
        K.set_lsm(np.load(B.STORE / 'lsm.npy'))
        _ST['mk'] = K.masks('g15')
    return _ST['E'], _ST['mk']


def era5_pl_window(init):
    O = np.full((K.NLEAD, len(B.UA), len(B.LEVELS), 121, 240), np.nan, np.float32)
    for k in range(K.NLEAD):
        f = OUT / 'era5_pl' / f'{init + timedelta(days=k):%Y%m%d}.npy'
        if f.exists():
            O[k] = np.load(f)
    return O


def task_start(init, systems=tuple(B.SYSTEMS)):
    E, mk = store()
    todo = [s for s in systems if not (OUT / 'start' / s / f'{init:%Y%m%d}.npz').exists()]
    if not todo:
        return 'cached'
    t0 = time.time()
    O = era5_pl_window(init)
    fw = OUT / 'win' / 'era5' / f'{init:%Y%m%d}.npy'
    if not fw.exists():
        save_npy(fw, B.windows_of(O[:, [B.UA.index(v) for v in B.WIN_UA]]))
    msg = []
    cw = K.cosw('g15')
    hmask = {r: mk[r] for r in B.HREG}
    for sy in todo:
        cfg = B.SYSTEMS[sy]
        r = B.read_ensstats(sy, init)
        if r is None:
            msg.append(f'{sy}: no ensstats')
            continue
        em_ua, va_ua, em_s, va_s = r
        # ERA5 surface in the package sampling of the system
        Os = np.full_like(em_s, np.nan)
        for k in range(K.NLEAD):
            for j, v in enumerate(B.SFC):
                x = E.field(v, cfg['time'], init + timedelta(days=k))
                if x is not None:
                    Os[k, j] = x
        d_ua = em_ua - O
        ub, um, uv = B.region_stats(d_ua, va_ua, mk)
        d_s = em_s - Os
        sb, sm, sv = B.region_stats(d_s, va_s, mk)
        res = dict(ua_bias=ub, ua_mse=um, ua_var=uv, sfc_bias=sb, sfc_mse=sm, sfc_var=sv,
                   ua_zm_em=B.zm(em_ua), ua_zm_O=B.zm(O), sfc_zm_em=B.zm(np.where(np.isfinite(Os), em_s, np.nan)), sfc_zm_O=B.zm(Os))
        del d_ua
        save_npy(OUT / 'win' / sy / f'{init:%Y%m%d}.npy', B.windows_of(em_ua[:, [B.UA.index(v) for v in B.WIN_UA]]))
        del em_ua, va_ua
        # members 0..9: precipitation histograms and zonal spectra
        hm = np.zeros((len(B.HWIN), len(B.HREG), len(B.HEDGES) - 1))
        ho = np.zeros_like(hm)
        spec = {}
        nm = 0
        for m in range(B.NMEM_SPEC):
            f, ld = B.member_daily(sy, init, m)
            if f is None:
                continue
            nm += 1
            tp = K.to_g15(f['tp24'])
            lat = np.arange(-89.5, 90, 1.0) if f['z500'].shape[-2] == 180 else K.LAT['g15']
            for b, rng in B.BANDS.items():
                zs = B.zonal_spectrum(f['z500'], lat, rng)
                ks = 0.5 * (B.zonal_spectrum(f['u250'], lat, rng) + B.zonal_spectrum(f['v250'], lat, rng))
                spec[f'z500_{b}'] = spec.get(f'z500_{b}', 0) + zs
                spec[f'ke250_{b}'] = spec.get(f'ke250_{b}', 0) + ks
            for iw, w in enumerate(B.HWIN):
                a, bb = K.WINDOWS[w]
                sel = [int(np.flatnonzero(ld == k)[0]) for k in range(a, bb + 1) if (ld == k).any()]
                for ir, rn in enumerate(B.HREG):
                    hm[iw, ir] += B.wavg_hist(tp[sel], cw * hmask[rn])
        if nm:
            hm /= nm
            spec = {k: (v / nm).astype(np.float32) for k, v in spec.items()}
        it = B.SFC.index('tp24')
        for iw, w in enumerate(B.HWIN):
            a, bb = K.WINDOWS[w]
            for ir, rn in enumerate(B.HREG):
                ho[iw, ir] = B.wavg_hist(Os[a:bb + 1, it], cw * hmask[rn])
        res.update(hist_model=hm, hist_era5=ho, n_members_hist=nm, **{f'spec_{k}': v for k, v in spec.items()})
        save_atomic(OUT / 'start' / sy / f'{init:%Y%m%d}.npz', **res)
        msg.append(f'{sy}: ok')
    return f'{init} ' + '; '.join(msg) + f' {time.time() - t0:.0f}s'


# ------------------------------------------------------------------ phase profile (drift-corrected RMSE by level)
def task_profile(sy, var):
    f = OUT / 'profile' / f'{sy}_{var}.npz'
    if f.exists():
        return 'cached'
    t0 = time.time()
    _, mk = store()
    st = [s for s in starts_all() if (OUT / 'win' / sy / f'{s:%Y%m%d}.npy').exists()
          and (OUT / 'win' / 'era5' / f'{s:%Y%m%d}.npy').exists()]
    j = B.WIN_UA.index(var)
    S, Wn, L = len(st), len(B.WKEEP), len(B.LEVELS)
    F = np.full((S, Wn, L, 121, 240), np.nan, np.float32)
    O = np.full_like(F, np.nan)
    for i, s in enumerate(st):
        F[i] = np.load(OUT / 'win' / sy / f'{s:%Y%m%d}.npy', mmap_mode='r')[:, j]
        O[i] = np.load(OUT / 'win' / 'era5' / f'{s:%Y%m%d}.npy', mmap_mode='r')[:, j]
    seasons = ['all', 'DJF', 'JJA']
    shp = (len(seasons), Wn, L, len(B.REG4))
    out = {k: np.full(shp, np.nan) for k in ('rmse_raw', 'rmse_dc', 'bias_raw')}
    zmb = np.full((len(seasons), Wn, L, 121), np.nan)
    for l in range(L):
        em, ob = F[:, :, l], O[:, :, l]
        valid = np.isfinite(em).any((-2, -1)) & np.isfinite(ob).any((-2, -1))
        if not valid.any():
            continue
        b = K.drift_bias(em, ob, st, valid)
        draw, ddc = em - ob, em - b - ob
        for si, sn in enumerate(seasons):
            ix0 = K.subset_index(st, 'pooled', sn)
            for w in range(Wn):
                ix = ix0[valid[ix0, w] & np.isfinite(b[ix0, w]).any((-2, -1))]
                if ix.size < 3:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mr, md, br = np.nanmean(draw[ix, w] ** 2, 0), np.nanmean(ddc[ix, w] ** 2, 0), np.nanmean(draw[ix, w], 0)
                zmb[si, w, l] = np.nanmean(br, -1)
                for ri, rn in enumerate(B.REG4):
                    out['rmse_raw'][si, w, l, ri] = np.sqrt(K.amean(mr, mk[rn]))
                    out['rmse_dc'][si, w, l, ri] = np.sqrt(K.amean(md, mk[rn]))
                    out['bias_raw'][si, w, l, ri] = K.amean(br, mk[rn])
    save_atomic(f, **out, zm_bias_raw=zmb, seasons=np.array(seasons), windows=np.array(B.WKEEP), levels=np.array(B.LEVELS),
                regions=np.array(B.REG4), starts=np.array([str(s) for s in st]))
    return f'{sy} {var}: {S} starts {time.time() - t0:.0f}s'


# ------------------------------------------------------------------ phase members (members-needed curves)
MEMBER_FIELDS = ['t2m', 'z500', 'tp24', 'sst', 'olr', 'u850']
MSIZES = [5, 10, 20, 30]
NDRAW = 5
MWIN = ['w1', 'w34', 'w56']


def task_members(sy, field):
    f = OUT / 'members' / f'{sy}_{field}.json'
    if f.exists():
        return 'cached'
    t0 = time.time()
    from s41eval.bing_eval import data as D
    sd = V15 / 'score' / sy / f'{field}_g15'
    dj = json.loads((sd / 'done_dc.json').read_text())
    st = [date.fromisoformat(s) for s in dj['starts']]
    vd = Path(dj['verif'])
    if not vd.is_absolute() or 'v1.5' not in str(vd):
        vd = V15 / 'work' / 'verif' / Path(dj['verif']).name
    meta = json.loads((vd / 'meta.json').read_text())
    pos = {s: i for i, s in enumerate(meta['starts'])}
    ix = np.array([pos[str(s)] for s in st])
    wi = [K.WK.index(w) for w in MWIN]
    O = np.load(vd / 'O.npy', mmap_mode='r')[ix][:, wi]
    CC = {r: np.load(vd / f'CC{r}.npy', mmap_mode='r')[ix][:, wi] for r in ('fixed', 'trend')}
    C = {r: np.load(vd / f'C{r}.npy', mmap_mode='r')[ix][:, wi] for r in ('fixed', 'trend')}
    M = int(dj['members'])
    F = np.full((len(st), M, len(wi), 121, 240), np.nan, np.float32)
    rb = V15 / 'work' / 'reduce' / sy / field
    for i, s in enumerate(st):
        F[i] = np.load(rb / f'{s:%Y%m%d}.npy')[:M][:, wi]
    K.set_lsm(np.load(B.STORE / 'lsm.npy'))
    mk = {r: K.masks('g15')[r] for r in B.REG4}
    rng = np.random.default_rng(20261112)
    res = []
    for n in MSIZES:
        draws = [np.arange(M)] if n >= M else [np.sort(rng.choice(M, n, replace=False)) for _ in range(NDRAW)]
        for di, idx in enumerate(draws):
            sc = B.subset_scores(F, O, CC, C, st, idx, mk, clip=K.FIELDS[field].get('clip', False))
            for (met, ref, rn, w), v in sc.items():
                res.append(dict(system=sy, field=field, members=n, draw=di, metric=met, reference=ref, region=rn, window=MWIN[w],
                                value=v))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(rows=res, starts=len(st), members_total=M, ndraw=NDRAW, seed=20261112,
                                 seconds=round(time.time() - t0, 1)), indent=0))
    return f'{sy} {field}: {len(st)} starts {time.time() - t0:.0f}s'


# ------------------------------------------------------------------ phase memberdiag (are the members exchangeable?)
DIAG_FIELDS = ['t2m', 'sst', 'z500', 'tp24', 'u850']


def task_memberdiag(sy, field):
    """per-member systematic offsets: d_i = mean over starts of (member i - ensemble mean), weeks 3-4 window (raw); pattern correlation of
    d_i between odd and even starts (persistent offsets -> r > 0; exchangeable members -> r ~ 0); offset variance / ensemble variance;
    plus the subset-identity check: raw fair CRPSS of 5-member subsets (must equal 30 members in expectation)."""
    f = OUT / 'memberdiag' / f'{sy}_{field}.json'
    if f.exists():
        return 'cached'
    t0 = time.time()
    sd = V15 / 'score' / sy / f'{field}_g15'
    dj = json.loads((sd / 'done_raw.json').read_text())
    st = [date.fromisoformat(s) for s in dj['starts']]
    vd = V15 / 'work' / 'verif' / Path(dj['verif']).name
    meta = json.loads((vd / 'meta.json').read_text())
    pos = {s: i for i, s in enumerate(meta['starts'])}
    ix = np.array([pos[str(s)] for s in st])
    wi = [K.WK.index('w34')]
    O = np.load(vd / 'O.npy', mmap_mode='r')[ix][:, wi]
    CC = {'fixed': np.load(vd / 'CCfixed.npy', mmap_mode='r')[ix][:, wi]}
    C = {'fixed': np.load(vd / 'Cfixed.npy', mmap_mode='r')[ix][:, wi]}
    M = int(dj['members'])
    F = np.full((len(st), M, 1, 121, 240), np.nan, np.float32)
    rb = V15 / 'work' / 'reduce' / sy / field
    for i, s in enumerate(st):
        F[i] = np.load(rb / f'{s:%Y%m%d}.npy', mmap_mode='r')[:M][:, wi]
    K.set_lsm(np.load(B.STORE / 'lsm.npy'))
    mk = K.masks('g15')
    out = dict(system=sy, field=field, starts=len(st), members=M)
    for rn in ('global', 'tropics', 'nh_extratropics'):
        r, frac, bias_i = B.member_offsets(F[:, :, 0], mk[rn])
        out[rn] = dict(r_odd_even=r.tolist(), r_mean=float(np.mean(r)), offset_frac=frac, noise_frac_expected=1.0 / len(st),
                       member_area_mean_offset=bias_i.tolist())
    # subset identity (raw, no drift correction): mean over 5 draws of 5-member CRPSS vs 30 members
    rng = np.random.default_rng(7)
    gm = {'global': mk['global']}
    c30 = []
    for idx in [np.arange(M)] + [np.sort(rng.choice(M, 5, replace=False)) for _ in range(5)]:
        Xs = F[:, idx]
        cr = np.stack([K.fair_crps(Xs[s], O[s]) for s in range(len(st))])
        pc, pr = K.amean(cr[:, 0], gm['global']), K.amean(np.asarray(CC['fixed'][:, 0]), gm['global'])
        ok = np.isfinite(pc) & np.isfinite(pr)
        c30.append(float(1 - pc[ok].sum() / pr[ok].sum()))
    out['raw_crpss_w34_global'] = dict(m30=c30[0], m5_draws=c30[1:])
    out['seconds'] = round(time.time() - t0, 1)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    return f'{sy} {field}: r_mean global {out["global"]["r_mean"]:.2f} offset_frac {out["global"]["offset_frac"]:.3f} {time.time() - t0:.0f}s'


# ------------------------------------------------------------------ phase permember (per-member drift correction, exploratory)
def task_permember(sy, field):
    f = OUT / 'permember' / f'{sy}_{field}.json'
    if f.exists():
        return 'cached'
    t0 = time.time()
    sd = V15 / 'score' / sy / f'{field}_g15'
    dj = json.loads((sd / 'done_dc.json').read_text())
    st = [date.fromisoformat(s) for s in dj['starts']]
    vd = V15 / 'work' / 'verif' / Path(dj['verif']).name
    meta = json.loads((vd / 'meta.json').read_text())
    pos = {s: i for i, s in enumerate(meta['starts'])}
    ix = np.array([pos[str(s)] for s in st])
    wi = [K.WK.index(w) for w in MWIN]
    O = np.load(vd / 'O.npy', mmap_mode='r')[ix][:, wi]
    CC = {r: np.load(vd / f'CC{r}.npy', mmap_mode='r')[ix][:, wi] for r in ('fixed', 'trend')}
    M = int(dj['members'])
    F = np.full((len(st), M, len(wi), 121, 240), np.nan, np.float32)
    rb = V15 / 'work' / 'reduce' / sy / field
    for i, s in enumerate(st):
        F[i] = np.load(rb / f'{s:%Y%m%d}.npy', mmap_mode='r')[:M][:, wi]
    K.set_lsm(np.load(B.STORE / 'lsm.npy'))
    mk = {r: K.masks('g15')[r] for r in B.REG4}
    sc = B.permember_scores(F, O, CC, st, mk, clip=K.FIELDS[field].get('clip', False))
    rows = [dict(system=sy, field=field, correction=c, reference=r, region=rn, window=MWIN[w], value=v) for (c, r, rn, w), v in sc.items()]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(rows=rows, starts=len(st), members=M, seconds=round(time.time() - t0, 1)), indent=0))
    d = {(r['correction'], r['window']): r['value'] for r in rows if r['reference'] == 'fixed' and r['region'] == 'global'}
    return f"{sy} {field}: w34 dc_ens {d[('dc_ens', 'w34')]:.3f} dc_member {d[('dc_member', 'w34')]:.3f} {time.time() - t0:.0f}s"


# ------------------------------------------------------------------ runner
def run_one(args):
    ph, key = args
    try:
        if ph == 'era5':
            return task_era5(key)
        if ph == 'start':
            return task_start(key)
        if ph == 'profile':
            return task_profile(*key)
        if ph == 'members':
            return task_members(*key)
        if ph == 'memberdiag':
            return task_memberdiag(*key)
        if ph == 'permember':
            return task_permember(*key)
    except Exception as e:  # noqa: BLE001
        return f'FAIL {ph} {key}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}'


def tasks(ph):
    if ph == 'era5':
        return era5_days()
    if ph == 'start':
        return starts_all()
    if ph == 'profile':
        return [(s, v) for s in B.SYSTEMS for v in B.WIN_UA]
    if ph == 'members':
        return [(s, f) for f in MEMBER_FIELDS for s in B.SYSTEMS]
    if ph == 'memberdiag':
        return [(s, f) for f in DIAG_FIELDS for s in B.SYSTEMS]
    if ph == 'permember':
        return [(s, f) for f in MEMBER_FIELDS for s in B.SYSTEMS]
    raise SystemExit(ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase')
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--rank', type=int, default=int(os.environ.get('PALS_RANKID', 0)))
    ap.add_argument('--nranks', type=int, default=int(os.environ.get('NRANKS', 1)))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    T = tasks(a.phase)
    if a.only:
        T = [t for t in T if a.only in str(t)]
    T = [t for i, t in enumerate(T) if i % a.nranks == a.rank]
    if a.limit:
        T = T[:a.limit]
    print(f'{a.phase}: rank {a.rank}/{a.nranks} {len(T)} tasks nproc {a.nproc} out {OUT}', flush=True)
    t0 = time.time()
    if a.nproc <= 1:
        for t in T:
            print(run_one((a.phase, t)), flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.nproc, maxtasksperchild=8) as p:
            for r in p.imap_unordered(run_one, [(a.phase, t) for t in T]):
                print(f'[{time.time() - t0:.0f}s] {r}', flush=True)
    print(f'{a.phase}: rank {a.rank} done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
