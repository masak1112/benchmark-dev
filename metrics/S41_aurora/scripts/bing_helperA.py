"""Helper A of the S41 evaluation v1.5 (W1 scorecards, W2 maps; plan v2 §2). One sub-command = one independent task.

    PY=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python3 (PYTHONPATH=envs/bing_pylib:<worktree>)
    $PY scripts/bing_helperA.py scorecard --a ucs2s_sstanom --b ucs2s_plain --field t2m     # paired rows, all windows
    $PY scripts/bing_helperA.py system    --sys fuxi --field t2m                            # per-system CRPSS / RPSS with CI
    $PY scripts/bing_helperA.py edges     --field t2m --samp 00z                            # q10 / q90 of the reference ensembles
    $PY scripts/bing_helperA.py bss       --sys fuxi --field t2m                            # per-start region-mean Brier scores
    $PY scripts/bing_helperA.py diffmap   --a ucs2s_sstanom --b ucs2s_plain --field t2m     # per-point paired differences + FDR
    $PY scripts/bing_helperA.py landocean --sys fuxi --field t2m                            # per-start region x land/ocean CRPS sums

Outputs under ``--out`` (default runs/bing/eval/v1.5/helperA). Every task skips if its output exists (resume).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

for _k, _v in (('OMP_NUM_THREADS', '4'), ('MKL_NUM_THREADS', '4'), ('OPENBLAS_NUM_THREADS', '4'), ('HDF5_USE_FILE_LOCKING', 'FALSE')):
    os.environ.setdefault(_k, _v)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from s41eval.bing_eval import core as K  # noqa: E402
from s41eval.bing_eval import data as D  # noqa: E402
from s41eval.bing_eval import pipeline as P  # noqa: E402

R = Path('/lus/flare/projects/S2Storm/s2storm')
OD = R / 'runs/bing/eval/v1.5'
CFG = Path(__file__).resolve().parents[1] / 'configs/bing_eval/aurora.yaml'
SUBSETS = ['all', 'DJF', 'MAM', 'JJA', 'SON']
MAP_WINDOWS = ['w34', 'w56']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def done_or(path):
    if Path(path).exists():
        log('exists', path)
        return True
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return False


def sysinfo(system, field):
    sd = OD / 'score' / system / f'{field}_g15'
    dj = json.loads((sd / 'done_dc.json').read_text())
    return sd, dj


def cmd_scorecard(a):
    from s41eval.bing_eval import scorecard as SC
    out = Path(a.out) / 'scorecard' / f'paired_{a.a}_vs_{a.b}_{a.field}.csv'
    if done_or(out):
        return
    rows = SC.paired_rows(OD, a.a, a.b, a.field, log=log)
    SC.write_csv(rows, out)
    log(len(rows), 'rows ->', out)


def cmd_system(a):
    from s41eval.bing_eval import scorecard as SC
    out = Path(a.out) / 'scorecard' / f'system_{a.sys}_{a.field}.csv'
    if done_or(out):
        return
    rows = SC.system_rows(OD, a.sys, a.field)
    SC.write_csv(rows, out)
    log(len(rows), 'rows ->', out)


def cmd_edges(a):
    from s41eval.bing_eval import extremes as EX
    cfg = P.load_config(str(CFG))
    vd = OD / 'work' / 'verif' / f'{a.field}_{a.samp}'
    meta = json.loads((vd / 'meta.json').read_text())
    od = Path(a.out) / 'work' / 'qedges' / f'{a.field}_{a.samp}'
    if done_or(od / 'meta.json'):
        return
    clim = D.Climatology(cfg['climatology']['stage_dirs'], cfg['climatology']['trend_dirs'],
                         tuple(cfg['climatology'].get('member_years', [1991, 2020])))
    starts = [date.fromisoformat(s) for s in meta['starts']][:a.limit or None]
    S, W = len(starts), len(K.WK)
    E = {r: np.full((S, 2, W, 121, 240), np.nan, np.float32) for r in ('fixed', 'trend')}
    V = {r: np.zeros((S, 2, W, 121, 240), bool) for r in ('fixed', 'trend')}
    Q = {r: np.full((S, len(EX.QUINT), W, 121, 240), np.nan, np.float32) for r in ('fixed', 'trend')}
    t0 = time.time()
    for i, st in enumerate(starts):
        e = clim.ensembles(a.field, a.samp, st)
        for r in ('fixed', 'trend'):
            if e[r] is None or e[r].shape[0] < 2:
                continue
            E[r][i], V[r][i] = EX.quantile_edges(e[r])
            Q[r][i] = EX.quantiles(e[r])
        if i % 50 == 0:
            log(f'edges {a.field}_{a.samp} {i}/{S} {time.time() - t0:.0f}s')
    for r in E:
        np.save(od / f'E{r}.npy', E[r])
        np.save(od / f'V{r}.npy', V[r])
        np.save(od / f'Q{r}.npy', Q[r])
    (od / 'meta.json').write_text(json.dumps(dict(field=a.field, sampling=a.samp, starts=[str(x) for x in starts], qs=list(EX.QS),
                                                  verif=str(vd), seconds=round(time.time() - t0, 1)), indent=1))
    log('edges done', od)


def cmd_bss(a):
    from s41eval.bing_eval import extremes as EX
    cfg = P.load_config(str(CFG))
    P.set_lsm(cfg)
    out = Path(a.out) / 'bss' / f'bss_{a.sys}_{a.field}.npz'
    if done_or(out):
        return
    sd, dj = sysinfo(a.sys, a.field)
    vd = Path(dj['verif'])
    samp = vd.name.split('_', 1)[1]
    qd = Path(a.out) / 'work' / 'qedges' / f'{a.field}_{samp}'
    qm = json.loads((qd / 'meta.json').read_text())
    starts = dj['starts'][:a.limit or None]
    vpos = {s: i for i, s in enumerate(json.loads((vd / 'meta.json').read_text())['starts'])}
    qpos = {s: i for i, s in enumerate(qm['starts'])}
    iv = np.array([vpos[s] for s in starts])
    iq = np.array([qpos[s] for s in starts])
    O = np.asarray(np.load(vd / 'O.npy', mmap_mode='r')[iv])
    Eg = {r: np.load(qd / f'E{r}.npy', mmap_mode='r') for r in ('fixed', 'trend')}
    Vg = {r: np.load(qd / f'V{r}.npy', mmap_mode='r') for r in ('fixed', 'trend')}
    Qg = {r: np.load(qd / f'Q{r}.npy', mmap_mode='r') for r in ('fixed', 'trend')} if (qd / 'Qfixed.npy').exists() else None
    b = np.load(sd / 'drift_bias.npy', mmap_mode='r')
    import xarray as xr
    ps_used = xr.open_dataset(sd / 'perstart_dc.nc')['used_dc'].values.astype(bool)[:len(starts)]
    clip = K.FIELDS[a.field].get('clip', False)
    rd = OD / 'work' / 'reduce' / a.sys / a.field

    class Lazy:
        def __init__(self, src, ix):
            self.src, self.ix = src, ix

        def __getitem__(self, s):
            return np.asarray(self.src[self.ix[s]])
    M = int(dj['members'])

    def F_dc(s):
        if not ps_used[s].any():
            return None
        X = np.load(rd / f'{starts[s].replace("-", "")}.npy').astype(np.float64)
        X = X - np.nan_to_num(np.asarray(b[s]))[None]
        if clip:
            X = np.clip(X, 0, None)
        return X
    t0 = time.time()
    res = EX.score_system(F_dc, O, {r: Lazy(Eg[r], iq) for r in Eg}, {r: Lazy(Vg[r], iq) for r in Vg},
                          Q_edges={r: Lazy(Qg[r], iq) for r in Qg} if Qg else None)
    arrs = {f'{k}_{r}': v for r in res for k, v in res[r].items()}
    for v in arrs.values():
        v[~ps_used] = np.nan                                  # windows the package did not score (e.g. beyond ERA5)
    np.savez(out, starts=np.array(starts), used=ps_used, members=M, regions=np.array(K.RK), windows=np.array(K.WK),
             events=np.array(EX.EVENTS), **arrs)
    log(f'bss {a.sys} {a.field}: {len(starts)} starts {time.time() - t0:.0f}s ->', out)


def _common(sa, sb):
    ca, cb = sa[1]['starts'], sb[1]['starts']
    common = sorted(set(ca) & set(cb))
    return common, np.array([ca.index(s) for s in common]), np.array([cb.index(s) for s in common])


def cmd_diffmap(a):
    import xarray as xr

    from s41eval.bing_eval import diffmaps as DM
    out = Path(a.out) / 'diffmaps' / f'diff_{a.a}_vs_{a.b}_{a.field}.nc'
    if done_or(out):
        return
    sa, sb = sysinfo(a.a, a.field), sysinfo(a.b, a.field)
    common, ia, ib = _common(sa, sb)
    st = [date.fromisoformat(s) for s in common]
    va, vb = Path(sa[1]['verif']), Path(sb[1]['verif'])
    same = va == vb
    vpa = {s: i for i, s in enumerate(json.loads((va / 'meta.json').read_text())['starts'])}
    vpb = {s: i for i, s in enumerate(json.loads((vb / 'meta.json').read_text())['starts'])}
    iva, ivb = np.array([vpa[s] for s in common]), np.array([vpb[s] for s in common])
    ua = xr.open_dataset(sa[0] / 'perstart_dc.nc')['used_dc'].values[ia].astype(bool)
    ub = xr.open_dataset(sb[0] / 'perstart_dc.nc')['used_dc'].values[ib].astype(bool)
    P_ = 121 * 240
    keys = ['crpss_fixed', 'crpss_trend', 'acc_fixed', 'acc_trend']
    res = {f'{k}_{v}': np.full((len(SUBSETS), len(MAP_WINDOWS), 121, 240), np.nan, np.float32) for k in keys
           for v in ('d', 'p', 'a', 'b')}
    for k in keys:
        res[f'{k}_sig'] = np.zeros((len(SUBSETS), len(MAP_WINDOWS), 121, 240), np.int8)
    nst = np.zeros((len(SUBSETS), len(MAP_WINDOWS)), int)
    thr = np.full((len(keys), len(SUBSETS), len(MAP_WINDOWS)), np.nan)
    t0 = time.time()
    ND = 20 if a.quick else 1000
    for iw, wn in enumerate(MAP_WINDOWS[:1] if a.quick else MAP_WINDOWS):
        w = K.WK.index(wn)
        L = lambda f, ix: np.asarray(np.load(f, mmap_mode='r')[ix, w], np.float64).reshape(len(ix), P_)  # noqa: E731
        ca, cb = L(sa[0] / 'crps_dc.npy', ia), L(sb[0] / 'crps_dc.npy', ib)
        ea, eb = L(sa[0] / 'em_dc.npy', ia), L(sb[0] / 'em_dc.npy', ib)
        Oa = L(va / 'O.npy', iva)
        Ob = Oa if same else L(vb / 'O.npy', ivb)
        refs = {}
        for r in ('fixed', 'trend'):
            cra = L(va / f'CC{r}.npy', iva)
            crb = cra if same else L(vb / f'CC{r}.npy', ivb)
            Ca = L(va / f'C{r}.npy', iva)
            Cb = Ca if same else L(vb / f'C{r}.npy', ivb)
            refs[r] = (cra, crb, Ca, Cb)
        for isb, sn in enumerate(SUBSETS[:1] if a.quick else SUBSETS):
            ix = np.array([i for i, s in enumerate(st) if (sn == 'all' or s.month in K.SEASONS[sn]) and ua[i, w] and ub[i, w]])
            nst[isb, iw] = ix.size
            for r in ('fixed', 'trend'):
                cra, crb, Ca, Cb = refs[r]
                m = DM.crpss_diff_map(ca[ix], cra[ix], cb[ix], crb[ix], seed=20261112, n_draws=ND)
                m2 = DM.acc_diff_map(ea[ix] - Ca[ix], Oa[ix] - Ca[ix], eb[ix] - Cb[ix], Ob[ix] - Cb[ix], seed=20261112, n_draws=ND)
                for k, mm in ((f'crpss_{r}', m), (f'acc_{r}', m2)):
                    for v in ('d', 'p', 'a', 'b'):
                        res[f'{k}_{v}'][isb, iw] = mm[v].reshape(121, 240)
                    sig, th = DM.fdr_wilks(mm['p'], 0.1)
                    res[f'{k}_sig'][isb, iw] = sig.reshape(121, 240)
                    thr[keys.index(k), isb, iw] = th
            log(f'diffmap {a.a}-{a.b} {a.field} {wn} {sn}: {ix.size} starts {time.time() - t0:.0f}s')
    ds = xr.Dataset({k: (('subset', 'window', 'lat', 'lon'), v[..., ::-1, :]) for k, v in res.items()},
                    coords=dict(subset=SUBSETS, window=MAP_WINDOWS, lat=K.LAT['g15'][::-1], lon=K.LON['g15'], key=keys))
    ds['n_starts'] = (('subset', 'window'), nst)
    ds['fdr_p_threshold'] = (('key', 'subset', 'window'), thr)
    ds.attrs.update(a_sys=a.a, b_sys=a.b, field=a.field, same_verif=int(same), alpha_fdr=0.1, block=4, n_draws=1000,
                    description='per-point A - B (d), two-sided bootstrap-SE normal p (p), Wilks 2016 FDR significance alpha 0.1 (sig); '
                                'crpss each vs own reference CRPS; acc centred, anomalies vs each reference climatology; '
                                'drift-corrected; seasons by start month; lat descending (WB2 order)')
    ds.to_netcdf(out)
    log('->', out)


def cmd_landocean(a):
    cfg = P.load_config(str(CFG))
    P.set_lsm(cfg)
    out = Path(a.out) / 'landocean' / f'lo_{a.sys}_{a.field}.npz'
    if done_or(out):
        return
    sd, dj = sysinfo(a.sys, a.field)
    vd = Path(dj['verif'])
    vpos = {s: i for i, s in enumerate(json.loads((vd / 'meta.json').read_text())['starts'])}
    iv = np.array([vpos[s] for s in dj['starts']])
    mk = K.masks('g15')
    land = mk['land'] > 0
    regs = ['global', 'tropics', 'nh_extratropics', 'sh_extratropics', 'conus']
    names, wts = [], []
    for rn in regs:
        for sfx, lm in (('land', land), ('ocean', ~land)):
            names.append(f'{rn}|{sfx}')
            wts.append((K.cosw('g15') * mk[rn] * lm).ravel())
    Wm = np.stack(wts, 1)                                    # [P, nreg]
    S = len(dj['starts'])
    res = {}
    for k, f in (('crps', sd / 'crps_dc.npy'), ('cc_fixed', vd / 'CCfixed.npy'), ('cc_trend', vd / 'CCtrend.npy')):
        A = np.load(f, mmap_mode='r')
        out_k = np.full((S, len(K.WK), len(names)), np.nan)
        for w in range(len(K.WK)):
            X = np.asarray(A[iv if k != 'crps' else np.arange(S), w], np.float64).reshape(S, -1)
            ok = np.isfinite(X)
            num = np.where(ok, X, 0) @ Wm
            den = ok.astype(np.float64) @ Wm
            with np.errstate(invalid='ignore', divide='ignore'):
                out_k[:, w] = np.where(den > 0, num / den, np.nan)
        res[k] = out_k
    np.savez(out, starts=np.array(dj['starts']), regions=np.array(names), windows=np.array(K.WK), **res)
    log('->', out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['scorecard', 'system', 'edges', 'bss', 'diffmap', 'landocean'])
    p.add_argument('--a')
    p.add_argument('--b')
    p.add_argument('--sys')
    p.add_argument('--field')
    p.add_argument('--samp')
    p.add_argument('--out', default=str(OD / 'helperA'))
    p.add_argument('--limit', type=int, default=0, help='edges / bss: first N starts only (smoke test)')
    p.add_argument('--quick', action='store_true', help='diffmap: w34, all starts, 20 draws (smoke test)')
    a = p.parse_args(argv)
    globals()[f'cmd_{a.cmd}'](a)


if __name__ == '__main__':
    main()
