#!/usr/bin/env python
"""C03 (exploratory, W5b): Wheeler-Kiladis spectra and wave-band diagnostics of 15S-15N OLR and u850, forecasts vs ERA5.
    python scripts/helperc_c03_waves.py --system S [--workers N]
Per start (members 0-9 of the strip extract, drift-corrected by the LOYO +-30 d lead-dependent 30-member-mean bias):
  (a) WK power, days 1-45 (mean + trend removed, 10 % split-cosine taper), symmetric / antisymmetric, summed over members and starts;
      ERA5 same valid days, same processing;
  (b) WK band filtering (Kelvin, ER, MRG, MJO; bing_waves.BANDS) of anomaly series (minus an ERA5 2017-2026 3-harmonic climatology)
      spanning init-135 .. init+135 days: forecast = ERA5 up to the init day + forecast days 1-45 + zeros; ERA5 'truncated' = ERA5 up to
      day 45 + zeros (same edge as the forecast, for variance ratios); ERA5 'truth' = ERA5 through day 135 (zeros after 2026-11-02).
      Per lead day 1-45: sums over the strip of member variance, truncated-ERA5 variance, and of the 30-member-mean x truth products.
Output: v1.5/helperC/waves/<system>.npz"""
import argparse
import json
import sys
import time
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval import bing_helperc as HC  # noqa: E402
from s41eval import bing_modes as BM  # noqa: E402
from s41eval import bing_waves as W  # noqa: E402

OUT = HC.OUTC / 'waves'
NP = 135
VARS = ('olr', 'u850')
BANDS = list(W.BANDS)
G = {}


def era5_clim(E):
    first = E.first
    C = {}
    for name, A in E.A.items():
        n = A.shape[0]
        days = [first + timedelta(days=i) for i in range(n)]
        H = BM.harm([BM.doy365(d) for d in days], 3)
        X = A.reshape(n, -1).astype(np.float64)
        ok = np.isfinite(X).all(1)
        C[name] = np.linalg.lstsq(H[ok], X[ok], rcond=None)[0].reshape((H.shape[1],) + A.shape[1:])
    return C


def clim_of(name, days):
    H = BM.harm([BM.doy365(d) for d in days], 3)
    return np.tensordot(H, G['clim'][name], axes=(1, 0))


def obs_name(v, samp, days):
    E = G['E']
    if v == 'olr':
        dd = [d - timedelta(days=1) for d in days] if samp == '00z' else list(days)
        return 'olr_flux', dd
    return ('u850_00' if samp == '00z' else 'u850_mean4'), list(days)


def obs_anom(v, samp, days):
    name, dd = obs_name(v, samp, days)
    x = G['E'].get(name, dd).astype(np.float64) - clim_of(name, dd)
    return np.where(np.isfinite(x), x, 0.0), np.isfinite(G['E'].get(name, dd)).all((1, 2))


def one(i):
    d = G['starts'][i]
    samp = G['samp']
    z = np.load(W.eq_path(G['system'], d))
    vd = BM.valid_dates(d)
    res = {}
    for v in VARS:
        bias = G['bias'][v][i]                                                           # [46, 21, 240]
        mem = z[f'{v}_m'].astype(np.float64) - bias[None]
        em = z[f'{v}_em'].astype(np.float64) - bias
        o_name, o_days = obs_name(v, samp, vd)
        obs = G['E'].get(o_name, o_days).astype(np.float64)
        okobs = np.isfinite(obs[1:]).all()
        # (a) spectra, days 1..45
        P = {'f_sym': 0.0, 'f_anti': 0.0, 'o_sym': 0.0, 'o_anti': 0.0}
        for m in range(mem.shape[0]):
            s_, a_ = W.wk_power(mem[m, 1:])
            P['f_sym'] = P['f_sym'] + s_
            P['f_anti'] = P['f_anti'] + a_
        P['f_sym'] /= mem.shape[0]
        P['f_anti'] /= mem.shape[0]
        if okobs:
            P['o_sym'], P['o_anti'] = W.wk_power(obs[1:])
        else:
            P['o_sym'] = P['o_anti'] = np.full_like(P['f_sym'], np.nan)
        res[f'{v}_spec'] = P
        # (b) filtering
        past = [d + timedelta(days=k) for k in range(-NP, 1)]
        fut = [d + timedelta(days=k) for k in range(1, NP + 1)]
        pa, _ = obs_anom(v, samp, past)
        fo, okf = obs_anom(v, samp, fut)
        cl = clim_of(o_name, obs_name(v, samp, vd[1:])[1])                               # forecast anomaly vs the same climatology
        zeros = np.zeros((NP - 45,) + pa.shape[1:])
        truth = np.concatenate([pa, fo], 0)
        trunc = np.concatenate([pa, fo[:45], zeros], 0)
        stats = {}
        ftr = {b: W.wk_filter(truth, b)[NP + 1:NP + 46] for b in BANDS}
        fto = {b: W.wk_filter(trunc, b)[NP + 1:NP + 46] for b in BANDS}
        emser = np.concatenate([pa, em[1:] - cl, zeros], 0)
        fem = {b: W.wk_filter(emser, b)[NP + 1:NP + 46] for b in BANDS}
        vm = {b: np.zeros(45) for b in BANDS}
        for m in range(mem.shape[0]):
            ser = np.concatenate([pa, mem[m, 1:] - cl, zeros], 0)
            for b in BANDS:
                y = W.wk_filter(ser, b)[NP + 1:NP + 46]
                vm[b] += (y ** 2).sum((1, 2))
        for b in BANDS:
            stats[b] = np.stack([vm[b] / mem.shape[0], (fto[b] ** 2).sum((1, 2)), (ftr[b] ** 2).sum((1, 2)),
                                 (fem[b] * ftr[b]).sum((1, 2)), (fem[b] ** 2).sum((1, 2))])      # [5, 45]
        res[f'{v}_filt'] = stats
        res[f'{v}_ok'] = bool(okobs) and bool(okf[:45].all())
    return i, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--system', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    s = a.system
    starts = [d for d in BM.list_starts() if W.eq_path(s, d).exists()]
    if s.startswith('ucs2s'):                                          # 2026-09-21 rerun pending (days 44-45 suspect); keep common 402
        starts = [d for d in starts if d.isoformat() < '2026-09-21']
    if a.limit:
        starts = starts[:a.limit]
    E = W.Era5Eq()
    G.update(system=s, samp=BM.state_samp(BM.SAMPLING[s]), starts=starts, E=E, clim=era5_clim(E))
    # pass 1: drift (LOYO +-30 d lead-dependent mean of the 30-member mean minus ERA5)
    S = len(starts)
    G['bias'] = {}
    for v in VARS:
        err = np.full((S, BM.NLEAD, W.NEQ, 240), np.nan, np.float32)
        for i, d in enumerate(starts):
            z = np.load(W.eq_path(s, d))
            o_name, o_days = obs_name(v, G['samp'], BM.valid_dates(d))
            err[i] = z[f'{v}_em'] - E.get(o_name, o_days)
        err[:, 0] = 0.0                                                  # lead 0 is replaced by ERA5 (UC-S2S writes OLR 0 at lead 0)
        b = BM.loyo_bias(err, starts).astype(np.float32)
        G['bias'][v] = np.where(np.isfinite(b), b, 0.0)
        print(f'{s} bias {v} done {time.time() - t0:.0f}s', flush=True)
    del err
    freq, ks = W.wk_axes(45)
    acc = {f'{v}_{k}': np.zeros((len(freq), len(ks))) for v in VARS for k in ('f_sym', 'f_anti', 'o_sym', 'o_anti')}
    nacc = {v: 0 for v in VARS}
    bandp = {f'{v}_{w}_{b}': np.full(S, np.nan) for v in VARS for w in ('f', 'o') for b in BANDS}
    filt = {v: np.full((S, len(BANDS), 5, 45), np.nan) for v in VARS}
    okv = {v: np.zeros(S, bool) for v in VARS}
    with Pool(a.workers) as pool:
        for i, res in pool.imap_unordered(one, range(S), chunksize=2):
            for v in VARS:
                P = res[f'{v}_spec']
                if np.isfinite(P['o_sym']).all():
                    for k in ('f_sym', 'f_anti', 'o_sym', 'o_anti'):
                        acc[f'{v}_{k}'] += P[k]
                    nacc[v] += 1
                    for b in BANDS:
                        comp = 'sym' if W.BANDS[b]['sym'] == 'sym' else 'anti'
                        bandp[f'{v}_f_{b}'][i] = W.band_power(P[f'f_{comp}'], freq, ks, b)
                        bandp[f'{v}_o_{b}'][i] = W.band_power(P[f'o_{comp}'], freq, ks, b)
                for j, b in enumerate(BANDS):
                    filt[v][i, j] = res[f'{v}_filt'][b]
                okv[v][i] = res[f'{v}_ok']
    out = dict(starts=np.array([d.isoformat() for d in starts]), freq=freq, ks=ks, bands=np.array(BANDS),
               filt_rows=np.array(['var_members', 'var_era5_trunc', 'var_era5_truth', 'cov_em30_truth', 'var_em30']))
    for k, x in acc.items():
        out[f'spec_{k}'] = x / max(nacc[k.split('_')[0]], 1)
    for k, x in bandp.items():
        out[f'bandpow_{k}'] = x
    for v in VARS:
        out[f'filt_{v}'] = filt[v]
        out[f'ok_{v}'] = okv[v]
        out[f'nspec_{v}'] = nacc[v]
    np.savez(OUT / f'{s}.npz', **out)
    (OUT / f'{s}.json').write_text(json.dumps(dict(system=s, starts=S, nspec={v: nacc[v] for v in VARS}, seconds=time.time() - t0,
                                                   members='0-9 (spectra, member variance), 30-member mean (skill)'), indent=1))
    print(f'{s} done {S} starts {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
