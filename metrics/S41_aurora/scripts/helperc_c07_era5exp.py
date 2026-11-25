#!/usr/bin/env python
"""C07 ERA5 expectation (exploratory, W6): how much weeks 3-4 skill the init state alone gives, from ERA5 statistics.
Empirical composite forecasts trained on ERA5 1991-2018 (every day): weeks 3-4 (days 15-28) mean anomaly (00 UTC states / 24 h fluxes
ending 00 UTC, minus the modes harmonic 1991-2020 climatology), composited by the init state
  MJO: RMM phase group (active, amp >= 1; P8-1, P2-3, P4-5, P6-7) or inactive;  ENSO: el_nino / la_nina / neutral (30-day Nino-3.4);
separately for Nov-Apr and May-Oct inits. For every 2019-26 start the forecast is the composite of its group and half-year; skill =
per-start centred pattern ACC vs the package truth (O - Cfixed, w34), regions as the package. Run on a compute node (reads ~4 GB).
Output: v1.5/helperC/c07_era5/era5_expect.csv (+ npz of the composites)"""
import csv
import json
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s41eval import bing_helperc as HC  # noqa: E402
from s41eval import bing_modes as BM  # noqa: E402
import bing_modes_score as MS  # noqa: E402
from helperc_c06_enso import load_em  # noqa: E402
from helperc_c07_windows import REG, LAT, LON, pattern_acc  # noqa: E402

OUT = HC.OUTC / 'c07_era5'
NAMES = {'z500': 'z500_00', 't2m': 't2m_00', 'tp24': 'tp24_flux'}
T0, T1 = date(1991, 1, 2), date(2018, 12, 31)          # 1991-01-02: the 00z flux of day d is the cache day d-1 (starts 1991-01-01)


def half(d):
    return 'NDJFMA' if d.month in (11, 12, 1, 2, 3, 4) else 'MJJASO'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ix = pickle.load(open(HC.MODES / 'era5' / 'mjo_full.pkl', 'rb'))
    tdays = BM.daterange(T0, T1)
    r = ix.obs('day', tdays)
    amp = np.hypot(r[:, 0], r[:, 1])
    ph = HC.rmm_phase(r[:, 0], r[:, 1])
    mjo_g = np.full(len(tdays), 'inactive', object)
    for g, q in {'P8-1': [8, 1], 'P2-3': [2, 3], 'P4-5': [4, 5], 'P6-7': [6, 7]}.items():
        mjo_g[(amp >= 1) & np.isin(ph, q)] = g
    mjo_g[~np.isfinite(amp)] = ''
    x, ndays, clim_box = MS.era5_nino('00z')
    nix = {d: i for i, d in enumerate(ndays)}
    an = x - clim_box(ndays)
    m30 = np.array([np.nanmean([an[nix[d - timedelta(days=k)]] for k in range(30) if (d - timedelta(days=k)) in nix]) for d in tdays])
    enso_g = np.where(m30 >= 0.5, 'el_nino', np.where(m30 <= -0.5, 'la_nina', 'neutral'))
    hv = np.array([half(d) for d in tdays])
    # 2019-26 starts and their states (same definitions as helperc_c07_windows)
    stz = np.load(HC.OUTC / 'windows' / 'c07_init_states.npz', allow_pickle=True)
    starts = HC.to_dates(stz['starts'])
    s_amp, s_ph = stz['amp'], stz['phase']
    s_mjo = np.full(len(starts), 'inactive', object)
    for g, q in {'P8-1': [8, 1], 'P2-3': [2, 3], 'P4-5': [4, 5], 'P6-7': [6, 7]}.items():
        s_mjo[(s_amp >= 1) & np.isin(s_ph, q)] = g
    s_enso = stz['enso']
    s_half = np.array([half(d) for d in starts])
    rows, comps = [], {}
    for f, name in NAMES.items():
        D = BM.daily(name)
        cl = BM.fixed_clim(name)
        sh = -1 if f == 'tp24' else 0                                   # 00z sampling: flux of the 24 h ending 00 UTC = day d-1
        d0 = T0 + timedelta(days=sh)
        alld = BM.daterange(d0, T1 + timedelta(days=30 + sh))
        i0, i1 = D.idx(alld[0]), D.idx(alld[-1]) + 1
        X = D.slab(i0, i1).astype(np.float32)
        X -= cl(alld)
        X = np.nan_to_num(X)
        cs = np.concatenate([np.zeros((1,) + X.shape[1:], np.float64), np.cumsum(X, 0, dtype=np.float64)], 0)
        del X
        w34 = lambda i: (cs[i + 29] - cs[i + 15]) / 14.0  # noqa: E731  (training day index i -> days i+15..i+28)
        for kind, tg in (('mjo', mjo_g), ('enso', enso_g)):
            for hn in ('NDJFMA', 'MJJASO'):
                for g in sorted(set(tg) - {''}):
                    ii = np.flatnonzero((tg == g) & (hv == hn))
                    if len(ii) < 20:
                        continue
                    comps[f'{f}|{kind}|{hn}|{g}'] = (sum(w34(i) for i in ii) / len(ii)).astype(np.float32)
                    comps[f'n|{f}|{kind}|{hn}|{g}'] = np.int32(len(ii))
        del cs
        # skill on the 2019-26 starts, truth = package O - Cfixed (UC-S2S 00z verification)
        st_, em, O, C = load_em('ucs2s_sstanom', f, 'w34')
        q = {d: i for i, d in enumerate(st_)}
        jj = np.array([q[d] for d in starts])
        Oa = O[jj] - C[jj]
        for kind, sg in (('mjo', s_mjo), ('enso', s_enso), ('mjo+enso', None)):
            F = np.zeros_like(Oa)
            for i in range(len(starts)):
                if kind == 'mjo+enso':
                    F[i] = comps.get(f'{f}|mjo|{s_half[i]}|{s_mjo[i]}', 0) + comps.get(f'{f}|enso|{s_half[i]}|{s_enso[i]}', 0)
                else:
                    F[i] = comps.get(f'{f}|{kind}|{s_half[i]}|{sg[i]}', np.zeros(Oa.shape[1:]))
            for rn, (a_, b_, c_, d_) in REG.items():
                mk = (LAT[:, None] >= a_) & (LAT[:, None] <= b_) & (LON[None] >= c_) & (LON[None] <= d_)
                acc = pattern_acc(F, Oa, mk)
                for i, d in enumerate(starts):
                    rows.append(dict(field=f, predictor=kind, region=rn, start=str(d), mjo=s_mjo[i], enso=s_enso[i], half=s_half[i],
                                     acc=float(acc[i])))
        print(f, 'done', flush=True)
    HC.write_csv(rows, OUT / 'era5_expect_perstart.csv')
    np.savez(OUT / 'era5_composites.npz', **comps)
    (OUT / 'meta.json').write_text(json.dumps(dict(training=[str(T0), str(T1)], note=__doc__), indent=1))


if __name__ == '__main__':
    main()
