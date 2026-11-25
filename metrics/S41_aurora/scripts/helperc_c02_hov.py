#!/usr/bin/env python
"""C02 (exploratory): MJO propagation across the Maritime Continent. Hovmoller composites of 15S-15N OLR and u850 anomalies (RMM-input
anomalies: minus the ERA5 1979-2018 3-harmonic climatology, minus the mean of the previous 120 days with ERA5 before the init), 30-member
ensemble mean, drift-corrected (LOYO +-30 d, lead-dependent, per longitude), by initial RMM phase group (active MJO), leads 0-28, vs ERA5
for the same starts; plus RMM phase speed and MC-crossing fractions from the member RMM.
Inputs: modes extract band_olr/band_u850 (all members), modes/era5/mjo_full.pkl, modes/mjo/perstart_mjo_full_<system>.nc.
Outputs: v1.5/helperC/hov/*.npz, *.csv"""
import json
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval import bing_helperc as HC  # noqa: E402
from s41eval import bing_modes as BM  # noqa: E402

OUT = HC.OUTC / 'hov'
NL = 29
PG = {'P2-3': (2, 3), 'P4-5': (4, 5), 'P6-7': (6, 7), 'P8-1': (8, 1)}


def anom_em(ix, samp, init, band_em):
    """band_em {v: [46, 240]} ensemble-mean raw band -> RMM-input anomaly [46, 240] per field (lead 0 = ERA5)."""
    S = ix.series[samp]
    i0 = S['ix'][init]
    H = BM.harm([BM.doy365(d) for d in BM.valid_dates(init)], 3)
    out = {}
    for v in ('olr', 'u850'):
        fa = np.asarray(band_em[v], np.float64) - H @ S['coef'][v]
        past = S['anom'][v][i0 - 120:i0 + 1]
        seq = np.concatenate([past, fa[1:]], 0)
        f = np.full((BM.NLEAD, 240), np.nan)
        for k in range(BM.NLEAD):
            f[k] = seq[120 + k] - seq[k:k + 120].mean(0)
        out[v] = f
    return out


def obs_anom(ix, samp, init):
    S = ix.series[samp]
    i0 = S['ix'][init]
    out = {}
    for v in ('olr', 'u850'):
        a = S['anom'][v][i0 - 120:i0 + BM.NLEAD]
        out[v] = BM.prev120(a)[120:]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ix = pickle.load(open(HC.MODES / 'era5' / 'mjo_full.pkl', 'rb'))
    mj = {s: HC.load_mjo(s, 'full') for s in HC.SYSTEMS}
    starts = HC.common_starts([mj[s]['starts'] for s in HC.SYSTEMS])
    mj = {s: HC.subset(v, starts) for s, v in mj.items()}
    init = mj['ucs2s_sstanom']['init']
    amp = np.hypot(init[:, 0], init[:, 1])
    ph = HC.rmm_phase(init[:, 0], init[:, 1])
    S = len(starts)
    F, O = {}, {}
    for s in HC.SYSTEMS:
        samp = BM.state_samp(BM.SAMPLING[s])
        F[s] = {v: np.full((S, BM.NLEAD, 240), np.nan) for v in ('olr', 'u850')}
        O[s] = {v: np.full((S, BM.NLEAD, 240), np.nan) for v in ('olr', 'u850')}
        for i, d in enumerate(starts):
            z = np.load(BM.extract_path(s, d))
            em = {v: z[f'band_{v}'][:30].astype(np.float64).mean(0) for v in ('olr', 'u850')}
            a = anom_em(ix, samp, d, em)
            o = obs_anom(ix, samp, d)
            for v in a:
                F[s][v][i], O[s][v][i] = a[v], o[v]
        for v in ('olr', 'u850'):
            F[s][v] = F[s][v] - BM.loyo_bias(F[s][v] - O[s][v], starts)
        print(s, 'loaded', flush=True)
    rows = []
    comp = {}
    lon = BM.LON15
    reg = (lon >= 40) & (lon <= 220)
    for g, (a, b) in PG.items():
        m = (amp >= 1) & np.isin(ph, [a, b])
        for v in ('olr', 'u850'):
            for s in HC.SYSTEMS:
                fc = np.nanmean(F[s][v][m, :NL], 0)
                oc = np.nanmean(O[s][v][m, :NL], 0)
                comp[f'{g}_{v}_{s}'] = fc
                comp[f'{g}_{v}_era5_{BM.SAMPLING[s]}'] = oc
                for wk, (k0, k1) in {'w1': (1, 7), 'w2': (8, 14), 'w3': (15, 21), 'w4': (22, 28), 'w1-4': (1, 28)}.items():
                    x, y = fc[k0:k1 + 1][:, reg].ravel(), oc[k0:k1 + 1][:, reg].ravel()
                    rows.append(dict(group=g, field=v, system=HC.LABEL[s], window=wk, n=int(m.sum()),
                                     pattern_corr=float(np.corrcoef(x, y)[0, 1]),
                                     amp_ratio=float(np.sqrt((x ** 2).mean() / (y ** 2).mean()))))
    np.savez(OUT / 'c02_hov_composites.npz', lon=lon, leads=np.arange(NL), **comp)
    HC.write_csv(rows, OUT / 'c02_hov_scores.csv')

    # RMM propagation: phase speed (deg/day, leads 1-20, unwrapped) and MC crossing (init P3-4 active: reaches P6-7 by day 20 with amp >= 0.5)
    prow = []
    for s in HC.SYSTEMS:
        d = mj[s]
        bias = d['raw'] - d['dc']                                   # [S, 46, 2] lead-dependent ensemble-mean drift (LOYO)
        mem = d['members'] - bias[:, None]
        o = d['obs']
        ang_o = np.unwrap(np.arctan2(o[:, :21, 1], o[:, :21, 0]), axis=1)
        ang_m = np.unwrap(np.arctan2(mem[:, :, :21, 1], mem[:, :, :21, 0]), axis=2)
        sp_o = np.rad2deg(ang_o[:, 20] - ang_o[:, 1]) / 19.0
        sp_m = np.rad2deg(ang_m[:, :, 20] - ang_m[:, :, 1]) / 19.0
        for g, msk in {'active': amp >= 1, 'P2-3': (amp >= 1) & np.isin(ph, [2, 3]), 'P4-5': (amp >= 1) & np.isin(ph, [4, 5]),
                       'P6-7': (amp >= 1) & np.isin(ph, [6, 7]), 'P8-1': (amp >= 1) & np.isin(ph, [8, 1])}.items():
            sel = np.flatnonzero(msk)
            bidx = HC.mbb_indices(len(sel), 4, 1000, seed=5)
            st = lambda b: np.nanmean(sp_m[sel[b]]) - np.nanmean(sp_o[sel[b]])  # noqa: E731
            dr = np.array([st(b) for b in bidx])
            sA, sB = HC.halves([starts[i] for i in sel])
            prow.append(dict(system=HC.LABEL[s], group=g, n=len(sel), speed_obs_deg_per_day=float(np.nanmean(sp_o[sel])),
                             speed_members_deg_per_day=float(np.nanmean(sp_m[sel])), diff=float(st(np.arange(len(sel)))),
                             ci_lo=HC.ci_of(dr)[0], ci_hi=HC.ci_of(dr)[1],
                             grade=HC.grade(HC.ci_of(dr), float(np.nanmean(sp_m[sel[sA]]) - np.nanmean(sp_o[sel[sA]])),
                                            float(np.nanmean(sp_m[sel[sB]]) - np.nanmean(sp_o[sel[sB]])))))
        # MC crossing
        m34 = (amp >= 1) & np.isin(ph, [3, 4])
        sel = np.flatnonzero(m34)
        pho = HC.rmm_phase(o[sel, 1:21, 0], o[sel, 1:21, 1])
        ampo = np.hypot(o[sel, 1:21, 0], o[sel, 1:21, 1])
        cross_o = (np.isin(pho, [6, 7]) & (ampo >= 0.5)).any(1)
        phm = HC.rmm_phase(mem[sel, :, 1:21, 0], mem[sel, :, 1:21, 1])
        ampm = np.hypot(mem[sel, :, 1:21, 0], mem[sel, :, 1:21, 1])
        cross_m = (np.isin(phm, [6, 7]) & (ampm >= 0.5)).any(2).mean(1)
        bidx = HC.mbb_indices(len(sel), 4, 1000, seed=6)
        dr = np.array([cross_m[b].mean() - cross_o[b].mean() for b in bidx])
        hA, hB = HC.halves([starts[i] for i in sel])
        gr = HC.grade(HC.ci_of(dr), float(cross_m[hA].mean() - cross_o[hA].mean()), float(cross_m[hB].mean() - cross_o[hB].mean()))
        prow.append(dict(system=HC.LABEL[s], group='MC crossing P3-4 -> P6-7 by day 20 (amp >= 0.5)', n=len(sel),
                         speed_obs_deg_per_day=float(cross_o.mean()), speed_members_deg_per_day=float(cross_m.mean()),
                         diff=float(cross_m.mean() - cross_o.mean()), ci_lo=HC.ci_of(dr)[0], ci_hi=HC.ci_of(dr)[1], grade=gr))
    HC.write_csv(prow, OUT / 'c02_propagation.csv')
    (OUT / 'c02_meta.json').write_text(json.dumps(dict(
        starts=S, groups={g: int(((amp >= 1) & np.isin(ph, list(p))).sum()) for g, p in PG.items()},
        note='composites over active-MJO starts by ERA5 init phase (provisional 37.5 deg rotation); region for scores 40-220E; drift-corrected '
             '30-member mean; ERA5 per system sampling (UC 00z, FuXi day); for the MC-crossing row speed_obs/members columns hold fractions; '
             'members are drift-corrected by the ensemble-mean LOYO bias; exploratory'), indent=1))
    print(json.dumps([r for r in prow if r['group'] in ('active', 'MC crossing P3-4 -> P6-7 by day 20 (amp >= 0.5)')], indent=0, default=str))


if __name__ == '__main__':
    main()
