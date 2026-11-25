#!/usr/bin/env python
"""C04 (exploratory): NAO / PNA / AO / SAM weekly-index COR by lead week (w1..w6), DJF and NDJFM starts, 30-member mean drift-corrected,
with an ERA5 persistence baseline (index of the 7-day mean ending on the init day, init-month pattern set) and a 'skill day' = lead
where the weekly COR crosses 0.5 (linear interpolation between window mid-days 4, 11, 18, 25, 32, 39; weekly means, not daily COR).
NAO/PNA/AO: reference test failed (all-month r vs CPC 0.745/0.54/0.71; DJF 0.88/0.88/0.91) -> DJF is the main row.
Inputs: modes/tele/perstart_tele_<system>.nc, modes/era5/tele_patterns.pkl, ERA5 daily z500 caches. Output: v1.5/helperC/tele/"""
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval import bing_helperc as HC  # noqa: E402
from s41eval import bing_modes as BM  # noqa: E402

OUT = HC.OUTC / 'tele'
IDX = ['nao', 'pna', 'ao', 'sam']
WKS = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
MID = np.array([4, 11, 18, 25, 32, 39], float)


def load(s):
    import xarray as xr
    ds = xr.open_dataset(HC.MODES / 'tele' / f'perstart_tele_{s}.nc')
    win = [str(w) for w in ds.window.values]
    j = [win.index(w) for w in WKS + ['w34']]
    out = dict(starts=HC.to_dates(ds.start.values), obs=ds.obs.values[:, :, j], em=ds.members_dc.values[:, :, :, j].mean(2))
    ds.close()
    return out


def pcor(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else np.nan


def persistence(TP, starts, samp):
    name = 'z500_00' if samp == '00z' else 'z500_mean4'
    D = BM.daily(name)
    clim = BM.fixed_clim(name, base=BM.EOF_YEARS, nh=4)
    P = np.full((len(IDX), len(starts)), np.nan)
    for i, d in enumerate(starts):
        dd = [d - timedelta(days=k) for k in range(6, -1, -1)]
        X = (D.get(dd) - clim(dd)).mean(0)
        r = TP.raw_indices(X, d.month)
        for j, k in enumerate(IDX):
            P[j, i] = r[k] / TP.norm[(k, d.month)]
    return P


def skill_day(c):
    c = np.asarray(c, float)
    if not np.isfinite(c[0]) or c[0] < 0.5:
        return np.nan
    for j in range(1, len(c)):
        if c[j] < 0.5:
            return float(MID[j - 1] + (c[j - 1] - 0.5) / (c[j - 1] - c[j]) * (MID[j] - MID[j - 1]))
    return float(MID[-1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TP = pickle.load(open(HC.MODES / 'era5' / 'tele_patterns.pkl', 'rb'))['TP']
    D = {s: load(s) for s in HC.SYSTEMS}
    starts = HC.common_starts([D[s]['starts'] for s in HC.SYSTEMS])
    D = {s: HC.subset({k: (np.moveaxis(v, 1, 0) if k != 'starts' else v) for k, v in d.items()}, starts) for s, d in D.items()}
    # D[s]['obs'], ['em'] are now [S, index, window]
    per = {'00z': persistence(TP, starts, '00z'), 'day': persistence(TP, starts, 'day')}
    np.savez(OUT / 'c04_persistence.npz', starts=np.array([str(d) for d in starts]), p00z=per['00z'], pday=per['day'])
    groups = {'DJF': np.array([d.month in (12, 1, 2) for d in starts]), 'NDJFM': np.array([d.month in (11, 12, 1, 2, 3) for d in starts]),
              'all': np.ones(len(starts), bool)}
    hA, hB = HC.halves(starts)
    rows, prs = [], []
    for g, gm in groups.items():
        sel = np.flatnonzero(gm)
        bidx = HC.mbb_indices(len(sel), 4, 1000, seed=7)
        for j, k in enumerate(IDX):
            curves = {}
            for s in HC.SYSTEMS + ['persistence_00z', 'persistence_day']:
                if s.startswith('persistence'):
                    samp = s.split('_')[1]
                    osys = 'ucs2s_sstanom' if samp == '00z' else 'fuxi'
                    f = np.repeat(per[samp][j][:, None], len(WKS) + 1, 1)
                    o = D[osys]['obs'][:, j]
                else:
                    f, o = D[s]['em'][:, j], D[s]['obs'][:, j]

                def st(b, f=f, o=o):
                    c = [pcor(f[sel[b], w], o[sel[b], w]) for w in range(len(WKS) + 1)]
                    return c + [skill_day(c[:6])]
                full = np.array(st(np.arange(len(sel))))
                dr = np.array([st(b) for b in bidx])
                curves[s] = (full, dr)
                ha = np.array([pcor(f[sel[hA[sel]], w], o[sel[hA[sel]], w]) for w in range(7)])
                hb = np.array([pcor(f[sel[hB[sel]], w], o[sel[hB[sel]], w]) for w in range(7)])
                curves[s] = (full, dr, ha, hb)
                for w, wn in enumerate(WKS + ['w34', 'skill_day']):
                    lo, hi = HC.ci_of(dr[:, w])
                    rows.append(dict(group=g, index=k, system=HC.LABEL.get(s, s), window=wn, n=len(sel), value=float(full[w]),
                                     ci_lo=lo, ci_hi=hi))
            for a, b in (('ucs2s_sstanom', 'fuxi'), ('ucs2s_plain', 'fuxi'), ('ucs2s_sstanom', 'ucs2s_plain'),
                         ('ucs2s_sstanom', 'persistence_00z'), ('ucs2s_plain', 'persistence_00z'), ('fuxi', 'persistence_day')):
                fa, da, haa, hba = curves[a]
                fb, db, hab, hbb = curves[b]
                for w, wn in enumerate(WKS + ['w34', 'skill_day']):
                    ci = HC.ci_of(da[:, w] - db[:, w])
                    ma = haa[w] - hab[w] if w < 7 else np.nan
                    mb = hba[w] - hbb[w] if w < 7 else np.nan
                    prs.append(dict(group=g, index=k, pair=f'{HC.LABEL.get(a, a)} - {HC.LABEL.get(b, b)}', window=wn, n=len(sel),
                                    diff=float(fa[w] - fb[w]), ci_lo=ci[0], ci_hi=ci[1], diff_2019_24=ma, diff_2025_26=mb,
                                    grade=HC.grade(ci, ma, mb) if w < 7 else ('noise' if ci[0] <= 0 <= ci[1] else 'CI excludes 0'),
                                    label='exploratory' + ('; reference test failed' if k != 'sam' else '')))
    HC.write_csv(rows, OUT / 'c04_tele_cor.csv')
    HC.write_csv(prs, OUT / 'c04_tele_pairs.csv')
    print('done', len(starts))


if __name__ == '__main__':
    main()
