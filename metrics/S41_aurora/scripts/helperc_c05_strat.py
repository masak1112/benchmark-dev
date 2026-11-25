#!/usr/bin/env python
"""C05 (exploratory): stratosphere. (1) SSW hits: P(U60 reversal within +-3 d of the ERA5 onset) by lead for every 2019-26 ERA5 event
(10 hPa for UC-S2S, 50 hPa for all; FuXi has no 10 hPa), hit = P >= 0.5, hit rate by lead week with counts (from modes/strat/ssw_hits.csv);
(2) U60N10 / U60N50 and polar-cap (60-90N) height anomaly ACC by lead, NDJFM starts, 30-member mean drift-corrected, with block-bootstrap
CIs and the lead where ACC falls below 0.6. Inputs: modes extract zmu60/cap_z, ERA5 strat staging (bing_modes_score.era5_strat).
Output: v1.5/helperC/strat/"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s41eval import bing_helperc as HC  # noqa: E402
from s41eval import bing_modes as BM  # noqa: E402
import bing_modes_score as MS  # noqa: E402

OUT = HC.OUTC / 'strat'
LEADS = np.arange(1, 46)


def acc_curve(f, o, c):
    fa, oa = f - c, o - c
    out = np.full(f.shape[1], np.nan)
    for k in range(1, f.shape[1]):
        ok = np.isfinite(fa[:, k]) & np.isfinite(oa[:, k])
        if ok.sum() > 5:
            out[k] = np.corrcoef(fa[ok, k], oa[ok, k])[0, 1]
    return out


def cross(c, thr=0.6):
    for k in range(2, len(c)):
        if np.isfinite(c[k]) and c[k] < thr <= c[k - 1]:
            return float(k - 1 + (c[k - 1] - thr) / (c[k - 1] - c[k]))
    return float('nan') if not np.isfinite(c).any() else (45.0 if np.nanmin(c[1:]) >= thr else float('nan'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S5 = MS.era5_strat()
    clim = {k: MS.Clim1D(v) for k, v in S5.items()}
    rows, sd = [], []
    for s in HC.SYSTEMS:
        st = MS.starts_of(s)
        if s.startswith('ucs2s'):
            st = [d for d in st if d.isoformat() < '2026-09-21']
        nd = np.array([d.month in (11, 12, 1, 2, 3) for d in st])
        st = [d for d, m in zip(st, nd) if m]
        samp = BM.state_samp(BM.SAMPLING[s])
        X, have = BM.load_extract(s, st, ['zmu60', 'cap_z', 'zm_levels'])
        zl = [int(round(v)) for v in X['zm_levels'][have[0]]] if 'zm_levels' in X else []
        series = {}
        for L in (10, 50):
            if L in zl:
                series[f'U60N{L}'] = (np.nanmean(X['zmu60'][:, :, :, zl.index(L)], 1), f'u60_{L}')
        for jl, L in enumerate(BM.CAP_LEVELS):
            f = np.nanmean(X['cap_z'][:, :, :, jl], 1) / MS.G0
            if np.isfinite(f[:, 1:]).any():
                series[f'cap{L}'] = (f, f'cap_{L}')
        hA, hB = HC.halves(st)
        bidx = HC.mbb_indices(len(st), 4, 1000, seed=8)
        for nm, (F, key) in series.items():
            O = np.stack([MS.series_get(S5[key], BM.valid_dates(d), samp) for d in st])
            C = np.stack([clim[key](BM.valid_dates(d)) for d in st])
            Fd = F - BM.loyo_bias(F - O, st)
            full = acc_curve(Fd, O, C)
            dr = np.array([acc_curve(Fd[b], O[b], C[b]) for b in bidx])
            for k in range(1, BM.NLEAD):
                lo, hi = HC.ci_of(dr[:, k])
                rows.append(dict(system=HC.LABEL[s], index=nm, lead=k, n=len(st), acc=float(full[k]), ci_lo=lo, ci_hi=hi))
            dd = np.array([cross(c) for c in dr])
            lo, hi = HC.ci_of(dd)
            sd.append(dict(system=HC.LABEL[s], index=nm, n=len(st), lead_acc_0p6=cross(full), ci_lo=lo, ci_hi=hi,
                           h2019_24=cross(acc_curve(Fd[hA], O[hA], C[hA])), h2025_26=cross(acc_curve(Fd[hB], O[hB], C[hB]))))
        print(s, len(st), flush=True)
    HC.write_csv(rows, OUT / 'c05_acc_by_lead.csv')
    HC.write_csv(sd, OUT / 'c05_acc_0p6_lead.csv')
    # SSW hits by lead week (from the modes file; 2019-26 events with starts 1-42 d before onset)
    with open(HC.MODES / 'strat' / 'ssw_hits.csv') as fh:
        H = list(csv.DictReader(fh))
    hr = []
    for s in HC.SYSTEMS:
        for L in ('10', '50'):
            hh = [h for h in H if h['system_key'] == s and h['level'] == L]
            if not hh:
                continue
            for wk, (a, b) in {'days 1-7': (1, 7), 'days 8-14': (8, 14), 'days 15-21': (15, 21), 'days 22-28': (22, 28),
                               'days 29-42': (29, 42)}.items():
                x = [h for h in hh if a <= int(h['lead']) <= b]
                if x:
                    hr.append(dict(system=HC.LABEL[s], level=L, lead_bin=wk, n_cases=len(x), n_events=len({h['event'] for h in x}),
                                   hits=sum(int(h['hit']) for h in x), hit_rate=sum(int(h['hit']) for h in x) / len(x),
                                   mean_p=float(np.mean([float(h['p_reversal']) for h in x]))))
    HC.write_csv(hr, OUT / 'c05_ssw_hit_rate.csv')
    print('done')


if __name__ == '__main__':
    main()
