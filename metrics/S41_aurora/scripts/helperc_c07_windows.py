#!/usr/bin/env python
"""C07 (exploratory, W6): windows of opportunity. Weeks 3-4 skill conditioned on the state at init, per system, from package v1.5:
  CRPSS (fair, 30 members, drift-corrected) of a group = 1 - sum(crps) / sum(cc_ref) from perstart_dc.nc (refs fixed 1991-2020 and trend);
  pattern ACC = per-start centred cos-lat anomaly correlation over the region of em_dc - Cfixed vs O - Cfixed, mean over the group's starts.
Conditions (ERA5 at init): MJO active (RMM amp >= 1, calendar-day RMM, provisional rotation for phases) and phase group; ENSO phase
(30-day-mean Nino-3.4 >= +0.5 / <= -0.5 K, modes definition); SSW (10-hPa U60 reversal, ERA5) within the 30 days before the init (NDJFM
starts); large CONUS soil-moisture anomaly (|z| >= 1 of the ERA5 init-day CONUS land-mean swvl1 anomaly, z standardised across the
2019-26 starts of the same season). Contrasts: group A minus group B, block bootstrap over all starts in date order (block 4, 1,000 draws;
each draw recomputes both groups), grades as the package (halves 2019-24 / 2025-26).
If v1.5/helperC/c07_era5/era5_expect.csv exists (helperc_c07_era5exp.py), the ERA5 expectation is merged into the output.
Output: v1.5/helperC/windows/"""
import csv
import json
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s41eval import bing_helperc as HC  # noqa: E402
from s41eval import bing_modes as BM  # noqa: E402
from helperc_c06_enso import load_em, score_dir  # noqa: E402

OUT = HC.OUTC / 'windows'
FIELDS = ['t2m', 'z500', 'tp24']
REG = {'global': (-90, 90, 0, 360), 'tropics': (-20, 20, 0, 360), 'nh_extratropics': (20, 90, 0, 360), 'conus': (25, 50, 235, 295)}
LAT = np.linspace(-90.0, 90.0, 121)
LON = np.arange(240) * 1.5


def init_states(starts):
    ix = pickle.load(open(HC.MODES / 'era5' / 'mjo_full.pkl', 'rb'))
    r = ix.obs('day', starts)
    amp = np.hypot(r[:, 0], r[:, 1])
    ph = HC.rmm_phase(r[:, 0], r[:, 1])
    with open(HC.MODES / 'enso' / 'perstart_nino34.csv') as fh:
        en = {x['start']: x['enso_phase'] for x in csv.DictReader(fh)}
    enso = np.array([en.get(str(d), '') for d in starts])
    ev = [HC.to_dates([x])[0] for x in json.loads((HC.MODES / 'strat' / 'ssw_events_era5.json').read_text())['u60_10']]
    ssw = np.array([any(0 <= (d - e).days <= 30 for e in ev) for d in starts])
    # CONUS soil moisture (ERA5 swvl1 00 UTC, harmonic 1991-2020 climatology), land points of the box
    D = BM.daily('swvl1_00')
    cl = BM.fixed_clim('swvl1_00')
    a = D.get(starts) - cl(starts)
    s, n, w, e = REG['conus']
    lsm = np.load(BM.R / 'data' / 'era5_bing_1p5' / 'lsm.npy') >= 0.5
    mk = (LAT[:, None] >= s) & (LAT[:, None] <= n) & (LON[None] >= w) & (LON[None] <= e) & lsm
    wt = np.cos(np.deg2rad(LAT))[:, None] * mk
    sm = np.array([(np.nan_to_num(x) * wt).sum() / (np.isfinite(x) * wt).sum() for x in a])
    seas = np.array([HC.season_of(d) for d in starts])
    z = np.full(len(starts), np.nan)
    for sn in HC.SEASONS:
        m = seas == sn
        z[m] = (sm[m] - sm[m].mean()) / sm[m].std()
    return dict(amp=amp, phase=ph, enso=enso, ssw=ssw, sm_z=z, season=seas, rmm=r)


def pattern_acc(F, O, mk):
    """per start centred cos-lat correlation over mask; F, O [S, 121, 240]."""
    w = np.cos(np.deg2rad(LAT))[:, None] * mk
    out = np.full(F.shape[0], np.nan)
    for i in range(F.shape[0]):
        ok = np.isfinite(F[i]) & np.isfinite(O[i]) & (w > 0)
        if ok.sum() < 10:
            continue
        ww = w[ok]
        f, o = F[i][ok], O[i][ok]
        f = f - (f * ww).sum() / ww.sum()
        o = o - (o * ww).sum() / ww.sum()
        out[i] = (f * o * ww).sum() / np.sqrt((f * f * ww).sum() * (o * o * ww).sum())
    return out


def main():
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    # common starts over systems and fields
    lists = []
    for s in HC.SYSTEMS:
        for f in FIELDS:
            d = json.loads((score_dir(s, f) / 'done_dc.json').read_text())
            lists.append(HC.to_dates(d['starts']))
    starts = HC.common_starts(lists)
    ST = init_states(starts)
    np.savez(OUT / 'c07_init_states.npz', starts=np.array([str(d) for d in starts]), **{k: v for k, v in ST.items()})
    S = len(starts)
    nd = np.array([d.month in (11, 12, 1, 2, 3) for d in starts])
    jja = ST['season'] == 'JJA'
    act = ST['amp'] >= 1
    pg = {p: act & np.isin(ST['phase'], q) for p, q in {'P8-1': [8, 1], 'P2-3': [2, 3], 'P4-5': [4, 5], 'P6-7': [6, 7]}.items()}
    enso_on = np.isin(ST['enso'], ['el_nino', 'la_nina'])
    big_sm = np.abs(ST['sm_z']) >= 1
    allm = np.ones(S, bool)
    contrasts = {
        'MJO active - inactive (all)': (act, ~act),
        'MJO active - inactive (NDJFM)': (act & nd, ~act & nd),
        'MJO P2-3 - inactive (NDJFM)': (pg['P2-3'] & nd, ~act & nd),
        'MJO P4-5 - inactive (NDJFM)': (pg['P4-5'] & nd, ~act & nd),
        'MJO P6-7 - inactive (NDJFM)': (pg['P6-7'] & nd, ~act & nd),
        'MJO P8-1 - inactive (NDJFM)': (pg['P8-1'] & nd, ~act & nd),
        'ENSO active - neutral (all)': (enso_on, ST['enso'] == 'neutral'),
        'ENSO active - neutral (NDJFM)': (enso_on & nd, (ST['enso'] == 'neutral') & nd),
        'El Nino - neutral (all)': (ST['enso'] == 'el_nino', ST['enso'] == 'neutral'),
        'La Nina - neutral (all)': (ST['enso'] == 'la_nina', ST['enso'] == 'neutral'),
        'SSW in previous 30 d - none (NDJFM)': (ST['ssw'] & nd, ~ST['ssw'] & nd),
        'large CONUS soil-moisture anomaly - small (all)': (big_sm, ~big_sm & np.isfinite(ST['sm_z'])),
        'large CONUS soil-moisture anomaly - small (JJA)': (big_sm & jja, ~big_sm & jja),
    }
    hA, hB = HC.halves(starts)
    bidx = HC.mbb_indices(S, 4, 1000, seed=10)
    rows = []
    for s in HC.SYSTEMS:
        for f in FIELDS:
            ds = xr.open_dataset(score_dir(s, f) / 'perstart_dc.nc')
            pst = HC.to_dates(ds.start.values)
            p = {d: i for i, d in enumerate(pst)}
            j = np.array([p[d] for d in starts])
            wj = [str(x) for x in ds.window.values].index('w34')
            regs = [str(x) for x in ds.region.values]
            crps = ds.crps_dc.values[j, wj]
            ccf = ds.cc_fixed_dc.values[j, wj]
            cct = ds.cc_trend_dc.values[j, wj]
            ds.close()
            st_, em, O, C = load_em(s, f, 'w34')
            q = {d: i for i, d in enumerate(st_)}
            jj = np.array([q[d] for d in starts])
            Fa, Oa = em[jj] - C[jj], O[jj] - C[jj]
            for rn, (a_, b_, c_, d_) in REG.items():
                ri = regs.index(rn)
                mk = (LAT[:, None] >= a_) & (LAT[:, None] <= b_) & (LON[None] >= c_) & (LON[None] <= d_)
                acc = pattern_acc(Fa, Oa, mk)
                cr, cf, ct = crps[:, ri], ccf[:, ri], cct[:, ri]

                def metr(ix, cr=cr, cf=cf, ct=ct, acc=acc):
                    if len(ix) < 3:
                        return [np.nan] * 3
                    return [1 - np.nansum(cr[ix]) / np.nansum(cf[ix]), 1 - np.nansum(cr[ix]) / np.nansum(ct[ix]), float(np.nanmean(acc[ix]))]
                for cn, (mA, mB) in contrasts.items():
                    ia, ib = np.flatnonzero(mA), np.flatnonzero(mB)
                    va, vb = np.array(metr(ia)), np.array(metr(ib))
                    dr = []
                    for bb in bidx:
                        dr.append(np.array(metr(bb[mA[bb]])) - np.array(metr(bb[mB[bb]])))
                    dr = np.array(dr)
                    hdA = np.array(metr(np.flatnonzero(mA & hA))) - np.array(metr(np.flatnonzero(mB & hA)))
                    hdB = np.array(metr(np.flatnonzero(mA & hB))) - np.array(metr(np.flatnonzero(mB & hB)))
                    for k, mn in enumerate(('crpss_fixed', 'crpss_trend', 'pattern_acc')):
                        ci = HC.ci_of(dr[:, k])
                        rows.append(dict(system=HC.LABEL[s], field=f, region=rn, contrast=cn, metric=mn, n_a=len(ia), n_b=len(ib),
                                         value_a=float(va[k]), value_b=float(vb[k]), diff=float(va[k] - vb[k]), ci_lo=ci[0], ci_hi=ci[1],
                                         diff_2019_24=float(hdA[k]), diff_2025_26=float(hdB[k]), grade=HC.grade(ci, hdA[k], hdB[k]),
                                         label='exploratory'))
            print(s, f, flush=True)
    HC.write_csv(rows, OUT / 'c07_windows.csv')
    meta = dict(starts=S, n={k: [int(a.sum()), int(b.sum())] for k, (a, b) in contrasts.items()},
                note='exploratory; MJO phases provisional (37.5 deg rotation); soil-moisture z standardised within the 2019-26 starts of each season')
    (OUT / 'c07_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta['n']))


if __name__ == '__main__':
    main()
