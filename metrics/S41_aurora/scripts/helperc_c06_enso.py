#!/usr/bin/env python
"""C06 (exploratory): ENSO and land, from package v1.5 (land/sea-masked UC-S2S sst/swvl1).
(1) Nino-3.4 (5S-5N, 170-120W) window-mean SST anomaly vs the fixed 1991-2020 climatology of the package (em_dc - Cfixed, O - Cfixed):
    RMSE and ACC of the anomaly and of the tendency (window anomaly - ERA5 init-day anomaly; init anomaly from the modes ENSO part,
    harmonic 1991-2020 climatology) by window, block-bootstrap CIs, paired differences; consistency check vs the modes extract values.
(2) ENSO teleconnections: z500 and tp24 weeks 3-4 anomaly composites, El Nino minus La Nina starts (30-day-mean Nino-3.4 before the
    init >= +0.5 / <= -0.5 K, modes definition), NDJFM starts and all starts, forecast (30-member mean, drift-corrected) vs ERA5 same starts;
    pattern correlation and amplitude ratio over 20-90N (z500) and 30S-60N (tp24).
(3) Soil-moisture TCC (Bing's method: weekly ensemble-mean anomaly correlation per grid point, fixed climatology, no drift removal =
    maps acc_fixed_raw; and drift-corrected) land mean by window, vs persistence of the ERA5 init-day swvl1 anomaly.
Output: v1.5/helperC/enso/"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval import bing_helperc as HC  # noqa: E402
from s41eval import bing_modes as BM  # noqa: E402

OUT = HC.OUTC / 'enso'
V = HC.V15
WIN = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w34', 'w56']
LON = np.arange(240) * 1.5


def score_dir(s, f):
    return V / 'score' / s / f'{f}_g15'


def load_em(s, f, w=None):
    d = json.loads((score_dir(s, f) / 'done_dc.json').read_text())
    st = [HC.to_dates([x])[0] for x in d['starts']]
    vd = Path(d['verif'].replace('/eval/v1/', '/eval/v1.5/'))
    meta = json.loads((vd / 'meta.json').read_text())
    pos = {x: i for i, x in enumerate(meta['starts'])}
    ix = np.array([pos[str(x)] for x in st])
    em = np.load(score_dir(s, f) / 'em_dc.npy', mmap_mode='r')
    O = np.load(vd / 'O.npy', mmap_mode='r')
    C = np.load(vd / 'Cfixed.npy', mmap_mode='r')
    if w is None:
        return st, em, O, C, ix
    j = WIN.index(w)
    return st, np.asarray(em[:, j]), np.asarray(O[ix, j]), np.asarray(C[ix, j])


def lat_of_grid():
    # package arrays: check orientation from the verification metadata of bing_eval (g15 = lat ascending in core?)
    return None


def box_mean(x, lat, s=-5, n=5, w=190, e=240):
    mk = (lat[:, None] >= s) & (lat[:, None] <= n) & (LON[None] >= w) & (LON[None] <= e)
    wt = np.cos(np.deg2rad(lat))[:, None] * mk
    ok = np.isfinite(x)
    return (np.where(ok, x, 0) * wt).sum((-2, -1)) / (ok * wt).sum((-2, -1))


def pcor(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(HC.MODES / 'enso' / 'perstart_nino34.csv') as fh:
        mrows = [r for r in csv.DictReader(fh) if r['correction'] == 'dc']
    phase = {}
    a0 = {}
    mod = {}
    for r in mrows:
        phase[r['start']] = r['enso_phase']
        a0.setdefault(r['system_key'], {})[r['start']] = float(r['init_anom'])
        mod.setdefault(r['system_key'], {})[r['start']] = [float(r[f'em30_{w}']) for w in WIN]
    # grid orientation: the package g15 grid (bing_eval.core) -> read lat from a maps file
    import xarray as xr
    lat = np.linspace(-90.0, 90.0, 121)                  # package arrays (em, O, C): lat ascending (bing_eval.core.LAT['g15'])
    # ---------------------------------------------------------------- (1) Nino-3.4
    N = {}
    for s in HC.SYSTEMS:
        st, em, O, C, ix = load_em(s, 'sst')
        F = np.stack([box_mean(np.asarray(em[i]) - np.asarray(C[ix[i]]), lat) for i in range(len(st))])      # [S, 8]
        Ob = np.stack([box_mean(np.asarray(O[ix[i]]) - np.asarray(C[ix[i]]), lat) for i in range(len(st))])
        N[s] = (st, F, Ob)
    starts = HC.common_starts([N[s][0] for s in HC.SYSTEMS])
    starts = [d for d in starts if str(d) in a0['ucs2s_sstanom']]
    A0 = np.array([a0['ucs2s_sstanom'][str(d)] for d in starts])
    A0f = np.array([a0['fuxi'][str(d)] for d in starts])
    rows, prs, chk = [], [], []
    hA, hB = HC.halves(starts)
    bidx = HC.mbb_indices(len(starts), 4, 1000, seed=9)
    stat = {}
    for s in HC.SYSTEMS:
        st, F, Ob = N[s]
        p = {d: i for i, d in enumerate(st)}
        j = np.array([p[d] for d in starts])
        F, Ob = F[j], Ob[j]
        a = A0f if s == 'fuxi' else A0
        mm = np.array([mod[s][str(d)] for d in starts])
        for w, wn in enumerate(WIN):
            chk.append(dict(system=HC.LABEL[s], window=wn, r_v15_vs_modes=pcor(F[:, w], mm[:, w]),
                            mean_diff_v15_minus_modes=float(np.nanmean(F[:, w] - mm[:, w]))))

        def st_(b, F=F, Ob=Ob, a=a):
            out = []
            for w in range(len(WIN)):
                out += [float(np.sqrt(np.nanmean((F[b, w] - Ob[b, w]) ** 2))), pcor(F[b, w], Ob[b, w]), pcor(F[b, w] - a[b], Ob[b, w] - a[b])]
            return out
        full = np.array(st_(np.arange(len(starts))))
        dr = np.array([st_(b) for b in bidx])
        hav = np.array(st_(np.flatnonzero(hA)))
        hbv = np.array(st_(np.flatnonzero(hB)))
        stat[s] = (full, dr, hav, hbv)
        for w, wn in enumerate(WIN):
            for q, qn in enumerate(('rmse', 'acc', 'tendency_acc')):
                lo, hi = HC.ci_of(dr[:, 3 * w + q])
                rows.append(dict(system=HC.LABEL[s], window=wn, metric=qn, n=len(starts), value=float(full[3 * w + q]), ci_lo=lo, ci_hi=hi))
    for a, b in (('ucs2s_sstanom', 'ucs2s_plain'), ('ucs2s_sstanom', 'fuxi'), ('ucs2s_plain', 'fuxi')):
        for w, wn in enumerate(WIN):
            for q, qn in enumerate(('rmse', 'acc', 'tendency_acc')):
                k = 3 * w + q
                ci = HC.ci_of(stat[a][1][:, k] - stat[b][1][:, k])
                ma, mb = stat[a][2][k] - stat[b][2][k], stat[a][3][k] - stat[b][3][k]
                prs.append(dict(pair=f'{HC.LABEL[a]} - {HC.LABEL[b]}', window=wn, metric=qn, n=len(starts),
                                diff=float(stat[a][0][k] - stat[b][0][k]), ci_lo=ci[0], ci_hi=ci[1], diff_2019_24=ma, diff_2025_26=mb,
                                grade=HC.grade(ci, ma, mb), label='exploratory'))
    HC.write_csv(rows, OUT / 'c06_nino34.csv')
    HC.write_csv(prs, OUT / 'c06_nino34_pairs.csv')
    HC.write_csv(chk, OUT / 'c06_nino34_check_vs_modes.csv')
    print('nino done', flush=True)
    # ---------------------------------------------------------------- (2) teleconnection composites, w34
    comp, crow = {}, []
    for f in ('z500', 'tp24'):
        for s in HC.SYSTEMS:
            st, em, O, C = load_em(s, f, 'w34')
            ph = np.array([phase.get(str(d), '') for d in st])
            nd = np.array([d.month in (11, 12, 1, 2, 3) for d in st])
            Fa, Oa = em - C, O - C
            for g, gm in (('NDJFM', nd), ('all', np.ones(len(st), bool))):
                en, ln = gm & (ph == 'el_nino'), gm & (ph == 'la_nina')
                fd = np.nanmean(Fa[en], 0) - np.nanmean(Fa[ln], 0)
                od = np.nanmean(Oa[en], 0) - np.nanmean(Oa[ln], 0)
                comp[f'{f}_{g}_{s}'] = fd.astype(np.float32)
                comp[f'{f}_{g}_era5_{BM.SAMPLING[s]}'] = od.astype(np.float32)
                reg = (lat >= 20) if f == 'z500' else ((lat >= -30) & (lat <= 60))
                wt = np.cos(np.deg2rad(lat))[reg][:, None] * np.ones((1, 240))
                x, y = fd[reg], od[reg]
                ok = np.isfinite(x) & np.isfinite(y)
                xm, ym = (x[ok] * wt[ok]).sum() / wt[ok].sum(), (y[ok] * wt[ok]).sum() / wt[ok].sum()
                r = ((x[ok] - xm) * (y[ok] - ym) * wt[ok]).sum() / np.sqrt(((x[ok] - xm) ** 2 * wt[ok]).sum() * ((y[ok] - ym) ** 2 * wt[ok]).sum())
                amp = np.sqrt((x[ok] ** 2 * wt[ok]).sum() / (y[ok] ** 2 * wt[ok]).sum())
                crow.append(dict(field=f, group=g, system=HC.LABEL[s], n_el_nino=int(en.sum()), n_la_nina=int(ln.sum()),
                                 region='20-90N' if f == 'z500' else '30S-60N', pattern_corr=float(r), amp_ratio=float(amp)))
    np.savez(OUT / 'c06_enso_composites.npz', lat=lat, lon=LON, **comp)
    HC.write_csv(crow, OUT / 'c06_enso_composites.csv')
    print('composites done', flush=True)
    # ---------------------------------------------------------------- (3) soil-moisture TCC
    lsm = np.load(BM.R / 'data' / 'era5_bing_1p5' / 'lsm.npy') if (BM.R / 'data' / 'era5_bing_1p5' / 'lsm.npy').exists() else None
    trow = []
    for s in ('ucs2s_sstanom', 'ucs2s_plain'):
        mp = xr.open_dataset(score_dir(s, 'swvl1') / 'maps_raw.nc')
        md = xr.open_dataset(score_dir(s, 'swvl1') / 'maps_dc.nc')
        latm = mp.lat.values.astype(float)
        for corr, ds, key in (('raw (Bing)', mp, 'acc_fixed_raw'), ('dc', md, 'acc_fixed_dc')):
            for sub in ('pooled|all', 'pooled|JJA', 'pooled|DJF'):
                for wn in WIN:
                    m = ds[key].sel(subset=sub, window=wn).values
                    ok = np.isfinite(m) & (np.abs(latm)[:, None] <= 75)
                    landmask = np.isfinite(m)[::-1]                    # lat ascending, model-defined land points
                    wt = np.cos(np.deg2rad(latm))[:, None] * np.ones((1, 240))
                    trow.append(dict(system=HC.LABEL[s], correction=corr, subset=sub, window=wn, land_mean_tcc=float((m[ok] * wt[ok]).sum() / wt[ok].sum()),
                                     n_points=int(ok.sum())))
        mp.close()
        md.close()
    # persistence of the ERA5 init-day anomaly (swvl1 00 UTC, harmonic 1991-2020 climatology of the modes caches)
    st, em, O, C, ix = load_em('ucs2s_sstanom', 'swvl1')
    D = BM.daily('swvl1_00')
    cl = BM.fixed_clim('swvl1_00')
    a0m = D.get(st) - cl(st)                                    # [S, 121, 240] lat ascending (modes grid)
    for wn in WIN:
        j = WIN.index(wn)
        Oa = np.asarray(O[ix, j]) - np.asarray(C[ix, j])
        for sub, msk in (('pooled|all', np.ones(len(st), bool)), ('pooled|JJA', np.array([d.month in (6, 7, 8) for d in st])),
                         ('pooled|DJF', np.array([d.month in (12, 1, 2) for d in st]))):
            x, y = a0m[msk], Oa[msk]
            xm, ym = np.nanmean(x, 0), np.nanmean(y, 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                r = np.nansum((x - xm) * (y - ym), 0) / np.sqrt(np.nansum((x - xm) ** 2, 0) * np.nansum((y - ym) ** 2, 0))
            ok = np.isfinite(r) & (np.abs(lat)[:, None] <= 75) & np.isfinite(y).all(0) & landmask
            wt = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 240))
            trow.append(dict(system='ERA5 persistence (init-day anomaly)', correction='-', subset=sub, window=wn,
                             land_mean_tcc=float((r[ok] * wt[ok]).sum() / wt[ok].sum()), n_points=int(ok.sum())))
    HC.write_csv(trow, OUT / 'c06_soil_tcc.csv')
    print('soil done')


if __name__ == '__main__':
    main()
