"""Scores of the modes evaluation (called by ``scripts/bing_modes.py score``). Parts: mjo, tele, enso, sst, strat, qbo, comp, land.
Outputs under runs/bing/eval/modes/<part>/ (CSV of scores, NetCDF/CSV of per-start values for the paired bootstrap)."""
from __future__ import annotations

import csv
import json
import pickle
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from s41eval import bing_modes as BM
from s41eval import bing_scores as BS

OUT = BM.OUT


def log(*a):
    print(f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}', *a, flush=True)


SUFFIX = ['']          # '__<system>' when a part runs for a subset of the systems (merged by bing_modes_summary.merge_parts)


def write_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path = path.with_name(path.stem + SUFFIX[0] + path.suffix)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def pdir(part):
    return BS.mkdir_striped(OUT / part)


def starts_of(system):
    st = BM.list_starts()
    return [d for d in st if BM.extract_path(system, d).exists()]


def save_nc(path, data_vars, coords, attrs):
    import xarray as xr
    path = Path(path)
    if SUFFIX[0] and not any(s in path.name for s in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi')):
        path = path.with_name(path.stem + SUFFIX[0] + path.suffix)
    ds = xr.Dataset({k: (dims, np.asarray(v)) for k, (dims, v) in data_vars.items()}, coords=coords, attrs=attrs)
    tmp = path.with_suffix('.partial.nc')
    ds.to_netcdf(tmp)
    tmp.rename(path)


# ===================================================================== MJO
def part_mjo(systems, commit):
    od = pdir('mjo')
    E = OUT / 'era5'
    IX = {nm: pickle.load(open(E / f'mjo_{nm}.pkl', 'rb')) for nm in ('full', 'winds')}
    rows = []
    for sysn in systems:
        st = starts_of(sysn)
        samp = BM.SAMPLING[sysn]
        X, have = BM.load_extract(sysn, st, ['band_olr', 'band_u850', 'band_u200', 'u200_src'])
        S = len(st)
        for nm, ix in IX.items():
            F = np.full((S, 30, BM.NLEAD, 2), np.nan)
            O = np.full((S, BM.NLEAD, 2), np.nan)
            init = np.full((S, 2), np.nan)
            ms = BM.state_samp(samp)
            for i, d in enumerate(st):
                band = {v: X[f'band_{v}'][i][:30] for v in ix.fields}
                M = band[ix.fields[0]].shape[0]
                F[i, :M] = ix.forecast(ms, d, band)
                O[i] = ix.obs(ms, BM.valid_dates(d))
            if sysn == 'ifs':
                F = F[:, :11]                                    # IFS: ensemble-mean scores from members 0-10 (plan section 2)
                init[i] = ix.obs('day', [d])[0]
            r, per = BM.mjo_scores(F, O, st, init, BM.SYSTEM_LABEL[sysn])
            for x in r:
                x.update(index=nm, system_key=sysn)
            rows += r
            dv = {'rmm_obs': (('start', 'lead', 'rmm'), O), 'rmm_init_obs': (('start', 'rmm'), init)}
            for k, v in per.items():
                dv[f'rmm_{k}'] = ((('draw',) if v.ndim == 4 else ()) + ('start', 'lead', 'rmm'), v)
            dv['rmm_members_raw'] = (('start', 'member', 'lead', 'rmm'), F.astype(np.float32))
            save_nc(od / f'perstart_mjo_{nm}_{sysn}.nc', dv,
                    coords=dict(start=np.array(st, 'datetime64[D]'), lead=np.arange(BM.NLEAD), rmm=[1, 2]),
                    attrs=dict(system=BM.SYSTEM_LABEL[sysn], index=nm, sampling=samp, commit=commit,
                               u200=str(X['u200_src'][have[0]]) if 'u200_src' in X else 'n/a',
                               note='em30/em11 = ensemble mean of 30 members / of 10 random 11-member draws (rng 11); dc = LOYO +-30 d '
                                    'lead-dependent mean RMM error removed; lead 0 = ERA5 (init day). COR for a resampled set of '
                                    'starts: bing_modes.mjo_cor_from_components(f, o).'))
            log('mjo', sysn, nm, 'starts', S)
    write_csv(rows, od / 'scores_mjo.csv')
    return rows


# ===================================================================== teleconnections
def part_tele(systems, commit):
    od = pdir('tele')
    T = pickle.load(open(OUT / 'era5' / 'tele_patterns.pkl', 'rb'))
    TP = T['TP']
    clims = {'day': BM.fixed_clim('z500_mean4', base=BM.EOF_YEARS, nh=4), '00z': BM.fixed_clim('z500_00', base=BM.EOF_YEARS, nh=4)}
    dser = {}
    for samp, nm in (('day', 'z500_mean4'), ('00z', 'z500_00')):
        ser, days = BM.era5_daily_tele(TP, nm, clims[samp], first=date(1991, 1, 1))
        dser[samp] = (ser, {d: i for i, d in enumerate(days)})
    log('tele ERA5 daily series done')
    rows = []
    W = BM.WK
    for sysn in systems:
        st = starts_of(sysn)
        samp = BM.SAMPLING[sysn]
        S = len(st)
        F = {k: np.full((S, 30, len(W)), np.nan) for k in ('nao', 'pna', 'ao', 'sam')}
        O = {k: np.full((S, len(W)), np.nan) for k in F}
        Cl = {k: np.full((S, 30, len(W)), np.nan) for k in F}
        ser, dix = dser[BM.state_samp(samp)]
        for i, d in enumerate(st):
            z = np.load(BM.extract_path(sysn, d))
            Z = z['w_z500']                                                       # [M, W, 121, 240]
            cw = np.stack([clims[BM.state_samp(samp)]([d + timedelta(days=k) for k in range(a, b + 1)]).mean(0)
                           for a, b in BM.WINDOWS.values()])
            r = BM.tele_window_index(TP, Z - cw[None], d)
            for k in F:
                F[k][i, :Z.shape[0]] = r[k]
            for j, w in enumerate(W):
                m = BM.window_mid(d, w).month
                for k in F:
                    O[k][i, j] = BM.window_from_daily(ser[k], dix, d, w, m) / TP.norm[(k, m)]
                    for iy, y in enumerate(range(1991, 2021)):
                        dy = date(y, d.month, min(d.day, 28) if d.month == 2 else d.day)
                        Cl[k][i, iy, j] = BM.window_from_daily(ser[k], dix, dy, w, m) / TP.norm[(k, m)]
        log('tele', sysn, 'indices done')
        per = {}
        for k in F:
            for corr in ('raw', 'dc'):
                Fk = F[k]
                if corr == 'dc':
                    b = BM.loyo_bias(Fk.mean(1) - O[k], st)
                    Fk = Fk - b[:, None]
                per[f'{k}_{corr}'] = Fk
                em30 = Fk.mean(1)
                em11 = [Fk[:, ix].mean(1) for ix in BM.draws11(30)]
                crps_f = np.stack([BM.fair_crps_1d(Fk[:, :, j], O[k][:, j]) for j in range(len(W))], 1)    # [S, W]
                crps_c = np.stack([BM.fair_crps_1d(Cl[k][:, :, j], O[k][:, j]) for j in range(len(W))], 1)
                per[f'{k}_{corr}_crps'] = crps_f
                per[f'{k}_clim_crps'] = crps_c
                groups = {'all': np.ones(S, bool)}
                groups.update({sn: np.array([BM.season_of(d) == sn for d in st]) for sn in BM.SEASONS})
                groups['NDJFM'] = np.array([d.month in (11, 12, 1, 2, 3) for d in st])
                for pn in BM.PERIODS:
                    pm = BM.period_mask(st, pn)
                    for g, gm in groups.items():
                        sel = pm & gm
                        if sel.sum() < 8:
                            continue
                        for j, w in enumerate(W):
                            ok = sel & np.isfinite(O[k][:, j]) & np.isfinite(em30[:, j])
                            rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, index=k, correction=corr, period=pn,
                                             group=g, window=w, n=int(ok.sum()),
                                             cor30=BM.pearson(em30[ok, j], O[k][ok, j]),
                                             cor11=float(np.nanmean([BM.pearson(e[ok, j], O[k][ok, j]) for e in em11])),
                                             rmse30=float(np.sqrt(np.nanmean((em30[ok, j] - O[k][ok, j]) ** 2))),
                                             spread30=float(np.sqrt(np.nanmean(Fk[ok, :, j].var(1, ddof=1)))),
                                             crps=float(np.nanmean(crps_f[ok, j])), crps_clim=float(np.nanmean(crps_c[ok, j])),
                                             crpss=float(1 - np.nanmean(crps_f[ok, j]) / np.nanmean(crps_c[ok, j]))))
        dv = {'obs': (('index', 'start', 'window'), np.stack([O[k] for k in F]))}
        for corr in ('raw', 'dc'):
            dv[f'members_{corr}'] = (('index', 'start', 'member', 'window'), np.stack([per[f'{k}_{corr}'] for k in F]).astype(np.float32))
            dv[f'crps_{corr}'] = (('index', 'start', 'window'), np.stack([per[f'{k}_{corr}_crps'] for k in F]))
        dv['crps_clim'] = (('index', 'start', 'window'), np.stack([per[f'{k}_clim_crps'] for k in F]))
        dv['clim_members'] = (('index', 'start', 'year', 'window'), np.stack([Cl[k] for k in F]).astype(np.float32))
        save_nc(od / f'perstart_tele_{sysn}.nc', dv,
                coords=dict(index=list(F), start=np.array(st, 'datetime64[D]'), window=W, member=np.arange(30),
                            year=np.arange(1991, 2021)),
                attrs=dict(system=BM.SYSTEM_LABEL[sysn], sampling=samp, commit=commit,
                           note='indices standardised by the ERA5 7-day-mean index std (1979-2018) of the window-middle month; '
                                'clim_members = ERA5 index of the same calendar window in 1991-2020 (fixed reference for CRPSS)'))
    write_csv(rows, od / 'scores_tele.csv')
    return rows


# ===================================================================== ENSO (Nino-3.4)
def nino_box(x):
    s, n, w, e = BM.NINO34
    mk = (BM.LAT15[:, None] >= s) & (BM.LAT15[:, None] <= n) & (BM.LON15[None] >= w) & (BM.LON15[None] <= e)
    wt = BM.cosw_lat()[:, None] * mk
    ok = np.isfinite(x)
    return (np.where(ok, x, 0) * wt).sum((-2, -1)) / (ok * wt).sum((-2, -1))


def era5_nino(samp):
    """ERA5 Nino-3.4 daily series (box mean) and its fixed 1991-2020 climatology (box mean of the harmonic clim) by date."""
    name, sh = BM.obs_name('sst', samp)
    D = BM.daily(name)
    days = [d - timedelta(days=sh) for d in D.dates()]                       # label dates (IFS: daily mean of D-1 labelled D)
    s, n, w, e = BM.NINO34
    i0, i1 = np.searchsorted(BM.LAT15, s), np.searchsorted(BM.LAT15, n, 'right')
    j0, j1 = np.searchsorted(BM.LON15, w), np.searchsorted(BM.LON15, e, 'right')
    sub = np.asarray(D.A[:, i0:i1, j0:j1], np.float64)
    full = np.full((len(days), 121, 240), np.nan)
    x = np.full(len(days), np.nan)
    wt = BM.cosw_lat()[i0:i1][:, None] * np.ones((1, j1 - j0))
    ok = np.isfinite(sub)
    x = (np.where(ok, sub, 0) * wt).sum((1, 2)) / (ok * wt).sum((1, 2))
    c = BM.fixed_clim(name)
    clim_box = lambda dd: nino_box(c([d + timedelta(days=sh) for d in dd]))  # noqa: E731
    return x, days, clim_box


def damped_fit(anom, days, years=(1991, 2018), half=30, step=2):
    """ERA5 damped-persistence coefficients by init day of year: r[doy, W] = sum a0 aw / sum a0^2 over inits within +-half days."""
    ix = {d: i for i, d in enumerate(days)}
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(anom))])
    cn = np.concatenate([[0], np.cumsum(np.isfinite(anom))])
    inits = [i for i, d in enumerate(days) if years[0] <= d.year <= years[1] and i % step == 0]
    r = np.full((365, len(BM.WINDOWS)), np.nan)
    a0 = anom[inits]
    dd = np.array([BM.doy365(days[i]) for i in inits])
    aw = np.full((len(inits), len(BM.WINDOWS)), np.nan)
    for j, (a, b) in enumerate(BM.WINDOWS.values()):
        for q, i in enumerate(inits):
            if i + b + 1 < len(cs) and cn[i + b + 1] - cn[i + a] == b - a + 1:
                aw[q, j] = (cs[i + b + 1] - cs[i + a]) / (b - a + 1)
    for doy in range(365):
        dist = np.minimum(np.abs(dd - doy), 365 - np.abs(dd - doy))
        m = (dist <= half) & np.isfinite(a0)
        for j in range(len(BM.WINDOWS)):
            mm = m & np.isfinite(aw[:, j])
            r[doy, j] = (a0[mm] * aw[mm, j]).sum() / (a0[mm] ** 2).sum()
    return r


def part_enso(systems, commit):
    od = pdir('enso')
    rows, out_rows = [], []
    W = BM.WK
    for sysn in systems:
        st = starts_of(sysn)
        samp = BM.SAMPLING[sysn]
        x, days, clim_box = era5_nino(samp)
        dix = {d: i for i, d in enumerate(days)}
        cl = clim_box(days)
        anom = x - cl
        r = damped_fit(anom, days)
        S = len(st)
        X, have = BM.load_extract(sysn, st, ['nino34'])
        Fd = X['nino34']                                                         # [S, M, 46]
        F = np.full((S, 30, len(W)), np.nan)
        O = np.full((S, len(W)), np.nan)
        P = np.full((S, len(W)), np.nan)
        Pp = np.full((S, len(W)), np.nan)
        a0s = np.full(S, np.nan)
        phase = np.full(S, '', object)
        for i, d in enumerate(st):
            vd = BM.valid_dates(d)
            cv = clim_box(vd)
            fa = Fd[i] - cv[None]                                                # [M, 46]
            F[i, :fa.shape[0]] = BM.window_mean_daily(fa, lead_axis=1)
            oi = [dix.get(v) for v in vd]
            oa = np.array([anom[j] if j is not None else np.nan for j in oi])
            O[i] = BM.window_mean_daily(oa)
            a0 = anom[dix[d]] if d in dix else np.nan
            a0s[i] = a0
            P[i] = r[BM.doy365(d)] * a0
            Pp[i] = a0
            prev30 = [anom[dix[d - timedelta(days=k)]] for k in range(30) if (d - timedelta(days=k)) in dix]
            m30 = np.nanmean(prev30) if prev30 else np.nan
            phase[i] = 'el_nino' if m30 >= 0.5 else ('la_nina' if m30 <= -0.5 else 'neutral')
        for corr in ('raw', 'dc'):
            Fk = F if corr == 'raw' else F - BM.loyo_bias(F.mean(1) - O, st)[:, None]
            em30 = Fk.mean(1)
            em11 = [Fk[:, ix].mean(1) for ix in BM.draws11(30)]
            groups = {'all': np.ones(S, bool), 'el_nino': phase == 'el_nino', 'la_nina': phase == 'la_nina', 'neutral': phase == 'neutral'}
            groups.update({sn: np.array([BM.season_of(d) == sn for d in st]) for sn in BM.SEASONS})
            for pn in BM.PERIODS:
                pm = BM.period_mask(st, pn)
                for g, gm in groups.items():
                    sel = pm & gm
                    if sel.sum() < 8:
                        continue
                    for j, w in enumerate(W):
                        ok = sel & np.isfinite(O[:, j]) & np.isfinite(em30[:, j])
                        rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, correction=corr, period=pn, group=g, window=w,
                                         n=int(ok.sum()), acc30=BM.pearson(em30[ok, j], O[ok, j]),
                                         acc11=float(np.nanmean([BM.pearson(e[ok, j], O[ok, j]) for e in em11])),
                                         rmse30=float(np.sqrt(np.nanmean((em30[ok, j] - O[ok, j]) ** 2))),
                                         acc_damped=BM.pearson(P[ok, j], O[ok, j]),
                                         rmse_damped=float(np.sqrt(np.nanmean((P[ok, j] - O[ok, j]) ** 2))),
                                         acc_persist=BM.pearson(Pp[ok, j], O[ok, j]),
                                         rmse_persist=float(np.sqrt(np.nanmean((Pp[ok, j] - O[ok, j]) ** 2))),
                                         spread30=float(np.sqrt(np.nanmean(Fk[ok, :, j].var(1, ddof=1)))),
                                         acc_tendency30=BM.pearson(em30[ok, j] - a0s[ok], O[ok, j] - a0s[ok]),
                                         acc_tendency_damped=BM.pearson(P[ok, j] - a0s[ok], O[ok, j] - a0s[ok])))
            for i, d in enumerate(st):
                out_rows.append(dict(system_key=sysn, correction=corr, start=d.isoformat(), enso_phase=phase[i], init_anom=a0s[i],
                                     **{f'em30_{w}': float(em30[i, j]) for j, w in enumerate(W)},
                                     **{f'obs_{w}': float(O[i, j]) for j, w in enumerate(W)},
                                     **{f'damped_{w}': float(P[i, j]) for j, w in enumerate(W)}))
        log('enso', sysn, S)
    write_csv(rows, od / 'scores_nino34.csv')
    write_csv(out_rows, od / 'perstart_nino34.csv')
    return rows


PARTS = {'mjo': part_mjo, 'tele': part_tele, 'enso': part_enso}


def main(a):
    commit = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    parts = a.parts or list(PARTS)
    if sorted(a.systems) != sorted(['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']):
        SUFFIX[0] = '__' + '_'.join(a.systems)
    for p in parts:
        t0 = time.time()
        PARTS[p](a.systems, commit)
        log('part', p, f'{time.time() - t0:.0f}s')


# ===================================================================== stratosphere + QBO
G0 = 9.80665


def era5_strat():
    """ERA5 daily 00 UTC series by date: u60_10, u60_50, ueq_10, ueq_50 (m s-1), cap_<L> (m) -> dict name -> {date: value}."""
    S = {}
    i60 = int(np.argmin(np.abs(BM.LAT15 - 60)))
    eq = np.abs(BM.LAT15) <= 5 + 1e-6
    weq = BM.cosw_lat()[eq]
    for f in sorted((BM.MD / 'strat').glob('[0-9][0-9][0-9][0-9].npz')):
        z = np.load(f)
        days = [date.fromisoformat(str(x)) for x in z['days']]
        ul = [int(x) for x in z['u_levels']]
        zl = [int(x) for x in z['z_levels']]
        for j, L in enumerate(ul):
            for nm, v in ((f'u60_{L}', z['zm_u'][:, j, i60]), (f'ueq_{L}', (z['zm_u'][:, j][:, eq] * weq).sum(1) / weq.sum())):
                S.setdefault(nm, {}).update({d: float(x) for d, x in zip(days, v) if np.isfinite(x)})
        for j, L in enumerate(zl):
            S.setdefault(f'cap_{L}', {}).update({d: float(x) / G0 for d, x in zip(days, z['cap_z'][:, j]) if np.isfinite(x)})
    return S


def series_get(ser, days, samp):
    """values for label dates in sampling samp ('day' = (00 UTC D + 00 UTC D+1) / 2)."""
    def one(d):
        if samp == '00z':
            return ser.get(d, np.nan)
        return 0.5 * (ser.get(d, np.nan) + ser.get(d + timedelta(days=1), np.nan))
    return np.array([one(d) for d in days])


class Clim1D:
    def __init__(self, ser, base=BM.CLIM_BASE, nh=4):
        dd = [d for d in ser if base[0] <= d.year <= base[1]]
        H = BM.harm([BM.doy365(d) for d in dd], nh)
        self.c = np.linalg.lstsq(H, np.array([ser[d] for d in dd]), rcond=None)[0]
        self.nh = nh
        self.n = len(dd)

    def __call__(self, days):
        return BM.harm([BM.doy365(d) for d in days], self.nh) @ self.c


def ssw_events(ser, first=date(1979, 11, 1), last=date(2026, 4, 30)):
    """Charlton & Polvani (2007) style: onset = first day of easterly U60 in Nov-Mar; 20 consecutive westerly days must separate
    events; final warmings (no return to westerlies for 10 consecutive days before 30 April) are excluded."""
    ev = []
    d = first
    last_end = None
    while d <= last:
        u = ser.get(d)
        if u is not None and u < 0 and d.month in (11, 12, 1, 2, 3):
            if last_end is None or (d - last_end).days >= 20:
                e = d + timedelta(days=1)
                run, ok_final = 0, False
                end_apr = date(d.year if d.month <= 4 else d.year + 1, 4, 30)
                while e <= end_apr:
                    v = ser.get(e)
                    if v is None:
                        break
                    run = run + 1 if v > 0 else 0
                    if run >= 10:
                        ok_final = True
                        break
                    e += timedelta(days=1)
                if ok_final:
                    ev.append(d)
            # skip to the end of this easterly spell
            while ser.get(d) is not None and ser.get(d) < 0:
                d += timedelta(days=1)
            last_end = d
        d += timedelta(days=1)
    return ev


def part_strat(systems, commit):
    od = pdir('strat')
    S5 = era5_strat()
    if not S5:
        log('strat: no ERA5 strat staging yet')
        return []
    clim = {k: Clim1D(v) for k, v in S5.items()}
    ev10 = ssw_events(S5['u60_10'])
    ev50 = ssw_events(S5['u60_50'])
    (od / 'ssw_events_era5.json').write_text(json.dumps(dict(u60_10=[d.isoformat() for d in ev10], u60_50=[d.isoformat() for d in ev50],
                                                             rule='Charlton & Polvani 2007 (Nov-Mar reversal of zonal-mean U at 60N, 20-day '
                                                                  'separation, final warmings excluded), ERA5 00 UTC'), indent=1))
    rows, hits, briers, qbo_rows, comp = [], [], [], [], {}
    leads = np.arange(BM.NLEAD)
    for sysn in systems:
        st = starts_of(sysn)
        samp = BM.SAMPLING[sysn]
        X, have = BM.load_extract(sysn, st, ['zmu60', 'zmueq', 'cap_z', 'zm_levels'])
        zl = [int(round(v)) for v in X['zm_levels'][have[0]]]
        Sn = len(st)
        for L in (10, 50):
            if L not in zl:
                continue
            j = zl.index(L)
            F = X['zmu60'][:, :, :, j]                                                    # [S, M, 46]
            O = np.stack([series_get(S5[f'u60_{L}'], BM.valid_dates(d), BM.state_samp(samp)) for d in st])
            C = np.stack([clim[f'u60_{L}'](BM.valid_dates(d)) for d in st])
            Fe = np.nanmean(F, 1)
            dcb = BM.loyo_bias(Fe - O, st)
            for corr, Fm in (('raw', Fe), ('dc', Fe - dcb)):
                for g, gm in (('NDJFM', np.array([d.month in (11, 12, 1, 2, 3) for d in st])), ('all', np.ones(Sn, bool)),
                              ('DJF', np.array([d.month in (12, 1, 2) for d in st]))):
                    for pn in ('2019-26',):
                        sel = gm & BM.period_mask(st, pn)
                        for k in range(1, BM.NLEAD):
                            ok = sel & np.isfinite(O[:, k]) & np.isfinite(Fm[:, k])
                            rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, index=f'u60n{L}', correction=corr, group=g,
                                             period=pn, lead=int(k), n=int(ok.sum()),
                                             acc=BM.pearson(Fm[ok, k] - C[ok, k], O[ok, k] - C[ok, k]),
                                             rmse=float(np.sqrt(np.nanmean((Fm[ok, k] - O[ok, k]) ** 2))),
                                             bias=float(np.nanmean(Fm[ok, k] - O[ok, k]))))
            # SSW hit rate by lead bin
            evs = ev10 if L == 10 else ev50
            for e in evs:
                for i, d in enumerate(st):
                    lead = (e - d).days
                    if not 1 <= lead <= 42:
                        continue
                    ks = [k for k in range(lead - 3, lead + 4) if 1 <= k < BM.NLEAD]
                    p = float(np.mean(np.nanmin(F[i][:, ks], 1) < 0))
                    hits.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, level=L, event=e.isoformat(), start=d.isoformat(),
                                     lead=lead, week=(lead - 1) // 7 + 1, p_reversal=p, hit=int(p >= 0.5)))
            # Brier score of daily P(u < 0), Nov-Mar valid days; reference = ERA5 1991-2020 frequency for the calendar day (+-15 d)
            freq = {}
            ser = S5[f'u60_{L}']
            base = [d for d in ser if 1991 <= d.year <= 2020]
            bd = np.array([BM.doy365(d) for d in base])
            bv = np.array([ser[d] < 0 for d in base])
            for doy in range(365):
                dist = np.minimum(np.abs(bd - doy), 365 - np.abs(bd - doy))
                freq[doy] = float(bv[dist <= 15].mean())
            for wk in range(1, 7):
                bs, br, n = 0.0, 0.0, 0
                for i, d in enumerate(st):
                    for k in range(7 * (wk - 1) + 1, 7 * wk + 1):
                        v = d + timedelta(days=k)
                        if v.month not in (11, 12, 1, 2, 3) or not np.isfinite(O[i, k]) or not np.isfinite(F[i, :, k]).any():
                            continue
                        p = float(np.mean(F[i, :, k] < 0))
                        o = float(O[i, k] < 0)
                        bs += (p - o) ** 2
                        br += (freq[BM.doy365(v)] - o) ** 2
                        n += 1
                briers.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, level=L, week=wk, n=n, bs=bs / max(n, 1),
                                   bs_clim=br / max(n, 1), bss=1 - bs / br if br > 0 else np.nan))
        # polar-cap height coupling composites after ERA5 10-hPa SSWs: starts with init in [e-14, e+2]
        cz = X['cap_z'] / G0                                                              # [S, M, 46, 5]
        rel = np.arange(-15, 46)
        accf = np.zeros((len(rel), len(BM.CAP_LEVELS)))
        acco = np.zeros_like(accf)
        cnt = np.zeros_like(accf)
        for e in ev10:
            for i, d in enumerate(st):
                if not -14 <= (d - e).days <= 2:
                    continue
                vd = BM.valid_dates(d)
                for jl, L in enumerate(BM.CAP_LEVELS):
                    fa = np.nanmean(cz[i, :, :, jl], 0) - clim[f'cap_{L}'](vd)
                    oa = series_get(S5[f'cap_{L}'], vd, BM.state_samp(samp)) - clim[f'cap_{L}'](vd)
                    for k in range(1, BM.NLEAD):
                        r = (vd[k] - e).days
                        q = r - rel[0]
                        if 0 <= q < len(rel) and np.isfinite(fa[k]) and np.isfinite(oa[k]):
                            accf[q, jl] += fa[k]
                            acco[q, jl] += oa[k]
                            cnt[q, jl] += 1
        with np.errstate(invalid='ignore'):
            comp[sysn] = dict(fc=accf / cnt, obs=acco / cnt, n=cnt)
        # QBO drift (equatorial zonal-mean u, 5S-5N)
        for L in (10, 50):
            if L not in zl or f'ueq_{L}' not in S5:
                continue
            j = zl.index(L)
            Fe = np.nanmean(X['zmueq'][:, :, :, j], 1)
            O = np.stack([series_get(S5[f'ueq_{L}'], BM.valid_dates(d), BM.state_samp(samp)) for d in st])
            init_ph = np.array([S5[f'ueq_50'].get(d, np.nan) for d in st])
            for g, gm in (('all', np.ones(Sn, bool)), ('qbo50_westerly', init_ph > 0), ('qbo50_easterly', init_ph < 0)):
                for k in range(1, BM.NLEAD):
                    ok = gm & np.isfinite(O[:, k]) & np.isfinite(Fe[:, k])
                    qbo_rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, level=L, group=g, lead=int(k), n=int(ok.sum()),
                                         drift=float(np.nanmean(Fe[ok, k] - O[ok, k])),
                                         rmse=float(np.sqrt(np.nanmean((Fe[ok, k] - O[ok, k]) ** 2)))))
        log('strat', sysn, Sn)
    write_csv(rows, od / 'scores_u60.csv')
    write_csv(hits, od / 'ssw_hits.csv')
    write_csv(briers, od / 'ssw_brier.csv')
    write_csv(qbo_rows, BS.mkdir_striped(OUT / 'qbo') / 'qbo_drift.csv')
    save_nc(od / 'ssw_capz_composites.nc',
            {f'{k}_{s}': (('rel_day', 'level'), v[k]) for s, v in comp.items() for k in ('fc', 'obs', 'n')},
            coords=dict(rel_day=rel, level=BM.CAP_LEVELS),
            attrs=dict(commit=commit, events=','.join(d.isoformat() for d in ev10),
                       note='polar-cap (60-90N) geopotential height anomaly (m) vs ERA5 1991-2020 harmonic climatology, ensemble mean '
                            '(raw), starts initialised 14 days before to 2 days after each ERA5 10-hPa SSW onset, by day relative to onset'))
    return rows


# ===================================================================== field helpers (z500, sst, land)
def obs_windows_field(var, samp, starts, clim=None, points=None):
    """ERA5 window means [S, W, 121, 240] (or [S, W, npoints] with ``points`` mask) and the fixed-climatology window means."""
    name, sh = BM.obs_name(var, samp)
    D = BM.daily(name)
    c = clim or BM.fixed_clim(name)
    shp = (len(starts), len(BM.WK)) + ((int(points.sum()),) if points is not None else (121, 240))
    O = np.full(shp, np.nan, np.float32)
    C = np.full(shp, np.nan, np.float32)
    for i, d in enumerate(starts):
        vd = [v + timedelta(days=sh) for v in BM.valid_dates(d)]
        x = D.get(vd)
        cc = c(vd)
        if var == 'tp24':
            x, cc = x * 1000.0, cc * 1000.0
        ow, cw = BM.window_mean_daily(x), BM.window_mean_daily(cc)
        O[i] = ow[..., points] if points is not None else ow
        C[i] = cw[..., points] if points is not None else cw
    return O, C


def acc_map(fa, oa):
    """[S, ...] -> per-point Pearson across starts (NaN-aware)."""
    ok = np.isfinite(fa) & np.isfinite(oa)
    n = ok.sum(0)
    f = np.where(ok, fa, 0.0)
    o = np.where(ok, oa, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        fm, om = f.sum(0) / n, o.sum(0) / n
        fc, oc = np.where(ok, fa - fm, 0), np.where(ok, oa - om, 0)
        r = (fc * oc).sum(0) / np.sqrt((fc ** 2).sum(0) * (oc ** 2).sum(0))
    return np.where(n >= 8, r, np.nan)


def area_mean(x, mask):
    w = BM.cosw_lat()[:, None] * np.ones((1, 240)) * mask
    ok = np.isfinite(x)
    return float((np.where(ok, x, 0) * w).sum() / (ok * w).sum()) if (ok * w).sum() > 0 else np.nan


def region_mask(s, n, w=0, e=360):
    la, lo = BM.LAT15[:, None], BM.LON15[None]
    return ((la >= s) & (la <= n) & (((lo >= w) & (lo <= e)) if w <= e else ((lo >= w) | (lo <= e)))).astype(float)


def pattern_corr(a, b, mask):
    w = BM.cosw_lat()[:, None] * np.ones((1, 240)) * mask
    ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
    a0 = a[ok] - (a[ok] * w[ok]).sum() / w[ok].sum()
    b0 = b[ok] - (b[ok] * w[ok]).sum() / w[ok].sum()
    return float((w[ok] * a0 * b0).sum() / np.sqrt((w[ok] * a0 ** 2).sum() * (w[ok] * b0 ** 2).sum()))


# ===================================================================== MJO -> extratropics composites
def era5_lagged_composite(groups_days, name, clim, windows=('w2', 'w3', 'w4', 'w34')):
    """long-record ERA5 composite: mean z500 anomaly over window w after each day in a group (NDJFM 1979-2018 active MJO)."""
    D = BM.daily(name)
    days = D.dates()
    sel = [i for i, d in enumerate(days) if BM.EOF_YEARS[0] <= d.year <= BM.EOF_YEARS[1] + 1]
    i0 = sel[0]
    X = np.asarray(D.A[sel[0]:sel[-1] + 1], np.float64)
    X -= clim(days[sel[0]:sel[-1] + 1])
    X = np.nan_to_num(X)
    cs = np.concatenate([np.zeros((1,) + X.shape[1:]), np.cumsum(X, 0)], 0)
    out = {}
    for g, gd in groups_days.items():
        for w in windows:
            a, b = BM.WINDOWS[w]
            ii = [(d - days[i0]).days for d in gd]
            ii = [i for i in ii if 0 <= i and i + b + 1 < cs.shape[0]]
            acc = sum((cs[i + b + 1] - cs[i + a]) / (b - a + 1) for i in ii)
            out[(g, w)] = (acc / max(len(ii), 1)).astype(np.float32), len(ii)
    return out


def part_comp(systems, commit):
    od = pdir('comp')
    IX = pickle.load(open(OUT / 'era5' / 'mjo_full.pkl', 'rb'))
    nh = region_mask(20, 90)
    # ERA5 long-record composites (truth), NDJFM, by init phase group (WH04 phase from our ERA5 RMM, amplitude >= 1)
    Sd = IX.series['day']
    gdays = {'P2-3': [], 'P6-7': []}
    for d, r in zip(Sd['days'], Sd['rmm']):
        if not (BM.EOF_YEARS[0] <= d.year <= BM.EOF_YEARS[1]) or d.month not in (11, 12, 1, 2, 3) or not np.isfinite(r).all():
            continue
        if np.hypot(*r) >= 1:
            ph = int(BM.rmm_phase(np.array([r[0]]), np.array([r[1]]))[0])
            for g, pp in (('P2-3', (2, 3)), ('P6-7', (6, 7))):
                if ph in pp:
                    gdays[g].append(d)
    climz = {'day': BM.fixed_clim('z500_mean4'), '00z': BM.fixed_clim('z500_00')}
    truth = era5_lagged_composite(gdays, 'z500_mean4', climz['day'])
    log('comp: ERA5 long-record composites', {g: len(v) for g, v in gdays.items()})
    rows, maps = [], {}
    for (g, w), (m, n) in truth.items():
        maps[f'era5long_{g}_{w}'] = m
    W = BM.WK
    for sysn in systems:
        st = starts_of(sysn)
        samp = BM.SAMPLING[sysn]
        X, have = BM.load_extract(sysn, st, ['em30_z500', 'em11_z500'])
        O, C = obs_windows_field('z500', samp, st, clim=climz[BM.state_samp(samp)])
        init = np.array([IX.obs('day', [d])[0] for d in st])
        amp = np.hypot(init[:, 0], init[:, 1])
        ph = BM.rmm_phase(init[:, 0], init[:, 1])
        ndjfm = np.array([d.month in (11, 12, 1, 2, 3) for d in st])
        for msz in ('30', '11'):
            E = X[f'em{msz}_z500']
            b = BM.loyo_bias(E - O, st)
            Fdc = (E - b - C).astype(np.float32)
            Oa = O - C
            for g, pp in (('P2-3', (2, 3)), ('P6-7', (6, 7))):
                for seas, sm in (('NDJFM', ndjfm), ('all', np.ones(len(st), bool))):
                    sel = (amp >= 1) & np.isin(ph, pp) & sm
                    for w in ('w2', 'w3', 'w4', 'w34'):
                        j = W.index(w)
                        ok = sel & np.isfinite(Fdc[:, j]).any((1, 2)) & np.isfinite(Oa[:, j]).any((1, 2))
                        if ok.sum() < 3:
                            continue
                        fc, ob = np.nanmean(Fdc[ok, j], 0), np.nanmean(Oa[ok, j], 0)
                        tr = truth.get((g, w), (None, 0))[0]
                        rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, members=int(msz), group=g, season=seas, window=w,
                                         n=int(ok.sum()), pc_fc_vs_era5_same_starts=pattern_corr(fc, ob, nh),
                                         pc_fc_vs_era5_longrecord=pattern_corr(fc, tr, nh) if tr is not None else np.nan,
                                         pc_era5_same_starts_vs_longrecord=pattern_corr(ob, tr, nh) if tr is not None else np.nan,
                                         amp_ratio_fc_over_era5=float(np.sqrt(area_mean(fc ** 2, nh) / area_mean(ob ** 2, nh)))))
                        if msz == '30':
                            maps[f'{sysn}_{g}_{seas}_{w}_fc'] = fc
                            maps[f'{sysn}_{g}_{seas}_{w}_era5'] = ob
            # skill conditioned on the init MJO state: NH z500 ACC (centred per point) and mean uncentred pattern ACC per start
            for state, sm in (('active', amp >= 1), ('inactive', amp < 1)):
                for seas, smm in (('NDJFM', ndjfm), ('all', np.ones(len(st), bool))):
                    sel = sm & smm
                    for w in ('w2', 'w3', 'w4', 'w34'):
                        j = W.index(w)
                        ok = sel & np.isfinite(Fdc[:, j]).any((1, 2))
                        if ok.sum() < 8:
                            continue
                        am = acc_map(Fdc[ok, j], Oa[ok, j])
                        wgt = BM.cosw_lat()[:, None] * np.ones((1, 240)) * nh
                        f, o = np.nan_to_num(Fdc[ok, j]), np.nan_to_num(Oa[ok, j])
                        pacc = (wgt * f * o).sum((1, 2)) / np.sqrt((wgt * f * f).sum((1, 2)) * (wgt * o * o).sum((1, 2)))
                        rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, members=int(msz), group=f'init_{state}',
                                         season=seas, window=w, n=int(ok.sum()), acc_nh_centred=area_mean(am, nh),
                                         acc_nh_pattern_mean=float(np.nanmean(pacc))))
        log('comp', sysn, len(st))
    write_csv(rows, od / 'scores_mjo_teleconnection.csv')
    keys = sorted(maps)
    save_nc(od / 'composites_z500.nc', {k.replace('-', ''): (('lat', 'lon'), maps[k]) for k in keys},
            coords=dict(lat=BM.LAT15, lon=BM.LON15),
            attrs=dict(commit=commit, n_longrecord=json.dumps({g: len(v) for g, v in gdays.items()}),
                       note='z500 anomaly composites (m2 s-2), drift-corrected ensemble mean (30) vs ERA5 for the same starts; '
                            'era5long = ERA5 NDJFM 1979-2018 lagged composite after active-MJO days of the phase group'))
    return rows


# ===================================================================== global SST vs damped persistence
def damped_fit_field(anom, days, doys, half=30, step=2, years=(1991, 2018)):
    """anom [n, P] daily ERA5 anomalies -> {doy: r [W, P]} damped-persistence coefficients for the requested init days of year."""
    ix_tr = [i for i, d in enumerate(days) if years[0] <= d.year <= years[1] and i % step == 0 and i + 43 < len(days)]
    dd = np.array([BM.doy365(days[i]) for i in ix_tr])
    A = np.nan_to_num(anom).astype(np.float64)
    fin = np.isfinite(anom)
    cs = np.concatenate([np.zeros((1, A.shape[1])), np.cumsum(A, 0)], 0)
    out = {}
    for doy in sorted(set(doys)):
        dist = np.minimum(np.abs(dd - doy), 365 - np.abs(dd - doy))
        ii = np.array(ix_tr)[dist <= half]
        a0 = A[ii]
        r = np.full((len(BM.WINDOWS), A.shape[1]), np.nan)
        for j, (a, b) in enumerate(BM.WINDOWS.values()):
            aw = (cs[ii + b + 1] - cs[ii + a]) / (b - a + 1)
            with np.errstate(invalid='ignore', divide='ignore'):
                r[j] = (a0 * aw).sum(0) / (a0 ** 2).sum(0)
        r[:, ~fin[ii].all(0)] = np.nan
        out[doy] = r.astype(np.float32)
    return out


def part_sst(systems, commit):
    od = pdir('sst')
    rows, maps = [], {}
    regions = {'global_60s60n': region_mask(-60, 60), 'tropics_20s20n': region_mask(-20, 20), 'nh_20n60n': region_mask(20, 60),
               'sh_60s20s': region_mask(-60, -20), 'nino34': region_mask(-5, 5, 190, 240)}
    W = BM.WK
    cache = {}
    for sysn in systems:
        st = starts_of(sysn)
        samp = BM.SAMPLING[sysn]
        X, have = BM.load_extract(sysn, st, ['em30_sst', 'em11_sst'])
        if 'em30_sst' not in X:
            continue
        name = BM.obs_name('sst', samp)[0]
        c = BM.fixed_clim(name)
        O, C = obs_windows_field('sst', samp, st, clim=c)
        ocean = ~BM.land15() & (np.nanstd(O[:, 0], 0) > 0.05)                    # ERA5 sea points, no constant sea ice (NaN windows of
        # the last starts beyond the ERA5 record are skipped per point by acc_map)
        if name not in cache:
            D = BM.daily(name)
            days = D.dates()
            A = np.asarray(D.A, np.float32).reshape(len(days), -1)
            cl = c(days).reshape(len(days), -1)
            an = A - cl
            del A, cl
            cache[name] = (an, days, {d: i for i, d in enumerate(days)})
        an, days, dix = cache[name]
        rr = damped_fit_field(an, days, [BM.doy365(d) for d in st])
        P = np.full_like(O, np.nan)
        for i, d in enumerate(st):
            if d in dix:
                P[i] = (rr[BM.doy365(d)] * an[dix[d]][None]).reshape(len(W), 121, 240)
        Oa = O - C
        for msz in ('30', '11'):
            E = X[f'em{msz}_sst']
            for corr in ('raw', 'dc'):
                Fa = E - C if corr == 'raw' else E - BM.loyo_bias(E - O, st) - C
                for pn in ('2019-26', '2019-24', '2025-26'):
                    pm = BM.period_mask(st, pn)
                    for j, w in enumerate(W):
                        am = np.where(ocean, acc_map(Fa[pm, j], Oa[pm, j]), np.nan)
                        ap = np.where(ocean, acc_map(P[pm, j], Oa[pm, j]), np.nan)
                        rm = np.where(ocean, np.sqrt(np.nanmean((Fa[pm, j] - Oa[pm, j]) ** 2, 0)), np.nan)
                        rp = np.where(ocean, np.sqrt(np.nanmean((P[pm, j] - Oa[pm, j]) ** 2, 0)), np.nan)
                        for rn, mk in regions.items():
                            rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, members=int(msz), correction=corr, period=pn,
                                             region=rn, window=w, n=int(pm.sum()), acc=area_mean(am, mk), acc_damped=area_mean(ap, mk),
                                             rmse=float(np.sqrt(area_mean(rm ** 2, mk))), rmse_damped=float(np.sqrt(area_mean(rp ** 2, mk))),
                                             frac_ocean_acc_gt_damped=area_mean((am > ap).astype(float) + 0 * am, mk)))
                        if pn == '2019-26' and msz == '30' and w in ('w1', 'w2', 'w34', 'w56'):
                            maps[f'{sysn}_{corr}_{w}_acc'] = am
                            maps[f'{sysn}_{corr}_{w}_acc_damped'] = ap
        log('sst', sysn, len(st))
    write_csv(rows, od / 'scores_sst.csv')
    save_nc(od / 'sst_acc_maps.nc', {k: (('lat', 'lon'), v) for k, v in maps.items()}, coords=dict(lat=BM.LAT15, lon=BM.LON15),
            attrs=dict(commit=commit, note='centred ACC per grid point across starts of SST window anomalies (fixed 1991-2020 harmonic '
                                           'climatology); damped persistence fitted on ERA5 1991-2018 (+-30 d of the init day of year)'))
    return rows


PARTS.update({'strat': part_strat, 'comp': part_comp, 'sst': part_sst})


# ===================================================================== land
HOTSPOTS = {   # Koster et al. 2004 soil moisture - precipitation/temperature coupling hotspots (boxes, deg N / deg E)
    'great_plains': (32, 45, 255, 265), 'sahel': (8, 17, 345, 30), 'india': (15, 28, 72, 85), 'pampas': (-38, -25, 295, 305),
    'central_asia': (40, 52, 55, 80), 'europe_east': (45, 55, 25, 45)}


def clim_ens_points(var, samp, starts, points, years=range(1991, 2021)):
    """trend-adjusted climatological ensemble of window means at ``points``: ERA5 of the same calendar dates in each year y
    (daily caches), shifted along the 1991-2025 trend b(doy) of ``data/clim_trend`` to the start's year (the S33 / bing_scores
    convention, ``bing_scores.clim_ensemble``) -> [S, len(years), W, P] (tp24 in mm/day, clipped at 0)."""
    name, sh = BM.obs_name(var, samp)
    D = BM.daily(name)
    A = D.A.reshape(D.shape[0], -1)[:, points.ravel()]                              # [n, P]
    hrs = [0] if (samp == '00z' or var in ('tp24', 'olr')) else [0, 1, 2, 3]
    b = BS._slope(var)[:, hrs].mean(1)                                               # [365, 240, 121]
    b = np.swapaxes(b, 1, 2).reshape(365, -1)[:, points.ravel()]
    sc = 1000.0 if var == 'tp24' else 1.0
    out = np.full((len(starts), len(years), len(BM.WK), A.shape[1]), np.nan, np.float32)
    for i, d in enumerate(starts):
        for iy, y in enumerate(years):
            vals = np.full((BM.NLEAD, A.shape[1]), np.nan, np.float32)
            for k in range(1, BM.NLEAD):
                v = d + timedelta(days=k + sh)
                try:
                    vy = date(y + (v.year - d.year), v.month, min(v.day, 28) if v.month == 2 else v.day)
                except ValueError:
                    continue
                j = D.idx(vy)
                if 0 <= j < A.shape[0]:
                    vals[k] = A[j] + b[BM.doy365(v)] * (d.year - y)
            if var == 'tp24':
                vals = np.clip(vals, 0, None)
            out[i, iy] = BM.window_mean_daily(vals * sc)
    return out


def land_points():
    return BM.land15()


def part_land(systems, commit):
    od = pdir('land')
    lm = land_points()
    lat_l = np.repeat(BM.LAT15[:, None], 240, 1)[lm]
    lon_l = np.repeat(BM.LON15[None], 121, 0)[lm]
    wl = np.cos(np.deg2rad(lat_l))
    use = (lat_l > -60) & (lat_l < 75)                                          # no Antarctica / central Greenland
    regs = {'land_60s75n': use, 'nh_land_20n75n': use & (lat_l >= 20), 'tropics_land': np.abs(lat_l) <= 20,
            'conus': (lat_l >= 25) & (lat_l <= 50) & (lon_l >= 235) & (lon_l <= 295)}
    W = BM.WK
    rows, coup, ext = [], [], []

    def amean_l(x, m):
        ok = np.isfinite(x) & m
        return float((x[ok] * wl[ok]).sum() / wl[ok].sum()) if ok.any() else np.nan

    for sysn in systems:
        st = starts_of(sysn)
        samp = BM.SAMPLING[sysn]
        X, have = BM.load_extract(sysn, st, ['em30_swvl1', 'em30_stl1', 'em30_t2m', 'land_t2m', 'land_tp24', 'land_swvl1'])
        Sn = len(st)
        # (a) anomaly ACC of swvl1, stl1, t2m (ensemble mean 30, per land point across starts)
        for var in ('swvl1', 'stl1', 't2m'):
            if f'em30_{var}' not in X:
                continue
            if var == 'stl1' and not (BM.DAILY / 'stl1_00.npy').exists():
                continue
            if samp != '00z' and var in ('swvl1', 'stl1'):
                continue
            O, C = obs_windows_field(var, samp, st, points=lm)
            E = X[f'em30_{var}'][..., lm]
            for corr in ('raw', 'dc'):
                Fa = E - C if corr == 'raw' else E - BM.loyo_bias(E - O, st) - C
                for j, w in enumerate(W):
                    am = acc_map(Fa[:, j], O[:, j] - C[:, j])
                    for rn, rm in regs.items():
                        rows.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, var=var, correction=corr, region=rn, window=w,
                                         n=Sn, acc=amean_l(am, rm),
                                         rmse=float(np.sqrt(amean_l(np.nanmean((Fa[:, j] - O[:, j] + C[:, j]) ** 2, 0), rm)))))
        # (b) soil moisture - t2m coupling in JJA over hotspots (UC-S2S only: FuXi has no soil)
        if 'land_swvl1' in X:
            jja = np.array([d.month in (6, 7, 8) for d in st])
            Os, Cs = obs_windows_field('swvl1', samp, st, points=lm)
            Ot, Ct = obs_windows_field('t2m', samp, st, points=lm)
            Fs = X['land_swvl1'] - Cs[:, None]
            Ft = X['land_t2m'] - Ct[:, None]
            Fs = Fs - BM.loyo_bias(Fs.mean(1) - (Os - Cs), st)[:, None]
            Ft = Ft - BM.loyo_bias(Ft.mean(1) - (Ot - Ct), st)[:, None]
            for hs, (s, n, w0, e0) in HOTSPOTS.items():
                inb = (lat_l >= s) & (lat_l <= n) & (((lon_l >= w0) & (lon_l <= e0)) if w0 <= e0 else ((lon_l >= w0) | (lon_l <= e0)))
                if inb.sum() == 0:
                    continue
                for (ws, wt) in (('w1', 'w1'), ('w2', 'w2'), ('w34', 'w34'), ('w1', 'w34'), ('w2', 'w34')):
                    js, jt = W.index(ws), W.index(wt)
                    ro = acc_map((Os - Cs)[jja, js][:, inb], (Ot - Ct)[jja, jt][:, inb])
                    rf = np.nanmean([acc_map(Fs[jja, m, js][:, inb], Ft[jja, m, jt][:, inb]) for m in range(Fs.shape[1])], 0)
                    coup.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, hotspot=hs, swvl1_window=ws, t2m_window=wt,
                                     n_starts=int(jja.sum()), n_points=int(inb.sum()),
                                     corr_era5=float(np.nanmean(ro)), corr_model_members=float(np.nanmean(rf))))
        # (c) heat extremes (weekly t2m > 90th pct) and dry / wet weeks (tp terciles): BSS vs climatology, ROC area
        for var, qs in (('t2m', (0.9,)), ('tp24', (1 / 3, 2 / 3))):
            if f'land_{var}' not in X:
                continue
            CE = clim_ens_points(var, samp, st, lm)                                         # [S, 30, W, P]
            O, _ = obs_windows_field(var, samp, st, points=lm)
            Fm = X[f'land_{var}']                                                           # [S, M, W, P]
            for corr in ('raw', 'dc'):
                Fk = Fm if corr == 'raw' else Fm - BM.loyo_bias(Fm.mean(1) - O, st)[:, None]
                if var == 'tp24':
                    Fk = np.clip(Fk, 0, None)
                for q in qs:
                    thr = np.quantile(CE, q, axis=1)          # [S, W, P]; NaN if any member is missing (nanquantile is ~100x slower)
                    above = q >= 0.5
                    p = (Fk > thr[:, None]).mean(1) if above else (Fk < thr[:, None]).mean(1)
                    o = (O > thr) if above else (O < thr)
                    pc = (1 - q) if above else q
                    ok = np.isfinite(O) & np.isfinite(thr) & np.isfinite(Fk).all(1)
                    for j, w in enumerate(W):
                        for rn, rm in regs.items():
                            m = ok[:, j] & rm[None]
                            if m.sum() < 100:
                                continue
                            pw = p[:, j][m]
                            ow = o[:, j][m].astype(float)
                            ww = np.broadcast_to(wl[None], m.shape)[m]
                            bs = float((ww * (pw - ow) ** 2).sum() / ww.sum())
                            bsr = float((ww * (pc - ow) ** 2).sum() / ww.sum())
                            ext.append(dict(system=BM.SYSTEM_LABEL[sysn], system_key=sysn, var=var,
                                            event=('t2m>p90' if var == 't2m' else ('tp<tercile1' if q < 0.5 else 'tp>tercile2')),
                                            correction=corr, region=rn, window=w, n=int(m.sum()), base_rate=float((ww * ow).sum() / ww.sum()),
                                            bs=bs, bs_clim=bsr, bss=1 - bs / bsr, roc_auc=roc_auc(pw, ow, ww)))
        log('land', sysn, Sn)
    write_csv(rows, od / 'scores_land_acc.csv')
    write_csv(coup, od / 'coupling_swvl1_t2m_jja.csv')
    write_csv(ext, od / 'scores_extremes.csv')
    return rows


def roc_auc(p, o, w=None):
    """area under the ROC curve of probabilities p for binary outcomes o (weighted Mann-Whitney)."""
    o = o.astype(bool)
    w = np.ones_like(p) if w is None else w
    if o.all() or (~o).all():
        return np.nan
    order = np.argsort(p, kind='mergesort')
    p, o, w = p[order], o[order], w[order]
    auc = 0.0
    cum_neg = 0.0
    i = 0
    n = len(p)
    while i < n:
        j = i
        while j < n and p[j] == p[i]:
            j += 1
        wp = w[i:j][o[i:j]].sum()
        wn = w[i:j][~o[i:j]].sum()
        auc += wp * (cum_neg + 0.5 * wn)
        cum_neg += wn
        i = j
    return float(auc / (w[o].sum() * w[~o].sum()))


PARTS.update({'land': part_land})
