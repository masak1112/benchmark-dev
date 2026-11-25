"""Figures of the modes evaluation (login node, light): reads runs/bing/eval/modes/<part>/*.csv|nc, writes runs/bing/eval/modes/figures/.
Colours: Okabe-Ito subset without green (vermillion, blue, reddish purple, orange, black), plus line styles as a second encoding;
maps RdBu_r (diverging, grey-white midpoint).

    python scripts/bing_modes_figs.py [--parts mjo tele enso sst comp strat qbo land]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from s41eval import bing_modes as BM  # noqa: E402

OUT = BM.OUT
FIG = OUT / 'figures'
COL = {'ucs2s_sstanom': '#D55E00', 'ucs2s_plain': '#0072B2', 'fuxi': '#CC79A7', 'ifs': '#000000', 'damped': '#E69F00',
       'persist': '#777777', 'era5': '#000000'}
LS = {'ucs2s_sstanom': '-', 'ucs2s_plain': '--', 'fuxi': '-.', 'ifs': ':', 'damped': ':', 'persist': (0, (1, 3))}
LAB = dict(BM.SYSTEM_LABEL, damped='damped persistence', persist='persistence', era5='ERA5')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#dddddd', 'grid.linewidth': 0.6, 'lines.linewidth': 2})


def rd(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIG / name)


def fig_mjo():
    rows = rd(OUT / 'mjo' / 'scores_mjo.csv')
    if not rows:
        return
    for idx in ('full', 'winds'):
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for sk in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
            for msz, alpha in (('30', 1.0), ('11', 0.45)):
                rr = [r for r in rows if r['index'] == idx and r['system_key'] == sk and r['members'] == msz and r['correction'] == 'dc'
                      and r['period'] == '2019-26' and r['group'] == 'all' and r['lead'] != 'w34']
                if not rr:
                    continue
                L = [int(r['lead']) for r in rr]
                lab = f'{LAB[sk]} ({msz} m)'
                ax[0].plot(L, [fnum(r['cor']) for r in rr], color=COL[sk], ls=LS[sk], alpha=alpha, label=lab)
                ax[1].plot(L, [fnum(r['rmse']) for r in rr], color=COL[sk], ls=LS[sk], alpha=alpha)
                ax[2].plot(L, [fnum(r['amp_err']) for r in rr], color=COL[sk], ls=LS[sk], alpha=alpha)
        ax[0].axhline(0.5, color='#555555', lw=1)
        ax[1].axhline(np.sqrt(2), color='#555555', lw=1)
        ax[2].axhline(0, color='#555555', lw=1)
        for a, t in zip(ax, ('bivariate COR', 'bivariate RMSE', 'amplitude error |F| - |O|')):
            a.set_title(t)
            a.set_xlabel('lead (days)')
        ax[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f'MJO {"RMM (OLR, u850, u200)" if idx == "full" else "winds-only RMM (u850, u200)"}, drift-corrected, '
                     f'2019-26 starts, ensemble mean')
        save(fig, f'mjo_{idx}_cor_rmse_lead.png')
    # by init state
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, groups in enumerate((('all', 'active', 'inactive'), ('P8-1', 'P2-3', 'P4-5', 'P6-7'))):
        x = np.arange(len(groups))
        for q, sk in enumerate(('ucs2s_sstanom', 'ucs2s_plain', 'fuxi')):
            vals = []
            for g in groups:
                rr = [r for r in rows if r['index'] == 'full' and r['system_key'] == sk and r['members'] == '30' and r['correction'] == 'dc'
                      and r['period'] == '2019-26' and r['group'] == g and r['lead'] == 'w34']
                vals.append(fnum(rr[0]['skill_day']) if rr else np.nan)
            ax[j].bar(x + (q - 1) * 0.27, vals, 0.25, color=COL[sk], label=LAB[sk])
        ax[j].set_xticks(x, groups)
        ax[j].set_ylabel('skill day (COR = 0.5)')
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle('MJO skill day by ERA5 state at the init (RMM, 30 members, drift-corrected, 2019-26)')
    save(fig, 'mjo_skill_day_by_init.png')
    # phase error
    fig, ax = plt.subplots(figsize=(6, 4))
    for sk in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
        rr = [r for r in rows if r['index'] == 'full' and r['system_key'] == sk and r['members'] == '30' and r['correction'] == 'dc'
              and r['period'] == '2019-26' and r['group'] == 'active' and r['lead'] != 'w34']
        if rr:
            ax.plot([int(r['lead']) for r in rr], [fnum(r['phase_err']) for r in rr], color=COL[sk], ls=LS[sk], label=LAB[sk])
    ax.axhline(0, color='#555555', lw=1)
    ax.set_xlabel('lead (days)')
    ax.set_ylabel('phase error (deg, > 0 = forecast ahead)')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('MJO phase error, active MJO at init')
    save(fig, 'mjo_phase_error.png')


def fig_tele():
    rows = rd(OUT / 'tele' / 'scores_tele.csv')
    if not rows:
        return
    W = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    for g in ('DJF', 'all'):
        fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), sharey='row')
        for i, idx in enumerate(('nao', 'pna', 'ao', 'sam')):
            for sk in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
                rr = {r['window']: r for r in rows if r['index'] == idx and r['system_key'] == sk and r['correction'] == 'dc'
                      and r['period'] == '2019-26' and r['group'] == g}
                if not rr:
                    continue
                ax[0, i].plot(range(1, 7), [fnum(rr[w]['cor30']) if w in rr else np.nan for w in W], color=COL[sk], ls=LS[sk],
                              marker='o', ms=4, label=LAB[sk])
                ax[1, i].plot(range(1, 7), [fnum(rr[w]['crpss']) if w in rr else np.nan for w in W], color=COL[sk], ls=LS[sk],
                              marker='o', ms=4)
            ax[0, i].set_title(idx.upper() + (' (ref. test failed)' if idx in ('nao', 'pna') else ''))
            ax[1, i].axhline(0, color='#555555', lw=1)
            ax[1, i].set_xlabel('week')
        ax[0, 0].set_ylabel('COR (ens. mean, 30 m)')
        ax[1, 0].set_ylabel('CRPSS vs ERA5 1991-2020')
        ax[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle(f'Weekly teleconnection indices, drift-corrected, starts {g} 2019-26')
        save(fig, f'tele_indices_{g}.png')


def fig_enso():
    rows = rd(OUT / 'enso' / 'scores_nino34.csv')
    if not rows:
        return
    W = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sk in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
        rr = {r['window']: r for r in rows if r['system_key'] == sk and r['correction'] == 'dc' and r['period'] == '2019-26'
              and r['group'] == 'all'}
        if not rr:
            continue
        ax[0].plot(range(1, 7), [fnum(rr[w]['acc30']) for w in W], color=COL[sk], ls=LS[sk], marker='o', ms=4, label=LAB[sk])
        ax[1].plot(range(1, 7), [fnum(rr[w]['rmse30']) for w in W], color=COL[sk], ls=LS[sk], marker='o', ms=4)
        last = rr
    ax[0].plot(range(1, 7), [fnum(last[w]['acc_damped']) for w in W], color=COL['damped'], ls=LS['damped'], marker='s', ms=4,
               label=LAB['damped'])
    ax[1].plot(range(1, 7), [fnum(last[w]['rmse_damped']) for w in W], color=COL['damped'], ls=LS['damped'], marker='s', ms=4)
    ax[0].set_ylabel('ACC')
    ax[1].set_ylabel('RMSE (K)')
    for a in ax:
        a.set_xlabel('week')
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Nino-3.4 SST anomaly, drift-corrected ensemble mean (30 m), 2019-26 starts')
    save(fig, 'enso_nino34.png')


def fig_sst():
    f = OUT / 'sst' / 'sst_acc_maps.nc'
    if not f.exists():
        return
    import xarray as xr
    ds = xr.merge([xr.open_dataset(g) for g in sorted((OUT / 'sst').glob('sst_acc_maps*.nc'), reverse=True)], compat='override')
    sk = [s for s in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi') if f'{s}_dc_w34_acc' in ds and np.isfinite(ds[f'{s}_dc_w34_acc']).any()]
    fig, ax = plt.subplots(1, len(sk) + 1, figsize=(5 * (len(sk) + 1), 3.2))
    im = None
    for i, s in enumerate(sk):
        d = ds[f'{s}_dc_w34_acc'] - ds[f'{s}_dc_w34_acc_damped']
        im = ax[i].pcolormesh(ds.lon, ds.lat, d, cmap='RdBu_r', vmin=-0.5, vmax=0.5, shading='auto')
        ax[i].set_title(f'{LAB[s]}: ACC - damped ACC')
    im2 = ax[-1].pcolormesh(ds.lon, ds.lat, ds[f'{sk[0]}_dc_w34_acc_damped'], cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    ax[-1].set_title('damped persistence ACC')
    for a in ax:
        a.set_ylim(-60, 60)
        a.grid(False)
    fig.colorbar(im, ax=ax[:-1], shrink=0.8)
    fig.colorbar(im2, ax=ax[-1], shrink=0.8)
    fig.suptitle('SST anomaly ACC, weeks 3-4, drift-corrected, 2019-26')
    save(fig, 'sst_acc_w34_vs_damped.png')


def fig_comp():
    f = OUT / 'comp' / 'composites_z500.nc'
    if not f.exists():
        return
    import xarray as xr
    ds = xr.open_dataset(f)
    for w in ('w3', 'w34'):
        sk = [s for s in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi') if f'{s}_P23_NDJFM_{w}_fc' in ds]
        fig, ax = plt.subplots(2, 2 + len(sk), figsize=(4 * (2 + len(sk)), 5.5))
        for r, g in enumerate(('P23', 'P67')):
            panels = [(f'era5long_{g}_{w}', 'ERA5 1979-2018 (NDJFM)')]
            if sk:
                panels.append((f'{sk[0]}_{g}_NDJFM_{w}_era5', 'ERA5, same starts'))
            panels += [(f'{s}_{g}_NDJFM_{w}_fc', LAB[s]) for s in sk]
            for c, (k, t) in enumerate(panels):
                if k not in ds:
                    ax[r, c].axis('off')
                    continue
                x = ds[k].values / 9.80665
                nh = ds.lat.values >= 20
                wgt = np.cos(np.deg2rad(ds.lat.values[nh]))[:, None] * np.ones((1, x.shape[1]))
                x = x - np.nansum(x[nh] * wgt) / np.sum(wgt * np.isfinite(x[nh]))      # 20-90N mean removed (warming offset)
                ax[r, c].pcolormesh(ds.lon, ds.lat, x, cmap='RdBu_r', vmin=-60, vmax=60, shading='auto')
                ax[r, c].set_xlabel('longitude (E)', fontsize=8)
                ax[r, c].set_ylim(20, 90)
                ax[r, c].set_title(f'{g[0]}{g[1]}-{g[2]}: {t}', fontsize=9)
                ax[r, c].grid(False)
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-60, 60))
        fig.colorbar(sm, ax=ax, shrink=0.7, label='z500 anomaly (m)')
        fig.suptitle(f'z500 anomaly {w} after init by MJO phase at init (active, amplitude >= 1; WH04 phases after the fixed 37.5 deg '
                     f'EOF-plane rotation), NDJFM starts; 20-90N mean removed; drift-corrected 30-member means')
        save(fig, f'mjo_teleconnection_z500_{w}.png')


def fig_strat():
    rows = rd(OUT / 'strat' / 'scores_u60.csv')
    if rows:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for idx, a in (('u60n10', ax[0]), ('u60n50', ax[1])):
            for sk in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
                rr = [r for r in rows if r['index'] == idx and r['system_key'] == sk and r['correction'] == 'dc' and r['group'] == 'NDJFM']
                if rr:
                    a.plot([int(r['lead']) for r in rr], [fnum(r['acc']) for r in rr], color=COL[sk], ls=LS[sk], label=LAB[sk])
            a.set_title(f'{idx.upper()} anomaly ACC, NDJFM starts')
            a.set_xlabel('lead (days)')
            a.axhline(0.5, color='#555555', lw=1)
        ax[0].legend(fontsize=8, frameon=False)
        save(fig, 'strat_u60_acc.png')
    f = OUT / 'strat' / 'ssw_capz_composites.nc'
    if f.exists():
        import xarray as xr
        ds = xr.open_dataset(f)
        sk = [s for s in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi') if f'fc_{s}' in ds]
        fig, ax = plt.subplots(1, len(sk) + 1, figsize=(4.2 * (len(sk) + 1), 3.6), sharey=True)
        lev = ds.level.values
        for i, (k, t) in enumerate([(f'obs_{sk[0]}', 'ERA5 (same starts)')] + [(f'fc_{s}', LAB[s]) for s in sk]):
            cf = ax[i].contourf(ds.rel_day, np.log(lev), ds[k].T, levels=np.arange(-450, 451, 50), cmap='RdBu_r', extend='both')
            ax[i].set_title(t, fontsize=9)
            ax[i].axvline(0, color='k', lw=1)
            ax[i].set_xlabel('days after SSW onset')
        ax[0].set_yticks(np.log(lev), [str(int(v)) for v in lev])
        ax[0].invert_yaxis()
        ax[0].set_ylabel('hPa')
        fig.colorbar(cf, ax=ax, shrink=0.8, label='m')
        fig.suptitle('Polar-cap (60-90N) height anomaly after ERA5 SSWs, ensemble mean (starts 14 d before to 2 d after onset; '
                     'FuXi has no 10 hPa)')
        save(fig, 'strat_ssw_capz.png')
    q = rd(OUT / 'qbo' / 'qbo_drift.csv')
    if q:
        fig, ax = plt.subplots(figsize=(6, 4))
        for sk in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
            for L, lw in (('50', 2), ('10', 1.2)):
                rr = [r for r in q if r['system_key'] == sk and r['level'] == L and r['group'] == 'all']
                if rr:
                    ax.plot([int(r['lead']) for r in rr], [fnum(r['drift']) for r in rr], color=COL[sk], ls=LS[sk], lw=lw,
                            label=f'{LAB[sk]} {L} hPa')
        ax.axhline(0, color='#555555', lw=1)
        ax.set_xlabel('lead (days)')
        ax.set_ylabel('drift of equatorial zonal-mean u (m/s)')
        ax.legend(fontsize=7, frameon=False)
        save(fig, 'qbo_drift.png')


def fig_land():
    rows = rd(OUT / 'land' / 'scores_extremes.csv')
    if not rows:
        return
    W = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    ev = sorted({r['event'] for r in rows})
    fig, ax = plt.subplots(1, len(ev), figsize=(4.5 * len(ev), 3.8), sharey=True)
    ax = np.atleast_1d(ax)
    for i, e in enumerate(ev):
        for sk in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
            rr = {r['window']: r for r in rows if r['event'] == e and r['system_key'] == sk and r['correction'] == 'dc'
                  and r['region'] == 'land_60s75n'}
            if rr:
                ax[i].plot(range(1, 7), [fnum(rr[w]['bss']) if w in rr else np.nan for w in W], color=COL[sk], ls=LS[sk], marker='o',
                           ms=4, label=LAB[sk])
        ax[i].axhline(0, color='#555555', lw=1)
        ax[i].set_title(e)
        ax[i].set_xlabel('week')
    ax[0].set_ylabel('BSS vs climatological probability')
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Weekly extremes over land (60S-75N), drift-corrected; thresholds from the trend-adjusted ERA5 climatology')
    save(fig, 'land_extremes_bss.png')


FIGS = {'mjo': fig_mjo, 'tele': fig_tele, 'enso': fig_enso, 'sst': fig_sst, 'comp': fig_comp, 'strat': fig_strat, 'land': fig_land}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parts', nargs='*')
    a = p.parse_args()
    for k in (a.parts or FIGS):
        try:
            FIGS[k]()
        except Exception as e:  # noqa: BLE001
            import traceback
            print('FIG ERROR', k, traceback.format_exc()[-800:])
