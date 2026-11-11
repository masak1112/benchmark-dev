"""
Weekly TCC of near-surface SOIL MOISTURE for UC-S2S V2 and the ECMWF 2025-model hindcasts, over land.

The two models carry different layers, so each is verified against the matching ERA5 layer:
  UC-S2S V2       volumetric_soil_water_layer_1 (ERA5 layer 1, 0-7 cm), instantaneous 00 UTC
                  -> ERA5 swvl1 at 00 UTC, 00Z climatology of swvl1
  ECMWF hindcast  soil_moisture_top_20_cm (sm20, kg m-3 /1000 -> m3 m-3), daily mean (step 24(L+1) = day L)
                  -> ERA5 0-20 cm = (7 swvl1 + 13 swvl2) / 20 (layer 2 is 7-28 cm), daily mean,
                     with the matching daily-mean climatology
TCC is a correlation, so units do not matter, but a deeper layer varies more slowly and is more persistent,
so the two TCCs are not strictly like-for-like.

Same method as tcc_weekly.py: ensemble mean (20 / 11 members), weekly means (week k = lead days 7k-6..7k,
plus weeks 3-4 and 5-6), anomalies vs the ERA5 1979-2018 day-of-year climatology, Pearson (centered) TCC
across inits at each grid point, cos(lat)-weighted means over LAND points (ERA5 land-sea mask >= 0.5, points
where the TCC is undefined - ocean, ice sheets - drop out), 95% month-block bootstrap CIs.

Stages:
  clim       ERA5 1979-2018 climatologies -> ACC/clim_00z/clim_00z_swvl1.npy, ACC/clim_dailymean/clim_dailymean_sm0_20.npy
  extract    per-init weekly means -> /glade/derecho/scratch/bgong/tcc_cache_soil/{model}/{init}.npz
  aggregate  TCC + bootstrap for the own-record and shared-date sets -> results/tcc_soil_*.pkl
  plot       paper-style figures -> /glade/work/bgong/plots/tcc_paper_soil_*.png/.pdf
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import netCDF4
import numpy as np

import tcc_weekly as t
from tcc_bootstrap import block_ids, resample_weights

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, 'results')
CACHE = '/glade/derecho/scratch/bgong/tcc_cache_soil'
PLOT_DIR = '/glade/work/bgong/plots'
ERA5_DIR = t.ERA5_DIR
L1, L2 = 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2'
CLIM_00Z = os.path.join(t.CLIM_ROOT, 'clim_00z', 'clim_00z_swvl1.npy')
CLIM_DM = os.path.join(t.CLIM_ROOT, 'clim_dailymean', 'clim_dailymean_sm0_20.npy')
N_WEEKS = t.N_WEEKS
WEEKS = t.WEEK_NAMES                     # week1..week6, week3-4, week5-6
MODELS = {
    'ucs2s_v2': {'name': 'UC-S2S V2', 'members': 20, 'daily': False,
                 'dates': 'dates_ucs2sv2_2019_2021.txt', 'desc': 'swvl1 (0-7 cm)'},
    'ecmwf_rf': {'name': 'ECMWF hindcast', 'members': 11, 'daily': True,
                 'dates': 'dates_ecmwf_rf_2019_2021.txt', 'desc': 'sm20 (0-20 cm)'},
}
SHARED = 'common_dates_ucs2sv2_ecmwf_rf.txt'

lsm = np.asarray(netCDF4.Dataset(f'{ERA5_DIR}/land_sea_mask_180x360.nc').variables['land_sea_mask'][:])
LAND = lsm >= 0.5
REGIONS = {'global land': (-90, 90), 'tropics land': (-30, 30), 'NH land': (30, 90), 'SH land': (-90, -30)}
MASKS = {r: LAND & np.outer((t.lat >= a) & (t.lat <= b), np.ones(360, dtype=bool)) for r, (a, b) in REGIONS.items()}


# ── ERA5 truth ────────────────────────────────────────────────────────────────

class Truth:
    """ERA5 swvl1 at 00 UTC (daily=False) or the daily-mean 0-20 cm layer (daily=True)."""

    def __init__(self, daily):
        self.daily = daily
        self.h = {}

    def _open(self, var, year):
        if (var, year) not in self.h:
            nc = netCDF4.Dataset(f'{ERA5_DIR}/{var}/{year}_180x360.nc')
            self.h[(var, year)] = (nc, {d: i for i, d in enumerate(t.era5_dates(nc.variables['time']))})
        return self.h[(var, year)]

    def _field(self, var, date, hours):
        nc, idx = self._open(var, date.year)
        ii = [idx.get(date + timedelta(hours=h)) for h in hours]
        if None in ii:
            return None
        return np.ma.filled(nc.variables[var][ii].astype(np.float64), np.nan).mean(axis=0)

    def get(self, date):
        if not self.daily:
            return self._field(L1, date, (0,))
        a, b = self._field(L1, date, (0, 6, 12, 18)), self._field(L2, date, (0, 6, 12, 18))
        return None if a is None or b is None else (7.0 * a + 13.0 * b) / 20.0

    def close(self):
        for nc, _ in self.h.values():
            nc.close()


# ── climatology ───────────────────────────────────────────────────────────────

def clim_year(year):
    """Per-DOY sums and counts for one year: (00Z swvl1, daily-mean 0-20 cm)."""
    s = {k: np.zeros((366, 180, 360)) for k in ('z', 'd')}
    n = {k: np.zeros((366, 180, 360)) for k in ('z', 'd')}
    with netCDF4.Dataset(f'{ERA5_DIR}/{L1}/{year}_180x360.nc') as a, netCDF4.Dataset(f'{ERA5_DIR}/{L2}/{year}_180x360.nc') as b:
        dates = t.era5_dates(a.variables['time'])
        assert dates == t.era5_dates(b.variables['time'])
        x1 = np.ma.filled(a.variables[L1][:].astype(np.float64), np.nan)
        x2 = np.ma.filled(b.variables[L2][:].astype(np.float64), np.nan)
    days = np.array([d.toordinal() for d in dates])
    hours = np.array([d.hour for d in dates])
    for od in np.unique(days):
        sel = days == od
        k = t.doy_index(datetime.fromordinal(int(od)))
        z = sel & (hours == 0)
        for key, v in (('z', x1[z][0] if z.any() else None),
                       ('d', ((7.0 * x1[sel] + 13.0 * x2[sel]) / 20.0).mean(axis=0))):
            if v is not None:
                ok = np.isfinite(v)
                s[key][k][ok] += v[ok]
                n[key][k][ok] += 1
    return s, n


def stage_clim():
    years = range(1979, 2019)
    tot_s = {k: np.zeros((366, 180, 360)) for k in ('z', 'd')}
    tot_n = {k: np.zeros((366, 180, 360)) for k in ('z', 'd')}
    with Pool(8) as p:
        for y, (s, n) in zip(years, p.imap(clim_year, years)):
            for k in tot_s:
                tot_s[k] += s[k]
                tot_n[k] += n[k]
            print(f'  clim {y}', flush=True)
    for k, path in (('z', CLIM_00Z), ('d', CLIM_DM)):
        with np.errstate(invalid='ignore', divide='ignore'):
            c = np.where(tot_n[k] > 0, tot_s[k] / tot_n[k], np.nan)
        np.save(path, c.astype(np.float32))
        print(f'Saved {path}', flush=True)


# ── forecasts ─────────────────────────────────────────────────────────────────

def read_uc(init):
    d = t.MODELS['ucs2s_v2']['dir']
    pre = t.uc_prefix('ucs2s_v2')
    files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith(f'{pre}{init}') and f.endswith('.nc')),
                   key=lambda f: t._member_num(f, '_ens_'))[:MODELS['ucs2s_v2']['members']]
    acc, n = np.zeros((7 * N_WEEKS, 180, 360)), 0
    for fp in files:
        with netCDF4.Dataset(fp) as nc:
            tt = np.asarray(nc.variables['time'][:]).astype(int)
            ti = [int(np.where(tt == L)[0][0]) for L in range(1, 7 * N_WEEKS + 1)]
            acc += np.asarray(nc.variables[L1][ti, :, :], dtype=np.float64)
        n += 1
    return acc / max(n, 1), n


def read_ec(init):
    import xarray as xr
    L = np.arange(1, 7 * N_WEEKS + 1)
    steps = [np.timedelta64(int(h), 'h') for h in 24 * (L + 1)]       # daily mean over day L
    out = []
    for e in ('cf', 'pf'):
        ds = xr.open_dataset(t.hc_file('sm20', e, init), engine='cfgrib', backend_kwargs={'indexpath': ''})
        a = ds['sm20'].sel(time=np.datetime64(f'{init[:4]}-{init[4:6]}-{init[6:]}')).sel(step=steps)
        a = a.expand_dims(number=[0]) if e == 'cf' else a.sortby('number').isel(number=slice(0, MODELS['ecmwf_rf']['members'] - 1))
        out.append(np.asarray(a.values, dtype=np.float64))
        ds.close()
    x = np.concatenate(out, axis=0)
    mean = x.mean(axis=0) / 1000.0                                   # kg m-3 -> m3 m-3
    reg = t._regrid_fuxi(mean[None], np.linspace(90.0, -90.0, 121), np.arange(0.0, 360.0, 1.5))[0]
    return reg, x.shape[0]


def extract_one(args):
    model, init = args
    out = f'{CACHE}/{model}/{init}.npz'
    if os.path.exists(out):
        return init, 'cached'
    try:
        fc, n = (read_uc if model == 'ucs2s_v2' else read_ec)(init)
        if n < MODELS[model]['members']:
            return init, f'SKIPPED: {n} members'
        truth = Truth(MODELS[model]['daily'])
        t0 = datetime.strptime(init, '%Y%m%d')
        try:
            obs = [truth.get(t0 + timedelta(days=L)) for L in range(1, 7 * N_WEEKS + 1)]
        finally:
            truth.close()
        if any(o is None for o in obs):
            return init, 'SKIPPED: truth missing'
        obs = np.stack(obs)
        f_wk = np.stack([fc[7 * k:7 * (k + 1)].mean(axis=0) for k in range(N_WEEKS)]).astype(np.float32)
        o_wk = np.stack([obs[7 * k:7 * (k + 1)].mean(axis=0) for k in range(N_WEEKS)]).astype(np.float32)
        np.savez(out + '.tmp.npz', fc=f_wk, obs=o_wk, n_members=n)
        os.replace(out + '.tmp.npz', out)
        return init, f'ok ({n} members)'
    except Exception as e:
        return init, f'ERROR {type(e).__name__}: {e}'


def stage_extract(workers):
    for model, cfg in MODELS.items():
        os.makedirs(f'{CACHE}/{model}', exist_ok=True)
        dates = [l.strip() for l in open(os.path.join(HERE, cfg['dates'])) if l.strip()]
        with Pool(workers) as p:
            for k, (init, msg) in enumerate(p.imap_unordered(extract_one, [(model, d) for d in dates]), 1):
                if not msg.startswith(('ok', 'cached')) or k % 50 == 0:
                    print(f'[{model} {k}/{len(dates)}] {init}: {msg}', flush=True)


# ── TCC ───────────────────────────────────────────────────────────────────────

def anomalies(model, dates, monthly_mean):
    clim = np.load(CLIM_DM if MODELS[model]['daily'] else CLIM_00Z)
    used, fa, oa = [], [], []
    for d in dates:
        p = f'{CACHE}/{model}/{d}.npz'
        if not os.path.exists(p):
            continue
        z = np.load(p)
        c = t.weekly_clim(clim, d)
        fa.append(z['fc'] - c)
        oa.append(z['obs'] - c)
        used.append(d)
    fa, oa = np.stack(fa).astype(np.float64), np.stack(oa).astype(np.float64)    # (N, week, lat, lon)
    if monthly_mean:
        months = np.array([int(d[4:6]) for d in used])
        for mo in np.unique(months):
            i = months == mo
            fa[i] -= fa[i].mean(axis=0)
            oa[i] -= oa[i].mean(axis=0)
    fa = np.concatenate([fa, fa[:, 2:4].mean(1, keepdims=True), fa[:, 4:6].mean(1, keepdims=True)], axis=1)
    oa = np.concatenate([oa, oa[:, 2:4].mean(1, keepdims=True), oa[:, 4:6].mean(1, keepdims=True)], axis=1)
    return used, fa, oa


def boot_regional(fa, oa, W):
    """(R, n_regions) land-mean TCC for each resample row of W (row 0 = full sample)."""
    n = W.sum(axis=1, keepdims=True)
    fa = np.where(np.isfinite(fa), fa, 0.0)
    oa = np.where(np.isfinite(oa), oa, 0.0)
    sx, sy = W @ fa, W @ oa
    sxx, syy, sxy = W @ fa ** 2, W @ oa ** 2, W @ (fa * oa)
    cov = sxy / n - sx * sy / n ** 2
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where((vx > 1e-14) & (vy > 1e-14), cov / np.sqrt(vx * vy), np.nan)
    out = np.empty((W.shape[0], len(MASKS)))
    for j, m in enumerate(MASKS.values()):
        rm = r[:, m.ravel()]
        w = np.broadcast_to(t.W2D[m], rm.shape)
        ok = np.isfinite(rm)
        out[:, j] = np.where(ok, rm * w, 0).sum(1) / np.where(ok, w, 0).sum(1)
    return out


def stage_aggregate(n_boot=1000):
    sets = {f'own_{k}': ([k], [l.strip() for l in open(os.path.join(HERE, c['dates'])) if l.strip()])
            for k, c in MODELS.items()}
    sets['shared'] = (list(MODELS), [l.strip() for l in open(os.path.join(HERE, SHARED)) if l.strip()])
    for mm in (False, True):
        for name, (models, dates) in sets.items():
            an = {m: anomalies(m, dates, mm) for m in models}
            common = sorted(set.intersection(*[set(a[0]) for a in an.values()]))
            bid = block_ids(common, 'month')
            W = resample_weights(bid, n_boot, np.random.default_rng(0))
            res = {'models': models, 'weeks': WEEKS, 'regions': list(REGIONS), 'init_dates': common,
                   'boot': {}, 'n_init': len(common), 'monthly_mean': mm}
            for m in models:
                used, fa, oa = an[m]
                keep = [used.index(d) for d in common]
                fa, oa = fa[keep], oa[keep]
                res['boot'][m] = np.stack([boot_regional(fa[:, k].reshape(len(common), -1),
                                                         oa[:, k].reshape(len(common), -1), W)
                                           for k in range(len(WEEKS))], axis=1)       # (R+1, week, region)
            out = os.path.join(RES_DIR, f'tcc_soil_{name}{"_mm" if mm else ""}.pkl')
            pickle.dump(res, open(out, 'wb'))
            g = res['boot']
            print(f'{name}{" (month mean removed)" if mm else ""}: n={len(common)}  global-land TCC '
                  + '  '.join(f'{MODELS[m]["name"]}: ' + ' '.join(f'{x:.3f}' for x in g[m][0, :, 0]) for m in models),
                  flush=True)


# ── figures ───────────────────────────────────────────────────────────────────

def stage_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, '/glade/work/bgong')
    from model_colors import model_color
    ticks = ['1', '2', '3', '4', '5', '6', '3–4', '5–6']
    x = np.array([0, 1, 2, 3, 4, 5, 6.6, 7.6])
    for mm in ('', '_mm'):
        for kind in ('own', 'shared'):
            if kind == 'own':
                res = {m: pickle.load(open(os.path.join(RES_DIR, f'tcc_soil_own_{m}{mm}.pkl'), 'rb')) for m in MODELS}
                series = [(m, res[m]) for m in MODELS]
            else:
                r = pickle.load(open(os.path.join(RES_DIR, f'tcc_soil_shared{mm}.pkl'), 'rb'))
                series = [(m, r) for m in MODELS]
            fig, ax = plt.subplots(figsize=(11, 6))
            w = 0.38
            for s_i, (m, r) in enumerate(series):
                b = r['boot'][m][:, :, 0]                        # global land
                v = b[0]
                lo, hi = np.nanpercentile(b[1:], [2.5, 97.5], axis=0)
                xs = x + (s_i - 0.5) * w
                label = f"{MODELS[m]['name']} – {MODELS[m]['desc']}" + (f" (n={r['n_init']})" if kind == 'own' else '')
                ax.bar(xs, v, w * 0.92, color=model_color(MODELS[m]['name']), label=label, zorder=2,
                       yerr=np.abs([v - lo, hi - v]), error_kw=dict(ecolor='0.2', elinewidth=1.2, capsize=3))
                if kind == 'shared' and s_i == 1:
                    b0 = r['boot'][series[0][0]][1:, :, 0]
                    d_lo, d_hi = np.nanpercentile(b[1:] - b0, [2.5, 97.5], axis=0)
                    for xi, vi, e, sig in zip(xs, v, hi - v, (d_lo > 0) | (d_hi < 0)):
                        if sig:
                            ax.text(xi, vi + e + 0.01, '★', ha='center', va='bottom', fontsize=16, color='0.15')
            ax.axvline(5.8, color='0.75', linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(ticks, fontsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.set_xlabel('Lead time (weeks)', fontsize=20)
            ax.set_ylabel('TCC (land)', fontsize=20)
            ax.set_ylim(0, 1.0)
            title = 'Near-surface soil moisture' + (', month mean removed' if mm else '')
            ax.set_title(title + ('' if kind == 'own' else f" — same {series[0][1]['n_init']} inits"), fontsize=20, pad=10)
            ax.grid(True, axis='y', color='0.85')
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            fig.legend(loc='lower center', ncol=1, fontsize=18, frameon=False, bbox_to_anchor=(0.5, 1.0))
            fig.tight_layout()
            stem = os.path.join(PLOT_DIR, f'tcc_paper_soil_{kind}_ucs2s_v2_ecmwf_rf{mm}')
            for ext in ('png', 'pdf'):
                fig.savefig(f'{stem}.{ext}', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {stem}.png / .pdf', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['clim', 'extract', 'aggregate', 'plot', 'all'])
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    if a.stage in ('clim', 'all') and not (os.path.exists(CLIM_00Z) and os.path.exists(CLIM_DM)):
        stage_clim()
    if a.stage in ('extract', 'all'):
        stage_extract(a.workers)
    if a.stage in ('aggregate', 'all'):
        stage_aggregate()
    if a.stage in ('plot', 'all'):
        stage_plot()
