"""summary.md tables and first figures for an evaluation tag (plain names; colour-blind-safe colours, no green)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from s41eval.bing_eval import core as K
from s41eval.bing_eval import pipeline as P

COLS = ['#0072B2', '#D55E00', '#CC79A7', '#000000', '#E69F00', '#56B4E9']        # Okabe-Ito without the green
NAMES = {'ucs2s_sstanom': 'UC-S2S SST-anomaly', 'ucs2s_plain': 'plain UC-S2S', 'fuxi': 'FuXi-S2S', 'ifs_rt': 'IFS real-time',
         'ifs_bing': 'IFS reforecast'}
REFN = {'fixed': 'fixed 1991-2020', 'trend': 'trend-adjusted'}
WIN = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w34', 'w56']


def _f(x, d=2):
    return '' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


class Tab:
    def __init__(self, df):
        self.df = df
        key = ['system', 'variable', 'grid', 'period', 'season', 'region', 'window', 'correction', 'members', 'reference', 'metric']
        self.ix = {tuple(r[:-2]): (r[-2], r[-1]) for r in df[key + ['value', 'n_starts']].itertuples(index=False, name=None)}

    def get(self, system, var, metric, window='w34', corr='dc', ref='', members=30, region='global', period='pooled', season='all',
            grid='g15'):
        v = self.ix.get((system, var, grid, period, season, region, window, corr, members, ref, metric))
        return (v[0], v[1]) if v else (float('nan'), 0)


def write_summary(cfg, tag, figures=True):
    import pandas as pd
    od = P.out_dir(cfg, tag)
    df = pd.read_csv(od / 'scores.csv', keep_default_na=False, na_values=['nan', 'NaN', ''])
    df['reference'] = df['reference'].fillna('').astype(str)
    df['members'] = df['members'].astype(int)
    T = Tab(df)
    systems = [s for s in cfg['run_systems'] if s in set(df['system'])]
    fields = [f for f in cfg['fields'] if f in set(df['variable'])]
    prov = json.loads((od / 'provenance.json').read_text()) if (od / 'provenance.json').exists() else {}
    fails = pd.read_csv(od / 'fail_summary.csv') if (od / 'fail_summary.csv').exists() else None
    L = [f'# S41 evaluation `{tag}`: UC-S2S SST-anomaly, plain UC-S2S, FuXi-S2S', '',
         f"Code `{prov.get('git_commit', '?')}` (s41eval.bing_eval), config `{cfg.get('_config_file')}`, written "
         f"{prov.get('written', '?')}. Full table `scores.csv` ({len(df)} rows; columns system, variable, grid, period, season, region, "
         'window, correction, members, reference, metric, value, n_starts, sampling), per-start values `score/<system>/<field>_<grid>/'
         'perstart_<raw|dc>.nc`, maps `maps_<raw|dc>.nc`, inputs `provenance.json`, problems `fail_summary.csv`.', '',
         '**How to read.** Scores against ERA5 on the common conservative 1.5-degree grid (forecasts and ERA5 alike). Windows: week 1 = '
         'days 1-7 ... week 6 = days 36-42; w34 = days 15-28, w56 = days 29-42 (window means). "raw" = as forecast; "dc" = drift '
         'corrected (lead-dependent mean error removed, leave-one-year-out, starts within +-30 days). Skill scores against two '
         'climatologies (30-member same-calendar-date ERA5 ensembles, years 1991-2020): **fixed** (as is) and **trend-adjusted** '
         '(each year shifted along the 1991-2025 trend to the start year). CRPSS > 0 = better than the climatology. ACC = centred '
         'anomaly correlation per grid point across starts, area mean (anomalies vs the reference climatology named). Spread/error '
         '= sqrt((M+1)/M) spread / RMSE of the ensemble mean (1 = well dispersed, < 1 = too confident). 30 members unless stated; '
         '"11" = mean over 10 random 11-member draws. No paired differences, confidence intervals or grades here (the pre-registered '
         'headline is computed separately).', '']
    kp = od / 'key_points.md'
    if kp.exists():
        L += [kp.read_text().strip(), '']
    # coverage
    L += ['## 1. What was scored', '', '| system | starts scored (g15, t2m) | members | verification sampling |', '|---|---|---|---|']
    for sy in systems:
        dj = od / 'score' / sy / 't2m_g15' / 'done_raw.json'
        d = json.loads(dj.read_text()) if dj.exists() else {}
        L.append(f"| {NAMES.get(sy, sy)} | {len(d.get('starts', []))} | {d.get('members', '')} | "
                 f"{'daily means of the calendar day' if cfg['systems'][sy].get('time') == 'day' else '00 UTC states; tp24/olr 24 h ending 00 UTC'} |")
    if fails is not None:
        L += ['', f'Fail summary: {len(fails)} lines (`fail_summary.csv`).']
        if len(fails):
            g = fails.groupby(['system', 'problem']).size()
            L += [f'- {a} / {b}: {n}' for (a, b), n in g.items()]
    cc = sorted((od / 'work' / 'climcheck').glob('*.json')) if (od / 'work' / 'climcheck').exists() else []
    if cc:
        ok = [json.loads(f.read_text()) for f in cc]
        npass = sum(x['pass_self_crpss'] and x['pass_era5_crps0'] and x['pass_30_members'] for x in ok)
        worst = max(abs(v) for x in ok for r in P.REFS if isinstance(x.get(f'{r}_as_forecast_crpss'), dict)
                    for v in x[f'{r}_as_forecast_crpss'][f'vs_{r}'].values() if v is not None)
        L += ['', f'Reference checks (`work/climcheck/`): {npass}/{len(ok)} field-samplings pass (both references load with 30 members; '
              f'climatology as forecast has CRPSS {worst:.1e} at most against itself; ERA5 as forecast has CRPS 0).']
    L.append('')

    def family(title, metric, refs, corrs=('raw', 'dc'), d=2, members=30):
        out = [f'## {title}', '']
        for f in fields:
            hdr = [f'**{f}** ({K.FIELDS[f]["units"]})', '', '| system | reference | corr | ' + ' | '.join(WIN) + ' | n |',
                   '|---|---|---|' + '---|' * (len(WIN) + 1)]
            n0 = len(out)
            out += hdr
            for sy in systems:
                for r in refs:
                    for c in corrs:
                        vals = [T.get(sy, f, metric, w, c, r, members) for w in WIN]
                        if not any(np.isfinite(v[0]) for v in vals):
                            continue
                        out.append(f'| {NAMES.get(sy, sy)} | {REFN.get(r, "-")} | {c} | ' + ' | '.join(_f(v[0], d) for v in vals) +
                                   f' | {vals[6][1]} |')
            if len(out) == n0 + len(hdr):
                del out[n0:]
                continue
            out.append('')
        return out
    L += family('2. CRPSS (fair CRPS skill score), global, pooled 2019-26', 'crpss', ('fixed', 'trend'))
    L += family('3. ACC (centred, anomalies vs the trend-adjusted climatology), global, pooled', 'acc', ('trend',))
    L += family('4. RMSE of the ensemble mean, global, pooled', 'rmse', ('',), d=3)
    L += family('5. Spread/error ratio, global, pooled', 'ssr', ('',))
    L += family('6. Mean bias by week (raw), global, pooled', 'bias', ('',), corrs=('raw',), d=3)
    L += family('7. Tercile RPSS (debiased, Weigel 2007), global, pooled', 'rpss', ('fixed', 'trend'), corrs=('dc',))
    # regions
    L += ['## 8. Regions: weeks 3-4, drift corrected, pooled (CRPSS trend-adjusted / fixed; ACC)', '']
    for f in fields:
        L += [f'**{f}**', '', '| system | ' + ' | '.join(K.RK) + ' |', '|---|' + '---|' * len(K.RK)]
        for sy in systems:
            cells = []
            for rn in K.RK:
                a = T.get(sy, f, 'crpss', ref='trend', region=rn)[0]
                b = T.get(sy, f, 'crpss', ref='fixed', region=rn)[0]
                c = T.get(sy, f, 'acc', ref='trend', region=rn)[0]
                cells.append(f'{_f(a)} / {_f(b)}; {_f(c)}')
            L.append(f'| {NAMES.get(sy, sy)} | ' + ' | '.join(cells) + ' |')
        L.append('')
    # seasons and periods
    L += ['## 9. Seasons (start month) and periods: weeks 3-4, dc, global, CRPSS vs trend-adjusted; ACC', '']
    cols = [('pooled', s) for s in K.SEASONS] + [(p, 'all') for p in ('2019-24', '2025-26', 'pooled_no2019')]
    L += ['| field | system | ' + ' | '.join(f'{p} {s}' for p, s in cols) + ' |', '|---|---|' + '---|' * len(cols)]
    for f in fields:
        for sy in systems:
            cells = []
            for p, s in cols:
                a, n = T.get(sy, f, 'crpss', ref='trend', period=p, season=s)
                c = T.get(sy, f, 'acc', ref='trend', period=p, season=s)[0]
                cells.append(f'{_f(a)}; {_f(c)} ({n})')
            L.append(f'| {f} | {NAMES.get(sy, sy)} | ' + ' | '.join(cells) + ' |')
    L.append('')
    # ensemble size
    L += ['## 10. Ensemble size: weeks 3-4, dc, global, pooled (30 members | mean of 10 random 11-member draws)', '',
          '| field | system | CRPSS trend 30 / 11 | ACC 30 / 11 | RMSE 30 / 11 | spread/error 30 / 11 |', '|---|---|---|---|---|---|']
    for f in fields:
        for sy in systems:
            c = [T.get(sy, f, 'crpss', ref='trend', members=m)[0] for m in (30, 11)]
            a = [T.get(sy, f, 'acc', ref='trend', members=m)[0] for m in (30, 11)]
            r = [T.get(sy, f, 'rmse', members=m)[0] for m in (30, 11)]
            s = [T.get(sy, f, 'ssr', members=m)[0] for m in (30, 11)]
            L.append(f'| {f} | {NAMES.get(sy, sy)} | {_f(c[0])} / {_f(c[1])} | {_f(a[0])} / {_f(a[1])} | {_f(r[0], 3)} / {_f(r[1], 3)} | '
                     f'{_f(s[0])} / {_f(s[1])} |')
    L.append('')
    # diagnosis
    L += ['## 11. Why is weeks 3-4 CRPSS <= 0? (plan §3 D; inflation = ensemble scaled about its mean to spread/error 1 per grid point, '
          'a diagnostic, not a product)', '',
          '| field | system | corr | CRPSS fixed | CRPSS trend | inflated fixed | inflated trend | spread/error | reading |', '|---|---|---|---|---|---|---|---|---|']
    for f in fields:
        for sy in systems:
            for c in ('raw', 'dc'):
                cf, ct = (T.get(sy, f, 'crpss', corr=c, ref=r)[0] for r in ('fixed', 'trend'))
                inf_f, inf_t = (T.get(sy, f, 'crpss_inflated', corr=c, ref=r)[0] for r in ('fixed', 'trend'))
                sr = T.get(sy, f, 'ssr', corr=c)[0]
                why = diagnose(cf, ct, inf_f, inf_t)
                if c == 'raw':
                    dcf, dct = (T.get(sy, f, 'crpss', corr='dc', ref=r)[0] for r in ('fixed', 'trend'))
                    if (cf <= 0 and dcf > 0) or (ct <= 0 and dct > 0):
                        why = 'drift (drift correction lifts it above 0)' + ('' if why.startswith(('no skill', 'not skil')) else '; ' + why)
                L.append(f'| {f} | {NAMES.get(sy, sy)} | {c} | {_f(cf)} | {_f(ct)} | {_f(inf_f)} | {_f(inf_t)} | {_f(sr)} | {why} |')
    L.append('')
    # 3-degree sensitivity
    g30 = [f for f in fields if ((df['variable'] == f) & (df['grid'] == 'g30')).any()]
    if g30:
        L += ['## 12. Resolution sensitivity: conservative 3 degrees vs 1.5 degrees (weeks 3-4, dc, pooled; CRPSS trend / ACC)', '',
              '| field | region | ' + ' | '.join(f'{NAMES.get(s, s)} 1.5 / 3' for s in systems) + ' | ranking 1.5 | ranking 3 | same? |',
              '|---|---|' + '---|' * (len(systems) + 3)]
        nchange = 0
        for f in g30:
            for rn in ('global', 'tropics', 'nh_extratropics', 'conus'):
                for met, ref in (('crpss', 'trend'), ('acc', 'trend')):
                    v15 = {s: T.get(s, f, met, ref=ref, region=rn)[0] for s in systems}
                    v30 = {s: T.get(s, f, met, ref=ref, region=rn, grid='g30')[0] for s in systems}
                    r15 = ' > '.join(sorted(systems, key=lambda s: -np.nan_to_num(v15[s], nan=-9)))
                    r30 = ' > '.join(sorted(systems, key=lambda s: -np.nan_to_num(v30[s], nan=-9)))
                    same = r15 == r30
                    nchange += not same
                    L.append(f'| {f} {met} | {rn} | ' + ' | '.join(f'{_f(v15[s])} / {_f(v30[s])}' for s in systems) +
                             f' | {_short(r15)} | {_short(r30)} | {"yes" if same else "**no**"} |')
        L += ['', f'Rankings that change between 1.5 and 3 degrees: {nchange}.', '']
    reg = od / 'regression_v0.json'
    if reg.exists():
        r = json.loads(reg.read_text())
        L += ['## 13. Regression against v0_ucs2s', '', f"{r.get('summary')}", '']
    figs = []
    if figures:
        try:
            figs = make_figures(cfg, tag, T, systems, fields)
        except Exception as e:  # noqa: BLE001
            L += [f'(figures failed: {type(e).__name__}: {e})']
    if figs:
        L += ['## Figures', ''] + [f'- `figs/{f}`' for f in figs] + ['']
    (od / 'summary.md').write_text('\n'.join(L))
    return od / 'summary.md'


def _short(r):
    for k, v in (('ucs2s_sstanom', 'SST'), ('ucs2s_plain', 'plain'), ('fuxi', 'FuXi')):
        r = r.replace(k, v)
    return r


def diagnose(cf, ct, inf_f, inf_t, tol=0.0):
    """plain-language reading of plan §3 D for one row."""
    if not all(np.isfinite(x) for x in (cf, ct)):
        return ''
    if cf > 0 and ct > 0:
        return 'skilful vs both'
    parts = []
    if (cf > 0) != (ct > 0):
        parts.append('reference matters (' + ('positive vs fixed only' if cf > 0 else 'positive vs trend-adjusted only') + ')')
    if np.isfinite(inf_t) and ct <= 0 < inf_t or np.isfinite(inf_f) and cf <= 0 < inf_f:
        parts.append('under-dispersion (inflation lifts it above 0)')
    if not parts:
        parts.append('no skill even with inflation' if (np.isfinite(inf_t) and inf_t <= 0) else 'not skilful')
    return '; '.join(parts)


def make_figures(cfg, tag, T, systems, fields):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import xarray as xr
    od = P.out_dir(cfg, tag)
    fd = P.mkdir(od / 'figs')
    lsm = np.load(cfg['verification']['lsm']) if Path(cfg['verification'].get('lsm', '')).exists() else None
    out = []
    # 1. CRPSS / ACC by window, per field
    for metric, ref in (('crpss', 'trend'), ('crpss', 'fixed'), ('acc', 'trend')):
        n = len(fields)
        fig, axs = plt.subplots(int(math.ceil(n / 4)), 4, figsize=(15, 3.0 * math.ceil(n / 4)), squeeze=False)
        for a, f in zip(axs.ravel(), fields):
            for i, sy in enumerate(systems):
                for c, ls in (('dc', '-'), ('raw', ':')):
                    y = [T.get(sy, f, metric, w, c, ref)[0] for w in WIN[:6]]
                    a.plot(range(1, 7), y, ls, color=COLS[i], label=f'{NAMES.get(sy, sy)} {c}' if f == fields[0] else None)
            a.axhline(0, color='0.5', lw=0.6)
            a.set_title(f, fontsize=9)
            a.set_xticks(range(1, 7))
            if metric == 'crpss':
                a.set_ylim(max(-1.0, a.get_ylim()[0]), min(1.0, a.get_ylim()[1]))
        for a in axs.ravel()[n:]:
            a.axis('off')
        axs[0, 0].legend(fontsize=6)
        fig.suptitle(f'{metric.upper()} vs {REFN[ref]} climatology by week (global, pooled 2019-26; solid dc, dotted raw)', fontsize=10)
        fig.tight_layout()
        nm = f'{metric}_{ref}_by_week.png'
        fig.savefig(fd / nm, dpi=110)
        plt.close(fig)
        out.append(nm)
    # 2. maps weeks 3-4: CRPSS trend dc, ACC dc, bias raw
    for f in fields:
        for key, corr, vmax, cm in (('crpss_trend', 'dc', 0.5, 'RdBu_r'), ('acc_trend', 'dc', 1.0, 'RdBu_r'), ('bias', 'raw', None, 'RdBu_r')):
            ds = {}
            for sy in systems:
                p = od / 'score' / sy / f'{f}_g15' / f'maps_{corr}.nc'
                if p.exists():
                    with xr.open_dataset(p) as x:
                        if f'{key}_{corr}' in x:
                            ds[sy] = x[f'{key}_{corr}'].sel(subset='pooled|all', window='w34').values
            if not ds:
                continue
            vm = vmax or float(np.nanpercentile(np.abs(np.concatenate([v.ravel() for v in ds.values()])), 98))
            fig, axs = plt.subplots(1, len(ds), figsize=(4.6 * len(ds) + 1, 2.7), squeeze=False)
            lat, lon = K.LAT['g15'][::-1], K.LON['g15']
            for a, (sy, x) in zip(axs[0], ds.items()):
                im = a.pcolormesh(lon, lat, x, cmap=cm, vmin=-vm, vmax=vm, shading='nearest')
                if lsm is not None:
                    a.contour(K.LON['g15'], K.LAT['g15'], lsm, levels=[0.5], colors='0.25', linewidths=0.5)
                a.set_title(NAMES.get(sy, sy), fontsize=9)
                a.set_aspect('equal')
            fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label=f'{key} ({corr})')
            fig.suptitle(f'{f}: weeks 3-4 {key.replace("_", " vs ")} ({corr}), pooled 2019-26', fontsize=10)
            nm = f'map_{f}_{key}_{corr}_w34.png'
            fig.savefig(fd / nm, dpi=100, bbox_inches='tight')
            plt.close(fig)
            out.append(nm)
    # 3. rank histograms and reliability (weeks 3-4, global, pooled, dc)
    for f in fields:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        any_ = False
        for i, sy in enumerate(systems):
            p = od / 'score' / sy / f'{f}_g15' / 'perstart_dc.nc'
            if not p.exists():
                continue
            with xr.open_dataset(p) as x:
                h = np.nansum(x['rank_hist_dc'].sel(window='w34', region='global').values, 0)
                rel = np.nansum(x['reliability_w34_dc'].sel(region='global', reference='trend', event='above_upper_tercile').values, 0)
            if h.sum() <= 0:
                continue
            any_ = True
            axs[0].plot(np.arange(len(h)) / (len(h) - 1), h / h.sum() * len(h), color=COLS[i], label=NAMES.get(sy, sy))
            with np.errstate(invalid='ignore', divide='ignore'):
                axs[1].plot(rel[:, 1] / rel[:, 0], rel[:, 2] / rel[:, 0], 'o-', color=COLS[i], ms=3)
        if not any_:
            plt.close(fig)
            continue
        axs[0].axhline(1, color='0.5', lw=0.6)
        axs[0].set(xlabel='normalised rank of ERA5', ylabel='relative frequency', title=f'{f}: rank histogram, weeks 3-4 (dc)')
        axs[0].legend(fontsize=7)
        axs[1].plot([0, 1], [0, 1], color='0.5', lw=0.6)
        axs[1].set(xlabel='forecast probability', ylabel='observed frequency', title='reliability, upper tercile (trend-adjusted)',
                   xlim=(0, 1), ylim=(0, 1))
        fig.tight_layout()
        nm = f'rank_reliability_{f}_w34.png'
        fig.savefig(fd / nm, dpi=110)
        plt.close(fig)
        out.append(nm)
    return out


def compare_v0(cfg, tag, v0_csv, tol=1e-3):
    """regression: rows of v0_ucs2s (bing_scores) vs this tag (run with v0 settings) -> regression_v0.json/.csv."""
    import pandas as pd
    od = P.out_dir(cfg, tag)
    v0 = pd.read_csv(v0_csv)
    new = pd.read_csv(od / 'scores.csv', keep_default_na=False, na_values=['nan', 'NaN', ''])
    new['reference'] = new['reference'].fillna('').astype(str)
    new = new[(new.grid == 'g15') & (new.period == 'pooled') & (new.season == 'all') & (new.members == 30)]
    mmap = {'acc_spatial': 'acc_pattern'}
    refm = {'crps_clim': 'trend', 'crpss': 'trend', 'acc': 'trend', 'acc_pattern': 'trend'}
    ix = {(r.system, r.variable, r.region, r.window, r.correction, r.metric, r.reference): r.value for r in new.itertuples()}
    rows = []
    for r in v0.itertuples():
        if r.correction not in ('raw', 'dc') or r.region not in K.RK or r.variable == 'olr':
            continue
        m = mmap.get(r.metric, r.metric)
        k = (r.system, r.variable, r.region, r.window, r.correction, m, refm.get(m, ''))
        if k not in ix:
            continue
        rows.append(dict(system=r.system, variable=r.variable, region=r.region, window=r.window, correction=r.correction, metric=m,
                         v0=r.value, new=ix[k], absdiff=abs(r.value - ix[k]) if np.isfinite(r.value) and np.isfinite(ix[k]) else
                         (0.0 if not np.isfinite(r.value) and not np.isfinite(ix[k]) else float('inf'))))
    d = pd.DataFrame(rows)
    d.to_csv(od / 'regression_v0.csv', index=False)
    worst = d.groupby(['variable', 'metric'])['absdiff'].max().reset_index() if len(d) else d
    n_bad = int((d['absdiff'] > tol).sum()) if len(d) else -1
    summ = f'{len(d)} v0 rows compared (non-OLR; raw and dc; regions in both): {n_bad} differ by more than {tol}; max abs diff ' \
           f'{d["absdiff"].max() if len(d) else float("nan"):.2e}.'
    res = dict(summary=summ, n=len(d), n_over_tol=n_bad, tol=tol, worst=worst.to_dict('records') if len(d) else [])
    (od / 'regression_v0.json').write_text(json.dumps(res, indent=1, default=float))
    return res
