#!/usr/bin/env python
"""S41 v1.5 helper-B figures and key numbers (light: plotting from cached arrays; run on a quiet login node).

    python scripts/bing_weak_figs.py <fig> [<fig> ...]      figs: b01 b02 b03 b04 b05 b06 b07 b08 all
Every number written to ``slides/key_numbers/helperB.md`` comes from the file named next to it.
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
from s41eval import bing_weak as B                      # noqa: E402
from s41eval.bing_eval import core as K                 # noqa: E402
from s41eval.sci import helperB_style as S                      # noqa: E402

V15 = S.V15
HB = V15 / 'helperB'
SC = V15 / 'score'
SYSL = list(S.SYS)
KN = S.SLIDES / 'key_numbers' / 'helperB.md'
UNITS = {'t2m': 'K', 'sst': 'K', 'tp24': 'mm/day', 'olr': 'W m⁻²', 'z500': 'm', 'u250': 'm/s', 'msl': 'hPa', 'u850': 'm/s',
         'z50': 'm', 'u50': 'm/s', 't850': 'K'}
FOOT_E = ('S41 v1.5 · per-start ensemble-mean files (ensstats): 402 UC-S2S starts (2019-01-07 … 2026-09-14; 2026-09-21 has none) × 30 members, '
          'FuXi-S2S 404 × 30 · 1.5° grid · ERA5 · IFS not yet in · exploratory (not pre-registered)')
SCALE = {'z500': 1 / 9.80665, 'z50': 1 / 9.80665, 'msl': 0.01}
NAME = {'t2m': '2-m temperature', 'sst': 'SST', 'tp24': 'precipitation', 'olr': 'OLR (positive up)', 'z500': 'z500', 'u250': 'u250',
        'msl': 'MSLP', 'u850': 'u850', 't850': 't850'}
LIM = {'t2m': 4, 'sst': 2, 'tp24': 3, 'olr': 20, 'z500': 60, 'u250': 6, 'msl': 6}
FOOT = ('S41 v1.5 · 403 UC-S2S starts (2019-01-07 … 2026-09-21) × 30 members, FuXi-S2S 404 × 30 · common 1.5° grid · ERA5 verification · '
        'IFS not yet in · exploratory (not pre-registered)')


def kn(section, lines):
    """replace the section ``## <section>`` of the key-numbers file."""
    KN.parent.mkdir(parents=True, exist_ok=True)
    txt = KN.read_text() if KN.exists() else '# Key numbers, helper B (S41 W3/W4; exploratory unless marked)\n'
    parts = txt.split('\n## ')
    head, secs = parts[0], {p.split('\n', 1)[0]: p for p in parts[1:]}
    secs[section] = section + '\n' + '\n'.join(f'- {l}' for l in lines) + '\n'
    KN.write_text(head.rstrip('\n') + '\n' + ''.join('\n## ' + secs[k].rstrip('\n') + '\n' for k in sorted(secs)))


def rows(sys, field, corr):
    f = SC / sys / f'{field}_g15' / f'rows_{corr}.csv'
    with open(f) as fh:
        return list(csv.DictReader(fh)), f


def pick(rs, **kw):
    out = [r for r in rs if all(r[k] == str(v) for k, v in kw.items())]
    return out


def maps(sys, field, corr='raw'):
    import xarray as xr
    f = SC / sys / f'{field}_g15' / f'maps_{corr}.nc'
    return xr.open_dataset(f), f


def area_stats(m):
    """bias map on lat-descending g15 -> (cos-lat mean, cos-lat RMS) over finite cells."""
    lat = np.linspace(90, -90, 121)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 240))
    ok = np.isfinite(m)
    return float((m[ok] * w[ok]).sum() / w[ok].sum()), float(np.sqrt((m[ok] ** 2 * w[ok]).sum() / w[ok].sum()))


def mapax(ax, m, lim, cmap, title=None):
    lat = np.linspace(90, -90, 121)
    lon = np.arange(240) * 1.5
    im = ax.pcolormesh(lon, lat, m, vmin=-lim, vmax=lim, cmap=cmap, shading='nearest', rasterized=True)
    S.coast(ax, lw=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-80, 85)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=15)
    return im


# ------------------------------------------------------------------ B02 bias maps
def b02():
    p = S.plt()
    fields = ['t2m', 'sst', 'tp24', 'z500']
    f, axs = S.fig(len(fields), 3, figsize=(S.W, S.H), gridspec_kw=dict(left=0.02, right=0.9, top=0.94, bottom=0.07, wspace=0.03, hspace=0.12))
    lines = []
    for i, fd in enumerate(fields):
        sc = SCALE.get(fd, 1.0)
        cmap = 'RdBu' if fd == 'tp24' else S.DIV
        for j, sy in enumerate(SYSL):
            ds, fn = maps(sy, fd)
            m = ds['bias_raw'].sel(subset='pooled|all', window='w34').values * sc
            mu, rms = area_stats(m)
            im = mapax(axs[i, j], m, LIM[fd], cmap, S.SYS[sy] if i == 0 else None)
            axs[i, j].text(0.01, 0.02, f'mean {mu:+.2f}, RMS {rms:.2f}', transform=axs[i, j].transAxes, fontsize=12,
                           bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
            lines.append(f'B02 weeks 3–4 raw bias {fd} {S.SYS[sy]}: area mean {mu:+.3f} {UNITS[fd]}, area RMS {rms:.3f} '
                         f'(cos-lat over valid cells) · `{fn}` bias_raw subset pooled|all window w34')
        cax = f.add_axes([0.915, axs[i, 2].get_position().y0 + 0.01, 0.012, axs[i, 2].get_position().height - 0.02])
        cb = f.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(f'{NAME[fd]} ({UNITS[fd]})', fontsize=12)
    S.save(f, 'B02', 'bias_maps_w34', FOOT + ' · raw ensemble mean − ERA5, weeks 3–4 (days 15–28), all seasons; SST v1.5 (land masked)')
    rs = []
    for fd in ['t2m', 'sst', 'tp24', 'z500', 'olr', 'u250', 'msl']:
        for sy in SYSL:
            r, fn = rows(sy, fd, 'raw')
            for reg in ('global', 'tropics', 'nh_extratropics', 'conus'):
                for w in ('w1', 'w34', 'w56'):
                    q = pick(r, period='pooled', season='all', region=reg, window=w, metric='bias', members=30)
                    if q:
                        rs.append(f'raw bias {fd} {S.SYS[sy]} {reg} {w}: {float(q[0]["value"]) * SCALE.get(fd, 1):+.3f} {UNITS[fd]} · '
                                  f'`{fn}` period=pooled season=all region={reg} window={w} metric=bias members=30')
    S.caption('B02', 'Weeks 3–4 raw bias maps (t2m, SST, precipitation, z500), three systems: UC-S2S has a broad warm bias (t2m and '
              'SST, both checkpoints alike) and a wet tropics, FuXi is close to ERA5 · numbers in key_numbers/helperB.md §B02 · exploratory · '
              '`v1.5/score/<sys>/<field>_g15/maps_raw.nc`')
    kn('B02', lines + rs)
    # extras: every field, seasons x windows, UC-S2S SST-anomaly and FuXi; and the three systems for olr, u250, msl
    for sy in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
        for fd in ['t2m', 'sst', 'tp24', 'olr', 'z500', 'u250', 'msl']:
            ds, fn = maps(sy, fd)
            sc = SCALE.get(fd, 1.0)
            cmap = 'RdBu' if fd == 'tp24' else S.DIV
            f, axs = S.fig(4, 3, gridspec_kw=dict(left=0.06, right=0.9, top=0.94, bottom=0.07, wspace=0.03, hspace=0.12))
            for i, sn in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
                for j, w in enumerate(['w1', 'w34', 'w56']):
                    m = ds['bias_raw'].sel(subset=f'pooled|{sn}', window=w).values * sc
                    mu, rms = area_stats(m)
                    im = mapax(axs[i, j], m, LIM[fd], cmap, {'w1': 'week 1', 'w34': 'weeks 3–4', 'w56': 'weeks 5–6'}[w] if i == 0 else None)
                    axs[i, j].text(0.01, 0.02, f'{mu:+.2f} / RMS {rms:.2f}', transform=axs[i, j].transAxes, fontsize=11,
                                   bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
                axs[i, 0].text(-0.03, 0.5, sn, transform=axs[i, 0].transAxes, rotation=90, va='center', ha='right')
            cax = f.add_axes([0.915, 0.2, 0.012, 0.6])
            f.colorbar(im, cax=cax).set_label(f'{NAME[fd]} raw bias ({UNITS[fd]})')
            S.save(f, f'B02x_{sy}', f'{fd}_bias_season_week', FOOT + f' · {S.SYS[sy]} · seasons by start month', slide=False)


# ------------------------------------------------------------------ B06 ensemble quality
def b06():
    p = S.plt()
    fields = ['t2m', 'z500', 'tp24', 'sst', 'olr', 'u850']
    f, axs = S.fig(2, 3, sharex=True, gridspec_kw=dict(left=0.07, right=0.98, top=0.93, bottom=0.17, wspace=0.25, hspace=0.3))
    lines = []
    for k, fd in enumerate(fields):
        ax = axs.flat[k]
        for sy in SYSL:
            r, fn = rows(sy, fd, 'dc')
            for reg, ls in (('global', '-'), ('tropics', '--')):
                y = []
                for w in ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']:
                    q = pick(r, period='pooled', season='all', region=reg, window=w, metric='ssr', members=30)
                    y.append(float(q[0]['value']) if q else np.nan)
                ax.plot(range(1, 7), y, ls, color=S.SYSC[sy], lw=2.2, marker='o' if reg == 'global' else None, ms=5)
                for w in ('w1', 'w34'):
                    q = pick(r, period='pooled', season='all', region=reg, window=w, metric='ssr', members=30)
                    if q:
                        lines.append(f'spread/error (M+1)/M {fd} {S.SYS[sy]} {reg} {w}: {float(q[0]["value"]):.3f} · `{fn}` '
                                     f'period=pooled season=all region={reg} window={w} metric=ssr members=30 (drift-corrected)')
        ax.axhline(1, color=S.GREY, lw=1)
        ax.set_title(NAME[fd])
        ax.set_ylim(0.2, 1.3)
        if k >= 3:
            ax.set_xlabel('lead week')
        if k % 3 == 0:
            ax.set_ylabel('spread / error')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=2.5, label=S.SYS[s]) for s in SYSL] + \
        [Line2D([], [], color='k', ls='-', marker='o', label='globe'), Line2D([], [], color='k', ls='--', label='tropics')]
    f.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.045), fontsize=13)
    S.save(f, 'B06a', 'spread_error', FOOT + ' · drift-corrected (LOYO) · spread/error = sqrt((M+1)/M · mean var) / RMSE, 1 = calibrated')
    kn('B06a', lines)
    # rank histograms (cos-lat weighted, drift-corrected, global)
    import xarray as xr
    fields = ['t2m', 'z500', 'tp24', 'sst']
    f, axs = S.fig(2, 4, sharey=True, gridspec_kw=dict(left=0.07, right=0.98, top=0.92, bottom=0.2, wspace=0.08, hspace=0.35))
    lines = []
    for j, fd in enumerate(fields):
        for i, w in enumerate(['w1', 'w34']):
            ax = axs[i, j]
            for sy in SYSL:
                fn = SC / sy / f'{fd}_g15' / 'perstart_dc.nc'
                ds = xr.open_dataset(fn)
                h = ds['rank_hist_dc'].sel(window=w, region='global').values
                h = np.nansum(h, 0)
                pr, dl, u = B.rank_flatness(h)
                M1 = len(pr)
                ax.step(np.arange(M1), pr * M1, where='mid', color=S.SYSC[sy], lw=2)
                lines.append(f'rank histogram {fd} {S.SYS[sy]} global {w}: outer-bin ratio {u:.2f} (1 = flat), first bin {pr[0] * M1:.2f}, '
                             f'last bin {pr[-1] * M1:.2f} (× uniform) · `{fn}` rank_hist_dc summed over starts')
            ax.axhline(1, color=S.GREY, lw=1)
            ax.set_title(f'{NAME[fd]}, {"week 1" if w == "w1" else "weeks 3–4"}', fontsize=15)
            ax.set_xticks([0, 15, 30])
            if j == 0:
                ax.set_ylabel('frequency / uniform')
            ax.set_yscale('log')
            ax.set_ylim(0.15, 30)
            ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20], ['0.2', '0.5', '1', '2', '5', '10', '20'])
    f.text(0.52, 0.125, 'rank of ERA5 among the 30 members', ha='center')
    h = [Line2D([], [], color=S.SYSC[s], lw=2.5, label=S.SYS[s]) for s in SYSL]
    f.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035))
    S.save(f, 'B06b', 'rank_histograms', FOOT + ' · drift-corrected · counts cos-lat weighted · random tie-breaking · ERA5 without '
           'observation error (outer bins inflated for any ensemble)')
    kn('B06b', lines)
    # reliability diagrams, terciles weeks 3-4
    fields = ['t2m', 'tp24']
    lines = []
    for ref in ('fixed', 'trend'):
        f, axs = S.fig(1, 4, gridspec_kw=dict(left=0.06, right=0.98, top=0.88, bottom=0.27, wspace=0.28))
        for j, (fd, ev) in enumerate([(a, b) for a in fields for b in ('below_lower_tercile', 'above_upper_tercile')]):
            ax = axs[j]
            ax.plot([0, 1], [0, 1], color=S.GREY, lw=1)
            for sy in SYSL:
                fn = SC / sy / f'{fd}_g15' / 'perstart_dc.nc'
                ds = xr.open_dataset(fn)
                c = np.nansum(ds['reliability_w34_dc'].sel(region='global', reference=ref, event=ev).values, 0)
                pm, of, wf, rel, res, ob = B.reliability_curve(c)
                ok = wf > 0.002
                ax.plot(pm[ok], of[ok], '-o', color=S.SYSC[sy], lw=2, ms=4)
                ax.text(0.03, 0.95 - 0.08 * SYSL.index(sy), f'REL {rel * 1000:.1f}·10⁻³', color=S.SYSC[sy], transform=ax.transAxes,
                        fontsize=12, va='top')
                lines.append(f'reliability w34 {fd} {ev} ref={ref} {S.SYS[sy]} global: REL {rel:.4f}, RES {res:.4f}, base rate {ob:.3f}, '
                             f'forecast p>=0.6 used {wf[6:].sum():.3f} of area-weight · `{fn}` reliability_w34_dc summed over starts')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'{NAME[fd]}\n{"lower" if ev.startswith("below") else "upper"} tercile', fontsize=15)
            ax.set_xlabel('forecast probability')
            if j == 0:
                ax.set_ylabel('observed frequency')
        f.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.06))
        S.save(f, 'B06c' if ref == 'fixed' else 'B06c_trend', 'reliability_w34' + ('' if ref == 'fixed' else '_trendref'),
               FOOT + f' · weeks 3–4, drift-corrected, tercile edges from the {"fixed 1991–2020" if ref == "fixed" else "trend-adjusted"} '
               '30-member ERA5 reference · cos-lat weighted, global', slide=(ref == 'fixed'))
    kn('B06c', lines)


# ------------------------------------------------------------------ aggregation of the per-start outputs (job 'start' phase)
AGG = HB / 'cache'
KEYS = ['ua_bias', 'ua_mse', 'ua_var', 'sfc_bias', 'sfc_mse', 'sfc_var', 'ua_zm_em', 'ua_zm_O', 'sfc_zm_em', 'sfc_zm_O', 'hist_model',
        'hist_era5']


def agg(sy):
    f = AGG / f'agg_{sy}.npz'
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return {k: z[k] for k in z.files}
    files = sorted((HB / 'start' / sy).glob('*.npz'))
    out = defaultdict(list)
    starts = []
    for p in files:
        z = np.load(p)
        starts.append(p.stem)
        for k in KEYS:
            out[k].append(z[k])
        for k in z.files:
            if k.startswith('spec_'):
                out[k].append(z[k])
        # ERA5 spectra on the valid dates of this start (same convention as the system)
        d0 = date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:]))
        es = defaultdict(list)
        for k in range(K.NLEAD):
            g = HB / 'era5_spec' / f'{(d0 + __import__("datetime").timedelta(days=k)):%Y%m%d}.npz'
            e = np.load(g) if g.exists() else None
            for b in B.BANDS:
                for v in ('z500', 'ke250'):
                    if sy == 'fuxi':
                        kk = f'{v}_g15day_{b}' if v == 'z500' else f'{v}_g15_{b}'
                    else:
                        kk = f'{v}_1deg_{b}'
                    n = 121 if sy == 'fuxi' else 181
                    es[f'era5spec_{v}_{b}'].append(e[kk] if (e is not None and kk in e.files) else np.full(n, np.nan))
        for k, v in es.items():
            out[k].append(np.stack(v))
    res = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    res['starts'] = np.array(starts)
    AGG.mkdir(parents=True, exist_ok=True)
    np.savez(f, **res)
    return res


def dstarts(A):
    return [date(int(s[:4]), int(s[4:6]), int(s[6:])) for s in A['starts']]


def band(ax, x, Y, c, lw=2.2, ls='-', label=None, alpha=0.15):
    m = np.nanmean(Y, 0)
    lo, hi = np.nanpercentile(Y, 10, 0), np.nanpercentile(Y, 90, 0)
    ax.fill_between(x, lo, hi, color=c, alpha=alpha, lw=0)
    ax.plot(x, m, ls, color=c, lw=lw, label=label)
    return m


LEADS = np.arange(K.NLEAD)
L42 = LEADS <= 42


def b01():
    S.plt()
    f, axs = S.fig(1, 2, sharey=True, gridspec_kw=dict(left=0.08, right=0.98, top=0.9, bottom=0.22, wspace=0.08))
    j = B.SFC.index('sst')
    lines = []
    for i, (reg, ri) in enumerate((('global ocean', 0), ('tropical ocean (20°S–20°N)', 1))):
        ax = axs[i]
        for sy in SYSL:
            A = agg(sy)
            Y = A['sfc_bias'][:, :, j, ri][:, L42]
            m = band(ax, LEADS[L42], Y, S.SYSC[sy], label=S.SYS[sy])
            for d in (0, 1, 7, 14, 21, 28, 42):
                lines.append(f'SST bias {reg} {S.SYS[sy]} lead {d}: mean {m[d]:+.3f} K (10–90 % over starts '
                             f'{np.nanpercentile(Y[:, d], 10):+.2f} … {np.nanpercentile(Y[:, d], 90):+.2f}; n = {np.isfinite(Y[:, d]).sum()}) · '
                             f'`helperB/cache/agg_{sy}.npz` sfc_bias[:, {d}, sst, {B.REG4[ri]}]')
        ax.axhline(0, color=S.GREY, lw=1)
        ax.set_title(reg)
        ax.set_xlabel('lead day')
        ax.set_xticks([0, 7, 14, 21, 28, 35, 42])
    axs[0].set_ylabel('raw SST bias vs ERA5 (K)')
    axs[0].legend(loc='upper right', frameon=False)
    S.save(f, 'B01', 'sst_bias_by_lead', FOOT_E + ' · raw ensemble mean (ensstats, 30 members), land masked at 1° before the 1.5° remap (v1.5 fix) · '
           'line = mean over starts, band = 10–90 % of starts · FuXi verified against daily-mean ERA5, UC-S2S against 00 UTC')
    kn('B01', lines)


def b03():
    S.plt()
    panels = [('sfc', 't2m', 'globe 2-m temperature (K)', 1.0), ('sfc', 'sst', 'global-ocean SST (K)', 1.0),
              ('sfc', 'tp24', 'globe precipitation (mm/day)', 1.0), ('sfc', 'olr', 'globe OLR (W m⁻²)', 1.0),
              ('sfc', 'msl', 'globe MSLP (hPa)', 0.01), ('ua', ('z', 500), 'globe z500 (m)', 1 / 9.80665)]
    f, axs = S.fig(2, 3, sharex=True, gridspec_kw=dict(left=0.07, right=0.98, top=0.93, bottom=0.17, wspace=0.28, hspace=0.3))
    lines = []
    for k, (kind, var, title, sc) in enumerate(panels):
        ax = axs.flat[k]
        for sy in SYSL:
            A = agg(sy)
            if kind == 'sfc':
                Y = A['sfc_bias'][:, :, B.SFC.index(var), 0] * sc
                lab = var
            else:
                Y = A['ua_bias'][:, :, B.UA.index(var[0]), B.LEVELS.index(var[1]), 0] * sc
                lab = f'{var[0]}{var[1]}'
            Y = Y.copy()
            if lab in ('tp24', 'olr'):
                Y[:, 0] = np.nan                                     # lead 0 has no 24-h flux in the UC-S2S files
            m = band(ax, LEADS[L42], Y[:, L42], S.SYSC[sy], label=S.SYS[sy], alpha=0.12)
            for d in (1, 7, 14, 28, 42):
                lines.append(f'global-mean raw drift {lab} {S.SYS[sy]} lead {d}: {m[d]:+.3f} · `helperB/cache/agg_{sy}.npz` '
                             f'{"sfc_bias" if kind == "sfc" else "ua_bias"} region global (× {sc:g})')
        ax.axhline(0, color=S.GREY, lw=1)
        ax.set_title(title, fontsize=16)
        if k >= 3:
            ax.set_xlabel('lead day')
        ax.set_xticks([0, 7, 14, 21, 28, 35, 42])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=2.5, label=S.SYS[s]) for s in SYSL]
    f.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.045))
    S.save(f, 'B03', 'global_drift_curves', FOOT_E + ' · raw ensemble mean − ERA5, cos-lat global mean by lead day; band = 10–90 % of starts · '
           'package samplings (UC-S2S 00 UTC / 24 h ending 00 UTC, FuXi daily means)')
    kn('B03', lines)


def b04():
    """vertical bias and drift-corrected RMSE profiles (window means) from the 'profile' phase."""
    S.plt()
    lines = []
    f, axs = S.fig(2, 4, sharey=True, gridspec_kw=dict(left=0.07, right=0.98, top=0.94, bottom=0.23, wspace=0.12, hspace=0.42))
    lev = np.array(B.LEVELS)
    var = [('t', 'T (K)', 1.0), ('u', 'u (m/s)', 1.0), ('z', 'z (m)', 1 / 9.80665), ('q', 'q (g/kg)', 1000.0)]
    for j, (v, lab, sc) in enumerate(var):
        for sy in SYSL:
            fn = HB / 'profile' / f'{sy}_{v}.npz'
            if not fn.exists():
                continue
            P = np.load(fn)
            wi = list(P['windows']).index('w34')
            w1 = list(P['windows']).index('w1')
            b = P['bias_raw'][0, :, :, 0] * sc
            r_raw, r_dc = P['rmse_raw'][0, :, :, 0] * sc, P['rmse_dc'][0, :, :, 0] * sc
            ok = np.isfinite(b[wi])
            axs[0, j].plot(b[wi][ok], lev[ok], '-o', color=S.SYSC[sy], lw=2, ms=3)
            axs[0, j].plot(b[w1][ok], lev[ok], ':', color=S.SYSC[sy], lw=1.8)
            axs[1, j].plot(r_dc[wi][ok], lev[ok], '-o', color=S.SYSC[sy], lw=2, ms=3)
            axs[1, j].plot(r_raw[wi][ok], lev[ok], '--', color=S.SYSC[sy], lw=1.5)
            for L in (10, 50, 100, 250, 500, 850, 1000):
                li = B.LEVELS.index(L)
                if np.isfinite(b[wi, li]):
                    lines.append(f'profile global w34 {v}{L} {S.SYS[sy]}: raw bias {b[wi, li]:+.3f}, RMSE raw {r_raw[wi, li]:.3f}, '
                                 f'drift-corrected {r_dc[wi, li]:.3f}; w1 bias {b[w1, li]:+.3f} (× {sc:g}) · `{fn}` season all region global')
        axs[0, j].axvline(0, color=S.GREY, lw=1)
        axs[0, j].set_title(lab, fontsize=16)
        axs[1, j].set_xlabel(f'RMSE {lab}')
        axs[0, j].set_xlabel(f'bias {lab}')
    for a in axs[:, 0]:
        a.set_yscale('log')
        a.set_ylim(1000, 5)
        a.set_yticks([1000, 500, 250, 100, 50, 20, 10, 5], ['1000', '500', '250', '100', '50', '20', '10', '5'])
        a.set_ylabel('hPa')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=2.5, label=S.SYS[s]) for s in SYSL] + \
        [Line2D([], [], color='k', marker='o', label='weeks 3–4 (bottom: drift-corrected)'),
         Line2D([], [], color='k', ls=':', label='week 1 bias'), Line2D([], [], color='k', ls='--', label='raw RMSE')]
    f.legend(handles=h, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.075), fontsize=12)
    S.save(f, 'B04', 'vertical_profiles', FOOT_E.replace(' · exploratory', ' · ERA5 00 UTC (UC-S2S input h5) for all levels; FuXi '
           '(daily means, 13 levels) vs 00 UTC → caveat below 850 hPa · exploratory'))
    kn('B04', lines)
    # extra: zonal-mean bias sections (w34) for T and u
    for v, lab, sc, lim in (('t', 'T bias (K)', 1.0, 4), ('u', 'u bias (m/s)', 1.0, 8)):
        f, axs = S.fig(1, 3, sharey=True, gridspec_kw=dict(left=0.07, right=0.9, top=0.9, bottom=0.18, wspace=0.08))
        for j, sy in enumerate(SYSL):
            fn = HB / 'profile' / f'{sy}_{v}.npz'
            if not fn.exists():
                continue
            P = np.load(fn)
            wi = list(P['windows']).index('w34')
            Z = P['zm_bias_raw'][0, wi] * sc
            okl = np.isfinite(Z).any(1)
            im = axs[j].pcolormesh(K.LAT['g15'], lev[okl], Z[okl], cmap=S.DIV, vmin=-lim, vmax=lim, shading='nearest')
            axs[j].set_title(S.SYS[sy])
            axs[j].set_yscale('log')
            axs[j].set_ylim(1000, 5)
            axs[j].set_xlabel('latitude')
        axs[0].set_yticks([1000, 500, 250, 100, 50, 20, 10, 5], ['1000', '500', '250', '100', '50', '20', '10', '5'])
        axs[0].set_ylabel('hPa')
        cax = f.add_axes([0.915, 0.2, 0.012, 0.65])
        f.colorbar(im, cax=cax).set_label(lab)
        S.save(f, 'B04x', f'zonal_mean_{v}_bias_w34', FOOT_E + ' · weeks 3–4 raw zonal-mean bias, all seasons', slide=False)


def b05():
    """QBO and polar-vortex drift: zonal-mean u, equator (5°S–5°N) at 50 and 10 hPa, 60°N at 10 hPa."""
    S.plt()
    lat = K.LAT['g15']
    eq = (lat >= -5) & (lat <= 5)
    i60 = int(np.argmin(np.abs(lat - 60)))
    lines = []
    f, axs = S.fig(1, 4, gridspec_kw=dict(left=0.06, right=0.98, top=0.88, bottom=0.3, wspace=0.3))
    iu = B.UA.index('u')
    for sy in SYSL:
        A = agg(sy)
        st = dstarts(A)
        E, O = A['ua_zm_em'][:, :, iu], A['ua_zm_O'][:, :, iu]          # [S, 46, 18, 121]
        for k, (L, la, title, sel) in enumerate(((50, 'eq', 'u 50 hPa, 5°S–5°N', None), (10, 'eq', 'u 10 hPa, 5°S–5°N', None),
                                                 (10, '60', 'u 10 hPa, 60°N (Nov–Mar starts)', (11, 12, 1, 2, 3)))):
            li = B.LEVELS.index(L)
            if la == 'eq':
                e, o = np.nanmean(E[:, :, li][..., eq], -1), np.nanmean(O[:, :, li][..., eq], -1)
            else:
                e, o = E[:, :, li, i60], O[:, :, li, i60]
            ix = np.array([i for i, s in enumerate(st) if sel is None or s.month in sel])
            Y = (e - o)[ix]
            if not np.isfinite(Y[:, 5:]).any():
                continue
            m = band(axs[k], LEADS[L42], Y[:, L42], S.SYSC[sy], label=S.SYS[sy])
            for d in (7, 14, 28, 42):
                lines.append(f'{title} {S.SYS[sy]} raw bias lead {d}: {m[d]:+.2f} m/s (n = {np.isfinite(Y[:, d]).sum()}) · '
                             f'`helperB/cache/agg_{sy}.npz` ua_zm_em − ua_zm_O')
            axs[k].set_title(title, fontsize=15)
            axs[k].axhline(0, color=S.GREY, lw=1)
            axs[k].set_xlabel('lead day')
            axs[k].set_xticks([0, 14, 28, 42])
        # QBO damping: day-28 forecast vs initial ERA5 u_eq50
        li = B.LEVELS.index(50)
        e, o = np.nanmean(E[:, :, li][..., eq], -1), np.nanmean(O[:, :, li][..., eq], -1)
        x0, y28, o28 = o[:, 0], e[:, 28], o[:, 28]
        ok = np.isfinite(x0) & np.isfinite(y28) & np.isfinite(o28)
        axs[3].scatter(o28[ok], y28[ok], s=8, color=S.SYSC[sy], alpha=0.5)
        sl = np.polyfit(o28[ok], y28[ok], 1)
        lines.append(f'QBO u50 eq day 28 {S.SYS[sy]}: regression of forecast on ERA5 slope {sl[0]:.2f}, intercept {sl[1]:+.2f} m/s, '
                     f'r = {np.corrcoef(o28[ok], y28[ok])[0, 1]:.3f} (n = {ok.sum()}) · `helperB/cache/agg_{sy}.npz`')
    lim = [-35, 20]
    axs[3].plot(lim, lim, color=S.GREY, lw=1)
    axs[3].set_xlim(lim)
    axs[3].set_ylim(lim)
    axs[3].set_xlabel('ERA5 u 50 hPa eq., day 28')
    axs[3].set_ylabel('forecast')
    axs[3].set_title('QBO wind at day 28', fontsize=15)
    axs[0].set_ylabel('raw bias (m/s)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=2.5, label=S.SYS[s]) for s in SYSL]
    f.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.1))
    S.save(f, 'B05', 'qbo_vortex_drift', FOOT_E.replace(' · exploratory', ' · zonal-mean u of the ensemble mean vs ERA5 00 UTC; FuXi has no '
           '10 hPa · band = 10–90 % of starts · exploratory'))
    kn('B05', lines)


def b07():
    """precipitation distribution: wet-day frequency, drizzle, 99th percentile by region and window (members 0-9, daily, 1.5 deg)."""
    S.plt()
    lines = []
    e = B.HEDGES
    f, axs = S.fig(1, 3, gridspec_kw=dict(left=0.07, right=0.98, top=0.88, bottom=0.27, wspace=0.3))
    regs = ['tropics', 'nh_extratropics', 'land', 'ocean']
    stats = {}
    for sy in SYSL:
        A = agg(sy)
        Hm, He = np.nansum(A['hist_model'], 0), np.nansum(A['hist_era5'], 0)       # [3 win, 6 reg, nb]
        for iw, w in enumerate(B.HWIN):
            for rn in regs:
                ir = B.HREG.index(rn)
                for src, Hh in (('model', Hm[iw, ir]), ('era5', He[iw, ir])):
                    stats[(sy, w, rn, src)] = (float(Hh[(e[:-1] >= 0.1 - 1e-9) & (e[1:] <= 1.0 + 1e-9)].sum() / Hh.sum()),
                                                float(Hh[e[:-1] >= 1.0 - 1e-9].sum() / Hh.sum()), B.hist_quantile(Hh, 0.99))
    x = np.arange(len(regs))
    wd = 0.2
    for k, (ttl, si) in enumerate((('drizzle days (0.1–1 mm/day)', 0), ('wet days (≥ 1 mm/day)', 1), ('99th percentile (mm/day)', 2))):
        ax = axs[k]
        for j, sy in enumerate(SYSL):
            ym = [stats[(sy, 'w34', r, 'model')][si] for r in regs]
            ye = [stats[(sy, 'w34', r, 'era5')][si] for r in regs]
            ax.bar(x + (j - 1) * wd, ym, wd, color=S.SYSC[sy])
            ax.scatter(x + (j - 1) * wd, ye, marker='_', s=300, color=S.GREY, zorder=3, lw=3)
        ax.set_xticks(x, ['tropics', 'NH extra-\ntropics', 'land', 'ocean'], fontsize=13)
        ax.set_title(ttl, fontsize=15)
    for (sy, w, rn, src), v in sorted(stats.items()):
        lines.append(f'precip {S.SYS[sy]} {w} {rn} {src}: drizzle {v[0]:.3f}, wet-day {v[1]:.3f}, p99 {v[2]:.1f} mm/day · '
                     f'`helperB/cache/agg_{sy}.npz` hist_{"model" if src == "model" else "era5"} (cos-lat weighted cell-days, members 0–9)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=8, label=S.SYS[s]) for s in SYSL] + \
        [Line2D([], [], color=S.GREY, marker='_', ms=18, lw=0, mew=3, label='ERA5 (same days, system sampling)')]
    f.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.08), fontsize=13)
    S.save(f, 'B07', 'precip_distribution_w34', FOOT_E + ' · daily precipitation of members 0–9, weeks 3–4 (days 15–28), 1.5° grid · '
           'fractions of cos-lat weighted cell-days', slide=True)
    kn('B07', lines)


def b08():
    """zonal spectra of z500 and 250 hPa KE by lead: forecast members / ERA5 power ratio, 30–60°N."""
    S.plt()
    lines = []
    f, axs = S.fig(1, 2, gridspec_kw=dict(left=0.08, right=0.98, top=0.88, bottom=0.3, wspace=0.22))
    leads = [1, 5, 10, 20, 40]
    sty = dict(zip(leads, [(0, (1, 1)), '--', '-.', '-', (0, (5, 1, 1, 1, 1, 1))]))
    for k, v in enumerate(('z500', 'ke250')):
        ax = axs[k]
        for sy in SYSL:
            if v == 'ke250' and sy == 'fuxi':
                continue                                   # FuXi daily-mean winds vs instantaneous ERA5: not comparable
            A = agg(sy)
            Fm = np.nanmean(A[f'spec_{v}_nh_mid'], 0)                  # [46, nk]
            Em = np.nanmean(A[f'era5spec_{v}_nh_mid'], 0)
            nk = 61
            kk = np.arange(Fm.shape[1])
            for d in leads:
                r = Fm[d, 1:nk] / Em[d, 1:nk]
                ax.plot(kk[1:nk], r, ls=sty[d], color=S.SYSC[sy], lw=1.8)
                for kw in (5, 10, 20, 40):
                    if kw < nk:
                        lines.append(f'spectral ratio {v} 30–60°N {S.SYS[sy]} lead {d} k={kw}: {Fm[d, kw] / Em[d, kw]:.3f} · '
                                     f'`helperB/cache/agg_{sy}.npz` spec_{v}_nh_mid / era5spec_{v}_nh_mid (members 0–9)')
        ax.axhline(1, color=S.GREY, lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.3, 3)
        ax.set_yticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3], ['0.3', '0.5', '0.7', '1', '1.5', '2', '3'])
        ax.set_xlabel('zonal wavenumber')
        ax.set_ylabel('member power / ERA5 power')
        ax.set_title({'z500': 'z500, 30–60°N', 'ke250': 'kinetic energy 250 hPa, 30–60°N'}[v], fontsize=15)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=2.5, label=S.SYS[s]) for s in SYSL] + \
        [Line2D([], [], color='k', ls=sty[d], lw=1.8, label=f'day {d}') for d in leads]
    f.legend(handles=h, loc='lower center', ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.1), fontsize=12)
    S.save(f, 'B08', 'spectra_ratio', FOOT_E + ' · zonal power spectra of members 0–9 (not the ensemble mean) vs ERA5 on the same valid days; UC-S2S '
           '1° vs ERA5 00 UTC 1°; FuXi 1.5° daily means vs ERA5 1.5° daily-mean z500 (no FuXi KE: daily means vs instantaneous ERA5)', slide=False)
    kn('B08', lines)


def b06d():
    """members-needed curves (fair CRPSS and ensemble-mean ACC vs ensemble size, drift-corrected, weeks 3-4)."""
    S.plt()
    lines = []
    fields = ['t2m', 'z500', 'tp24', 'sst', 'olr', 'u850']
    f, axs = S.fig(2, 6, sharex=True, gridspec_kw=dict(left=0.06, right=0.99, top=0.92, bottom=0.17, wspace=0.45, hspace=0.25))
    for j, fd in enumerate(fields):
        for sy in SYSL:
            fn = HB / 'members' / f'{sy}_{fd}.json'
            if not fn.exists():
                continue
            rr = json.loads(fn.read_text())['rows']
            for i, met in enumerate(('crpss', 'acc')):
                xs, ms, lo, hi = [], [], [], []
                for n in (5, 10, 20, 30):
                    v = [r['value'] for r in rr if r['members'] == n and r['metric'] == met and r['reference'] == 'fixed'
                         and r['region'] == 'global' and r['window'] == 'w34']
                    if v:
                        xs.append(n)
                        ms.append(np.mean(v))
                        lo.append(min(v))
                        hi.append(max(v))
                axs[i, j].plot(xs, ms, '-o', color=S.SYSC[sy], lw=2, ms=4)
                axs[i, j].fill_between(xs, lo, hi, color=S.SYSC[sy], alpha=0.15, lw=0)
                if len(ms) == 4:
                    lines.append(f'members-needed {fd} {S.SYS[sy]} global w34 {met} (fixed ref): M=5 {ms[0]:.3f}, 10 {ms[1]:.3f}, 20 {ms[2]:.3f}, '
                                 f'30 {ms[3]:.3f} (mean of 5 draws; 30 = all) · `{fn}`')
        axs[0, j].set_title(NAME[fd], fontsize=14)
        axs[1, j].set_xlabel('members')
        axs[1, j].set_xticks([5, 10, 20, 30])
    axs[0, 0].set_ylabel('fair CRPSS')
    axs[1, 0].set_ylabel('ens.-mean ACC')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=2.5, label=S.SYS[s]) for s in SYSL]
    f.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.03))
    S.save(f, 'B06d', 'members_needed', FOOT + ' · weeks 3–4, drift-corrected with the subset\'s own LOYO mean, vs the fixed 1991–2020 '
           'reference, globe; band = min–max of 5 random subsets', slide=True)
    kn('B06d', lines)



def b09():
    """member exchangeability: share of the weeks-3-4 ensemble variance that is a fixed member-specific offset (memberdiag phase)."""
    S.plt()
    lines = []
    fields = ['sst', 't2m', 'tp24', 'u850', 'z500']
    f, axs = S.fig(1, 2, gridspec_kw=dict(left=0.08, right=0.98, top=0.88, bottom=0.3, wspace=0.25))
    x = np.arange(len(fields))
    wd = 0.27
    for k, reg in enumerate(('global', 'tropics')):
        ax = axs[k]
        for j, sy in enumerate(SYSL):
            vals, rs = [], []
            for fd in fields:
                fn = HB / 'memberdiag' / f'{sy}_{fd}.json'
                d = json.loads(fn.read_text())
                vals.append(d[reg]['offset_frac'])
                rs.append(d[reg]['r_mean'])
                lines.append(f'member offsets {fd} {S.SYS[sy]} {reg} w34: offset variance / ensemble variance {d[reg]["offset_frac"]:.3f} '
                             f'(expected for exchangeable members ≈ 1/S = {d[reg]["noise_frac_expected"]:.4f}); odd/even-start pattern r of the '
                             f'member offsets, mean over members {d[reg]["r_mean"]:.2f} · `{fn}`')
            bars = ax.bar(x + (j - 1) * wd, vals, wd, color=S.SYSC[sy])
            for xi, v, r in zip(x + (j - 1) * wd, vals, rs):
                if sy != 'fuxi':
                    ax.text(xi, v + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=10, color=S.SYSC[sy], rotation=90)
        ax.set_xticks(x, ['SST', 't2m', 'precip.', 'u850', 'z500'], fontsize=14)
        ax.set_ylim(0, 0.75)
        ax.set_title({'global': 'globe', 'tropics': 'tropics (20°S–20°N)'}[reg], fontsize=16)
        ax.set_ylabel('fixed member offsets / ensemble variance')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=S.SYSC[s], lw=8, label=S.SYS[s]) for s in SYSL]
    f.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.11), fontsize=13)
    S.save(f, 'B09', 'member_offsets_w34', FOOT + ' · weeks 3–4 member window means; offset = mean over starts of (member − ensemble mean); '
           'numbers above bars = pattern correlation of each member\'s offset map between odd and even starts (0 = noise)')
    kn('B09', lines)


if __name__ == '__main__':
    for a in sys.argv[1:]:
        globals()[a]()
