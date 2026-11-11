"""
Plot weekly TCC for a set of models on a common date set: regional-mean TCC vs week
(per variable x region), a global all-variable summary, TCC maps (week 1, week 2,
weeks 3-4) with first-model-minus-other differences, paper-style bars, and the paired
bootstrap difference (first model minus each other model).

Usage:
  python plot_TCC.py                                              # UC-S2S V2 vs FuXi, 86 inits
  python plot_TCC.py --models ucs2s_v2 fuxi si_s2s --tag common39
  python plot_TCC.py --models ucs2s_v2 si_s2s --tag si104
Figures for the default tag keep their original names; other tags get a _{tag} suffix.
"""

import argparse
import os
import pickle
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import netCDF4
import numpy as np
from scipy import stats

sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_color, model_style  # noqa: E402

from tcc_weekly import MODELS as MODEL_CFG  # noqa: E402

HERE     = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(HERE, 'results')
PLOT_DIR = '/glade/work/bgong/plots'
DEFAULT_TAG = 'common86'
# Each model on its own full record (not the common dates), shown as hatched bars in the paper-style
# figure: FuXi 2017-2021 = FuXi-S2S paper test period; UC-S2S V2 = all of its inits.
FULL_TAGS = {'ucs2s_v2': ('cached151', 'all extracted'), 'fuxi': ('y2017_2021', '2017-2021'),
             'ngcm': ('y2019_2021', '2019-2021')}
BOOT_BLOCK = 'month'
LEGEND_FS = 22          # legend text size (pt)


def legend_ncol(n):
    """columns for a 22-pt legend: one row for up to two entries, otherwise two columns"""
    return n if n <= 2 else 2

VAR_LABELS = {
    't2m':  '2m Temperature',
    'z500': 'Geopotential 500 hPa',
    'tp':   'Total Precipitation',
    'u10':  '10m U-wind',
    'v10':  '10m V-wind',
    'u850': 'U-wind 850 hPa',
    'v850': 'V-wind 850 hPa',
    'olr':  'Outgoing Longwave Radiation',
}
REGIONS = ['global', 'tropics', 'NH', 'SH']
REGION_TITLES = {
    'global':  'Global',
    'tropics': 'Tropics (30S-30N)',
    'NH':      'Extratropics NH (30N-90N)',
    'SH':      'Extratropics SH (30S-90S)',
}
MAP_WEEKS = ['week1', 'week2', 'week3-4']
MAP_WEEK_TITLES = {'week1': 'Week 1 (d1-7)', 'week2': 'Week 2 (d8-14)', 'week3-4': 'Weeks 3-4 (d15-28)'}
LINE_WEEKS = [f'week{k}' for k in range(1, 7)]

ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', default=['ucs2s_v2', 'fuxi'], choices=list(MODEL_CFG))
ap.add_argument('--tag', default=DEFAULT_TAG)
ap.add_argument('--no-own-record', action='store_true',
                help='paper-style bars: only the common-date bars (no hatched own-record bars)')
ap.add_argument('--bars-only', action='store_true', help='only make the paper-style bar figure')
ap.add_argument('--mem-suffix', default='',
                help="e.g. '_m20': appended to --tag and to each model's own-record tag")
args = ap.parse_args()
FULL_TAGS = {k: (t + args.mem_suffix, d) for k, (t, d) in FULL_TAGS.items()}
TAG = args.tag + args.mem_suffix
SUFFIX = '' if TAG == DEFAULT_TAG else f'_{TAG}'
MODELS = {MODEL_CFG[k]['name']: k for k in args.models}      # display name -> key, in order
MODEL_STYLES = {name: model_style(name) for name in MODELS}
names = list(MODELS)

os.makedirs(PLOT_DIR, exist_ok=True)

reg, maps = {}, {}
for name, key in MODELS.items():
    with open(os.path.join(RES_DIR, f'tcc_weekly_{key}_{TAG}.pkl'), 'rb') as f:
        reg[name] = pickle.load(f)
    with netCDF4.Dataset(os.path.join(RES_DIR, f'tcc_weekly_{key}_{TAG}.nc')) as nc:
        maps[name] = {
            'tcc': nc.variables['tcc'][:].filled(np.nan),
            'var': list(nc.variables['var'][:]),
            'week': list(nc.variables['week'][:]),
        }
        lat, lon = nc.variables['lat'][:], nc.variables['lon'][:]
n_inits = {reg[n]['n_init'] for n in names}
assert len(n_inits) == 1, f'models were aggregated on different date sets: {n_inits}'
n_init = n_inits.pop()


# Block-bootstrap resamples from tcc_bootstrap.py (optional): boot[key][label] = (R+1, week, region)
def load_boot(tag):
    p = os.path.join(RES_DIR, f'tcc_bootstrap_{tag}_{BOOT_BLOCK}.pkl')
    if not os.path.exists(p):
        print(f'no bootstrap file {p}; plotting without CIs')
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


boot = load_boot(TAG)


def boot_ci(bt, key, label, region, weeks_sel, diff_key=None):
    """(lo, hi) 95% percentile CI for the listed weeks; diff_key -> CI of (key - diff_key)."""
    if bt is None or key not in bt['boot'] or (diff_key is not None and diff_key not in bt['boot']):
        return None
    ri = bt['regions'].index(region)
    wi = [bt['weeks'].index(w) for w in weeks_sel]
    b = bt['boot'][key][label][1:, :, ri][:, wi]
    if diff_key is not None:
        b = b - bt['boot'][diff_key][label][1:, :, ri][:, wi]
    if np.isnan(b).all():
        return None
    return np.nanpercentile(b, [2.5, 97.5], axis=0)


def available(name, label):
    return not np.all(np.isnan(reg[name]['regional'][label]['global']))


# only variables that at least one plotted model outputs (e.g. no OLR panel without FuXi)
# OLR is left out of the figures (only FuXi has it); it is still in the result files
EXCLUDE_VARS = {'olr'}
PLOT_VARS = {v: full for v, full in VAR_LABELS.items()
             if v not in EXCLUDE_VARS and any(available(n, v) for n in names)}
n_line = len(LINE_WEEKS)
x = np.arange(1, n_line + 1)
# nominal 95% two-sided threshold for r with n_init samples (inits overlap in
# time, so effective n is smaller and this is a lower bound on the true threshold)
t = stats.t.ppf(0.975, n_init - 2)
R_CRIT = t / np.sqrt(n_init - 2 + t ** 2)
ci_note = f' (shading: 95% {BOOT_BLOCK}-block bootstrap CI)' if boot else ''


def plot_lines(ax, label, region):
    for name in names:
        if not available(name, label):
            continue
        ax.plot(x, reg[name]['regional'][label][region][:n_line], label=name, **MODEL_STYLES[name])
        ci = boot_ci(boot, MODELS[name], label, region, LINE_WEEKS)
        if ci is not None:
            ax.fill_between(x, ci[0], ci[1], color=MODEL_STYLES[name]['color'], alpha=0.18, linewidth=0)


if not args.bars_only:
    # ---- Per-variable: regional-mean TCC vs week ----
    for label, full in PLOT_VARS.items():
        fig, axes = plt.subplots(1, len(REGIONS), figsize=(5 * len(REGIONS), 4), sharey=True)
        for ax, region in zip(axes, REGIONS):
            plot_lines(ax, label, region)
            ax.axhline(R_CRIT, color='0.4', linewidth=1, linestyle=':', label=f'r=0 test, 95% (n={n_init})')
            ax.set_xticks(x)
            ax.set_xlim(0.6, n_line + 0.4)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('Lead week', fontsize=13)
            ax.set_title(REGION_TITLES[region], fontsize=14)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('TCC (area mean)', fontsize=14)
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc='upper center', ncol=legend_ncol(len(handles)), fontsize=LEGEND_FS,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f'Weekly TCC — {full} — ensemble mean, {n_init} common inits{ci_note}', fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        out = os.path.join(PLOT_DIR, f'tcc_weekly_{label}_regions{SUFFIX}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    # ---- Global summary, all variables ----
    labels = list(PLOT_VARS)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for ax, label in zip(axes, labels):
        plot_lines(ax, label, 'global')
        ax.axhline(R_CRIT, color='0.4', linewidth=1, linestyle=':')
        ax.set_title(VAR_LABELS[label], fontsize=12)
        ax.set_xticks(x)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Lead week', fontsize=10)
        ax.set_ylabel('TCC', fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(labels):]:
        ax.set_visible(False)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower right', fontsize=LEGEND_FS, bbox_to_anchor=(0.95, 0.08))
    fig.suptitle(f'Weekly TCC vs lead week — Global — ensemble mean ({n_init} common inits){ci_note}', fontsize=14)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'tcc_weekly_global_all_vars{SUFFIX}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # ---- Maps: rows = each model, then first model minus each other; cols = week1, week2, weeks3-4 ----
    lev_tcc  = np.arange(-1.0, 1.01, 0.1)
    lev_diff = np.arange(-0.5, 0.51, 0.05)
    n_rows = 2 * len(names) - 1
    for label, full in PLOT_VARS.items():
        fig, axes = plt.subplots(n_rows, len(MAP_WEEKS), figsize=(6 * len(MAP_WEEKS), 2.5 * n_rows),
                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        for j, wk in enumerate(MAP_WEEKS):
            fields = [maps[n]['tcc'][maps[n]['var'].index(label), maps[n]['week'].index(wk)] for n in names]
            rows = [(n, f, lev_tcc, 'RdBu_r', True) for n, f in zip(names, fields)]
            rows += [(f'{names[0]} − {n}', fields[0] - f, lev_diff, 'PuOr_r', False)
                     for n, f in zip(names[1:], fields[1:])]
            for i, (title, fld, lev, cmap, is_tcc) in enumerate(rows):
                ax = axes[i, j]
                ax.coastlines(linewidth=0.5, color='0.25')
                ax.set_title(f'{title} — {MAP_WEEK_TITLES[wk]}', fontsize=11)
                if np.isnan(fld).all():   # variable the model does not output (e.g. OLR)
                    ax.text(0.5, 0.5, 'not available', transform=ax.transAxes, ha='center', va='center',
                            fontsize=12, color='0.3', bbox=dict(facecolor='white', edgecolor='none'))
                else:
                    ax.contourf(lon, lat, fld, levels=lev, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
                    if is_tcc:
                        ax.contour(lon, lat, fld, levels=[R_CRIT], colors='k', linewidths=0.4,
                                   transform=ccrs.PlateCarree())
                if j == len(MAP_WEEKS) - 1:
                    sm = plt.cm.ScalarMappable(norm=BoundaryNorm(lev, plt.get_cmap(cmap).N, extend='both'), cmap=cmap)
                    cb = fig.colorbar(sm, ax=axes[i, :].tolist(), orientation='vertical', shrink=0.85, pad=0.02,
                                      extend='both')
                    cb.set_label('TCC' if is_tcc else 'TCC difference', fontsize=11)
        fig.suptitle(f'Weekly TCC — {full} ({n_init} common inits; black contour = nominal 95% level r={R_CRIT:.2f})',
                     fontsize=14)
        out = os.path.join(PLOT_DIR, f'tcc_weekly_{label}_maps{SUFFIX}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

# ---- Paper-style bars (FuXi-S2S Fig. 1 layout): weeks 3,4,5,6,3-4,5-6 ----
bar_series = [(name, reg[name], model_color(name), None, boot, MODELS[name])
              for name in names]
for name, key in ({} if args.no_own_record else MODELS).items():
    if key not in FULL_TAGS:
        continue
    ftag, fdesc = FULL_TAGS[key]
    full_pkl = os.path.join(RES_DIR, f'tcc_weekly_{key}_{ftag}.pkl')
    if not os.path.exists(full_pkl):
        continue
    with open(full_pkl, 'rb') as f:
        full = pickle.load(f)
    bar_series.append((f'{name} {fdesc}', full, model_color(name), '//', load_boot(ftag), key))
bar_weeks = ['week3', 'week4', 'week5', 'week6', 'week3-4', 'week5-6']
bar_ticks = ['3', '4', '5', '6', '3-4', '5-6']
bar_vars = [(v, y) for v, y in [('tp', 'TCC TP'), ('t2m', 'TCC T2M'), ('z500', 'TCC Z500'), ('olr', 'TCC OLR')]
            if v in PLOT_VARS]
nrow, ncol = (2, 2) if len(bar_vars) == 4 else (1, len(bar_vars))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.5 * nrow + 0.5), squeeze=False)
wbar = 0.8 / len(bar_series)
for ax, (label, ylab) in zip(axes.flatten(), bar_vars):
    for s_i, (sname, res, color, hatch, bt, bkey) in enumerate(bar_series):
        vals = np.array([res['regional'][label]['global'][res['weeks'].index(w)] for w in bar_weeks])
        if np.isnan(vals).all():
            continue
        xs = np.arange(len(bar_weeks)) + (s_i - (len(bar_series) - 1) / 2) * wbar
        ci = boot_ci(bt, bkey, label, 'global', bar_weeks)
        yerr = None if ci is None else np.abs(np.array([vals - ci[0], ci[1] - vals]))
        ax.bar(xs, vals, width=wbar * 0.9, color=color if hatch is None else 'white',
               edgecolor=color, hatch=hatch, linewidth=1.2, label=sname,
               yerr=yerr, error_kw=dict(ecolor='0.15', elinewidth=1, capsize=2))
    ax.set_xticks(np.arange(len(bar_weeks)))
    ax.set_xticklabels(bar_ticks)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_xlabel('Lead time (weeks)', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
handles, lbls = [], []
for ax in axes.flatten():                 # every series once, even if absent from the first panel
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in lbls:
            handles.append(h)
            lbls.append(l)
fig.legend(handles, lbls, loc='lower center', ncol=legend_ncol(len(handles)), fontsize=LEGEND_FS,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()   # legend sits above the panels (outside), kept by bbox_inches='tight'
out = os.path.join(PLOT_DIR, f'tcc_weekly_paper_style_bars{SUFFIX}{"_common_only" if args.no_own_record else ""}.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ---- Paired difference: first model minus each other model, 95% block-bootstrap CI ----
if boot is not None and len(names) > 1 and not args.bars_only:
    ref, others = names[0], names[1:]
    diff_weeks = LINE_WEEKS + ['week3-4', 'week5-6']
    diff_ticks = ['1', '2', '3', '4', '5', '6', '3-4', '5-6']
    xd = np.arange(len(diff_weeks))
    dvars = [v for v in VAR_LABELS if available(ref, v)]
    fig, axes = plt.subplots(len(dvars), len(REGIONS), figsize=(4.2 * len(REGIONS), 2.3 * len(dvars)),
                             sharex=True, sharey='row', squeeze=False)
    bt = boot['boot']
    wi = [boot['weeks'].index(w) for w in diff_weeks]
    off = (np.arange(len(others)) - (len(others) - 1) / 2) * 0.22
    for i, label in enumerate(dvars):
        for j, region in enumerate(REGIONS):
            ax = axes[i, j]
            ri = boot['regions'].index(region)
            ax.axhline(0, color='0.3', linewidth=0.8)
            ax.axvline(5.5, color='0.8', linewidth=0.8)
            for o_i, other in enumerate(others):
                if not available(other, label):
                    continue
                d0 = (bt[MODELS[ref]][label][0, :, ri] - bt[MODELS[other]][label][0, :, ri])[wi]
                lo, hi = boot_ci(boot, MODELS[ref], label, region, diff_weeks, diff_key=MODELS[other])
                sig = (lo > 0) | (hi < 0)
                c = model_color(other)
                xo = xd + off[o_i]
                ax.errorbar(xo, d0, yerr=[d0 - lo, hi - d0], fmt='none', ecolor=c, elinewidth=1.2, capsize=2)
                ax.scatter(xo[sig], d0[sig], s=34, color=c, zorder=3)
                ax.scatter(xo[~sig], d0[~sig], s=34, facecolor='white', edgecolor=c, zorder=3)
            ax.grid(True, axis='y', alpha=0.3)
            if i == 0:
                ax.set_title(REGION_TITLES[region], fontsize=12)
            if j == 0:
                ax.set_ylabel(f'{label}\nΔTCC', fontsize=11)
            if i == len(dvars) - 1:
                ax.set_xticks(xd)
                ax.set_xticklabels(diff_ticks)
                ax.set_xlabel('Lead week', fontsize=11)
    handles = [Line2D([], [], color=model_color(o), marker='o', linestyle='-', label=f'{ref} − {o}')
               for o in others]
    handles += [Line2D([], [], color='0.3', marker='o', linestyle='none', label='CI excludes 0 (filled)'),
                Line2D([], [], color='0.3', marker='o', markerfacecolor='white', linestyle='none',
                       label='CI includes 0 (open)')]
    fig.legend(handles=handles, loc='upper center', ncol=legend_ncol(len(handles)), fontsize=LEGEND_FS,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'{ref} minus other models, weekly TCC (>0: {ref} better); {boot["n_boot"]} paired '
                 f'{boot["block"]}-block bootstrap resamples, n={len(boot["init_dates"])}', fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    stem = 'ucs2s_minus_fuxi' if args.models == ['ucs2s_v2', 'fuxi'] else '_'.join(['minus'] + args.models[1:])
    stem = stem if stem == 'ucs2s_minus_fuxi' else f'{args.models[0]}_{stem}'
    out = os.path.join(PLOT_DIR, f'tcc_weekly_diff_{stem}_bootstrap{SUFFIX}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
