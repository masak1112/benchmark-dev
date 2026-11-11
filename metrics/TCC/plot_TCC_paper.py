"""
Publication-style weekly-TCC bar figure: (a) TP, (b) T2M, (c) Z500, global cos-lat mean TCC for
weeks 3, 4, 5, 6 and the 3-4 / 5-6 averages, all models on the SAME init dates.

Error bars: 95% month-block bootstrap CI. A star above a bar marks a significant paired difference from the
first (reference) model: the 95% bootstrap CI of (model - reference) excludes 0.

With --own-record each model is instead verified on ALL of its own 2019-2021 inits (different samples), so
the bars carry their own CIs but no paired-difference stars.

Usage:
  python plot_TCC_paper.py                                            # UC-S2S V2 vs ECMWF, 263 inits
  python plot_TCC_paper.py --own-record                               # all five models, own 2019-2021 inits
  python plot_TCC_paper.py --models ucs2s_v2 fuxi ngcm ecmwf --tag ucfxngec46_m20
Writes PNG (300 dpi) and PDF to /glade/work/bgong/plots/tcc_paper_{tag}_{models}.{png,pdf}
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_color  # noqa: E402

from tcc_weekly import MODELS as MODEL_CFG  # noqa: E402

HERE     = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(HERE, 'results')
PLOT_DIR = '/glade/work/bgong/plots'

WEEKS = ['week3', 'week4', 'week5', 'week6', 'week3-4', 'week5-6']
TICKS = ['3', '4', '5', '6', '3–4', '5–6']
PANELS = [('tp', 'Total precipitation'), ('t2m', '2-m temperature'), ('z500', 'Geopotential 500 hPa'),
          ('u850', 'Zonal wind 850 hPa'), ('v850', 'Meridional wind 850 hPa')]
DEFAULT_VARS = ['tp', 't2m', 'z500']   # u850 / v850 only on request (--vars u850 v850)

FS_LEGEND, FS_TITLE, FS_LABEL, FS_TICK, FS_PANEL = 22, 22, 20, 18, 24
# each model's full 2019-2021 record (tag without the member / month-mean suffix)
OWN_RECORD_1921 = {'ucs2s_v2': 'ucall1921', 'fuxi': 'fx1921', 'si_s2s': 'si20m1921',
                   'ngcm': 'y2019_2021', 'ecmwf': 'ec1921',
                   'ecmwf_rf': 'ecrf1921',
                   'ucs2s_v2_fbcorr': 'fb1921'}   # ECMWF 2025-model hindcasts: always its 11 members

ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', default=['ucs2s_v2', 'ecmwf'], choices=list(MODEL_CFG),
                help='first model is the reference for the significance stars')
ap.add_argument('--tag', default='ucec263_m20', help='result tag incl. member/mm suffix, e.g. ucec263_m20_mm')
ap.add_argument('--region', default='global', choices=['global', 'tropics', 'NH', 'SH', 'extratropics', 'land', 'land_interior', 'ocean'])
ap.add_argument('--vars', nargs='+', default=DEFAULT_VARS, choices=[l for l, _ in PANELS],
                help='panels to draw, e.g. --vars t2m --region land')
ap.add_argument('--t2m-land', action='store_true',
                help='T2M panel over land only; other panels keep --region')
ap.add_argument('--own-record', action='store_true',
                help='each model on all its own 2019-2021 inits (default models: all five); no stars')
ap.add_argument('--suffix', default='_m20', help="with --own-record: '_m20' or '_m20_mm'")
ap.add_argument('--fuxi-m51', action='store_true',
                help='with --own-record: add FuXi with all 51 members as a lighter bar next to FuXi (20)')
ap.add_argument('--ec-m11', action='store_true',
                help='with --own-record: add ECMWF with 11 members (as in the FuXi-S2S paper) as a lighter bar')
ap.add_argument('--series', nargs='+', default=None,
                help="explicit bars 'model:tag[:label[:alpha]]', each on its own inits like --own-record (no stars)")
ap.add_argument('--block', default='month', choices=['month', 'season', 'none'],
                help='bootstrap block of the CI files (tcc_bootstrap_{tag}_{block}.pkl)')
ap.add_argument('--name', default=None, help="with --series: figure name, tcc_paper_{name}.png")
args = ap.parse_args()
if args.series:
    args.own_record = True
if args.own_record and args.models == ['ucs2s_v2', 'ecmwf']:
    args.models = ['ucs2s_v2', 'fuxi', 'si_s2s', 'ngcm', 'ecmwf']


def load_pkl(name):
    p = os.path.join(RES_DIR, name)
    if not os.path.exists(p):
        print(f'WARNING: missing {p}')
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


# bar series: id -> (model key, result tag, legend label, alpha)
SERIES = {}
for k in args.models:
    label = MODEL_CFG[k]['name']
    if k == 'fuxi' and not args.own_record and '_m51' in args.tag:
        label = 'FuXi (51 members)'          # same-date sets computed with FuXi's full ensemble
    SERIES[k] = (k, OWN_RECORD_1921[k] + args.suffix if args.own_record else args.tag, label, 1.0)
    if k == 'fuxi' and args.own_record and args.fuxi_m51:
        SERIES['fuxi_m51'] = ('fuxi', 'fx1921' + args.suffix.replace('_m20', ''), 'FuXi (51 members)', 0.5)
    if k == 'ecmwf' and args.own_record and args.ec_m11:
        SERIES['ecmwf_m11'] = ('ecmwf', 'ec1921' + args.suffix.replace('_m20', '_m11'), 'ECMWF (11 members)', 0.5)
if args.series:
    SERIES = {}
    for spec in args.series:
        k, tag, *rest = spec.split(':')
        SERIES[f'{k}:{tag}'] = (k, tag, rest[0] if rest else MODEL_CFG[k]['name'],
                                float(rest[1]) if len(rest) > 1 else 1.0)
    args.models = list(dict.fromkeys(v[0] for v in SERIES.values()))
series_ids = list(SERIES)
res = {s_id: load_pkl(f'tcc_weekly_{k}_{tag}.pkl') for s_id, (k, tag, _, _) in SERIES.items()}
boots = {s_id: load_pkl(f'tcc_bootstrap_{tag}_{args.block}.pkl') for s_id, (k, tag, _, _) in SERIES.items()}
n_inits = {k: r['n_init'] for k, r in res.items()}
if not args.own_record:
    assert len(set(n_inits.values())) == 1, f'models were aggregated on different init sets: {n_inits}'


def reg(label):
    return 'land' if args.t2m_land and label == 't2m' else args.region


def values(key, label):
    r = res[key]
    if reg(label) in r['regional'][label]:
        return np.array([r['regional'][label][reg(label)][r['weeks'].index(w)] for w in WEEKS])
    # regions added after the aggregate run (land / ocean): full-sample estimate = bootstrap row 0
    boot = boots[key]
    if boot is None or reg(label) not in boot['regions']:
        return np.full(len(WEEKS), np.nan)
    b = boot['boot'][SERIES[key][0]][label][0, :, boot['regions'].index(reg(label))]
    return b[[boot['weeks'].index(w) for w in WEEKS]]


def boot_samples(s_id, label):
    boot, key = boots[s_id], SERIES[s_id][0]
    if boot is None or key not in boot['boot']:
        return None
    b = boot['boot'][key][label][1:, :, boot['regions'].index(reg(label))]
    return b[:, [boot['weeks'].index(w) for w in WEEKS]]


ref = series_ids[0]
labels = [l for l, _ in PANELS if l in args.vars and any(not np.isnan(values(k, l)).all() for k in series_ids)]
panels = [(l, t) for l, t in PANELS if l in labels]
# drop series with no data in any drawn panel (e.g. NeuralGCM has no t2m) so they leave no empty bar slot
series_ids = [k for k in series_ids if k == ref or any(not np.isnan(values(k, l)).all() for l in labels)]

plt.rcParams.update({'font.size': FS_TICK, 'axes.linewidth': 1.0})
NROW = 2 if len(panels) == 4 else 1            # 4 panels -> 2 x 2
NCOL = len(panels) // NROW
fig, axes = plt.subplots(NROW, NCOL, figsize=(7.0 * NCOL if len(panels) > 1 else 9.0, 6.0 * NROW))
axes = np.atleast_1d(axes).ravel()
x = np.array([0, 1, 2, 3, 4.6, 5.6])           # a gap separates single weeks from 2-week means
width = 0.8 / len(series_ids)

for p_i, (ax, (label, title)) in enumerate(zip(axes, panels)):
    ref_b = boot_samples(ref, label)
    top = 0.0
    for m_i, key in enumerate(series_ids):
        v = values(key, label)
        if np.isnan(v).all():
            continue
        xs = x + (m_i - (len(series_ids) - 1) / 2) * width
        b = boot_samples(key, label)
        yerr = None
        if b is not None:
            lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
            yerr = np.abs(np.array([v - lo, hi - v]))
            top = max(top, np.nanmax(hi))
        top = max(top, np.nanmax(v))
        model_key, _, name, alpha = SERIES[key]
        ax.bar(xs, v, width=width * 0.92, color=model_color(MODEL_CFG[model_key]['name']), alpha=alpha,
               label=name, zorder=2,
               yerr=yerr, error_kw=dict(ecolor='0.2', elinewidth=1.2, capsize=3, zorder=3))
        # significance of the paired difference from the reference model
        # stars only for paired samples: same-date mode, or --series bars sharing the reference's bootstrap file
        paired = not args.own_record or (args.series and SERIES[key][1] == SERIES[ref][1])
        if paired and key != ref and b is not None and ref_b is not None:
            d_lo, d_hi = np.nanpercentile(b - ref_b, [2.5, 97.5], axis=0)
            sig = (d_lo > 0) | (d_hi < 0)
            for xi, vi, e, s in zip(xs, v, yerr[1], sig):
                if s:
                    ax.text(xi, vi + e + 0.006, '★', ha='center', va='bottom', fontsize=FS_TICK - 2,
                            color='0.15')
    ax.axvline(3.8, color='0.75', linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(TICKS, fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_xlim(-0.6, 6.2)
    ax.set_ylim(0, top * 1.15)
    if p_i >= len(panels) - NCOL:
        ax.set_xlabel('Lead time (weeks)', fontsize=FS_LABEL)
    if p_i % NCOL == 0:
        ax.set_ylabel('TCC', fontsize=FS_LABEL)
    r_ = reg(label)
    ax.set_title(title if r_ == 'global' else title.replace('Meridional wind', 'Merid. wind') + f' ({r_})', fontsize=FS_TITLE, pad=10)
    if len(panels) > 1:
        ax.text(-0.02, 1.04, f'({chr(97 + p_i)})', transform=ax.transAxes, fontsize=FS_PANEL,
                fontweight='bold', ha='right', va='bottom')
    ax.grid(True, axis='y', color='0.85', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

handles, lbls = axes[0].get_legend_handles_labels()
for ax in axes[1:]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in lbls:
            handles.append(h)
            lbls.append(l)
ncol = 2 if len(panels) == 1 else len(handles) if len(handles) <= 4 else (len(handles) + 1) // 2
# matplotlib fills legend columns first; reorder so the legend reads row by row in bar order
nrow = -(-len(handles) // ncol)
order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < len(handles)]
handles, lbls = [handles[i] for i in order], [lbls[i] for i in order]
fig.legend(handles, lbls, loc='lower center', ncol=ncol,
           fontsize=FS_LEGEND, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.8)
fig.tight_layout(w_pad=2.5)
name = (f'own_record_2019_2021{args.suffix}' if args.own_record else args.tag)
stem = os.path.join(PLOT_DIR, f'tcc_paper_{args.name}') if args.name else os.path.join(PLOT_DIR, f'tcc_paper_{name}_{"_".join(args.models)}' + ('_fxm51' if 'fuxi_m51' in SERIES else '') + ('_ecm11' if 'ecmwf_m11' in SERIES else '')
                    + ('' if args.vars == DEFAULT_VARS else '_' + '_'.join(args.vars))
                    + ('' if args.region == 'global' else f'_{args.region}'))
for ext in ('png', 'pdf'):
    fig.savefig(f'{stem}.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {stem}.png / .pdf   (inits: {n_inits}, region {args.region})')
