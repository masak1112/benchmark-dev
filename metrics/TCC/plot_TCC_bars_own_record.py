"""
Paper-style bars (FuXi-S2S Fig. 1 layout: weeks 3,4,5,6,3-4,5-6) with every model on ALL of
its own init dates rather than a common date set. Error bars: 95% month-block bootstrap CI.

Because each model is verified on a different sample (different years/seasons), this shows
each model's typical skill; for a head-to-head use plot_TCC.py on a common date set.

Usage:
  python plot_TCC_bars_own_record.py                          # UC-S2S V2, FuXi, NeuralGCM
  python plot_TCC_bars_own_record.py --models ucs2s_v2 fuxi si_s2s ngcm
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
BOOT_BLOCK = 'month'

# model key -> (result tag, description of its record)
OWN_RECORD = {
    'ucs2s_v2': ('cached151', 'extracted inits'),
    'fuxi':     ('y2017_2021', '2017-2021'),
    'ngcm':     ('y2019_2021', '2019-2021'),
    'si_s2s':   ('si104', 'May-Jul 2019-2022'),
    'ecmwf':    ('ec1921', '2019-2021'),
}
# same, restricted to the 2019-2021 period that all four models cover (--period 2019-2021)
OWN_RECORD_1921 = {
    'ucs2s_v2': ('ucall1921', '2019-2021'),   # all 302 UC-S2S V2 inits in 2019-2021
    'fuxi':     ('fx1921', '2019-2021'),
    'ngcm':     ('y2019_2021', '2019-2021'),
    'si_s2s':   ('si20m1921', '2019-2021'),   # 129 inits with 20 members: May-Jul 2019/2021 + most of 2020
    'ecmwf':    ('ec1921', '2019-2021'),      # all 313 ECMWF inits in 2019-2021
}
BAR_WEEKS = ['week3', 'week4', 'week5', 'week6', 'week3-4', 'week5-6']
BAR_TICKS = ['3', '4', '5', '6', '3-4', '5-6']
BAR_VARS = [('tp', 'TCC TP'), ('t2m', 'TCC T2M'), ('z500', 'TCC Z500')]   # OLR left out (only FuXi has it)

ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', default=['ucs2s_v2', 'fuxi', 'ngcm'], choices=list(OWN_RECORD),
                help='non-default model lists get a _{models} suffix on the figure name')
ap.add_argument('--mem-suffix', default='', help="e.g. '_m20' for the 20-member results")
ap.add_argument('--period', default='all', choices=['all', '2019-2021'],
                help="'2019-2021': every model restricted to 2019-2021 inits")
args = ap.parse_args()
if args.period == '2019-2021':
    OWN_RECORD = OWN_RECORD_1921

series = []
for key in args.models:
    tag, desc = OWN_RECORD[key]
    tag += args.mem_suffix
    with open(os.path.join(RES_DIR, f'tcc_weekly_{key}_{tag}.pkl'), 'rb') as f:
        res = pickle.load(f)
    bp = os.path.join(RES_DIR, f'tcc_bootstrap_{tag}_{BOOT_BLOCK}.pkl')
    boot = None
    if os.path.exists(bp):
        with open(bp, 'rb') as f:
            boot = pickle.load(f)
    name = MODEL_CFG[key]['name']
    series.append((f'{name} {desc}', key, res, boot, model_color(name)))


def global_vals(res, label):
    return np.array([res['regional'][label]['global'][res['weeks'].index(w)] for w in BAR_WEEKS])


def global_ci(boot, key, label):
    if boot is None or key not in boot['boot']:
        return None
    b = boot['boot'][key][label][1:, :, boot['regions'].index('global')]
    b = b[:, [boot['weeks'].index(w) for w in BAR_WEEKS]]
    return None if np.isnan(b).all() else np.nanpercentile(b, [2.5, 97.5], axis=0)


# drop variables that none of the listed models outputs (e.g. OLR without FuXi)
bar_vars = [(lbl, ylab) for lbl, ylab in BAR_VARS
            if any(not np.isnan(global_vals(res, lbl)).all() for _, _, res, _, _ in series)]
nrow, ncol = (2, 2) if len(bar_vars) == 4 else (1, len(bar_vars))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.5 * nrow + 0.5), squeeze=False)
wbar = 0.8 / len(series)
for ax, (label, ylab) in zip(axes.flatten(), bar_vars):
    for s_i, (sname, key, res, boot, color) in enumerate(series):
        vals = global_vals(res, label)
        if np.isnan(vals).all():          # variable the model does not output
            continue
        xs = np.arange(len(BAR_WEEKS)) + (s_i - (len(series) - 1) / 2) * wbar
        ci = global_ci(boot, key, label)
        yerr = None if ci is None else np.abs(np.array([vals - ci[0], ci[1] - vals]))
        ax.bar(xs, vals, width=wbar * 0.9, color=color, label=sname,
               yerr=yerr, error_kw=dict(ecolor='0.15', elinewidth=1, capsize=2))
    ax.set_xticks(np.arange(len(BAR_WEEKS)))
    ax.set_xticklabels(BAR_TICKS)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_xlabel('Lead time (weeks)', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
handles, lbls = [], []
for ax in axes.flatten():                 # collect every series once (some panels lack a model)
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in lbls:
            handles.append(h)
            lbls.append(l)
LEGEND_FS = 22          # legend text size (pt); two columns once there are more than two entries
fig.legend(handles, lbls, loc='lower center', ncol=len(handles) if len(handles) <= 2 else 2, fontsize=LEGEND_FS,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()   # legend sits above the panels (outside), kept by bbox_inches='tight'
DEFAULT_MODELS = ['ucs2s_v2', 'fuxi', 'ngcm']
suffix = '' if args.models == DEFAULT_MODELS else '_' + '_'.join(args.models)   # never overwrite the default figure
period = '' if args.period == 'all' else '_2019_2021'
out = os.path.join(PLOT_DIR, f'tcc_weekly_paper_style_bars_own_record{suffix}{period}{args.mem_suffix}.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
