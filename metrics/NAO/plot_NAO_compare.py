"""
NAO comparison figure combining models from different score files (nao_daily_scores_{tag}.pkl),
e.g. the ML models on the 2019-21 winters and the ECMWF reforecasts. One panel per metric
(--metrics, any of acc cor rmse ssr); the skill day (last lead >= SKILL_THR) is taken from the
first panel when it is acc or cor.

Usage:  python plot_NAO_compare.py --entries y1921:ucs2s_v2,ngcm,fuxi ecmwf_ndjfm:ecmwf_rf2018 \
            --metrics acc --out nao_compare_acc_all1921.png
Output: /glade/work/bgong/plots/{out}
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_style  # noqa: E402

from plot_NAO_daily import MARKERS, MIN_MONTHS, PLOT_DIR, RES_DIR  # noqa: E402

SKILL_THR = 0.5          # skill day = last lead with ACC / correlation >= SKILL_THR
PANELS = {   # metric -> (title, ylim, reference line)
    'acc':  ('NAO index ACC (ensemble mean vs ERA5)', (-0.2, 1.0), 0.0),
    'cor':  ('Correlation of ensemble-mean NAO index with ERA5', (-0.2, 1.0), 0.0),
    'rmse': ('RMSE of ensemble-mean NAO index (σ units; 1 ≈ climatology)', (0, 1.6), 1.0),
    'ssr':  ('Spread / RMSE (1 = reliable ensemble)', (0, 1.6), 1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entries', nargs='+', required=True, help='tag:model1,model2 ...')
    ap.add_argument('--metrics', nargs='+', choices=list(PANELS), default=['cor', 'rmse'])
    ap.add_argument('--out', default='nao_compare.png')
    a = ap.parse_args()
    skill_m = a.metrics[0] if a.metrics[0] in ('acc', 'cor') else None

    n = len(a.metrics)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n if n > 1 else 8, 4.8), squeeze=False)
    axes = axes[0]
    for entry in a.entries:
        tag, models = entry.split(':')
        with open(os.path.join(RES_DIR, f'nao_daily_scores_{tag}.pkl'), 'rb') as fh:
            S = pickle.load(fh)
        x, own, names = S['leads'], S['own'], S['names']
        for key in models.split(','):
            r = own[key]
            st = model_style(names[key])
            label = f'{names[key]} (n={r["n"]})'
            if skill_m:
                below = np.where(r[skill_m]['val'] < SKILL_THR)[0]
                day = int(below[0]) if len(below) else len(x)    # last lead >= threshold
                label += f': skill to day {day}'
                axes[0].plot([day, day], [-0.2, SKILL_THR], color=st['color'], lw=1.0, ls='--')
                axes[0].plot(day, SKILL_THR, marker='o', ms=5, color=st['color'], zorder=4)
            for ax, m in zip(axes, a.metrics):
                if len({d[:6] for d in r['dates']}) >= MIN_MONTHS:
                    ax.fill_between(x, r[m]['lo'], r[m]['hi'], color=st['color'], alpha=0.10, lw=0)
                ax.plot(x, r[m]['val'], marker=MARKERS.get(key, 'o'), ms=4, markevery=7, label=label, **st)

    for ax, m in zip(axes, a.metrics):
        title, ylim, ref = PANELS[m]
        if m == skill_m:
            title += f'; skill day = last day ≥ {SKILL_THR}'
            ax.axhline(SKILL_THR, color='0.6', lw=0.8, ls='--')
        ax.axhline(ref, color='0.5', lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(0, 43)
        ax.set_xticks(range(0, 43, 7))
        ax.set_xlabel('Lead day')
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle('Daily winter NAO skill; shading = 95% month-block bootstrap CI', fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    ncol = 2 if len(h) <= 4 else 3
    fig.legend(h, l, loc='lower center', ncol=ncol, fontsize=12 if ncol == 2 else 11, frameon=False,
               bbox_to_anchor=(0.5, -0.15 if n > 1 else -0.2))
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, a.out)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
