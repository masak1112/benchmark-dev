"""
Figures for the daily DJF NAO evaluation (reads results/nao_daily_scores_{tag}.pkl).

  nao_daily_skill_djf_{tag}.png   correlation / RMSE & spread / CRPSS / spread-skill ratio vs lead day
  nao_daily_paired_djf_{tag}.png  paired correlation differences vs lead day
  nao_daily_acc_djf_{tag}.png     ACC vs lead day + paired ACC differences
Output: /glade/work/bgong/plots/
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
from model_colors import model_color, model_style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, 'results')
PLOT_DIR = '/glade/work/bgong/plots'
MARKERS = {'ucs2s_v2': 'o', 'fuxi': 's', 'si_s2s': '^', 'ngcm': 'D', 'ecmwf_rf2018': 'v', 'ecmwf_rt': 'v', 'ecmwf_rf2025': 'v', 'ucs2s_v2_fbcorr': 'P', 'ucs2s_multicond': 'X', 'ucs2s_mc12800': 'X', 'ucs2s_sstanom': 'P'}
MIN_MONTHS = 6          # fewer month blocks -> no CI band


def describe(S):
    yrs, st, en = S.get('years'), S.get('start'), S.get('end')
    if st or en:
        period = f'DJF inits {st or ""}–{en or ""}'
    elif yrs:
        period = f'DJF inits {yrs[0]}–{yrs[1]}'
    elif S.get('months'):
        period = ''
    else:
        period = 'all DJF inits per model'
    if S.get('months'):
        names = {'11': 'Nov', '12': 'Dec', '01': 'Jan', '02': 'Feb', '03': 'Mar'}
        period += ('' if not period else ', ') + 'inits ' + '/'.join(names.get(m, m) for m in S['months'])
    if S.get('match_to'):
        period += f', matched to {S["names"][S["match_to"]]} inits (±{S["match_tol"]} d)'
    return period


def winters(dates):
    return len({int(d[:4]) + (d[4:6] == '12') for d in dates})


def plot_skill(S, tag):
    names, own, x = S['names'], S['own'], S['leads']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    panels = [('cor', 'Correlation of ensemble mean with ERA5', (-0.2, 1.0)),
              ('rmse', 'RMSE (solid) and ensemble spread (dashed), σ units', (0, 1.6)),
              ('crpss', 'CRPSS vs climatology N(0,1)', (-0.6, 1.0)),
              ('ssr', 'Spread / RMSE', (0, 1.6))]
    for ax, (m, title, ylim) in zip(axes.ravel(), panels):
        for key, r in own.items():
            st = model_style(names[key])
            if len({d[:6] for d in r['dates']}) >= MIN_MONTHS:
                ax.fill_between(x, r[m]['lo'], r[m]['hi'], color=st['color'], alpha=0.12, lw=0)
            ax.plot(x, r[m]['val'], marker=MARKERS.get(key, 'o'), ms=4, markevery=7,
                    label=f'{names[key]} (n={r["n"]}, {winters(r["dates"])} winters)', **st)
            if m == 'rmse':
                spread = r['ssr']['val'] * r['rmse']['val']
                ax.plot(x, spread, color=st['color'], ls='--', lw=1.0)
        if m == 'cor':
            big = max(own, key=lambda q: own[q]['n'])
            ax.plot(x, own[big]['pers_cor']['val'], color='black', ls=':', lw=1.5,
                    label=f'Persistence of init-day ERA5 ({names[big]} inits)')
            ax.axhline(0.5, color='0.6', lw=0.8, ls='--')
        ax.axhline({'cor': 0, 'rmse': 1, 'crpss': 0, 'ssr': 1}[m], color='0.5', lw=0.8)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 43)
        ax.set_xticks(np.arange(0, 43, 7))
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel('Lead day')
    axes[0, 0].legend(fontsize=8, loc='upper right', frameon=False)
    period = describe(S)
    fig.suptitle(f'Daily NAO index, {period}; shading = 95% month-block bootstrap CI', fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'nao_daily_skill_djf_{tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


def plot_pairs(S, tag):
    names, pairs, x = S['names'], S['pairs'], S['leads']
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=True, squeeze=False)
    for ax, ((a, b), r) in zip(axes[0], pairs.items()):
        d = r['diff']['cor']
        ax.fill_between(x, d['lo'], d['hi'], color=model_color(names[a]), alpha=0.18, lw=0)
        ax.plot(x, d['val'], color=model_color(names[a]), lw=1.8)
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set_title(f'{names[a]} − {names[b]} (n={r["n"]})', fontsize=10)
        ax.set_xlabel('Lead day')
        ax.set_xticks(np.arange(0, 43, 7))
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].set_ylabel('Δ correlation')
    axes[0, 0].set_ylim(-0.8, 0.8)
    fig.suptitle('Paired DJF NAO correlation differences on common inits (95% month-block CI)', fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'nao_daily_paired_djf_{tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


def plot_acc(S, tag):
    """ACC vs lead day (own records) and paired ACC differences."""
    names, own, pairs, x = S['names'], S['own'], S['pairs'], S['leads']
    if pairs:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.25, 1]})
    else:                                               # single model: no paired panel
        fig, ax0 = plt.subplots(figsize=(7.5, 4.6))
        axes = [ax0]
    ax = axes[0]
    for key, r in own.items():
        st = model_style(names[key])
        if len({d[:6] for d in r['dates']}) >= MIN_MONTHS:
            ax.fill_between(x, r['acc']['lo'], r['acc']['hi'], color=st['color'], alpha=0.12, lw=0)
        k = np.where(r['acc']['val'] < 0.5)[0]
        day = f', ACC≥0.5 to day {k[0]}' if len(k) else ''
        ax.plot(x, r['acc']['val'], marker=MARKERS.get(key, 'o'), ms=4, markevery=7,
                label=f'{names[key]} (n={r["n"]}{day})', **st)
    big = max(own, key=lambda q: own[q]['n'])
    ax.plot(x, own[big]['pers_acc']['val'], color='black', ls=':', lw=1.5,
            label=f'Persistence of init-day ERA5 ({names[big]} inits)')
    ax.axhline(0.5, color='0.6', lw=0.8, ls='--')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_ylim(-0.2, 1.0)
    ax.set_title('NAO index ACC (ensemble mean vs ERA5)', fontsize=10.5)
    ax.legend(fontsize=8, loc='upper right', frameon=False)
    for i, ((a, b), r) in enumerate(pairs.items()):
        ax = axes[1]
        d = r['diff']['acc']
        sig = (d['lo'] > 0) | (d['hi'] < 0)
        ax.plot(x, d['val'], color=model_color(names[a]), lw=1.5, ls=['-', '--', '-.', ':'][i % 4],
                label=f'{names[a]} − {names[b]} (n={r["n"]})')
        ax.scatter(x[sig], d['val'][sig], s=22, facecolor='none', edgecolor='black', lw=0.8, zorder=3)
    if pairs:
        ax = axes[1]
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_title('Paired ACC difference on common inits (○ = 95% CI excludes 0)', fontsize=10.5)
        ax.legend(fontsize=8, frameon=False, loc='lower left')
    for ax in axes:
        ax.set_xlim(0, 43)
        ax.set_xticks(np.arange(0, 43, 7))
        ax.set_xlabel('Lead day')
        ax.grid(alpha=0.25, lw=0.5)
    period = describe(S)
    fig.suptitle(f'Daily NAO ACC, {period}; shading = 95% month-block bootstrap CI', fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'nao_daily_acc_djf_{tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='alldjf')
    tag = ap.parse_args().tag
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, f'nao_daily_scores_{tag}.pkl'), 'rb') as fh:
        S = pickle.load(fh)
    plot_skill(S, tag)
    plot_pairs(S, tag)
    plot_acc(S, tag)
