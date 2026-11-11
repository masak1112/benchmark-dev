"""
UC-S2S multicond vs UC-S2S V2 on their common init dates: weekly TCC, spatial ACC and RMSE for
T2M (land), TP and Z500 (global), weeks 1-6.

multicond and V2 both have 20 members: shown with all 20 and with the first 4 (plus SSTanom, 4 members).
TCC is a correlation ACROSS inits, so it is only meaningful once there are a few dozen common inits;
spatial ACC / RMSE are per-init scores averaged over the inits and are usable with few inits.

Needs the caches from
  tcc_weekly.py --model ucs2s_multicond20 --stage extract --dates-file dates_ucs2s_multicond.txt
  tcc_weekly.py --model ucs2s_multicond   --stage extract --dates-file dates_ucs2s_multicond.txt --n-members 4
  (ucs2s_v2 at 20 and 4 members, ucs2s_sstanom at 4, on the same dates)
Usage:  python plot_multicond_compare.py [--year 2020]
Writes /glade/work/bgong/plots/tcc_acc_rmse_ucs2s_multicond_vs_v2[_2020].{png,pdf}
--year 2020: multicond 2020 inits (ucs2s_multicond2020) vs V2's nearest init (<=1 day apart, V2 runs Mon/Thu),
20 members each; each model is scored against ERA5 on its own init dates.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_style  # noqa: E402

import tcc_weekly as t  # noqa: E402

PLOT_DIR = '/glade/work/bgong/plots'
DATES = os.path.join(t.HERE, 'dates_common_v2_l_d_sst_mc.txt')
WEEKS = ['week1', 'week2', 'week3', 'week4', 'week5', 'week6']
PANELS = [('t2m', 'land', '2-m temperature (land)', 1.0, 'K'),
          ('tp', 'global', 'Total precipitation', 1000.0, 'mm day$^{-1}$'),
          ('z500', 'global', 'Geopotential 500 hPa', 1 / 9.80665, 'gpm')]
# (model key, n members, legend label, style label, linestyle override)
SERIES = [('ucs2s_multicond20', 20, 'UC-S2S multicond (20 members)', 'UC-S2S multicond', None),
          ('ucs2s_v2', 20, 'UC-S2S V2 (20 members)', 'UC-S2S V2', None),
          ('ucs2s_multicond', 4, 'UC-S2S multicond (4 members)', 'UC-S2S multicond', '--'),
          ('ucs2s_v2', 4, 'UC-S2S V2 (4 members)', 'UC-S2S V2', '--'),
          ('ucs2s_sstanom', 4, 'UC-S2S SSTanom (4 members)', 'UC-S2S SSTanom', '--')]

ap = argparse.ArgumentParser()
ap.add_argument('--year', default='2019', choices=['2019', '2020', 'pp12800'],
                help="'pp12800': multicond rollout_ckpt_1050 run, Jan 2019 + Jan 2020, vs V2 / multicond (nearest init)")
YEAR = ap.parse_args().year
if YEAR == '2020':
    SERIES = [('ucs2s_multicond2020', 20, 'UC-S2S multicond (20 members)', 'UC-S2S multicond', None),
              ('ucs2s_v2', 20, 'UC-S2S V2 (20 members, nearest init)', 'UC-S2S V2', None)]
    DATES_OF = {'ucs2s_multicond2020': 'dates_ucs2s_multicond2020.txt', 'ucs2s_v2': 'dates_ucs2sv2_2020_mc.txt'}
elif YEAR == 'pp12800':
    SERIES = [('ucs2s_mc_pp12800', 4, 'UC-S2S multicond pp12800 (4 members)', 'UC-S2S multicond pp12800', None),
              ('ucs2s_multicond_new', 4, 'UC-S2S multicond (4 members)', 'UC-S2S multicond', None),
              ('ucs2s_v2', 4, 'UC-S2S V2 (4 members)', 'UC-S2S V2', None),
              ('ucs2s_v2', 20, 'UC-S2S V2 (20 members)', 'UC-S2S V2', ':')]
    DATES_OF = {'ucs2s_mc_pp12800': 'dates_pp12800_paired.txt', 'ucs2s_multicond_new': 'dates_pp12800_paired.txt',
                'ucs2s_v2': 'dates_v2_pp12800_paired.txt'}
else:
    DATES_OF = {}

read = lambda f: sorted({l.strip() for l in open(os.path.join(t.HERE, f)) if l.strip()})
dates = sorted({l.strip() for l in open(DATES) if l.strip()}) if YEAR == '2019' else read(DATES_OF[SERIES[0][0]])
scores, n_init = {}, None
for key, n_mem, label, *_ in SERIES:
    dates_k = read(DATES_OF[key]) if key in DATES_OF else dates
    t.N_MEMBERS = n_mem
    used, fa, oa = t.load_anomalies(key, dates_k)      # (N, var, week, lat, lon)
    assert used == dates_k, f'{label}: cache missing for {sorted(set(dates_k) - set(used))}'
    n_init = len(used)
    tcc = t.pearson(fa, oa)                              # (var, week, lat, lon)
    s = {}
    for lbl, region, *_ in PANELS:
        i, m = t.LABELS.index(lbl), t.MASKS[region]
        w = t.W2D[m]
        res = np.full((3, len(WEEKS)), np.nan)          # TCC, ACC, RMSE
        for k, wk in enumerate(WEEKS):
            kk = t.WEEK_NAMES.index(wk)
            r = tcc[i, kk][m]
            ok = np.isfinite(r)
            res[0, k] = (w[ok] * r[ok]).sum() / w[ok].sum()
            acc, rmse = [], []
            for n in range(n_init):
                f, o = fa[n, i, kk][m], oa[n, i, kk][m]
                ok = np.isfinite(f) & np.isfinite(o)
                ww, f, o = w[ok], f[ok], o[ok]
                acc.append((ww * f * o).sum() / np.sqrt((ww * f ** 2).sum() * (ww * o ** 2).sum()))
                rmse.append(np.sqrt((ww * (f - o) ** 2).sum() / ww.sum()))
            res[1, k], res[2, k] = np.mean(acc), np.mean(rmse)
        s[lbl] = res
    scores[label] = s

ROWS = ['TCC', 'Spatial ACC', 'RMSE']
fig, axes = plt.subplots(3, len(PANELS), figsize=(5.2 * len(PANELS), 11), sharex=True)
x = np.arange(1, len(WEEKS) + 1)
for c, (lbl, region, title, scale, unit) in enumerate(PANELS):
    for r, row in enumerate(ROWS):
        ax = axes[r, c]
        for key, n_mem, label, style, ls in SERIES:
            st = model_style(style)
            if ls:
                st['linestyle'] = ls
            y = scores[label][lbl][r] * (scale if r == 2 else 1)
            ax.plot(x, y, marker='o', ms=5, label=label, **st)
        if r < 2:
            ax.axhline(0, color='0.6', linewidth=0.8, zorder=0)
        ax.grid(True, color='0.9', linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        if r == 0:
            ax.set_title(title, fontsize=15, pad=8)
        if c == 0:
            ax.set_ylabel(row, fontsize=14)
        if r == 2:
            ax.set_ylabel(f'RMSE ({unit})', fontsize=12)
            ax.set_xlabel('Lead time (weeks)', fontsize=13)
            ax.set_xticks(x)
        ax.tick_params(labelsize=11)
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', ncol=3, fontsize=13, frameon=False,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle(f'{n_init} {"common" if YEAR == "2019" else "paired (≤1 day apart)"} inits ({dates[0]}–{dates[-1]})' +
             ('  —  TCC is not meaningful with this few inits' if n_init < 20 else ''),
             fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
stem = os.path.join(PLOT_DIR, 'tcc_acc_rmse_ucs2s_multicond_vs_v2' + ('' if YEAR == '2019' else f'_{YEAR}'))
for ext in ('png', 'pdf'):
    fig.savefig(f'{stem}.{ext}', dpi=200, bbox_inches='tight')
print(f'Saved: {stem}.png / .pdf  ({n_init} inits)')
