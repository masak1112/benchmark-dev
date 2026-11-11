"""
Figures for the DJF NAO evaluation (reads results/nao_eof_djf.npz and results/nao_scores.pkl).

  nao_eof_djf.png          ERA5 DJF NAO pattern (Z500 per +1 std of the daily index)
  nao_skill_djf_{tag}.png        own-record correlation / RMSE / amplitude ratio vs lead week, 95% CIs
  nao_paired_diff_djf_{tag}.png  paired correlation differences on common inits
Output: /glade/work/bgong/plots/
"""

import os
import pickle
import sys

import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_color, model_style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, 'results')
PLOT_DIR = '/glade/work/bgong/plots'
G = 9.80665
# markers as a second cue next to colour (FuXi grey vs NeuralGCM purple are close)
MARKERS = {'ucs2s_v2': 'o', 'fuxi': 's', 'si_s2s': '^', 'ngcm': 'D'}
MIN_MONTHS = 6
WEEKS = ['week1', 'week2', 'week3', 'week4', 'week5', 'week6']


def winters(dates):
    return len({int(d[:4]) + (d[4:6] == '12') for d in dates})


def plot_eof():
    e = np.load(os.path.join(RES_DIR, 'nao_eof_djf.npz'))
    lat = np.linspace(-89.5, 89.5, 180)[e['lat_idx']]
    lon = np.where(e['lon_idx'] >= 180, e['lon_idx'] - 360, e['lon_idx']).astype(float)
    reg = e['eof'] / e['wgt'] * e['sd_daily'] / G          # gpm per +1 std of the daily index
    vmax = np.ceil(np.abs(reg).max() / 10) * 10
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=-25, central_latitude=50))
    ax.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    cf = ax.contourf(lon, lat, reg, levels=np.linspace(-vmax, vmax, 17), cmap='RdBu_r',
                     transform=ccrs.PlateCarree(), extend='both')
    ax.coastlines(lw=0.6, color='0.3')
    ax.gridlines(lw=0.3, color='0.7', linestyle=':')
    fig.colorbar(cf, ax=ax, shrink=0.8, label='Z500 (gpm) per +1σ NAO')
    fy = e['fit_years']
    ax.set_title(f'ERA5 DJF NAO (EOF1 of DJF-monthly Z500, {fy[0]}–{fy[1]}): {e["evr"][0]:.1%} of variance', fontsize=10)
    out = os.path.join(PLOT_DIR, 'nao_eof_djf.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


def plot_skill(S):
    names, own = S['names'], S['own']
    wn = S['week_names']
    k = [wn.index(w) for w in WEEKS]
    x = np.arange(1, len(WEEKS) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    panels = [('cor', 'Correlation with ERA5', (-0.2, 1.0)), ('rmse', 'RMSE (σ units)', (0, 1.6)),
              ('amp', 'Amplitude ratio σ(forecast)/σ(ERA5)', (0, 1.4))]
    for ax, (m, title, ylim) in zip(axes, panels):
        for i, (key, r) in enumerate(own.items()):
            dx = (i - (len(own) - 1) / 2) * 0.06                # small offset so CIs don't overlap
            st = model_style(names[key])
            few = len({d[:6] for d in r['dates']}) < MIN_MONTHS     # too few month blocks for a CI
            if not few:
                ax.fill_between(x + dx, r[m]['lo'][k], r[m]['hi'][k], color=st['color'], alpha=0.12, lw=0)
            ax.plot(x + dx, r[m]['val'][k], marker=MARKERS.get(key, 'o'), ms=6,
                    label=f'{names[key]} (n={r["n"]}, {winters(r["dates"])} winters' + (', no CI)' if few else ')'), **st)
        if m == 'cor':
            # persistence of the ERA5 days -6..0 index, on the largest init set
            big = max(own, key=lambda q: own[q]['n'])
            ax.plot(x, own[big]['pers_cor']['val'][k], color='black', ls=':', lw=1.5,
                    label=f'Persistence (ERA5, {names[big]} inits)')
            ax.axhline(0, color='0.6', lw=0.8)
        if m == 'rmse':
            ax.axhline(1, color='0.6', lw=0.8, ls='--')
            ax.text(0.75, 1.03, 'climatology', fontsize=8, color='0.4', ha='left', va='bottom')
        if m == 'amp':
            ax.axhline(1, color='0.6', lw=0.8, ls='--')
        ax.set_xticks(x)
        ax.set_xlabel('Lead week')
        ax.set_title(title, fontsize=11)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=8, loc='upper right', frameon=False)
    yrs = S.get('years')
    period = f'DJF inits {yrs[0]}–{yrs[1]}' if yrs else 'all DJF inits per model'
    fig.suptitle(f'Weekly DJF NAO index, ensemble mean (20 members), {period}; shading = 95% month-block bootstrap CI',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'nao_skill_djf_{TAG}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


def plot_pairs(S):
    names, pairs = S['names'], S['pairs']
    if not pairs:
        return
    wn = S['week_names']
    cols = WEEKS + ['week3-4']
    k = [wn.index(w) for w in cols]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for i, ((a, b), r) in enumerate(pairs.items()):
        dx = (i - (len(pairs) - 1) / 2) * 0.15
        d = r['diff']['cor']
        ax.errorbar(x + dx, d['val'][k], yerr=[d['val'][k] - d['lo'][k], d['hi'][k] - d['val'][k]],
                    fmt=MARKERS.get(a, 'o'), ms=6, capsize=3, color=model_color(names[a]),
                    mec=model_color(names[b]), mew=1.5,
                    label=f'{names[a]} − {names[b]} (n={r["n"]})')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xticks(x, [c.replace('week', 'wk ') for c in cols])
    ax.set_ylabel('Δ correlation (A − B)')
    ax.set_title('Paired DJF NAO correlation differences on common inits (95% month-block CI)', fontsize=11)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    ax.legend(fontsize=8, frameon=False, loc='best')
    out = os.path.join(PLOT_DIR, f'nao_paired_diff_djf_{TAG}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='alldjf')
    TAG = ap.parse_args().tag
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, f'nao_scores_{TAG}.pkl'), 'rb') as fh:
        S = pickle.load(fh)
    plot_eof()
    plot_skill(S)
    plot_pairs(S)
