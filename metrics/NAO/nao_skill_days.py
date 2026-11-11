"""
NAO skill days: last lead day before the ensemble-mean NAO ACC first drops below a threshold
(0.5 / 0.6), with 95% CIs from the month-block bootstrap (same resampling as nao_daily_score.py).
Resamples that never drop below the threshold within 42 days count as 42 (censored, flagged '>').

Samples: alldjf (every DJF init per model), w1921 (inits 2019-01-01..2021-02-28, own inits),
w1921match (same winters, FuXi / NeuralGCM matched to the UC-S2S V2 inits within 3 days).
Output: results/nao_skill_days.csv, /glade/work/bgong/plots/nao_skill_days_djf.png (with CIs) and
        nao_skill_days_djf_noci.png (point values only)
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nao_daily_score as N  # noqa: E402

sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_color  # noqa: E402

THRESHOLDS = (0.5, 0.6)
SAMPLES = [
    ('alldjf', 'All DJF inits per model', ['ucs2s_v2', 'ngcm', 'fuxi'], None, None, None),
    ('w1921', 'Same winters 2019–2021, own inits', ['ucs2s_v2', 'ngcm', 'fuxi'], '20190101', '20210228', None),
    ('w1921match', 'Same winters, matched to UC-S2S V2 dates', ['ucs2s_v2', 'ngcm', 'fuxi'], '20190101', '20210228', 'ucs2s_v2'),
]
SIMPLE_SAMPLES = ('w1921', 'w1921match')       # panels in the no-CI figure
MARKERS = {'ucs2s_v2': 'o', 'fuxi': 's', 'si_s2s': '^', 'ngcm': 'D', 'pers': 'x'}


def skill_day(acc, thr):
    """acc (R, L) -> (R,) last lead day with ACC >= thr before the first drop; L if it never drops."""
    below = acc < thr
    first = np.where(below.any(axis=1), below.argmax(axis=1), acc.shape[1])
    return first                     # index of first drop (0-based) == last skilful lead day (1-based)


def main():
    e = {k: v for k, v in np.load(N.EOF_FILE).items()}
    rows = []
    for tag, title, models, start, end, match in SAMPLES:
        rng = np.random.default_rng(N.SEED)
        data = {m: N.load_model(m, e, None) for m in models}
        for m, D in data.items():
            keep = np.array([(not start or d >= start) and (not end or d <= end) for d in D['dates']])
            data[m] = N.subset(D, np.where(keep)[0])
        if match:
            for m in data:
                if m != match:
                    data[m] = N.match_dates(data[m], data[match]['dates'], 3)
        for m, D in data.items():
            n = len(D['dates'])
            W = N.resample_weights(N.block_ids(D['dates'], 'month'), N.N_BOOT, rng)
            s = N.wstats(D, np.arange(n), W)
            for key, acc in ((m, s['acc']), ('pers', s['pers_acc'])):
                if key == 'pers' and m != (match or max(data, key=lambda q: len(data[q]['dates']))):
                    continue
                for thr in THRESHOLDS:
                    sd = skill_day(acc, thr)
                    lo, hi = np.percentile(sd[1:], [2.5, 97.5])
                    rows.append(dict(sample=tag, title=title, model=key,
                                     name='Persistence (ERA5)' if key == 'pers' else N.MODELS[m]['name'],
                                     n=n, thr=thr, day=int(sd[0]), lo=int(lo), hi=int(hi),
                                     censored=bool(hi >= acc.shape[1])))
    rows.sort(key=lambda r: r['model'] == 'pers')        # stable: persistence last within each sample
    out = os.path.join(N.RES_DIR, 'nao_skill_days.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}\n')
    for tag, title, *_ in SAMPLES:
        print(title)
        for r in [r for r in rows if r['sample'] == tag and r['thr'] == 0.5]:
            r6 = next(x for x in rows if x['sample'] == tag and x['model'] == r['model'] and x['thr'] == 0.6)
            print(f'  {r["name"]:<20} n={r["n"]:3d}  ACC>=0.5: day {r["day"]:2d} [{r["lo"]}–{">" if r["censored"] else ""}{r["hi"]}]'
                  f'   ACC>=0.6: day {r6["day"]:2d} [{r6["lo"]}–{">" if r6["censored"] else ""}{r6["hi"]}]')
        print()
    plot(rows)
    plot_simple(rows)


def plot(rows):
    fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5.2 * len(SAMPLES), 3.9), sharex=True)
    for ax, (tag, title, *_) in zip(axes, SAMPLES):
        sel = [r for r in rows if r['sample'] == tag and r['thr'] == 0.5]
        for i, r in enumerate(sel):
            y = len(sel) - 1 - i
            c = 'black' if r['model'] == 'pers' else model_color(r['name'])
            for thr, dy, fill in ((0.5, 0.12, True), (0.6, -0.12, False)):
                q = next(x for x in rows if x['sample'] == tag and x['model'] == r['model'] and x['thr'] == thr)
                ax.plot([q['lo'], q['hi']], [y + dy] * 2, color=c, lw=1.6, alpha=0.8)
                ax.plot(q['day'], y + dy, marker=MARKERS[r['model']], ms=8, color=c,
                        mfc=c if fill else 'white', mew=1.5, ls='none')
                if q['censored']:
                    ax.annotate('', xy=(43.5, y + dy), xytext=(41.5, y + dy),
                                arrowprops=dict(arrowstyle='->', color=c, lw=1.2))
                ax.text(q['day'], y + dy + 0.17 * np.sign(dy), str(q['day']), ha='center',
                        va='bottom' if dy > 0 else 'top', fontsize=8, color='0.25')
        ax.set_yticks(range(len(sel)), [f'{r["name"]}\n(n={r["n"]})' for r in sel][::-1], fontsize=9)
        ax.set_ylim(-0.6, len(sel) - 0.4)
        ax.set_xlim(0, 44)
        ax.set_xticks(range(0, 43, 7))
        ax.set_xlabel('Skill day (last lead with NAO ACC ≥ threshold)')
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.5, axis='x')
    h = [plt.Line2D([], [], color='0.3', marker='o', ls='none', ms=7, label='ACC ≥ 0.5'),
         plt.Line2D([], [], color='0.3', marker='o', mfc='white', ls='none', ms=7, label='ACC ≥ 0.6'),
         plt.Line2D([], [], color='0.3', lw=1.6, label='95% bootstrap CI (→ = beyond day 42)')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('DJF NAO skill days, ensemble mean (20 members)', fontsize=11.5)
    fig.tight_layout()
    out = '/glade/work/bgong/plots/nao_skill_days_djf.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


def plot_simple(rows):
    """Same skill days without the bootstrap intervals: bars for ACC >= 0.5 (solid) and 0.6 (light).
    Same-winter samples only (the all-DJF panel compares different winters per model)."""
    samples = [s for s in SAMPLES if s[0] in SIMPLE_SAMPLES]
    fig, axes = plt.subplots(1, len(samples), figsize=(5.2 * len(samples), 3.7), sharex=True)
    for ax, (tag, title, *_) in zip(axes, samples):
        sel = [r for r in rows if r['sample'] == tag and r['thr'] == 0.5]
        for i, r in enumerate(sel):
            y = len(sel) - 1 - i
            c = 'black' if r['model'] == 'pers' else model_color(r['name'])
            r6 = next(x for x in rows if x['sample'] == tag and x['model'] == r['model'] and x['thr'] == 0.6)
            for q, dy, alpha in ((r, 0.19, 0.9), (r6, -0.19, 0.4)):
                ax.barh(y + dy, q['day'], height=0.34, color=c, alpha=alpha, edgecolor='white', lw=1)
                ax.text(q['day'] + 0.6, y + dy, str(q['day']), va='center', fontsize=8.5, color='0.2')
        ax.set_yticks(range(len(sel)), [f'{r["name"]}\n(n={r["n"]})' for r in sel][::-1], fontsize=9)
        ax.set_ylim(-0.6, len(sel) - 0.4)
        ax.set_xlim(0, 44)
        ax.set_xticks(range(0, 43, 7))
        ax.set_xlabel('Skill day (last lead with NAO ACC ≥ threshold)')
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.5, axis='x')
        ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color='0.3', alpha=0.9, label='ACC ≥ 0.5'),
         plt.Rectangle((0, 0), 1, 1, color='0.3', alpha=0.4, label='ACC ≥ 0.6')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('DJF NAO skill days, ensemble mean (20 members)', fontsize=11.5)
    fig.tight_layout()
    out = '/glade/work/bgong/plots/nao_skill_days_djf_noci.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
