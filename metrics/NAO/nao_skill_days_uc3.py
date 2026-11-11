"""
NAO skill days with 95% month-block bootstrap CIs for the UC-S2S V2 / pp12800 / SSTanom comparison
(same method as nao_skill_days.py: last lead day before the ensemble-mean ACC first drops below 0.5 / 0.6;
resamples that never drop count as 42 and are flagged '>').

Samples
  common : the init dates shared by all three UC-S2S models (dates_common_uc3.txt); FuXi and ECMWF matched
           to them within 3 days
  own    : every model's own DJF inits, Jan 2019 - Feb 2022; ECMWF hindcast Monday/Thursday inits only
Usage:  [NAO_SEASON=ndjfm] python nao_skill_days_uc3.py [--samples common own] [--models ...] [--out F.png]
Output: results/{out}.csv, /glade/work/bgong/plots/{out}.png
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nao_daily_score as N  # noqa: E402
from nao_skill_days import THRESHOLDS, skill_day  # noqa: E402
from plot_NAO_daily import MARKERS  # noqa: E402

sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_color  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ['ucs2s_sstanom', 'ucs2s_mc12800', 'ucs2s_v2', 'fuxi', 'ecmwf_rf2025']
if N.SEASON == 'ndjfm':       # NAO_SEASON=ndjfm: Nov-Mar pattern / cache (no pp12800 cache)
    MONTHS = ('11', '12', '01', '02', '03')
    MODELS = ['ucs2s_sstanom', 'ucs2s_v2', 'fuxi', 'ecmwf_rf2025']
    SAMPLES = [   # tag, title, reference model, reference dates file, start, end, Monday/Thursday-only models
        ('common', 'Common UC-S2S V2 / SSTanom init dates (Nov–Mar, Jan 2019 – Mar 2021)', 'ucs2s_sstanom',
         'dates_common_v2_sstanom_ndjfm.txt', None, None, ()),
        ('own', 'Own inits per model (Nov–Mar, Nov 2018 – Mar 2022)', None, None, '20181101', '20220331',
         ('ecmwf_rf2025',)),
    ]
else:
    MONTHS = ('12', '01', '02')
    SAMPLES = [
        ('common', 'Common UC-S2S init dates (Jan 2019 – Feb 2021)', 'ucs2s_sstanom', 'dates_common_uc3.txt',
         None, None, ()),
        ('own', 'Own inits per model (DJF Jan 2019 – Feb 2022)', None, None, '20190101', '20220228', ('ecmwf_rf2025',)),
    ]


def load(sample, e, models):
    tag, title, ref, ref_file, start, end, monthu = sample
    data = {m: N.load_model(m, e, None) for m in models}
    for m, D in data.items():
        keep = [i for i, d in enumerate(D['dates'])
                if d[4:6] in MONTHS and (not start or d >= start) and (not end or d <= end)
                and (m not in monthu or datetime.strptime(d, '%Y%m%d').weekday() in (0, 3))]
        data[m] = N.subset(D, np.array(keep, dtype=int))
    if ref:
        want = {l.strip() for l in open(os.path.join(HERE, ref_file)) if l.strip()}
        R = data[ref]
        data[ref] = N.subset(R, np.array([i for i, d in enumerate(R['dates']) if d in want], dtype=int))
        for m in data:
            if m != ref:
                data[m] = N.match_dates(data[m], data[ref]['dates'], 3)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', nargs='+', choices=[s[0] for s in SAMPLES], default=[s[0] for s in SAMPLES])
    ap.add_argument('--out', default='nao_skill_days_uc3.png')
    ap.add_argument('--models', nargs='+', default=MODELS)
    a = ap.parse_args()
    samples = [s for s in SAMPLES if s[0] in a.samples]
    e = {k: v for k, v in np.load(N.EOF_FILE).items()}
    rows = []
    for sample in samples:
        tag, title = sample[:2]
        rng = np.random.default_rng(N.SEED)
        for m, D in load(sample, e, a.models).items():
            n = len(D['dates'])
            W = N.resample_weights(N.block_ids(D['dates'], 'month'), N.N_BOOT, rng)
            acc = N.wstats(D, np.arange(n), W)['acc']
            for thr in THRESHOLDS:
                sd = skill_day(acc, thr)
                lo, hi = np.percentile(sd[1:], [2.5, 97.5])
                rows.append(dict(sample=tag, title=title, model=m, name=N.MODELS[m]['name'], n=n,
                                 months=len({d[:6] for d in D['dates']}), thr=thr, day=int(sd[0]),
                                 lo=int(lo), hi=int(hi), censored=bool(hi >= acc.shape[1])))
    out = os.path.join(N.RES_DIR, a.out.replace('.png', '.csv'))
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}\n')
    for tag, title, *_ in samples:
        print(title)
        for r in [r for r in rows if r['sample'] == tag and r['thr'] == 0.5]:
            r6 = next(x for x in rows if x['sample'] == tag and x['model'] == r['model'] and x['thr'] == 0.6)
            print(f'  {r["name"]:<26} n={r["n"]:3d} ({r["months"]} months)  '
                  f'ACC>=0.5: day {r["day"]:2d} [{r["lo"]}–{">" if r["censored"] else ""}{r["hi"]}]   '
                  f'ACC>=0.6: day {r6["day"]:2d} [{r6["lo"]}–{">" if r6["censored"] else ""}{r6["hi"]}]')
        print()
    plot(rows, samples, a.out)


def plot(rows, samples, name):
    fig, axes = plt.subplots(1, len(samples), figsize=(6.2 * len(samples) + (1.0 if len(samples) == 1 else 0), 4.4),
                             sharex=True, squeeze=False)
    for ax, (tag, title, *_) in zip(axes[0], samples):
        sel = [r for r in rows if r['sample'] == tag and r['thr'] == 0.5]
        for i, r in enumerate(sel):
            y = len(sel) - 1 - i
            c = model_color(r['name'])
            for thr, dy, fill in ((0.5, 0.13, True), (0.6, -0.13, False)):
                q = next(x for x in rows if x['sample'] == tag and x['model'] == r['model'] and x['thr'] == thr)
                ax.plot([q['lo'], q['hi']], [y + dy] * 2, color=c, lw=1.8, alpha=0.8)
                ax.plot(q['day'], y + dy, marker=MARKERS.get(r['model'], 'o'), ms=8, color=c,
                        mfc=c if fill else 'white', mew=1.5, ls='none')
                if q['censored']:
                    ax.annotate('', xy=(43.5, y + dy), xytext=(41.5, y + dy),
                                arrowprops=dict(arrowstyle='->', color=c, lw=1.2))
                ax.text(q['day'], y + dy + 0.17 * np.sign(dy), str(q['day']), ha='center',
                        va='bottom' if dy > 0 else 'top', fontsize=8.5, color='0.25')
        ax.set_yticks(range(len(sel)), [f'{r["name"]}\n(n={r["n"]})' for r in sel][::-1], fontsize=9.5)
        ax.set_ylim(-0.6, len(sel) - 0.4)
        ax.set_xlim(0, 44)
        ax.set_xticks(range(0, 43, 7))
        ax.set_xlabel('Skill day (last lead with NAO ACC ≥ threshold)')
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.5, axis='x')
    h = [plt.Line2D([], [], color='0.3', marker='o', ls='none', ms=7, label='ACC ≥ 0.5'),
         plt.Line2D([], [], color='0.3', marker='o', mfc='white', ls='none', ms=7, label='ACC ≥ 0.6'),
         plt.Line2D([], [], color='0.3', lw=1.8, label='95% month-block bootstrap CI (→ = beyond day 42)')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f'{"Nov–Mar" if N.SEASON == "ndjfm" else "Winter (DJF)"} NAO skill days with 95% confidence intervals', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join('/glade/work/bgong/plots', name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
