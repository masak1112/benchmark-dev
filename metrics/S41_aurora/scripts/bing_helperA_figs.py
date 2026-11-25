"""Helper A slide figures (S41 evaluation v1.5; W1 scorecards, W2 maps) from the cached outputs of ``bing_helperA.py``.

    python scripts/bing_helperA_figs.py tables      # bss_*.csv, landocean_*.csv (numbers with block-bootstrap CIs)
    python scripts/bing_helperA_figs.py figs [IDs]  # slide PNGs + captions + key numbers

Style constants copied from ``s2storm/sci/slide_style.py`` (branch ``aurora-port``) with the fixed model colours of this set
(HELPERS.md): white, 16:9, Open Sans / DejaVu, base 16 pt, no green, no figure title, grey footer, grades on the figure.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

R = Path('/lus/flare/projects/S2Storm/s2storm')
OD = R / 'runs/bing/eval/v1.5'
HA = Path(os.environ.get('HELPERA_DIR', OD / 'helperA'))
SLIDES = Path(os.environ.get('HELPERA_SLIDES', OD / 'slides'))
EXTRA = HA / 'figs'
W, H = 13.33, 7.5
MODEL = {'ucs2s_sstanom': '#D55E00', 'ucs2s_plain': '#0072B2', 'fuxi': '#000000', 'ifs': '#CC79A7', 'era5': '#7a7a7a'}
LABEL = {'ucs2s_sstanom': 'UC-S2S SST-anomaly', 'ucs2s_plain': 'UC-S2S plain', 'fuxi': 'FuXi-S2S', 'ifs': 'IFS'}
SHORT = {'ucs2s_sstanom': 'SST-anomaly', 'ucs2s_plain': 'plain', 'fuxi': 'FuXi-S2S'}
GREY, LGREY = '#7a7a7a', '#bdbdbd'
FIELDS = ['t2m', 'tp24', 'z500', 'sst', 'olr', 'u850']
FNAME = {'t2m': 't2m', 'tp24': 'precip', 'z500': 'z500', 'sst': 'SST', 'olr': 'OLR', 'u850': 'u850', 'nino34': 'Niño-3.4',
         'mjo_full': 'MJO'}
REGIONS = ['global', 'tropics', 'nh_extratropics', 'conus']
RNAME = {'global': 'globe', 'tropics': 'tropics', 'nh_extratropics': 'NH extratropics', 'conus': 'CONUS', 'sh_extratropics': 'SH extratropics'}
WINS = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w34', 'w56']
WNAME = {'w1': 'wk 1', 'w2': 'wk 2', 'w3': 'wk 3', 'w4': 'wk 4', 'w5': 'wk 5', 'w6': 'wk 6', 'w34': 'wks 3–4', 'w56': 'wks 5–6'}
REFNAME = {'fixed': 'fixed 1991–2020 climatology', 'trend': 'trend-adjusted climatology'}
SETUP = ('UC-S2S 403 starts 2019-01 … 2026-09 × 30 members, FuXi-S2S 404 × 30; ERA5, conservative 1.5°; drift-corrected (LOYO ±30 d); '
         'block bootstrap over starts (block 4, 1,000 draws)')
GRADE_NOTE = 'robust = 95 % CI excludes 0 and same sign 2019–24 and 2025–26; suggestive = CI only; noise otherwise'
KEYNUM = []


def plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as p
    p.rcParams.update({'font.family': ['Open Sans', 'DejaVu Sans'], 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                       'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.facecolor': 'white',
                       'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    return p


def save(f, fid, name, footer, slide=True):
    import matplotlib.pyplot as p
    d = SLIDES if slide else EXTRA
    d.mkdir(parents=True, exist_ok=True)
    f.text(0.01, 0.01, footer, fontsize=10.5, color=GREY, ha='left', va='bottom', wrap=True)
    path = d / f'{fid}_{name}.png'
    for dpi in (150, 130, 110, 90):
        f.savefig(path, dpi=dpi)
        if path.stat().st_size < 1_000_000:
            break
    p.close(f)
    print('wrote', path, path.stat().st_size // 1024, 'kB')
    return path


def caption(fid, text):
    d = SLIDES / 'captions'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{fid}.md').write_text(text.strip() + '\n')


def key(fid, text, src):
    KEYNUM.append(f'- **{fid}** {text} — `{src}`')


def rows(path):
    return list(csv.DictReader(open(path)))


def fnum(x):
    return float(x) if x not in ('', 'nan', None) else float('nan')


def paired(a, b):
    out = []
    for f in FIELDS:
        p = HA / 'scorecard' / f'paired_{a}_vs_{b}_{f}.csv'
        if p.exists():
            out += rows(p)
    return out


def mjo_rows(a, b):
    p = HA / 'scorecard' / 'mjo_windows.csv'
    return [r for r in rows(p) if r['a_sys'] == a and r['b_sys'] == b and r['field'] == 'mjo_full'] if p.exists() else []


def ci_txt(r, k=1.0, nd=3):
    return f"{fnum(r['diff']) * k:+.{nd}f} ({fnum(r['ci_lo']) * k:+.{nd}f} … {fnum(r['ci_hi']) * k:+.{nd}f}), {r['grade']}"


# ================================================================== tables (numbers files)
def mbb(n, block=4, nd=1000, seed=20261112):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    o = rng.integers(0, max(n - block, 0) + 1, size=(nd, nb))
    return (o[:, :, None] + np.arange(block)[None, None]).reshape(nd, -1)[:, :n]


def grade(lo, hi, a, b):
    if not (np.isfinite(lo) and np.isfinite(hi)) or not (lo > 0 or hi < 0):
        return 'noise'
    s = np.sign(lo)
    return 'robust' if (np.sign(a) == s and np.sign(b) == s) else 'suggestive'


def periods(starts):
    y = np.array([int(s[:4]) for s in starts])
    return {'2019-24': y <= 2024, '2025-26': y >= 2025}


def table_bss():
    """per-system and paired debiased BSS (lo = below q10, hi = above q90) and RPSS-like numbers for weeks 1-6, w34, w56."""
    from s41eval.bing_eval import extremes as EX
    out_sys, out_pair = [], []
    D = {}
    for sy in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
        for f in ('t2m', 'tp24'):
            p = HA / 'bss' / f'bss_{sy}_{f}.npz'
            if p.exists():
                D[(sy, f)] = dict(np.load(p))
    if not D:
        return
    any_ = next(iter(D.values()))
    regs, wins, evs = list(any_['regions']), list(any_['windows']), list(any_['events'])
    for (sy, f), d in D.items():
        st = list(d['starts'])
        M = int(d['members'])
        for wi, wn in enumerate(wins):
            for ref in ('fixed', 'trend'):
                for ri, rn in enumerate(regs):
                    if rn not in REGIONS + ['land', 'ocean']:
                        continue
                    for ei, ev in enumerate(evs):
                        bf, bc, eo = d[f'bs_f_{ref}'][:, wi, ri, ei], d[f'bs_c_{ref}'][:, wi, ri, ei], d[f'ev_{ref}'][:, wi, ri, ei]
                        ok = np.flatnonzero(np.isfinite(bf) & np.isfinite(bc))
                        if ok.size < 8:
                            continue
                        stat = lambda ix: EX.bss_d(bf[ok][ix], bc[ok][ix], M)  # noqa: E731
                        v = stat(np.arange(ok.size))
                        bs = np.array([stat(ix) for ix in mbb(ok.size)])
                        lo, hi = np.quantile(bs, [0.025, 0.975])
                        pr = periods([st[i] for i in ok])
                        out_sys.append(dict(system=sy, field=f, window=wn, reference=ref, region=rn, event=ev, bss_d=v, ci_lo=lo,
                                            ci_hi=hi, bss_raw=1 - np.nanmean(bf[ok]) / np.nanmean(bc[ok]),
                                            base_rate=np.nanmean(eo[ok]), n_starts=ok.size, members=M,
                                            **{f'v_{k}': stat(np.flatnonzero(m)) for k, m in pr.items()}))
    out_q = []
    for (sy, f), d in D.items():
        if 'rps5_fixed' not in d:
            continue
        st = list(d['starts'])
        M = int(d['members'])
        for wi, wn in enumerate(wins):
            for ref in ('fixed', 'trend'):
                for ri, rn in enumerate(regs):
                    if rn not in REGIONS + ['land', 'ocean']:
                        continue
                    rf, rc = d[f'rps5_{ref}'][:, wi, ri, 0], d[f'rps5_{ref}'][:, wi, ri, 1]
                    ok = np.flatnonzero(np.isfinite(rf) & np.isfinite(rc))
                    if ok.size < 8:
                        continue
                    stat = lambda ix: 1 - rf[ok][ix].mean() / (rc[ok][ix].mean() + EX.rpss_D(M, 5))  # noqa: E731
                    bs = np.array([stat(ix) for ix in mbb(ok.size)])
                    lo, hi = np.quantile(bs, [0.025, 0.975])
                    pr = periods([st[i] for i in ok])
                    out_q.append(dict(system=sy, field=f, window=wn, reference=ref, region=rn, rpss5_d=stat(np.arange(ok.size)), ci_lo=lo,
                                      ci_hi=hi, n_starts=ok.size, members=M, **{f'v_{k}': stat(np.flatnonzero(m)) for k, m in pr.items()}))
    for a, b in (('ucs2s_sstanom', 'ucs2s_plain'), ('ucs2s_sstanom', 'fuxi'), ('ucs2s_plain', 'fuxi')):
        for f in ('t2m', 'tp24'):
            if (a, f) not in D or (b, f) not in D:
                continue
            da, db = D[(a, f)], D[(b, f)]
            common = sorted(set(da['starts']) & set(db['starts']))
            ia = np.array([list(da['starts']).index(s) for s in common])
            ib = np.array([list(db['starts']).index(s) for s in common])
            Ma, Mb = int(da['members']), int(db['members'])
            for wi, wn in enumerate(wins):
                for ref in ('fixed', 'trend'):
                    for ri, rn in enumerate(regs):
                        if rn not in REGIONS:
                            continue
                        for ei, ev in enumerate(evs):
                            xa = [da[f'{k}_{ref}'][ia, wi, ri, ei] for k in ('bs_f', 'bs_c')]
                            xb = [db[f'{k}_{ref}'][ib, wi, ri, ei] for k in ('bs_f', 'bs_c')]
                            ok = np.flatnonzero(np.isfinite(xa[0]) & np.isfinite(xb[0]) & np.isfinite(xa[1]) & np.isfinite(xb[1]))
                            if ok.size < 8:
                                continue

                            def stat(ix):
                                j = ok[ix]
                                return EX.bss_d(xa[0][j], xa[1][j], Ma) - EX.bss_d(xb[0][j], xb[1][j], Mb)
                            v = stat(np.arange(ok.size))
                            bs = np.array([stat(ix) for ix in mbb(ok.size)])
                            lo, hi = np.quantile(bs, [0.025, 0.975])
                            pr = periods([common[i] for i in ok])
                            m = {k: stat(np.flatnonzero(mm)) for k, mm in pr.items()}
                            out_pair.append(dict(a_sys=a, b_sys=b, field=f, window=wn, reference=ref, region=rn, event=ev,
                                                 a=EX.bss_d(xa[0][ok], xa[1][ok], Ma), b=EX.bss_d(xb[0][ok], xb[1][ok], Mb), diff=v,
                                                 ci_lo=lo, ci_hi=hi, n_starts=ok.size, **{f'diff_{k}': x for k, x in m.items()},
                                                 grade=grade(lo, hi, m['2019-24'], m['2025-26']), status='exploratory'))
    from s41eval.bing_eval.scorecard import write_csv
    write_csv(out_sys, HA / 'scorecard' / 'bss_system.csv')
    write_csv(out_pair, HA / 'scorecard' / 'bss_paired.csv')
    if out_q:
        write_csv(out_q, HA / 'scorecard' / 'rpss5_system.csv')
    print('bss tables', len(out_sys), len(out_pair))


def table_landocean():
    out = []
    for sy in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
        for f in FIELDS:
            p = HA / 'landocean' / f'lo_{sy}_{f}.npz'
            if not p.exists():
                continue
            d = np.load(p)
            st = list(d['starts'])
            for wi, wn in enumerate(d['windows']):
                if wn not in ('w34', 'w56'):
                    continue
                for ri, rn in enumerate(d['regions']):
                    c = d['crps'][:, wi, ri]
                    for ref in ('fixed', 'trend'):
                        cr = d[f'cc_{ref}'][:, wi, ri]
                        ok = np.flatnonzero(np.isfinite(c) & np.isfinite(cr))
                        if ok.size < 8:
                            continue
                        stat = lambda ix: 1 - c[ok][ix].sum() / cr[ok][ix].sum()  # noqa: E731
                        bs = np.array([stat(ix) for ix in mbb(ok.size)])
                        lo, hi = np.quantile(bs, [0.025, 0.975])
                        pr = periods([st[i] for i in ok])
                        out.append(dict(system=sy, field=f, window=str(wn), region=str(rn).split('|')[0], surface=str(rn).split('|')[1],
                                        reference=ref, crpss=stat(np.arange(ok.size)), ci_lo=lo, ci_hi=hi, n_starts=ok.size,
                                        **{f'v_{k}': stat(np.flatnonzero(m)) for k, m in pr.items()}))
    from s41eval.bing_eval.scorecard import write_csv
    write_csv(out, HA / 'scorecard' / 'landocean_crpss.csv')
    print('landocean table', len(out))


# ================================================================== A01 / A02 scorecards
def _card(ax, R_, metric, ref, vmax, title, rowlist, mjo=None, show_rows=True, unit=100):
    """heat-map scorecard on ax: rows = rowlist [(field, region)], columns = WINS; cell text = diff x unit; grade frames."""
    import matplotlib.colors as mc
    from matplotlib.patches import Rectangle
    cmap = plt().get_cmap('RdBu_r')
    norm = mc.Normalize(-vmax, vmax)
    idx = {(r['field'], r['region'], r['metric'], r['reference'], r['window']): r for r in R_}
    nr = len(rowlist) + (1 if mjo else 0)
    xs = {w: i + (0.35 if i >= 6 else 0) for i, w in enumerate(WINS)}
    labels = []
    for i, (f, rn) in enumerate(rowlist):
        y = nr - 1 - i
        m = 'acc' if f == 'nino34' else metric
        for w in WINS:
            r = idx.get((f, rn, m, ref, w))
            if r is None:
                continue
            _cell(ax, xs[w], y, fnum(r['diff']), r['grade'], cmap, norm, unit)
        labels.append((y, f"{FNAME[f]}{' ACC' if f == 'nino34' and metric != 'acc' else ''} · {RNAME.get(rn, '')}" if f != 'nino34'
                       else f"Niño-3.4{' ACC' if metric != 'acc' else ''}"))
    if mjo:
        y = 0
        mi = {r['window']: r for r in mjo}
        for w in WINS:
            if w in mi:
                _cell(ax, xs[w], y, fnum(mi[w]['diff']), mi[w]['grade'], cmap, norm, unit)
        labels.append((y, 'MJO RMM COR'))
    ax.set_xlim(-0.5, xs['w56'] + 0.5)
    ax.set_ylim(-0.5, nr - 0.5)
    ax.set_xticks([xs[w] for w in WINS])
    ax.set_xticklabels([WNAME[w] for w in WINS], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([t for _, t in labels] if show_rows else [], fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(1, len(rowlist) // 4 + 1):           # field separators
        ax.axhline(nr - 0.5 - 4 * i, color='k', lw=0.6)
    ax.add_patch(Rectangle((xs['w34'] - 0.5, -0.5), 1, nr, fill=False, ec='k', lw=1.6, ls='-'))
    ax.set_title(title, fontsize=14, pad=26)
    return cmap, norm


def _cell(ax, x, y, v, g, cmap, norm, unit):
    from matplotlib.patches import Rectangle
    if not np.isfinite(v):
        return
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=cmap(norm(v)), ec='white', lw=0.5))
    col = 'k' if abs(norm(v) - 0.5) < 0.35 else 'white'
    if g == 'robust':
        ax.add_patch(Rectangle((x - 0.44, y - 0.40), 0.88, 0.80, fill=False, ec='k', lw=1.4))
        ax.text(x, y, f'{v * unit:+.1f}', ha='center', va='center', fontsize=8.5, fontweight='bold', color=col)
    elif g == 'suggestive':
        ax.text(x, y, f'{v * unit:+.1f}', ha='center', va='center', fontsize=8.5, color=col)
    else:
        ax.text(x, y, f'{v * unit:+.1f}', ha='center', va='center', fontsize=7, color=LGREY if col == 'k' else '#eeeeee')


ROWS = [(f, rn) for f in FIELDS for rn in REGIONS]


def fig_A01():
    R_ = paired('ucs2s_sstanom', 'ucs2s_plain')
    if not R_:
        return
    mj = mjo_rows('ucs2s_sstanom', 'ucs2s_plain')
    p = plt()
    f, axs = p.subplots(1, 2, figsize=(W, H))
    rl = ROWS + [('nino34', 'nino34')]
    for k, ref in enumerate(('fixed', 'trend')):
        cmap, norm = _card(axs[k], R_, 'crpss', ref, 0.02, f'ΔCRPSS vs {REFNAME[ref]} (× 100)', rl, mjo=mj, show_rows=(k == 0))
    f.subplots_adjust(left=0.15, right=0.93, top=0.86, bottom=0.12, wspace=0.05)
    cax = f.add_axes([0.945, 0.25, 0.012, 0.5])
    sm = p.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = f.colorbar(sm, cax=cax)
    cb.set_ticks([-0.02, -0.01, 0, 0.01, 0.02])
    cb.set_ticklabels(['−2', '−1', '0', '+1', '+2'])
    cb.ax.tick_params(labelsize=11)
    cb.set_label('red: SST-anomaly better', fontsize=11)
    f.text(0.15, 0.995, 'UC-S2S SST-anomaly − UC-S2S plain, paired by start (403). Framed column (weeks 3–4) = pre-registered;\n'
           'other columns exploratory. Niño-3.4 row: ΔACC; MJO row: Δ mean daily RMM COR (402 starts).', fontsize=11.5, ha='left', va='top')
    save(f, 'A01', 'scorecard_sstanom_vs_plain',
         f'{SETUP}. Cells: Δ × 100; bold + frame = robust, plain text = suggestive, grey = noise ({GRADE_NOTE}). SST rows: v1.5 '
         '(land/sea mask fix). Source: helperA/scorecard/paired_ucs2s_sstanom_vs_ucs2s_plain_*.csv, mjo_windows.csv.')
    # key numbers: pre-registered w34 rows + counts by grade
    idx = {(r['field'], r['region'], r['metric'], r['reference'], r['window']): r for r in R_}
    nrob = {}
    for r in R_:
        if r['metric'] == 'crpss':
            nrob.setdefault(r['window'], []).append(r['grade'])
    src = 'v1.5/helperA/scorecard/paired_ucs2s_sstanom_vs_ucs2s_plain_<field>.csv'
    for k_ in [('t2m', 'tropics', 'crpss', 'trend'), ('tp24', 'tropics', 'crpss', 'trend'), ('sst', 'global', 'crpss', 'trend'),
               ('sst', 'tropics', 'crpss', 'trend'), ('olr', 'tropics', 'crpss', 'trend'), ('t2m', 'global', 'crpss', 'trend')]:
        for w in ('w34', 'w56', 'w1'):
            r = idx.get(k_ + (w,))
            if r:
                key('A01', f'SST-anomaly − plain ΔCRPSS ({REFNAME[k_[3]]}) {FNAME[k_[0]]} {RNAME[k_[1]]} {WNAME[w]}: {ci_txt(r)} '
                    f'[{r["status"]}]', src)
    for w in WINS:
        g = nrob.get(w, [])
        key('A01', f'{WNAME[w]}: of {len(g)} ΔCRPSS cells (6 fields × 4 regions × 2 refs) {g.count("robust")} robust, '
            f'{g.count("suggestive")} suggestive', src)
    for w in ('w34', 'w56', 'w6'):
        m = [r for r in mj if r['window'] == w]
        if m:
            key('A01', f'MJO mean daily COR SST-anomaly − plain {WNAME[w]}: {ci_txt(m[0])} [{m[0]["status"]}]',
                'v1.5/helperA/scorecard/mjo_windows.csv')
    r34 = [r for r in R_ if r['window'] == 'w34' and r['metric'] == 'crpss']
    nr_ = sum(r['grade'] == 'robust' for r in r34)
    mx = max(abs(fnum(r['diff'])) for r in r34)
    w56 = [r for r in R_ if r['window'] == 'w56' and r['metric'] == 'crpss']
    caption('A01', f"Scorecard UC-S2S SST-anomaly − plain (ΔCRPSS, drift-corrected, both references, weeks 1–6, 3–4, 5–6) · weeks 3–4 "
            f"(pre-registered): {nr_}/{len(r34)} cells robust, all |Δ| ≤ {mx:.3f}; weeks 5–6 (exploratory): "
            f"{sum(r['grade'] == 'robust' for r in w56)}/{len(w56)} robust · "
            f"`v1.5/helperA/scorecard/paired_ucs2s_sstanom_vs_ucs2s_plain_*.csv`")


def fig_A02(a='ucs2s_sstanom', fid='A02', slide=True):
    R_ = paired(a, 'fuxi')
    if not R_:
        return
    mj = mjo_rows(a, 'fuxi')
    p = plt()
    f, axs = p.subplots(1, 2, figsize=(W, H))
    rl = ROWS + [('nino34', 'nino34')]
    _card(axs[0], R_, 'crpss', 'trend', 0.10, 'ΔCRPSS vs trend-adjusted climatology (× 100)', rl, mjo=mj)
    cmap, norm = _card(axs[1], R_, 'acc', 'trend', 0.10, 'Δ centred ACC (× 100)', rl, mjo=mj, show_rows=False)
    f.subplots_adjust(left=0.15, right=0.93, top=0.86, bottom=0.12, wspace=0.05)
    cax = f.add_axes([0.945, 0.25, 0.012, 0.5])
    cb = f.colorbar(p.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
    cb.set_ticklabels(['−10', '−5', '0', '+5', '+10'])
    cb.ax.tick_params(labelsize=11)
    cb.set_label(f'red: {SHORT[a]} better', fontsize=11)
    f.text(0.15, 0.995, f'{LABEL[a]} − FuXi-S2S, paired on 403 common starts, each verified against its own ERA5 sampling\n'
           '(UC-S2S 00 UTC, FuXi daily mean). All columns exploratory (secondary comparison). IFS: not yet available.',
           fontsize=11.5, ha='left', va='top')
    save(f, fid, f'scorecard_{SHORT[a].lower().replace("-", "")}_vs_fuxi',
         f'{SETUP}. Cells: Δ × 100; bold + frame = robust, plain = suggestive, grey = noise ({GRADE_NOTE}). Niño-3.4: ΔACC; '
         f'MJO: Δ mean daily COR. Source: helperA/scorecard/paired_{a}_vs_fuxi_*.csv, mjo_windows.csv.', slide=slide)
    if not slide:
        return
    idx = {(r['field'], r['region'], r['metric'], r['reference'], r['window']): r for r in R_}
    src = f'v1.5/helperA/scorecard/paired_{a}_vs_fuxi_<field>.csv'
    for fl in FIELDS:
        for rn in ('global', 'tropics'):
            for m in ('crpss', 'acc'):
                r = idx.get((fl, rn, m, 'trend', 'w34'))
                if r:
                    key(fid, f'{SHORT[a]} − FuXi Δ{m.upper()} (trend ref.) {FNAME[fl]} {RNAME[rn]} wks 3–4: {ci_txt(r)} '
                        f'(UC {fnum(r["a"]):.3f}, FuXi {fnum(r["b"]):.3f})', src)
    for w in ('w34', 'w56'):
        m = [r for r in mj if r['window'] == w]
        if m:
            key(fid, f'MJO mean daily COR {SHORT[a]} − FuXi {WNAME[w]}: {ci_txt(m[0])} (UC {fnum(m[0]["a"]):.3f}, '
                f'FuXi {fnum(m[0]["b"]):.3f})', 'v1.5/helperA/scorecard/mjo_windows.csv')
    c34 = [r for r in R_ if r['window'] == 'w34' and r['metric'] == 'crpss' and r['reference'] == 'trend' and r['field'] != 'nino34']
    a34 = [r for r in R_ if r['window'] == 'w34' and r['metric'] == 'acc' and r['reference'] == 'trend' and r['field'] != 'nino34']
    pos = sum(r['grade'] == 'robust' and fnum(r['diff']) > 0 for r in c34)
    neg = sum(r['grade'] == 'robust' and fnum(r['diff']) < 0 for r in c34)
    apos = sum(r['grade'] == 'robust' and fnum(r['diff']) > 0 for r in a34)
    aneg = sum(r['grade'] == 'robust' and fnum(r['diff']) < 0 for r in a34)
    mj34 = [r for r in mj if r['window'] == 'w34']
    caption(fid, f"Scorecard {LABEL[a]} − FuXi-S2S (ΔCRPSS trend-adjusted ref., Δ centred ACC; weeks 1–6, 3–4, 5–6) · weeks 3–4: "
            f"CRPSS robustly better in {pos}/{len(c34)} field×region cells, worse in {neg}; ACC better {apos}, worse {aneg}; "
            f"MJO COR {ci_txt(mj34[0]) if mj34 else 'n/a'} (exploratory) · `{src}`")


# ================================================================== A03 lines
def fig_A03(field='t2m', fid='A03', slide=True):
    import matplotlib.lines as ml
    S = {}
    for sy in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
        p_ = HA / 'scorecard' / f'system_{sy}_{field}.csv'
        if p_.exists():
            S[sy] = rows(p_)
    if not S:
        return
    p = plt()
    f, axs = p.subplots(2, 4, figsize=(W, H), sharex=True)
    x = np.arange(6)
    for i, ref in enumerate(('fixed', 'trend')):
        for j, rn in enumerate(REGIONS):
            ax = axs[i, j]
            ax.axhline(0, color=GREY, lw=0.8)
            for k, (sy, rr) in enumerate(S.items()):
                d = {r['window']: r for r in rr if r['metric'] == 'crpss' and r['reference'] == ref and r['region'] == rn}
                v = np.array([fnum(d[w]['value']) for w in WINS[:6]])
                lo = np.array([fnum(d[w]['ci_lo']) for w in WINS[:6]])
                hi = np.array([fnum(d[w]['ci_hi']) for w in WINS[:6]])
                a_ = np.array([fnum(d[w]['v_2019-24']) for w in WINS[:6]])
                b_ = np.array([fnum(d[w]['v_2025-26']) for w in WINS[:6]])
                off = (k - 1) * 0.12
                ax.fill_between(x, lo, hi, color=MODEL[sy], alpha=0.15, lw=0)
                ax.plot(x, v, color=MODEL[sy], lw=2.2)
                ax.plot(x + off, a_, 'v', color=MODEL[sy], ms=5, mfc='white')
                ax.plot(x + off, b_, '^', color=MODEL[sy], ms=5)
            if i == 0:
                ax.set_title(RNAME[rn], fontsize=15)
            if j == 0:
                ax.set_ylabel(f'CRPSS vs\n{"fixed 1991–2020" if ref == "fixed" else "trend-adjusted"}', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{w[1]}' for w in WINS[:6]], fontsize=12)
            ax.tick_params(labelsize=11)
            if i == 1:
                ax.set_xlabel('lead week', fontsize=13)
    hs = [ml.Line2D([], [], color=MODEL[s], lw=2.5, label=LABEL[s]) for s in S]
    hs += [ml.Line2D([], [], color=MODEL['ifs'], lw=2.5, ls=':', label='IFS (not yet available)'),
           ml.Line2D([], [], color=GREY, marker='v', ls='', mfc='white', label='2019–24'),
           ml.Line2D([], [], color=GREY, marker='^', ls='', label='2025–26')]
    f.legend(handles=hs, loc='upper center', ncol=6, fontsize=11.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    f.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.14, hspace=0.18, wspace=0.28)
    save(f, fid, f'crpss_by_week_{field}',
         f'{FNAME[field]}: fair CRPSS of weekly means by lead week (line = pooled 2019–26, band = 95 % block-bootstrap CI; markers = '
         f'2019–24 / 2025–26). {SETUP}. Exploratory. Source: helperA/scorecard/system_<system>_{field}.csv.', slide=slide)
    if slide:
        src = f'v1.5/helperA/scorecard/system_<system>_{field}.csv'
        for sy, rr in S.items():
            d = {(r['window'], r['reference'], r['region']): r for r in rr if r['metric'] == 'crpss'}
            for ref in ('fixed', 'trend'):
                v = [fnum(d[(w, ref, 'global')]['value']) for w in WINS[:6]]
                key(fid, f'{LABEL[sy]} {FNAME[field]} global CRPSS ({REFNAME[ref]}) weeks 1–6: ' + ', '.join(f'{x:.3f}' for x in v), src)
            for ref in ('fixed', 'trend'):
                r = d[('w34', ref, 'global')]
                key(fid, f'{LABEL[sy]} {FNAME[field]} global CRPSS ({REFNAME[ref]}) wks 3–4: {fnum(r["value"]):.3f} '
                    f'({fnum(r["ci_lo"]):.3f} … {fnum(r["ci_hi"]):.3f}); 2019–24 {fnum(r["v_2019-24"]):.3f}, 2025–26 {fnum(r["v_2025-26"]):.3f}',
                    src)
        g = {sy: {(r['window'], r['reference'], r['region']): r for r in rr if r['metric'] == 'crpss'} for sy, rr in S.items()}
        caption(fid, f"{FNAME[field]} CRPSS by lead week per system, 4 regions × 2 references (pooled with CI; 2019–24 / 2025–26 markers) · "
                f"global wks 3–4 vs trend-adjusted ref.: " + ', '.join(f"{SHORT[s]} {fnum(g[s][('w34', 'trend', 'global')]['value']):.3f}"
                                                                   for s in g) + " (exploratory) · `" + src + "`")


# ================================================================== A04 RPSS + BSS
def fig_A04():
    bs = HA / 'scorecard' / 'bss_system.csv'
    if not bs.exists():
        return
    B = rows(bs)
    S = {sy: rows(HA / 'scorecard' / f'system_{sy}_{fl}.csv') for sy in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi') for fl in ('t2m',)}
    p = plt()
    f, axs = p.subplots(2, 2, figsize=(W, H), sharey='row')
    systems = ['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']
    cats = [('rpss', None, 'tercile\nRPSS'), ('bss', 'lo', 'BSS\n< 10th pct'), ('bss', 'hi', 'BSS\n> 90th pct')]
    out_rows = []
    for i, fl in enumerate(('t2m', 'tp24')):
        SR = {sy: rows(HA / 'scorecard' / f'system_{sy}_{fl}.csv') for sy in systems}
        for j, ref in enumerate(('fixed', 'trend')):
            ax = axs[i, j]
            ax.axhline(0, color=GREY, lw=0.8)
            for ci_, (m, ev, lab) in enumerate(cats):
                for k, sy in enumerate(systems):
                    if m == 'rpss':
                        r = [r for r in SR[sy] if r['metric'] == 'rpss' and r['window'] == 'w34' and r['reference'] == ref
                             and r['region'] == 'global'][0]
                        v, lo, hi = fnum(r['value']), fnum(r['ci_lo']), fnum(r['ci_hi'])
                    else:
                        r = [r for r in B if r['system'] == sy and r['field'] == fl and r['window'] == 'w34' and r['reference'] == ref
                             and r['region'] == 'global' and r['event'] == ev][0]
                        v, lo, hi = fnum(r['bss_d']), fnum(r['ci_lo']), fnum(r['ci_hi'])
                    xx = ci_ + (k - 1) * 0.26
                    ax.bar(xx, v, 0.24, color=MODEL[sy], label=LABEL[sy] if (ci_ == 0 and i == 0 and j == 0) else None)
                    ax.errorbar(xx, v, [[v - lo], [hi - v]], color=GREY if sy == 'fuxi' else 'k', lw=1.2, capsize=3)
                    ax.text(xx, max(hi, 0) + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
                    out_rows.append(f'{LABEL[sy]} {FNAME[fl]} global wks 3–4 {lab.replace(chr(10), " ")} ({REFNAME[ref]}): {v:.3f} '
                                    f'({lo:.3f} … {hi:.3f})')
            if fl == 't2m':
                ax.axhspan(0.07, 0.10, xmin=0.0, xmax=0.33, color=MODEL['ifs'], alpha=0.25, lw=0)
                ax.text(-0.42, 0.105, 'debiased ECMWF t2m RPSS\n0.07–0.10 (Guan et al. 2026)', fontsize=9.5, color='#8a3f6e', va='bottom')
            ax.set_xticks(range(3))
            ax.set_xticklabels([c[2] for c in cats], fontsize=12)
            ax.set_title(f'{FNAME[fl]} · vs {REFNAME[ref]}', fontsize=14)
            ax.tick_params(labelsize=11)
    axs[0, 0].legend(loc='upper right', fontsize=11, frameon=False)
    f.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.14, hspace=0.42, wspace=0.08)
    save(f, 'A04', 'rpss_bss_w34',
         'Weeks 3–4, global, drift-corrected members; tercile RPSS and BSS of 10th/90th-percentile events, thresholds from the 30-member '
         'ERA5 reference ensemble of the same calendar window; Weigel-debiased (climatological ensemble scores 0); 95 % block-bootstrap CI. '
         'Exploratory. Guan et al. band: their setup (domain, years, ensemble) differs. '
         'Source: helperA/scorecard/system_*.csv (rpss), bss_system.csv.')
    for t in out_rows:
        key('A04', t, 'v1.5/helperA/scorecard/system_<system>_<field>.csv (rpss) / bss_system.csv (bss)')
    g = {(r['system'], r['field'], r['reference'], r['event']): fnum(r['bss_d']) for r in B if r['window'] == 'w34' and r['region'] == 'global'}
    rp = {sy: [r for r in rows(HA / 'scorecard' / f'system_{sy}_t2m.csv') if r['metric'] == 'rpss' and r['window'] == 'w34'
               and r['reference'] == 'trend' and r['region'] == 'global'][0] for sy in systems}
    caption('A04', 'Weeks 3–4 global tercile RPSS and 10th/90th-percentile BSS (debiased, both references) for t2m and precipitation per system · '
            't2m RPSS vs trend-adjusted ref.: ' + ', '.join(f'{SHORT[s]} {fnum(rp[s]["value"]):.3f}' for s in systems) +
            f'; t2m BSS > 90th pct (trend): ' + ', '.join(f'{SHORT[s]} {g.get((s, "t2m", "trend", "hi"), np.nan):.3f}' for s in systems) +
            ' (exploratory; cf. debiased ECMWF t2m RPSS 0.07–0.10, Guan et al. 2026, different setup) · '
            '`v1.5/helperA/scorecard/bss_system.csv`, `system_*_t2m.csv`')


# ================================================================== maps
def lsm():
    m = np.load(R / 'data/era5_bing_1p5/lsm.npy')           # [121, 240] lat ascending
    return np.linspace(-90, 90, 121), np.arange(240) * 1.5, m


def _map(ax, z, vmax, cmap='RdBu_r', sig=None, lat_desc=True, vmin=None):
    la, lo, m = lsm()
    zz = z[::-1] if lat_desc else z
    lon = np.concatenate([lo, [360]])
    zz = np.concatenate([zz, zz[:, :1]], 1)
    im = ax.pcolormesh(lon - 0.75, la, zz, cmap=cmap, vmin=-vmax if vmin is None else vmin, vmax=vmax, shading='nearest',
                       rasterized=True)
    ax.contour(lo, la, m, levels=[0.5], colors='k', linewidths=0.4)
    if sig is not None:
        s = sig[::-1] if lat_desc else sig
        yy, xx = np.nonzero(s[::2, ::2] > 0)
        ax.scatter(lo[::2][xx], la[::2][yy], s=0.6, c='k', marker='.', lw=0)
    ax.set_xlim(0, 360)
    ax.set_ylim(-80, 85)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    return im


SEAS = ['DJF', 'MAM', 'JJA', 'SON']


def fig_maps_system(field='t2m', metric='crpss_trend', win='w34', fid='A05', slide=True):
    import xarray as xr
    p = plt()
    f, axs = p.subplots(3, 4, figsize=(W, H))
    systems = ['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']
    vals = {}
    for i, sy in enumerate(systems):
        ds = xr.open_dataset(OD / 'score' / sy / f'{field}_g15' / 'maps_dc.nc')
        for j, sn in enumerate(SEAS):
            z = ds[f'{metric}_dc'].sel(subset=f'pooled|{sn}', window=win).values
            vm = 0.5 if metric.startswith('crpss') else 1.0
            im = _map(axs[i, j], z, vm)
            w = np.cos(np.deg2rad(ds.lat.values))[:, None] * np.isfinite(z)
            vals[(sy, sn)] = float(np.nansum(np.nan_to_num(z) * w) / w.sum())
            axs[i, j].text(3, -77, f'mean {vals[(sy, sn)]:+.2f}', fontsize=10.5, va='bottom', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
            if i == 0:
                axs[i, j].set_title(sn, fontsize=15)
            if j == 0:
                axs[i, j].set_ylabel(SHORT[sy], fontsize=14, color=MODEL[sy])
    f.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.12, wspace=0.03, hspace=0.06)
    cax = f.add_axes([0.93, 0.25, 0.012, 0.55])
    cb = f.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=11)
    mname = {'crpss_trend': 'CRPSS vs trend-adjusted clim.', 'crpss_fixed': 'CRPSS vs fixed 1991–2020', 'acc_trend': 'centred ACC',
             'acc_fixed': 'centred ACC (fixed-clim. anomalies)'}[metric]
    cb.set_label(mname, fontsize=12)
    name = f'maps_{metric}_{field}_{win}'
    save(f, fid, name, f'{FNAME[field]} {WNAME[win]}, {mname}, per grid point by start season (start month), drift-corrected, pooled '
         f'2019–26 (≈ 100 starts per season). Exploratory. Source: score/<system>/{field}_g15/maps_dc.nc.', slide=slide)
    return vals


def fig_diffmaps(field='t2m', key_='crpss_trend', win='w34', fid='A06', slide=True, subset='all', pairs=None, fields=None):
    import xarray as xr
    pairs = pairs or [('ucs2s_sstanom', 'ucs2s_plain'), ('ucs2s_sstanom', 'fuxi')]
    fields = fields or [field]
    p = plt()
    f, axs = p.subplots(len(fields), len(pairs), figsize=(W, H), squeeze=False)
    info = []
    for i, fl in enumerate(fields):
        for j, (a, b) in enumerate(pairs):
            pth = HA / 'diffmaps' / f'diff_{a}_vs_{b}_{fl}.nc'
            if not pth.exists():
                continue
            ds = xr.open_dataset(pth)
            sel = dict(subset=subset, window=win)
            d = ds[f'{key_}_d'].sel(**sel).values
            s = ds[f'{key_}_sig'].sel(**sel).values
            vm = 0.05 if (a, b) == pairs[0] and 'sstanom' in a and 'plain' in b else 0.2
            im = _map(axs[i, j], d, vm, sig=s)
            fin = np.isfinite(d)
            w = np.cos(np.deg2rad(ds.lat.values))[:, None] * fin
            frac = float((w * (s > 0)).sum() / w.sum())
            pos = float((w * ((s > 0) & (d > 0))).sum() / w.sum())
            neg = float((w * ((s > 0) & (d < 0))).sum() / w.sum())
            n = int(ds['n_starts'].sel(**sel))
            axs[i, j].set_title(f'{FNAME[fl]}: {SHORT[a]} − {SHORT[b]}', fontsize=14)
            axs[i, j].text(2, -78, f'FDR-significant: {100 * pos:.0f} % of area better, {100 * neg:.0f} % worse (n = {n})', fontsize=10.5,
                           va='bottom', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
            cb = f.colorbar(im, ax=axs[i, j], fraction=0.035, pad=0.01)
            cb.ax.tick_params(labelsize=10)
            info.append(dict(field=fl, a=a, b=b, frac=frac, pos=pos, neg=neg, n=n, thr=float(ds['fdr_p_threshold'].sel(
                key=key_, **sel))))
    f.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.12, wspace=0.08, hspace=0.18)
    km = {'crpss_trend': 'ΔCRPSS (trend-adjusted reference)', 'crpss_fixed': 'ΔCRPSS (fixed reference)', 'acc_trend': 'Δ centred ACC',
          'acc_fixed': 'Δ centred ACC (fixed-clim. anomalies)'}[key_]
    sub = 'all starts' if subset == 'all' else f'{subset} starts'
    save(f, fid, f'diffmaps_{key_}_{"_".join(fields)}_{win}_{subset}',
         f'{km}, {WNAME[win]}, {sub}, per grid point, paired by start; dots = significant at Wilks (2016) FDR α = 0.1 (p from block-bootstrap '
         f'SE, 1,000 draws). Red = first system better. Exploratory. Source: helperA/diffmaps/diff_<a>_vs_<b>_<field>.nc.', slide=slide)
    return info


def fig_A07():
    p_ = HA / 'scorecard' / 'landocean_crpss.csv'
    if not p_.exists():
        return
    L = rows(p_)
    p = plt()
    flds = ['t2m', 'tp24', 'z500']
    f, axs = p.subplots(1, 3, figsize=(W, H), sharey=False)
    systems = ['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']
    regs = ['global', 'tropics', 'nh_extratropics', 'sh_extratropics', 'conus']
    out = []
    for i, fl in enumerate(flds):
        ax = axs[i]
        ax.axvline(0, color=GREY, lw=0.8)
        ylab = []
        yy = 0
        for rn in regs:
            for sf in ('land', 'ocean'):
                if rn == 'conus' and sf == 'ocean':
                    continue
                for k, sy in enumerate(systems):
                    r = [r for r in L if r['system'] == sy and r['field'] == fl and r['window'] == 'w34' and r['reference'] == 'trend'
                         and r['region'] == rn and r['surface'] == sf]
                    if not r:
                        continue
                    r = r[0]
                    v, lo, hi = fnum(r['crpss']), fnum(r['ci_lo']), fnum(r['ci_hi'])
                    y = yy + (1 - k) * 0.26
                    ax.barh(y, v, 0.24, color=MODEL[sy], label=LABEL[sy] if (i == 0 and yy == 0) else None)
                    ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], color=GREY if sy == 'fuxi' else 'k', lw=1, capsize=2)
                    out.append(f'{LABEL[sy]} {FNAME[fl]} {RNAME[rn]} {sf} wks 3–4 CRPSS (trend ref.): {v:.3f} ({lo:.3f} … {hi:.3f})')
                ylab.append((yy, f'{RNAME[rn]} {sf}'))
                yy -= 1
        ax.set_yticks([y for y, _ in ylab])
        ax.set_yticklabels([t for _, t in ylab] if i == 0 else [], fontsize=11)
        ax.set_title(FNAME[fl], fontsize=15)
        ax.tick_params(axis='x', labelsize=11)
        ax.set_xlabel('CRPSS vs trend-adjusted clim.', fontsize=12)
    axs[0].legend(loc='lower left', fontsize=10.5, frameon=False, bbox_to_anchor=(-0.6, -0.2), ncol=3)
    f.subplots_adjust(left=0.15, right=0.99, top=0.93, bottom=0.17, wspace=0.12)
    save(f, 'A07', 'land_ocean_crpss_w34',
         f'Weeks 3–4 fair CRPSS vs the trend-adjusted climatology over land (ERA5 lsm ≥ 0.5) and ocean cells of each region, 95 % block-bootstrap CI. '
         f'{SETUP}. Exploratory. Source: helperA/scorecard/landocean_crpss.csv.')
    for t in out:
        key('A07', t, 'v1.5/helperA/scorecard/landocean_crpss.csv')
    g = {(r['system'], r['field'], r['region'], r['surface']): fnum(r['crpss']) for r in L if r['window'] == 'w34' and r['reference'] == 'trend'}
    caption('A07', 'Weeks 3–4 CRPSS (trend-adjusted ref.) over land vs ocean per region and system (t2m, precipitation, z500) · t2m global land / ocean: '
            + ', '.join(f"{SHORT[s]} {g.get((s, 't2m', 'global', 'land'), np.nan):.3f} / {g.get((s, 't2m', 'global', 'ocean'), np.nan):.3f}"
                        for s in systems) + ' (exploratory) · `v1.5/helperA/scorecard/landocean_crpss.csv`')


def write_keys():
    """per-figure key-number parts in helperA/keys/<ID>.md (rewritten when the figure is redrawn), concatenated into
    slides/key_numbers/helperA.md."""
    kd = HA / 'keys'
    kd.mkdir(parents=True, exist_ok=True)
    byid = {}
    for t in KEYNUM:
        byid.setdefault(t.split('**')[1], []).append(t)
    for fid, ts in byid.items():
        (kd / f'{fid}.md').write_text('\n'.join(ts) + '\n')
    d = SLIDES / 'key_numbers'
    d.mkdir(parents=True, exist_ok=True)
    parts = ''.join(f'\n## {x.stem}\n' + x.read_text() for x in sorted(kd.glob('A*.md')))
    (d / 'helperA.md').write_text('# Key numbers, helper A (W1 scorecards, W2 maps), S41 evaluation v1.5\n'
                                  'Paths relative to `runs/bing/eval/`. Δ = first − second system, 95 % block-bootstrap CI, grade.\n'
                                  + parts)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[0] == 'tables':
        table_bss()
        table_landocean()
        return
    want = set(argv[1:]) or {'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'X'}
    if 'A01' in want:
        fig_A01()
    if 'A02' in want:
        fig_A02()
    if 'A03' in want:
        fig_A03('t2m', 'A03')
    if 'A04' in want:
        fig_A04()
    if 'A05' in want:
        v = fig_maps_system('t2m', 'crpss_trend', 'w34', 'A05')
        caption('A05', 'Weeks 3–4 t2m CRPSS (trend-adjusted ref.) maps per system and start season · area-mean of point CRPSS DJF/MAM/JJA/SON: '
                + '; '.join(f"{SHORT[s]} " + '/'.join(f'{v[(s, sn)]:.2f}' for sn in SEAS) for s in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'))
                + ' (exploratory) · `score/<system>/t2m_g15/maps_dc.nc`')
        for s in ('ucs2s_sstanom', 'ucs2s_plain', 'fuxi'):
            key('A05', f'{LABEL[s]} t2m wks 3–4 cos-lat mean of point CRPSS (trend) DJF/MAM/JJA/SON: '
                + ' / '.join(f'{v[(s, sn)]:.3f}' for sn in SEAS), f'v1.5/score/{s}/t2m_g15/maps_dc.nc')
    if 'A06' in want:
        info = fig_diffmaps(key_='crpss_trend', win='w34', fid='A06', fields=['t2m', 'tp24'])
        for x in info:
            key('A06', f'{FNAME[x["field"]]} wks 3–4 ΔCRPSS (trend) {SHORT[x["a"]]} − {SHORT[x["b"]]}: FDR-significant area {100 * x["pos"]:.1f} % '
                f'better, {100 * x["neg"]:.1f} % worse (n = {x["n"]}, FDR p threshold {x["thr"]:.2g})',
                f'v1.5/helperA/diffmaps/diff_{x["a"]}_vs_{x["b"]}_{x["field"]}.nc')
        if info:
            caption('A06', 'Difference maps weeks 3–4 ΔCRPSS (trend-adjusted ref.), SST-anomaly − plain and SST-anomaly − FuXi, t2m and precipitation, '
                    'Wilks FDR α = 0.1 · ' + '; '.join(f'{FNAME[x["field"]]} {SHORT[x["a"]]}−{SHORT[x["b"]]}: {100 * x["pos"]:.0f} % of area '
                                                       f'better / {100 * x["neg"]:.0f} % worse' for x in info)
                    + ' (exploratory) · `v1.5/helperA/diffmaps/diff_*.nc`')
    if 'A07' in want:
        fig_A07()
    if 'X' in want:                                         # extras into helperA/figs
        fig_A02('ucs2s_plain', 'X02', slide=False)
        for fl in ('tp24', 'z500', 'sst', 'olr', 'u850'):
            fig_A03(fl, 'X03', slide=False)
        for fl in ('t2m', 'tp24', 'z500', 'sst'):
            for m in ('crpss_trend', 'acc_trend', 'crpss_fixed'):
                for w in ('w34', 'w56'):
                    if (fl, m, w) != ('t2m', 'crpss_trend', 'w34'):
                        fig_maps_system(fl, m, w, 'X05', slide=False)
        for k_ in ('crpss_trend', 'acc_trend', 'crpss_fixed'):
            for w in ('w34', 'w56'):
                for sub in ['all'] + SEAS:
                    for fl in (['z500', 'sst'] if (k_, w, sub) == ('crpss_trend', 'w34', 'all') else ['t2m', 'tp24', 'z500', 'sst']):
                        fig_diffmaps(fl, k_, w, 'X06', slide=False, subset=sub,
                                     pairs=[('ucs2s_sstanom', 'ucs2s_plain'), ('ucs2s_sstanom', 'fuxi'), ('ucs2s_plain', 'fuxi')])
    if KEYNUM:
        write_keys()


if __name__ == '__main__':
    main()
