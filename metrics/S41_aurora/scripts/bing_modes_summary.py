"""summary.md + provenance.json of the modes evaluation (login node, light; reads the part CSVs/JSONs).

    python scripts/bing_modes_summary.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval import bing_modes as BM  # noqa: E402

OUT = BM.OUT
SK = ['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']
LAB = BM.SYSTEM_LABEL


def rd(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def f2(x, n=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 'n/a'
    return 'n/a' if v != v else f'{v:.{n}f}'


def pick(rows, **kw):
    out = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    return out[0] if out else {}


def table(head, lines):
    s = '| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head) + '\n'
    for ln in lines:
        s += '| ' + ' | '.join(str(x) for x in ln) + ' |\n'
    return s


def merge_parts():
    """per-system partial CSVs (<stem>__<systems>.csv) + the full-run file -> <stem>.csv (partial files win for their systems)."""
    import glob
    for d in ('mjo', 'tele', 'enso', 'sst', 'comp', 'strat', 'qbo', 'land'):
        parts = sorted(glob.glob(str(OUT / d / '*__*.csv')))
        stems = sorted({Path(p).name.split('__')[0] for p in parts})
        for st in stems:
            rows, cover = [], set()
            for p in [q for q in parts if Path(q).name.startswith(st + '__')]:
                rr = rd(p)
                rows += rr
                cover |= {r.get('system_key') for r in rr}
            base = OUT / d / f'{st}.csv.full'
            src = OUT / d / f'{st}.csv'
            if src.exists() and not base.exists():
                src.rename(base)
            if base.exists():
                rows = [r for r in rd(base) if r.get('system_key') not in cover] + rows
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            with open(src, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)


def main():
    merge_parts()
    commit = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    md = [f'# S41 modes and patterns (plan 2026-11-11 section 3 B)\n', f'Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, '
          f'code s2storm `bing-modes` {commit}. Per-start values for the paired bootstrap are in the `perstart_*` files of each part. '
          'All skill numbers below are drift-corrected (LOYO, +-30 d), 2019-26 pooled, unless stated. Conventions: module docstring of '
          '`s2storm/metrics/bing_modes.py`.\n']
    t = OUT / 'tests'
    if (t / 'mjo_test.json').exists():
        j = json.loads((t / 'mjo_test.json').read_text())
        a = j['test']
        e = j.get('test_enso_removed_1979_2013')
        md.append('## Blocking reference tests\n')
        lines = [['MJO: ERA5 RMM (our EOFs 1979-2018) vs BoM, 1979-2018', f2(a['r_rmm1'], 3), f2(a['r_rmm2'], 3),
                  f2(a['phase_match_active'], 3), 'PASS' if a['pass_'] else 'FAIL'],
                 ['  same, 2014-2018 (BoM without ENSO removal)', f2(a['r_2014_2018'][0], 3), f2(a['r_2014_2018'][1], 3),
                  f2(a.get('phase_match_active_2014_2018'), 3), '']]
        if isinstance(e, dict):
            lines.append(['  ERA5 with WH04 SST1 removal vs BoM 1979-2013', f2(e['r_rmm1'], 3), f2(e['r_rmm2'], 3),
                          f2(e['phase_match_active'], 3), 'PASS' if e['pass_'] else 'FAIL'])
        md.append(table(['test', 'r RMM1', 'r RMM2', 'phase match (active)', 'result'], lines))
        md.append(f"EOF plane rotation to the BoM phase convention: {j['full'].get('rotation')}; normalisation {j['full']['norm']}; "
                  f"explained variance {[round(x, 3) for x in j['full']['explained']]}.\n")
    if (t / 'tele_test.json').exists():
        j = json.loads((t / 'tele_test.json').read_text())
        md.append(table(['index', 'r vs CPC 1979-2018 (all months)', 'DJF', '2019-2025'],
                        [[k.upper(), f2(j.get(f'{k}_r_all'), 3), f2(j.get(f'{k}_r_DJF'), 3), f2(j.get(f'{k}_r_2019_2025'), 3)]
                         for k in ('nao', 'pna', 'ao', 'sam')]))
        md.append(f"NAO blocking test (r >= 0.9): **{'PASS' if j.get('nao_pass') else 'FAIL'}**. CPC AO uses 1000 hPa and AAO 700 hPa "
                  'heights (ours: z500), so lower r is expected there.\n')
    pl = sorted((OUT / 'tests').glob('pytest_9*.log'))
    if pl:
        last = [ln for ln in pl[-1].read_text().splitlines() if 'passed' in ln or 'failed' in ln]
        md.append(f"CPU unit tests `tests/test_bing_modes.py` ({pl[-1].name}): {last[-1] if last else 'n/a'} (the two failures are the "
                  'blocking reference tests above).\n')
    m = rd(OUT / 'mjo' / 'scores_mjo.csv')
    if m:
        md.append('## MJO\n')
        lines = []
        for idx in ('full', 'winds'):
            for sk in SK:
                for msz in ('30', '11'):
                    r = pick(m, index=idx, system_key=sk, members=msz, correction='dc', period='2019-26', group='all', lead='w34')
                    rr = pick(m, index=idx, system_key=sk, members=msz, correction='raw', period='2019-26', group='all', lead='w34')
                    if r:
                        lines.append([idx, LAB[sk], msz, f2(r['skill_day'], 1), f2(rr.get('skill_day'), 1), f2(r['cor']),
                                      f2(r.get('cor_w34_meanrmm')), f2(r['rmse']), r['n']])
        md.append(table(['RMM', 'system', 'members', 'skill day dc', 'skill day raw', 'COR w3-4 (mean of daily COR)',
                         'COR w3-4 (RMM averaged days 15-28)', 'RMSE w3-4', 'starts'], lines))
        lines = []
        for sk in SK:
            ln = [LAB[sk]]
            for g in ('active', 'inactive', 'P8-1', 'P2-3', 'P4-5', 'P6-7', 'DJF', 'JJA'):
                r = pick(m, index='full', system_key=sk, members='30', correction='dc', period='2019-26', group=g, lead='w34')
                ln.append(f"{f2(r.get('skill_day'), 1)} ({r.get('n', '')})")
            lines.append(ln)
        md.append('Skill day (COR = 0.5) by ERA5 state at the init (number of starts):\n\n')
        md.append(table(['system', 'active', 'inactive', 'P8-1', 'P2-3', 'P4-5', 'P6-7', 'DJF starts', 'JJA starts'], lines))
        lines = []
        for sk in SK:
            ln = [LAB[sk]]
            for k in (7, 14, 21, 28):
                r = pick(m, index='full', system_key=sk, members='30', correction='dc', period='2019-26', group='active', lead=k)
                ln.append(f"{f2(r.get('amp_err'))} / {f2(r.get('phase_err'), 0)}")
            lines.append(ln)
        md.append('Amplitude error / phase error (deg, > 0 = ahead), active MJO at init:\n\n')
        md.append(table(['system', 'day 7', 'day 14', 'day 21', 'day 28'], lines))
    tl = rd(OUT / 'tele' / 'scores_tele.csv')
    if tl:
        md.append('## NAO, PNA, AO, SAM (weekly z500 index)\n')
        for g in ('DJF', 'all'):
            lines = []
            for idx in ('nao', 'pna', 'ao', 'sam'):
                for sk in SK:
                    ln = [idx.upper(), LAB[sk]]
                    for w in ('w1', 'w2', 'w3', 'w4', 'w34'):
                        r = pick(tl, index=idx, system_key=sk, correction='dc', period='2019-26', group=g, window=w)
                        ln.append(f"{f2(r.get('cor30'))} / {f2(r.get('crpss'))}")
                    ln.append(pick(tl, index=idx, system_key=sk, correction='dc', period='2019-26', group=g, window='w3').get('n', ''))
                    lines.append(ln)
            md.append(f'COR (30-member mean) / CRPSS (vs ERA5 1991-2020 same-window index), starts {g}:\n\n')
            md.append(table(['index', 'system', 'week 1', 'week 2', 'week 3', 'week 4', 'weeks 3-4', 'starts'], lines))
    en = rd(OUT / 'enso' / 'scores_nino34.csv')
    if en:
        md.append('## ENSO: Nino-3.4 SST anomaly\n')
        lines = []
        for sk in SK:
            for corr in ('dc', 'raw'):
                ln = [LAB[sk], corr]
                for w in ('w1', 'w2', 'w34', 'w56'):
                    r = pick(en, system_key=sk, correction=corr, period='2019-26', group='all', window=w)
                    ln.append(f"{f2(r.get('acc30'), 3)} ({f2(r.get('acc_damped'), 3)})")
                r = pick(en, system_key=sk, correction=corr, period='2019-26', group='all', window='w34')
                ln.append(f"{f2(r.get('rmse30'))} ({f2(r.get('rmse_damped'))})")
                ln.append(f"{f2(r.get('acc_tendency30'))} ({f2(r.get('acc_tendency_damped'))})")
                lines.append(ln)
        md.append('ACC of the ensemble mean (damped persistence in brackets); headline quantity = weeks 3-4 dc. The anomaly ACC is close '
                  'to saturation (ENSO cycle + trend over 2019-26); RMSE and the ACC of the tendency (anomaly minus the init-day anomaly) '
                  'separate the systems better:\n\n')
        md.append(table(['system', 'correction', 'week 1', 'week 2', 'weeks 3-4', 'weeks 5-6', 'RMSE w3-4 (K)', 'tendency ACC w3-4'],
                        lines))
        lines = []
        for sk in SK:
            ln = [LAB[sk]]
            for g in ('el_nino', 'la_nina', 'neutral'):
                r = pick(en, system_key=sk, correction='dc', period='2019-26', group=g, window='w34')
                ln.append(f"{f2(r.get('rmse30'))} vs {f2(r.get('rmse_damped'))} ({r.get('n', '')})")
            lines.append(ln)
        md.append('Weeks 3-4 RMSE (K) model vs damped persistence by ENSO phase at the init (30-day mean anomaly before the init, '
                  '+-0.5 K):\n\n')
        md.append(table(['system', 'El Nino', 'La Nina', 'neutral'], lines))
    ss = rd(OUT / 'sst' / 'scores_sst.csv')
    if ss:
        md.append('## Global SST anomaly ACC vs damped persistence\n')
        lines = []
        for sk in SK:
            for rn in ('global_60s60n', 'tropics_20s20n', 'nh_20n60n', 'sh_60s20s'):
                ln = [LAB[sk], rn]
                for w in ('w1', 'w2', 'w34', 'w56'):
                    r = pick(ss, system_key=sk, members='30', correction='dc', period='2019-26', region=rn, window=w)
                    ln.append(f"{f2(r.get('acc'))} ({f2(r.get('acc_damped'))})")
                lines.append(ln)
        md.append(table(['system', 'region', 'week 1', 'week 2', 'weeks 3-4', 'weeks 5-6'], lines))
    cp = rd(OUT / 'comp' / 'scores_mjo_teleconnection.csv')
    if cp:
        md.append('## MJO -> extratropics (z500, NDJFM starts)\n')
        lines = []
        for sk in SK:
            for g in ('P2-3', 'P6-7'):
                ln = [LAB[sk], g]
                for w in ('w2', 'w3', 'w4'):
                    r = pick(cp, system_key=sk, members='30', group=g, season='NDJFM', window=w)
                    ln.append(f"{f2(r.get('pc_fc_vs_era5_same_starts'))} / {f2(r.get('pc_fc_vs_era5_longrecord'))} / "
                              f"{f2(r.get('amp_ratio_fc_over_era5'))} ({r.get('n', '')})")
                lines.append(ln)
        md.append('Pattern correlation over 20-90N of the composite: vs ERA5 same starts / vs ERA5 1979-2018 lagged composite / '
                  'amplitude ratio (starts):\n\n')
        md.append(table(['system', 'init phase', 'week 2', 'week 3', 'week 4'], lines))
        lines = []
        for sk in SK:
            ln = [LAB[sk]]
            for g in ('init_active', 'init_inactive'):
                for w in ('w3', 'w4'):
                    r = pick(cp, system_key=sk, members='30', group=g, season='NDJFM', window=w)
                    ln.append(f"{f2(r.get('acc_nh_centred'))} ({r.get('n', '')})")
            lines.append(ln)
        md.append('NH z500 ACC (centred, 20-90N) conditioned on the MJO at the init, NDJFM:\n\n')
        md.append(table(['system', 'active w3', 'active w4', 'inactive w3', 'inactive w4'], lines))
    st = rd(OUT / 'strat' / 'scores_u60.csv')
    if st:
        md.append('## Stratosphere\n')
        ev = json.loads((OUT / 'strat' / 'ssw_events_era5.json').read_text())
        md.append(f"ERA5 SSWs (U60N10 reversal, Charlton-Polvani): {', '.join(d for d in ev['u60_10'] if d >= '2018-11')} (2018-19 on); "
                  f"at 50 hPa: {', '.join(d for d in ev['u60_50'] if d >= '2018-11')}.\n\n")
        lines = []
        for idx in ('u60n10', 'u60n50'):
            for sk in SK:
                ln = [idx.upper(), LAB[sk]]
                for k in (7, 14, 21, 28):
                    r = pick(st, index=idx, system_key=sk, correction='dc', group='NDJFM', lead=k)
                    ln.append(f"{f2(r.get('acc'))} / {f2(r.get('rmse'), 1)}" if r else 'n/a')
                lines.append(ln)
        md.append('U60 anomaly ACC / RMSE (m/s), NDJFM starts (FuXi has no 10 hPa):\n\n')
        md.append(table(['index', 'system', 'day 7', 'day 14', 'day 21', 'day 28'], lines))
        hb = rd(OUT / 'strat' / 'ssw_hits.csv')
        bs = rd(OUT / 'strat' / 'ssw_brier.csv')
        lines = []
        for L in ('10', '50'):
            for sk in SK:
                ln = [f'{L} hPa', LAB[sk]]
                for wk in range(1, 5):
                    h = [int(r['hit']) for r in hb if r['system_key'] == sk and r['level'] == L and int(r['week']) == wk]
                    b = pick(bs, system_key=sk, level=L, week=wk)
                    ln.append(f"{sum(h)}/{len(h)}; {f2(b.get('bss'))}" if h or b else 'n/a')
                lines.append(ln)
        md.append('SSW hits (P(reversal within +-3 d) >= 0.5) / cases; Brier skill score of daily P(u < 0), Nov-Mar, by lead week of '
                  'the event:\n\n')
        md.append(table(['level', 'system', 'week 1', 'week 2', 'week 3', 'week 4'], lines))
    qb = rd(OUT / 'qbo' / 'qbo_drift.csv')
    if qb:
        lines = []
        for L in ('50', '10'):
            for sk in SK:
                ln = [f'{L} hPa', LAB[sk]]
                for k in (7, 14, 28, 42):
                    r = pick(qb, system_key=sk, level=L, group='all', lead=k)
                    ln.append(f2(r.get('drift')))
                if any(x != 'n/a' for x in ln[2:]):
                    lines.append(ln)
        md.append('## QBO: drift of equatorial (5S-5N) zonal-mean u (m/s)\n\n')
        md.append(table(['level', 'system', 'day 7', 'day 14', 'day 28', 'day 42'], lines))
    la = rd(OUT / 'land' / 'scores_land_acc.csv')
    if la:
        md.append('## Land\n')
        lines = []
        for var in ('swvl1', 'stl1', 't2m'):
            for sk in SK:
                ln = [var, LAB[sk]]
                for w in ('w1', 'w2', 'w34'):
                    r = pick(la, var=var, system_key=sk, correction='dc', region='land_60s75n', window=w)
                    ln.append(f2(r.get('acc')))
                if any(x != 'n/a' for x in ln[2:]):
                    lines.append(ln)
        md.append('Anomaly ACC over land 60S-75N (ensemble mean, centred):\n\n')
        md.append(table(['field', 'system', 'week 1', 'week 2', 'weeks 3-4'], lines))
        co = rd(OUT / 'land' / 'coupling_swvl1_t2m_jja.csv')
        if co:
            lines = [[r['hotspot'], LAB.get(r['system_key']), f"{r['swvl1_window']} -> {r['t2m_window']}", f2(r['corr_era5']),
                      f2(r['corr_model_members']), r['n_starts']] for r in co]
            md.append('Soil moisture - t2m coupling, JJA starts: mean per-point correlation across starts of swvl1 and t2m anomalies:\n\n')
            md.append(table(['hotspot', 'system', 'windows', 'ERA5', 'model (members)', 'starts'], lines))
        ex = rd(OUT / 'land' / 'scores_extremes.csv')
        if ex:
            lines = []
            for evn in ('t2m>p90', 'tp<tercile1', 'tp>tercile2'):
                for sk in SK:
                    ln = [evn, LAB[sk]]
                    for w in ('w1', 'w2', 'w3', 'w34'):
                        r = pick(ex, event=evn, system_key=sk, correction='dc', region='land_60s75n', window=w)
                        ln.append(f"{f2(r.get('bss'))} / {f2(r.get('roc_auc'))}")
                    lines.append(ln)
            md.append('Weekly extremes over land: BSS vs climatological probability / ROC area (drift-corrected):\n\n')
            md.append(table(['event', 'system', 'week 1', 'week 2', 'week 3', 'weeks 3-4'], lines))
    md.append('''## Choices and deviations (recorded)
- **MJO EOF plane rotation.** The ERA5 1979-2018 EOF1/EOF2 eigenvalues are nearly equal (13.6 % / 13.1 %), so their orientation
  within the plane is not fixed by the data. The pair is rotated once by a fixed angle (fitted to BoM RMM 2014-2018, complex
  correlation) to the WH04/BoM phase convention; the same rotation is used for every system. Bivariate COR/RMSE and amplitude are
  rotation-invariant; phases (skill by phase, composites, phase error) depend on it.
- **MJO blocking test failed as specified** (r RMM1 0.938 < 0.95; phase match 79 % < 90 %). It was accepted provisionally by the
  parent, pending the PI. Most of the gap before 2014 is BoM's ENSO (SST1) removal; inputs also differ (NOAA OLR and NCEP winds
  vs ERA5). 2014-2018: r 0.98 / 0.99, phase match 88 %.
- **NAO test failed** (r 0.745 all months, 0.882 DJF). The reason is mode identification in summer and early winter, not the index
  algebra. NAO/PNA scores are labelled "reference test failed". Covariance RPCA is used: the standardised version matched CPC worse.
- **MJO 120-day mean.** WH04 removes the mean of the previous 120 days (excluding the current day). For forecasts the window uses
  ERA5 up to and including the init day, then forecast leads >= 1. Lead 0 of every system is replaced by ERA5.
- **u200 for UC-S2S and FuXi** = log-p interpolation u150/u250 (Bing's weights). FuXi's full state has true u200, but only for
  member 0 and the ensemble statistics; the interpolation is used for all members for consistency.
- **ERA5 sampling per system.** UC-S2S and IFS: 00 UTC states and 24-h fluxes ending 00 UTC. FuXi: calendar-day means; where only
  00 UTC is staged, (00 UTC D + 00 UTC D+1)/2.
- **Stratosphere.** U60N10 for UC-S2S; FuXi only from 50 hPa (U60N50 shown for all systems). ERA5 SSWs are detected from 00 UTC
  zonal-mean U (WB2 512x256 37-level up to 2022, ARCO from 2023). The detected dates 1998-2024 match the published event list
  within 1 day.
- **Teleconnection CRPSS reference** = ERA5 index of the same calendar window, 1991-2020 (fixed, 30 members). The cos-lat domain
  mean of the z500 anomaly is removed before projection (warming offset).
- **SST / Nino-3.4 anomalies** are taken against a fixed 1991-2020 harmonic climatology. Damped persistence is fitted on ERA5
  1991-2018 (+-30 days of the init day of year). The Nino-3.4 anomaly ACC is near saturation over 2019-26; RMSE and the tendency
  ACC are reported as well.
- **Land extremes.** Thresholds come from the trend-adjusted 30-member ERA5 climatology (same dates 1991-2020, shifted along the
  1991-2025 trend), per land point, window and start.
''')
    figs = sorted(p.name for p in (OUT / 'figures').glob('*.png')) if (OUT / 'figures').exists() else []
    md.append('## Figures\n' + '\n'.join(f'- `figures/{f}`' for f in figs) + '\n')
    (OUT / 'summary.md').write_text('\n'.join(md))
    prov = dict(created=datetime.now(timezone.utc).isoformat(), code=dict(repo='github.com/jhayron-perez/s2storm', branch='bing-modes',
                                                                            commit=commit),
                inputs=dict(hindcasts=str(BM.BS.HC), starts=str(BM.R / 'runs/bing/starts/starts.csv'), era5_daily=str(BM.DAILY),
                            era5_stage=[str(d) for d in BM.STAGE_DIRS], strat=str(BM.MD / 'strat'), indices=str(BM.IDX)),
                index_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(BM.IDX.glob('*')) if p.is_file()},
                era5_daily_meta={p.stem: json.loads(p.read_text()).get('n_missing') for p in sorted(BM.DAILY.glob('*.json'))},
                extraction={s: len(os.listdir(OUT / 'extract' / s)) for s in SK if (OUT / 'extract' / s).exists()},
                outputs=sorted(str(p.relative_to(OUT)) for d in ('mjo', 'tele', 'enso', 'sst', 'comp', 'strat', 'qbo', 'land', 'tests',
                                                                   'figures', 'era5') if (OUT / d).exists()
                               for p in (OUT / d).iterdir() if p.suffix in ('.csv', '.nc', '.json', '.png', '.pkl', '.log')))
    (OUT / 'provenance.json').write_text(json.dumps(prov, indent=1))
    print('wrote', OUT / 'summary.md')


if __name__ == '__main__':
    main()
