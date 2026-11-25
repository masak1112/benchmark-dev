"""Driver for ``s41eval.bing_modes`` (modes and patterns of the S41 hindcasts, plan 2026-11-11 section 3 B).

    python scripts/bing_modes.py era5 [--names ...] [--last 2026-11-12] [--workers 4]     # daily ERA5 caches (data/era5_modes/daily)
    python scripts/bing_modes.py mjo_fit                                                  # RMM EOFs 1979-2018 + BoM test
    python scripts/bing_modes.py tele_fit                                                 # NAO/PNA/AO/SAM patterns + CPC test
    python scripts/bing_modes.py tasks --systems ucs2s_sstanom ucs2s_plain fuxi           # task-farm list for the extraction
    python scripts/bing_modes.py extract --system fuxi --start 2019-01-07                  # one start (task)
    python scripts/bing_modes.py score [--systems ...]                                     # all mode scores -> runs/bing/eval/modes
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s41eval import bing_modes as BM  # noqa: E402
from s41eval import bing_scores as BS  # noqa: E402

SYSTEMS = ['ucs2s_sstanom', 'ucs2s_plain', 'fuxi']


def log(*a):
    print(f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}', *a, flush=True)


def code_commit():
    try:
        return subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def _era5_one(args):
    name, last = args
    try:
        m = BM.build_daily(name, date.fromisoformat(last), log=None)
        return name, f'ok missing {m["n_missing"]}'
    except Exception as e:  # noqa: BLE001
        import traceback
        return name, 'ERROR ' + traceback.format_exc()[-800:]


def cmd_era5(a):
    names = a.names or list(BM.DAILY_SPEC)
    names = [n for n in names if a.force or not (BM.DAILY / f'{n}.npy').exists()]
    log('era5 caches', names)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for name, r in pool.imap_unordered(_era5_one, [(n, a.last) for n in names]):
            log(name, r)


def era5_dir():
    return BS.mkdir_striped(BM.OUT / 'era5')


def cmd_mjo_fit(a):
    last = date.fromisoformat(a.last)
    s_day, days = BM.mjo_series('day', last)
    s_00, _ = BM.mjo_series('00z', last)
    full = BM.MJOIndex(s_day, days)
    full.add_series('day', s_day, days)
    full.add_series('00z', s_00, days)
    res0 = dict(test_before_rotation=mjo_test(full))
    bom = BM.read_bom_rmm()
    full.align_rotation('day', bom, tuple(a.align_years))
    winds = BM.MJOIndex(s_day, days, fields=('u850', 'u200'), ref=full)
    winds.add_series('day', s_day, days)
    winds.add_series('00z', s_00, days)
    fo = full.series['day']
    winds.align_rotation('day', {d: r for d, r in zip(fo['days'], fo['rmm']) if np.isfinite(r).all()}, BM.EOF_YEARS)
    od = era5_dir()
    for nm, ix in (('full', full), ('winds', winds)):
        with open(od / f'mjo_{nm}.pkl', 'wb') as f:
            pickle.dump(ix, f)
    res = dict(created=datetime.now(timezone.utc).isoformat(), commit=code_commit(), full=full.to_json(), winds=winds.to_json(), **res0)
    res['test'] = mjo_test(full)
    try:
        res['test_enso_removed_1979_2013'] = mjo_test_enso(full)
    except Exception as e:  # noqa: BLE001
        import traceback
        res['test_enso_removed_1979_2013'] = 'ERROR ' + traceback.format_exc()[-600:]
    res['test_winds'] = mjo_test(winds)
    (BS.mkdir_striped(BM.OUT / 'tests') / 'mjo_test.json').write_text(json.dumps(res, indent=1, default=str))
    log(json.dumps(res['test'], indent=1))
    log('winds-only', json.dumps(res['test_winds'], indent=1))
    log('norms', full.norm, 'explained', full.explained, 'winds ref corr', winds.ref_corr)


def sst1_daily(days, box=(-30, 30, 30, 290), years=(1979, 2018)):
    """WH04-like ENSO index SST1: PC1 of monthly ERA5 SST anomalies over the Indo-Pacific (30S-30N, 30E-70W), 1979-2018,
    standardised, linearly interpolated from mid-month to daily."""
    s_, n_, w_, e_ = box
    la = (BM.LAT15 >= s_) & (BM.LAT15 <= n_)
    lo = (BM.LON15 >= w_) & (BM.LON15 <= e_)
    mons, X = [], []
    for y in range(years[0], years[1] + 1):
        A = BM.stage_arr('sst', y)
        x = np.asarray(A[:, 0], np.float32)[:, lo][:, :, la]                     # [365, lon, lat]
        dd = [d for d in BM.daterange(date(y, 1, 1), date(y, 12, 31)) if not (d.month == 2 and d.day == 29)]
        mo = np.array([d.month for d in dd])
        for m in range(1, 13):
            X.append(np.nanmean(x[mo == m], 0))
            mons.append((y, m))
    X = np.stack(X)
    mo = np.array([m for _, m in mons])
    for m in range(1, 13):
        X[mo == m] -= np.nanmean(X[mo == m], 0)
    ok = np.isfinite(X).all(0)
    w = np.sqrt(np.cos(np.deg2rad(BM.LAT15[la])))[None, None, :] * np.ones(X.shape[1:])[None]
    Z = (X * w)[:, ok]
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pc = U[:, 0] * S[0]
    pc = (pc - pc.mean()) / pc.std()
    e1 = np.zeros(ok.shape)
    e1[ok] = Vt[0]
    lat2 = np.repeat(BM.LAT15[la][None], lo.sum(), 0)
    lon2 = np.repeat(BM.LON15[lo][:, None], la.sum(), 1)
    nino = (np.abs(lat2) <= 5) & (lon2 >= 190) & (lon2 <= 240)
    if (e1[nino]).mean() < 0:
        pc = -pc
    mid = np.array([(date(y, m, 15) - date(1970, 1, 1)).days for y, m in mons], float)
    t = np.array([(d - date(1970, 1, 1)).days for d in days], float)
    out = np.interp(t, mid, pc, left=np.nan, right=np.nan)
    return out


def mjo_test_enso(ix, samp='day', years=(1979, 2013), reg_years=(1979, 2001)):
    """sensitivity of the BoM test: BoM removed the ENSO (SST1) component before 2014. Remove the part of each band anomaly
    linearly related to SST1 (regression per longitude over 1979-2001, as WH04), then the previous-120-day mean, and project on
    OUR EOFs (unchanged); compare with BoM 1979-2013."""
    S = ix.series[samp]
    days = S['days']
    e = sst1_daily(days)
    tr = np.array([reg_years[0] <= d.year <= reg_years[1] for d in days]) & np.isfinite(e)
    Z = []
    for v in ix.fields:
        A = S['anom'][v].copy()
        okr = tr & np.isfinite(A).all(1)
        b = (A[okr] * e[okr, None]).sum(0) / (e[okr] ** 2).sum()
        A = A - np.where(np.isfinite(e), e, np.nan)[:, None] * b[None]
        Z.append(BM.prev120(A) / ix.norm[v])
    R = np.concatenate(Z, 1) @ ix.eof.T
    keep = dict(S)
    S2 = dict(S, rmm=R)
    ix.series['_enso'] = S2
    try:
        out = mjo_test(ix, '_enso', years)
    finally:
        ix.series.pop('_enso')
    return out


def mjo_test(ix, samp='day', years=(1979, 2018)):
    bom = BM.read_bom_rmm()
    S = ix.series[samp]
    dd = [d for d in S['days'] if years[0] <= d.year <= years[1] and d in bom]
    ours = ix.obs(samp, dd)
    B = np.array([bom[d][:2] for d in dd])
    ok = np.isfinite(ours).all(1)
    r1 = float(np.corrcoef(ours[ok, 0], B[ok, 0])[0, 1])
    r2 = float(np.corrcoef(ours[ok, 1], B[ok, 1])[0, 1])
    amp_b = np.hypot(B[:, 0], B[:, 1])
    act = ok & (amp_b >= 1)
    ph_o = BM.rmm_phase(ours[:, 0], ours[:, 1])
    ph_b = np.array([bom[d][2] for d in dd])
    ph_b_calc = BM.rmm_phase(B[:, 0], B[:, 1])
    match = float((ph_o[act] == ph_b[act]).mean())
    amp_o = np.hypot(ours[:, 0], ours[:, 1])
    out = dict(n_days=int(ok.sum()), n_active=int(act.sum()), r_rmm1=r1, r_rmm2=r2, phase_match_active=match,
               phase_match_active_adjacent=float((np.minimum((ph_o[act] - ph_b[act]) % 8, (ph_b[act] - ph_o[act]) % 8) <= 1).mean()),
               bom_phase_vs_formula=float((ph_b_calc == ph_b).mean()), r_amplitude=float(np.corrcoef(amp_o[ok], amp_b[ok])[0, 1]),
               mean_amp_ours=float(amp_o[ok].mean()), mean_amp_bom=float(amp_b[ok].mean()),
               pass_=bool(r1 >= 0.95 and r2 >= 0.95 and match >= 0.90), criterion='r >= 0.95 (RMM1, RMM2) and phase match >= 90 % of '
               'active days (BoM amplitude >= 1), 1979-2018, ERA5 day sampling')
    for p0, p1 in ((1979, 2001), (2002, 2013), (1979, 2013), (2014, 2018)):
        s = np.array([p0 <= d.year <= p1 for d in dd]) & ok
        if s.sum() > 100:
            out[f'r_{p0}_{p1}'] = [float(np.corrcoef(ours[s, j], B[s, j])[0, 1]) for j in range(2)]
            out[f'phase_match_active_{p0}_{p1}'] = float((ph_o[s & act] == ph_b[s & act]).mean())
    return out


def monthly_means(name, y0, y1, clim):
    D = BM.daily(name)
    months, X = [], []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            a = date(y, m, 1)
            b = date(y + (m == 12), m % 12 + 1, 1)
            days = BM.daterange(a, b - BM.timedelta(days=1))
            x = D.get(days) - clim(days)
            X.append(np.nanmean(x, 0))
            months.append((y, m))
    return np.stack(X).astype(np.float32), months


def cmd_tele_fit(a):
    name = 'z500_mean4'
    clim = BM.fixed_clim(name, base=BM.EOF_YEARS, nh=4)
    log('clim done')
    Xm, months = monthly_means(name, 1979, 2025, clim)
    tr = [i for i, (y, m) in enumerate(months) if BM.EOF_YEARS[0] <= y <= BM.EOF_YEARS[1]]
    TP = BM.TelePatterns(Xm[tr], [months[i] for i in tr])
    log('patterns done')
    # monthly indices (all months 1979-2025), standardised per calendar month over 1979-2018
    mon = {k: np.full(len(months), np.nan) for k in ('nao', 'pna', 'ao', 'sam')}
    for i, (y, m) in enumerate(months):
        r = TP.raw_indices(Xm[i], m)
        for k in mon:
            mon[k][i] = r[k]
    mstd = {}
    for k in mon:
        for m in range(1, 13):
            sel = [i for i in tr if months[i][1] == m]
            mstd[(k, m)] = float(np.std(mon[k][sel]))
    mon_n = {k: np.array([mon[k][i] / mstd[(k, months[i][1])] for i in range(len(months))]) for k in mon}
    # 7-day-mean index normalisation per calendar month (ERA5 1979-2018, all 7-day windows; month of the middle day)
    dser, ddays = BM.era5_daily_tele(TP, name, clim, first=date(BM.EOF_YEARS[0], 1, 1), last=date(BM.EOF_YEARS[1], 12, 31))
    wk = {}
    mid_m = np.array([d.month for d in ddays])
    for k, X in dser.items():
        cs = np.concatenate([np.zeros((1, 12)), np.cumsum(X, 0)], 0)
        r7 = (cs[7:] - cs[:-7]) / 7.0                                    # window starting at day i, middle day i + 3
        mm = mid_m[3:3 + len(r7)]
        for m in range(1, 13):
            wk[(k, m)] = r7[mm == m, m - 1]
    TP.norm = {key: float(np.std(v)) for key, v in wk.items()}
    TP.clim_base = BM.EOF_YEARS
    with open(era5_dir() / 'tele_patterns.pkl', 'wb') as f:
        pickle.dump(dict(TP=TP, clim_coef=clim.coef, clim_nh=4, clim_base=BM.EOF_YEARS, monthly=mon_n, months=months), f)
    res = dict(created=datetime.now(timezone.utc).isoformat(), commit=code_commit())
    files = {'nao': 'norm.nao.monthly.b5001.current.ascii.table', 'pna': 'norm.pna.monthly.b5001.current.ascii.table',
             'ao': 'monthly.ao.index.b50.current.ascii.table', 'sam': 'monthly.aao.index.b79.current.ascii.table'}
    for k, fn in files.items():
        cpc = BM.read_cpc_table(BM.IDX / fn)
        for (p0, p1, mm, tag) in ((1979, 2018, range(1, 13), 'all'), (1979, 2018, (12, 1, 2), 'DJF'), (2019, 2025, range(1, 13), '2019_2025')):
            sel = [i for i, (y, m) in enumerate(months) if p0 <= y <= p1 and m in mm and (y, m) in cpc]
            o = np.array([mon_n[k][i] for i in sel])
            c = np.array([cpc[months[i]] for i in sel])
            res[f'{k}_r_{tag}'] = float(np.corrcoef(o, c)[0, 1])
            res[f'{k}_n_{tag}'] = len(sel)
        res[f'{k}_r_by_month'] = {}
        for m in range(1, 13):
            sel = [i for i, (y, mo) in enumerate(months) if 1979 <= y <= 2018 and mo == m and (y, mo) in cpc]
            res[f'{k}_r_by_month'][m] = round(float(np.corrcoef([mon_n[k][i] for i in sel], [cpc[months[i]] for i in sel])[0, 1]), 3)
    res['nao_pass'] = bool(res['nao_r_all'] >= 0.9)
    res['criterion'] = 'our ERA5 monthly NAO vs CPC norm.nao.monthly, 1979-2018 all months: r >= 0.9 (blocking)'
    res['modes'] = {m: {'nao_k': TP.sets[m]['nh']['nao_k'], 'pna_k': TP.sets[m]['nh']['pna_k']} for m in range(1, 13)}
    (BS.mkdir_striped(BM.OUT / 'tests') / 'tele_test.json').write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if not k.endswith('by_month') and k != 'modes'}, indent=1))


def cmd_tasks(a):
    st = BM.list_starts()
    lines = []
    for s in a.systems:
        todo = [d for d in st if not BM.extract_path(s, d).exists() and BM.start_dir(s, d).is_dir()]
        for i in range(0, len(todo), a.chunk):
            ch = todo[i:i + a.chunk]
            lines.append(json.dumps(dict(id=f'S41e_modes_{s}_{ch[0]:%Y%m%d}_{len(ch)}', est_s=60 * len(ch),
                                         cmd=f'python $C/scripts/bing_modes.py extract --system {s} --starts '
                                             + ' '.join(f'{d}' for d in ch))))
    p = Path(a.out)
    p.write_text('\n'.join(lines) + '\n')
    log(len(lines), 'tasks ->', p)


def cmd_extract(a):
    BS.mkdir_striped(BM.OUT / 'extract')
    BS.mkdir_striped(BM.OUT / 'extract' / a.system)
    bad = []
    for st in a.starts:
        d = date.fromisoformat(st)
        out = BM.extract_path(a.system, d)
        if out.exists() and not a.force:
            log('exists', out)
            continue
        if a.system == 'ifs':
            try:
                BM.extract_ifs_start(d, BM.start_dir('ifs', d), out, log=log)
            except Exception:  # noqa: BLE001
                import traceback
                log(f'ERROR ifs {d}: {traceback.format_exc()[-1500:]}')
                bad.append(st)
            continue
        files = BM.member_files(a.system, d)
        if len(files) < 30:
            log(f'WARNING {a.system} {d}: {len(files)} member files')
        if not files:
            bad.append(st)
            continue
        try:
            BM.extract_start(a.system, d, files, out, log=log)
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f'ERROR {a.system} {d}: {traceback.format_exc()[-1500:]}')
            bad.append(st)
    if bad:
        raise SystemExit(f'failed starts {bad}')


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('era5')
    e.add_argument('--names', nargs='*')
    e.add_argument('--last', default='2026-11-12')
    e.add_argument('--workers', type=int, default=4)
    e.add_argument('--force', action='store_true')
    m = sub.add_parser('mjo_fit')
    m.add_argument('--last', default='2026-11-12')
    m.add_argument('--align_years', nargs=2, type=int, default=[2014, 2018])
    sub.add_parser('tele_fit')
    t = sub.add_parser('tasks')
    t.add_argument('--systems', nargs='+', default=SYSTEMS)
    t.add_argument('--out', default=str(BM.OUT / 'tasks_extract.jsonl'))
    t.add_argument('--chunk', type=int, default=8)
    x = sub.add_parser('extract')
    x.add_argument('--system', required=True)
    x.add_argument('--starts', nargs='+', required=True)
    x.add_argument('--force', action='store_true')
    s = sub.add_parser('score')
    s.add_argument('--systems', nargs='+', default=SYSTEMS)
    s.add_argument('--parts', nargs='*')
    a = p.parse_args(argv)
    {'era5': cmd_era5, 'mjo_fit': cmd_mjo_fit, 'tele_fit': cmd_tele_fit, 'tasks': cmd_tasks, 'extract': cmd_extract,
     'score': lambda a: __import__('bing_modes_score').main(a)}[a.cmd](a)


if __name__ == '__main__':
    main()
