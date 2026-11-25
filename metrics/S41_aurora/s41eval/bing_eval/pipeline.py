"""Config-driven, resumable evaluation pipeline (brief 2026-11-11 tasks 4-6; plan §3 A and D).

Phases (each a list of independent tasks for ``s2storm.aurora.taskfarm``; a task is skipped when its ``done`` marker exists):
  reduce:<system>:<YYYYMMDD>   member files -> window means on g15, ``work/reduce/<system>/<field>/<date>.npy`` [M, W, 121, 240]
                               + ``work/reduce/<system>/meta/<date>.json`` (files read, missing, errors) = done marker
  verif:<field>:<sampling>     ERA5 window means and both references for the union of starts, g15 and g30,
                               ``work/verif/<field>_<sampling>/`` (O, Cfixed, Ctrend, CCfixed, CCtrend, Tfixed, Ttrend [+ _g30])
  score:<system>:<field>:<grid>:<corr>  all metrics for one correction (raw | dc), ``score/<system>/<field>_<grid>/``
                               (rows_<corr>.csv, perstart_<corr>.nc, maps_<corr>.nc, em_<corr>.npy, crps_<corr>.npy, done_<corr>.json)
``collect`` (login node, light) merges rows into ``scores.csv`` / ``scores.nc``, writes ``fail_summary.csv`` and ``provenance.json``.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from s41eval.bing_eval import core as K
from s41eval.bing_eval import data as D

warnings.filterwarnings('ignore', category=RuntimeWarning, message='(Mean of empty slice|All-NaN|invalid value|divide by zero|Degrees)')
GOOD_OSTS = [5, 19, 33, 47, 61, 77, 91, 107, 121, 139, 153]
_K = [int(time.time()) % 11]


# ------------------------------------------------------------------ config
def load_config(path, overrides=None):
    import yaml
    txt = Path(path).read_text()
    m = re.search(r'^root:\s*(\S+)', txt, re.M)
    root = m.group(1) if m else ''
    cfg = yaml.safe_load(txt.replace('${root}', root))
    for k, v in (overrides or {}).items():
        cfg[k] = v
    cfg['_config_file'] = str(Path(path).resolve())
    return cfg


def out_dir(cfg, tag):
    return Path(cfg['out_root']) / tag


def reduce_root(cfg, tag):
    """member window means; ``reduce_root`` in the config shares them between tags (same reader code and fields)."""
    return Path(cfg['reduce_root']) if cfg.get('reduce_root') else out_dir(cfg, tag) / 'work' / 'reduce'


def mkdir(d) -> Path:
    d = Path(d)
    if d.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    if str(d).startswith('/lus/'):
        import subprocess
        k = GOOD_OSTS[_K[0] % len(GOOD_OSTS)]
        _K[0] += 1
        subprocess.run(['lfs', 'setstripe', '-c', '1', '-i', str(k), str(d)], capture_output=True, timeout=60)
    return d


def save_npy(path, x):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial.npy')
    np.save(tmp, np.asarray(x))
    os.replace(tmp, path)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def system_starts(cfg, system):
    sc = cfg['systems'][system]
    st = D.list_starts(cfg['starts_csv'], cfg.get('first', '2019-01-01'), cfg.get('last', '2026-12-31'))
    if sc.get('first') or sc.get('last'):
        st = [s for s in st if str(s) >= sc.get('first', '0') and str(s) <= sc.get('last', '9')]
    return st


def all_starts(cfg):
    return sorted({s for sy in cfg['run_systems'] for s in system_starts(cfg, sy)})


def system_fields(cfg, system):
    sc = cfg['systems'][system]
    return [f for f in cfg['fields'] if f not in (sc.get('missing_fields') or [])]


def samplings(cfg):
    """(field, sampling) pairs needed by the systems of this run."""
    out = set()
    for sy in cfg['run_systems']:
        for f in system_fields(cfg, sy):
            out.add((f, D.sampling(cfg['systems'][sy], f)))
    return sorted(out)


# ------------------------------------------------------------------ phase 1a: reduce
def task_reduce(cfg, tag, system, day):
    init = date.fromisoformat(f'{day[:4]}-{day[4:6]}-{day[6:]}')
    sc = cfg['systems'][system]
    fields = system_fields(cfg, system)
    base = mkdir(reduce_root(cfg, tag) / system)
    mkdir(base / 'meta')
    meta_f = base / 'meta' / f'{day}.json'
    if meta_f.exists():
        return 'cached'
    t0 = time.time()
    meta = dict(system=system, start=str(init), format=sc['format'], fields=fields, files=[], missing=[], errors=[], members=0)
    reader = D.FORMATS[sc['format']]
    recs = []
    if sc['format'] == 'ifs_ecds_grib':
        d = sc['path'].format(date=day)
        if not os.path.isdir(d):
            meta['missing'].append(d)
        else:
            try:
                rr, info = reader(d, init, fields, sc)
                meta['files'] = sorted(glob_files(d))
                recs = rr
            except Exception as e:  # noqa: BLE001
                meta['errors'].append(dict(file=d, error=f'{type(e).__name__}: {e}'))
    else:
        for m, p in D.member_paths(sc, init):
            if p is None:
                meta['missing'].append(sc['path'].format(date=day, member=m, year=init.year))
                continue
            try:
                r, info = reader(p, init, fields, sc)
                recs.append(r)
                meta['files'].append(p)
                meta.setdefault('u200_src', info.get('u200_src'))
                meta.setdefault('lead_days', info.get('lead_days'))
            except Exception as e:  # noqa: BLE001
                meta['errors'].append(dict(file=p, error=f'{type(e).__name__}: {e}'))
    meta['members'] = len(recs)
    if not recs and not meta['errors']:
        return f"no member files ({len(meta['missing'])} missing): nothing written, task can be rerun when the files exist"
    meta['fields_written'] = []
    meta['fields_absent'] = []
    if recs and not meta['errors']:
        for f in fields:
            if not all(f in r for r in recs):
                meta['fields_absent'].append(f)
                continue
            mkdir(base / f)
            save_npy(base / f / f'{day}.npy', np.stack([r[f] for r in recs]).astype(np.float32))
            meta['fields_written'].append(f)
    meta['seconds'] = round(time.time() - t0, 1)
    write_json(meta_f, meta)
    return f"{meta['members']} members, {len(meta['fields_written'])} fields, {len(meta['missing'])} missing, {len(meta['errors'])} errors"


def glob_files(d):
    return [str(p) for p in Path(d).glob('*') if p.is_file()]


# ------------------------------------------------------------------ phase 1b: verification + references
def verif_dir(cfg, tag, field, samp):
    return out_dir(cfg, tag) / 'work' / 'verif' / f'{field}_{samp}'


def task_verif(cfg, tag, field, samp):
    vd = mkdir(verif_dir(cfg, tag, field, samp))
    if (vd / 'meta.json').exists():
        return 'cached'
    t0 = time.time()
    starts = [date.fromisoformat(s) for s in json.loads((out_dir(cfg, tag) / 'work' / 'starts.json').read_text())]
    S, W = len(starts), len(K.WINDOWS)
    vcfg = cfg['verification']
    clim = D.Climatology(cfg['climatology']['stage_dirs'], cfg['climatology']['trend_dirs'],
                         tuple(cfg['climatology'].get('member_years', [1991, 2020])))
    notes = {}
    if vcfg.get('source', 'store') == 'v0':
        O, Ct, CEt, used = D.V0Verif().windows_and_trend(field, samp, starts)
        notes['v0_sampling_used'] = used
    else:
        era = D.Era5(vcfg['store'])
        O, notes = era.windows(field, samp, starts)
        notes = {'missing_days': notes, 'era5_files': era.files(field)}
        Ct, CEt = None, None
    shp = (S, W, 121, 240)
    arr = {k: np.full(shp, np.nan, np.float32) for k in ('Cfixed', 'Ctrend', 'CCfixed', 'CCtrend')}
    T = {r: np.full((S, 2, W, 121, 240), np.nan, np.float32) for r in ('fixed', 'trend')}
    g3 = {k: np.full((S, W, 61, 120), np.nan, np.float32) for k in ('O', 'Cfixed', 'Ctrend', 'CCfixed', 'CCtrend')}
    nmem = {}
    for i, st in enumerate(starts):
        e = clim.ensembles(field, samp, st)
        if CEt is not None:                                # v0: trend ensemble and climatology from bing_scores
            e['trend'] = CEt[i] if CEt.shape[1] >= 2 else None
            e['trend_mean'] = Ct[i]
        nmem[str(st)] = e['n_members']
        o3 = K.regrid(O[i], 'g15', 'g30')
        g3['O'][i] = o3
        for r in ('fixed', 'trend'):
            E = e[r]
            if E is None or E.shape[0] < 2:
                continue
            arr[f'CC{r}'][i] = K.fair_crps(E, O[i])
            T[r][i] = K.tercile_edges(E)
            arr[f'C{r}'][i] = e[f'{r}_mean'] if e.get(f'{r}_mean') is not None else E.mean(0)
            E3 = K.regrid(E, 'g15', 'g30')
            g3[f'CC{r}'][i] = K.fair_crps(E3, o3)
            g3[f'C{r}'][i] = K.regrid(arr[f'C{r}'][i], 'g15', 'g30')
    save_npy(vd / 'O.npy', O)
    for k, x in arr.items():
        save_npy(vd / f'{k}.npy', x)
    for r, x in T.items():
        save_npy(vd / f'T{r}.npy', x)
    for k, x in g3.items():
        save_npy(vd / f'{k}_g30.npy', x)
    meta = dict(field=field, sampling=samp, starts=[str(s) for s in starts], n_ref_members=nmem, notes=notes,
                clim_inputs=sorted(clim.used), source=vcfg.get('source', 'store'), seconds=round(time.time() - t0, 1),
                written=f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}')
    write_json(vd / 'meta.json', meta)
    return f'{S} starts, {time.time() - t0:.0f}s'


def load_verif(cfg, tag, field, samp, starts, grid):
    vd = verif_dir(cfg, tag, field, samp)
    meta = json.loads((vd / 'meta.json').read_text())
    pos = {s: i for i, s in enumerate(meta['starts'])}
    ix = np.array([pos[str(s)] for s in starts])
    sfx = '' if grid == 'g15' else '_g30'
    out = {k: np.load(vd / f'{k}{sfx}.npy', mmap_mode='r')[ix] for k in ('O', 'Cfixed', 'Ctrend', 'CCfixed', 'CCtrend')}
    if grid == 'g15':
        for r in ('fixed', 'trend'):
            out[f'T{r}'] = np.load(vd / f'T{r}.npy', mmap_mode='r')[ix]
    return out, meta


# ------------------------------------------------------------------ phase 2: score
REFS = ('fixed', 'trend')


def _subsets(starts):
    out = []
    for p in K.PERIODS:
        for s in K.SEASONS:
            ix = K.subset_index(starts, p, s)
            if ix.size:
                out.append((p, s, ix))
    return out


MAP_SUBSETS = [('pooled', 'all'), ('2019-24', 'all'), ('2025-26', 'all'), ('pooled', 'DJF'), ('pooled', 'MAM'), ('pooled', 'JJA'),
               ('pooled', 'SON')]


class Scorer:
    """all metrics for one (system, field, grid) from the member window means F [S, M, W, y, x] and the verification arrays."""

    def __init__(self, F, V, starts, grid, clip=False, M_label=None, seed=0, tercile_grid=True):
        self.F, self.V, self.starts, self.grid, self.clip = F, V, list(starts), grid, clip
        self.S, self.M = F.shape[:2]
        self.Wn = F.shape[2]
        self.mk = K.masks(grid)
        self.cw = K.cosw(grid)
        self.valid = np.isfinite(V['O']).any((-2, -1)) & np.isfinite(F[:, 0]).any((-2, -1))
        self.rng = np.random.default_rng(seed)
        self.subs = _subsets(self.starts)
        self.do_terc = tercile_grid and 'Tfixed' in V

    # -- per-start maps for one ensemble (members index), one correction
    def corrected(self, s, b, corr, idx=None):
        X = np.asarray(self.F[s], np.float64) if idx is None else np.asarray(self.F[s][idx], np.float64)
        if corr == 'dc':
            X = X - np.nan_to_num(b[s])[None]
            if self.clip:
                X = np.clip(X, 0, None)
        return X

    def bias_for(self, em):
        return K.drift_bias(em, np.asarray(self.V['O']), self.starts, self.valid)

    def use(self, corr, b):
        return self.valid.copy() if corr == 'raw' else self.valid & np.isfinite(b).any((-2, -1))

    def ens_mean_stats(self, emc, varc, use, M, seasons=None):
        """subset aggregates of the ensemble-mean scores (v0 order: per-point mean over starts, then area mean)."""
        O = np.asarray(self.V['O'])
        res = {}
        se = (emc - O) ** 2
        er = emc - O
        an = {r: (emc - np.asarray(self.V[f'C{r}']), O - np.asarray(self.V[f'C{r}'])) for r in REFS}
        accp = {r: {rn: K.acc_pattern(an[r][0], an[r][1], m, self.grid) for rn, m in self.mk.items()} for r in REFS}   # [S, W]
        for p, sn, six in self.subs:
            if seasons is not None and sn not in seasons:
                continue
            for w in range(self.Wn):
                ix = six[use[six, w]]
                if ix.size < 1:
                    continue
                mse_m = np.nanmean(se[ix, w], 0)
                var_m = np.nanmean(varc[ix, w], 0) if varc is not None else None
                bia_m = np.nanmean(er[ix, w], 0)
                accm = {r: K.acc_map(an[r][0][ix, w], an[r][1][ix, w]) if ix.size >= 3 else None for r in REFS}
                for rn, m in self.mk.items():
                    mse = float(K.amean(mse_m, m, self.grid))
                    v = dict(rmse=math.sqrt(mse) if np.isfinite(mse) else float('nan'), bias=float(K.amean(bia_m, m, self.grid)))
                    if var_m is not None:
                        mv = float(K.amean(var_m, m, self.grid))
                        v['spread'] = math.sqrt(mv) if np.isfinite(mv) else float('nan')
                        v['ssr'] = K.ssr(M, mv, mse)
                    for r in REFS:
                        v[f'acc|{r}'] = float(K.amean(accm[r], m, self.grid)) if accm[r] is not None else float('nan')
                        v[f'acc_pattern|{r}'] = float(np.nanmean(accp[r][rn][ix, w]))
                    res[(p, sn, rn, K.WK[w])] = (v, int(ix.size))
        return res

    def maps_of(self, emc, varc, crps, use, M):
        O = np.asarray(self.V['O'])
        out = {}
        for p, sn in MAP_SUBSETS:
            six = K.subset_index(self.starts, p, sn)
            if not six.size:
                continue
            mp = {k: np.full((self.Wn,) + O.shape[-2:], np.nan, np.float32) for k in
                  ('crpss_fixed', 'crpss_trend', 'acc_fixed', 'acc_trend', 'bias', 'rmse', 'ssr')}
            for w in range(self.Wn):
                ix = six[use[six, w]]
                if ix.size < 1:
                    continue
                cf = np.nanmean(crps[ix, w], 0)
                for r in REFS:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        mp[f'crpss_{r}'][w] = 1 - cf / np.nanmean(np.asarray(self.V[f'CC{r}'][ix, w]), 0)
                    if ix.size >= 3:
                        C = np.asarray(self.V[f'C{r}'][ix, w])
                        mp[f'acc_{r}'][w] = K.acc_map(emc[ix, w] - C, O[ix, w] - C)
                mse = np.nanmean((emc[ix, w] - O[ix, w]) ** 2, 0)
                mp['bias'][w] = np.nanmean(emc[ix, w] - O[ix, w], 0)
                mp['rmse'][w] = np.sqrt(mse)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mp['ssr'][w] = np.sqrt((M + 1) / M * np.nanmean(varc[ix, w], 0)) / np.sqrt(mse)
            out[(p, sn)] = mp
        return out

    def run(self, log=print, n_draws=10, do_infl=True, do_maps=True, keep_maps=False, corrs=('raw', 'dc')):
        S, M, Wn = self.S, self.M, self.Wn
        O = np.asarray(self.V['O'])
        R = len(K.RK)
        em = self.F.mean(1, dtype=np.float64).astype(np.float32) if S else None
        b = self.bias_for(em)
        rows, per, maps, keep = [], {}, {}, {}
        iw34 = K.WK.index('w34') if 'w34' in K.WK and Wn == len(K.WK) else None
        for corr in corrs:
            t0 = time.time()
            use = self.use(corr, b)
            crps = np.full((S, Wn) + O.shape[-2:], np.nan, np.float32)
            emc = np.full_like(crps, np.nan)
            varc = np.full_like(crps, np.nan)
            rps = {r: np.full_like(crps, np.nan) for r in REFS}
            rpscl = {r: np.full_like(crps, np.nan) for r in REFS}
            rank = np.zeros((S, Wn, R, M + 1), np.float64)
            rel = np.zeros((S, R, len(REFS), 2, 11, 3), np.float64)
            for s in range(S):
                if not use[s].any():
                    continue
                X = self.corrected(s, b, corr)
                crps[s] = K.fair_crps(X, O[s])
                emc[s] = X.mean(0)
                varc[s] = X.var(0, ddof=1)
                rk = K.obs_rank(X, O[s], self.rng)
                for w in range(Wn):
                    if not use[s, w]:
                        continue
                    for j, rn in enumerate(K.RK):
                        if rn not in self.mk:
                            continue
                        wt = (self.cw * self.mk[rn])
                        ok = rk[w] >= 0
                        rank[s, w, j] = np.bincount(rk[w][ok], weights=wt[ok], minlength=M + 1)[:M + 1]
                if self.do_terc:
                    for ir, r in enumerate(REFS):
                        Tr = np.asarray(self.V[f'T{r}'][s])
                        plo, phi = K.tercile_probs(X, Tr)
                        rps[r][s] = K.rps3(plo, phi, O[s], Tr)
                        rpscl[r][s] = K.rps_clim3(O[s], Tr)
                        if iw34 is not None and use[s, iw34]:
                            o_lo = (O[s, iw34] < Tr[0, iw34]).astype(float)
                            o_hi = (O[s, iw34] > Tr[1, iw34]).astype(float)
                            okt = np.isfinite(O[s, iw34]) & np.isfinite(Tr[0, iw34])
                            for j, rn in enumerate(K.RK):
                                if rn not in self.mk:
                                    continue
                                wt = self.cw * self.mk[rn] * okt
                                rel[s, j, ir, 0] = K.reliability_counts(plo[iw34], o_lo, wt)
                                rel[s, j, ir, 1] = K.reliability_counts(phi[iw34], o_hi, wt)
            crps[~use] = np.nan
            log(f'  {corr}: members {M}, CRPS/ranks/terciles {time.time() - t0:.0f}s')
            # inflation diagnostic (plan §3 D): ensemble scaled about its mean so that spread/error = 1 per point and window
            crps_inf = np.full_like(crps, np.nan)
            alpha = None
            if do_infl:
                with np.errstate(invalid='ignore', divide='ignore'):
                    mse_p = np.nanmean(np.where(use[:, :, None, None], (emc - O) ** 2, np.nan), 0)
                    var_p = np.nanmean(np.where(use[:, :, None, None], varc, np.nan), 0)
                    alpha = np.clip(np.sqrt(mse_p / ((M + 1) / M * var_p)), 0.2, 10.0)
                alpha = np.where(np.isfinite(alpha), alpha, 1.0)
                for s in range(S):
                    if not use[s].any():
                        continue
                    X = self.corrected(s, b, corr)
                    Xi = emc[s][None] + alpha[None] * (X - emc[s][None])
                    if self.clip:
                        Xi = np.clip(Xi, 0, None)
                    crps_inf[s] = K.fair_crps(Xi, O[s])
                crps_inf[~use] = np.nan
            # 11-member draws (mean over draws of every score)
            dr = K.draws11(M, n_draws) if M > 11 else []
            crps11 = np.zeros_like(crps) if dr else None
            st11 = []

            def one_draw(idx):
                em_d = self.F[:, idx].mean(1, dtype=np.float64).astype(np.float32) if corr == 'dc' else None
                b_d = self.bias_for(em_d) if corr == 'dc' else None
                use_d = use if corr == 'raw' else self.valid & np.isfinite(b_d).any((-2, -1))
                c_d = np.zeros_like(crps)
                e_d = np.full_like(crps, np.nan)
                v_d = np.full_like(crps, np.nan)
                for s in range(S):
                    if not use_d[s].any():
                        continue
                    X = self.corrected(s, b_d, corr, idx)
                    c_d[s] = K.fair_crps(X, O[s])
                    e_d[s] = X.mean(0)
                    v_d[s] = X.var(0, ddof=1)
                return c_d, self.ens_mean_stats(e_d, v_d, use_d, len(idx), seasons=('all',))   # 11-member: all seasons only
            # draws in threads (numpy releases the GIL; one process per core-group otherwise leaves the cores idle)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, int(os.environ.get('BING_EVAL_THREADS', '4')))) as ex:
                for c_d, st in ex.map(one_draw, dr):          # results in draw order (deterministic sum)
                    crps11 += c_d / len(dr)
                    st11.append(st)
                    del c_d
            if crps11 is not None:
                crps11[~use] = np.nan
            log(f'  {corr}: 11-member draws {len(dr)} + inflation, total {time.time() - t0:.0f}s')
            # ---- aggregation
            st30 = self.ens_mean_stats(emc, varc, use, M)
            rm = {k: K.region_means(x, self.grid) for k, x in (('crps', crps), ('crps_infl', crps_inf), ('se', (emc - O) ** 2),
                                                                ('var', varc), ('bias', emc - O))}
            if crps11 is not None:
                rm['crps11'] = K.region_means(crps11, self.grid)
            for r in REFS:
                rm[f'cc_{r}'] = K.region_means(np.asarray(self.V[f'CC{r}']), self.grid)
                if self.do_terc:
                    rm[f'rps_{r}'] = K.region_means(rps[r], self.grid)
                    rm[f'rpscl_{r}'] = K.region_means(rpscl[r], self.grid)
            u3 = use[:, :, None]
            for key in rm:
                rm[key] = np.where(u3, rm[key], np.nan)
            Dm = K.rpss_debias_D(M)
            for (p, sn, six) in self.subs:
                for w in range(Wn):
                    ix = six[use[six, w]]
                    if ix.size < 1:
                        continue
                    for j, rn in enumerate(K.RK):
                        if rn not in self.mk:
                            continue
                        base = dict(period=p, season=sn, region=rn, window=K.WK[w], correction=corr, n_starts=int(ix.size))

                        def add(metric, value, members, ref=''):
                            rows.append(dict(base, metric=metric, value=float(value), members=members, reference=ref))
                        pc = rm['crps'][ix, w, j]
                        add('crps', np.nanmean(pc), M)
                        if crps11 is not None:
                            add('crps', np.nanmean(rm['crps11'][ix, w, j]), 11)
                        for r in REFS:
                            pr = rm[f'cc_{r}'][ix, w, j]
                            okr = np.isfinite(pr) & np.isfinite(pc)
                            if not okr.any():
                                continue
                            add('crps_clim', np.nanmean(pr[okr]), 30, r)
                            add('crpss', 1 - np.nansum(pc[okr]) / np.nansum(pr[okr]), M, r)
                            if crps11 is not None:
                                add('crpss', 1 - np.nansum(rm['crps11'][ix, w, j][okr]) / np.nansum(pr[okr]), 11, r)
                            if do_infl:
                                add('crpss_inflated', 1 - np.nansum(rm['crps_infl'][ix, w, j][okr]) / np.nansum(pr[okr]), M, r)
                            if self.do_terc:
                                a_, c_ = np.nanmean(rm[f'rps_{r}'][ix, w, j]), np.nanmean(rm[f'rpscl_{r}'][ix, w, j])
                                add('rpss', 1 - a_ / (c_ + Dm), M, r)
                        v30, n30 = st30.get((p, sn, rn, K.WK[w]), ({}, 0))
                        for k_, val in v30.items():
                            met, _, ref = k_.partition('|')
                            add(met, val, M, ref)
                        if st11:
                            vals = [x.get((p, sn, rn, K.WK[w]), ({}, 0))[0] for x in st11]
                            for k_ in (vals[0] if vals[0] else []):
                                met, _, ref = k_.partition('|')
                                add(met, np.nanmean([v.get(k_, np.nan) for v in vals]), 11, ref)
                        if do_infl and alpha is not None:
                            add('inflation_factor', K.amean(alpha[w], self.mk[rn], self.grid), M)
            per[corr] = dict(rm=rm, rank=rank, rel=rel, use=use)
            if do_maps:
                maps[corr] = self.maps_of(emc, varc, crps, use, M)
            if keep_maps:
                keep[corr] = dict(em=emc, crps=crps)
            log(f'  {corr}: aggregated {time.time() - t0:.0f}s')
        return rows, per, maps, keep, b


def task_score(cfg, tag, system, field, grid, corr='raw', log=print):
    sc = cfg['systems'][system]
    sd = mkdir(out_dir(cfg, tag) / 'score' / system / f'{field}_{grid}')
    if (sd / f'done_{corr}.json').exists():
        return 'cached'
    t0 = time.time()
    rb = reduce_root(cfg, tag) / system
    starts_all = system_starts(cfg, system)
    starts, dropped = [], []
    Ms = []
    for s in starts_all:
        mf = rb / 'meta' / f'{s:%Y%m%d}.json'
        if not mf.exists():
            dropped.append((str(s), 'no reduce output'))
            continue
        m = json.loads(mf.read_text())
        if field not in m.get('fields_written', []):
            why = 'errors: ' + '; '.join(e['error'] for e in m['errors'])[:300] if m['errors'] else \
                (f"{len(m['missing'])} member files missing" if m['missing'] else 'field absent')
            dropped.append((str(s), why))
            continue
        if m['missing']:
            dropped.append((str(s), f"{len(m['missing'])} member files missing"))
            continue
        starts.append(s)
        Ms.append(m['members'])
    if not starts:
        write_json(sd / f'done_{corr}.json', dict(system=system, field=field, grid=grid, starts=[], dropped=dropped, note='no data'))
        return 'no data'
    M = min(Ms)
    if len(set(Ms)) > 1:
        log(f'member counts differ {sorted(set(Ms))}: using the first {M}')
    samp = D.sampling(sc, field)
    V, vmeta = load_verif(cfg, tag, field, samp, starts, grid)
    S = len(starts)
    ny, nx = (121, 240) if grid == 'g15' else (61, 120)
    F = np.full((S, M, len(K.WINDOWS), ny, nx), np.nan, np.float32)
    for i, s in enumerate(starts):
        x = np.load(rb / field / f'{s:%Y%m%d}.npy')[:M]                 # one sequential read (mmap page faults were slow)
        F[i] = x if grid == 'g15' else K.regrid(x, 'g15', 'g30')
    log(f'{system} {field} {grid}: {S} starts x {M} members loaded {time.time() - t0:.0f}s (sampling {samp})')
    import zlib
    seed = zlib.crc32(f'{system}|{field}|{grid}'.encode())
    sco = Scorer(F, V, starts, grid, clip=K.FIELDS[field].get('clip', False), seed=seed)
    rows, per, maps, keep, b = sco.run(log=log, n_draws=int(cfg.get('n_draws', 10)), do_infl=True, do_maps=True,
                                       keep_maps=(grid == 'g15'), corrs=(corr,))
    lab = sc.get('label', system)
    for r in rows:
        r.update(system=system, variable=field, grid=grid, sampling=samp)
    cols = ['system', 'variable', 'grid', 'period', 'season', 'region', 'window', 'correction', 'members', 'reference', 'metric', 'value',
            'n_starts', 'sampling']
    with open(sd / f'rows_{corr}.csv.partial', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(sd / f'rows_{corr}.csv.partial', sd / f'rows_{corr}.csv')
    write_perstart(sd / f'perstart_{corr}.nc', per, starts, M, system, field, grid)
    write_maps(sd / f'maps_{corr}.nc', maps, grid, system, field)
    if keep:
        for corr, d in keep.items():
            save_npy(sd / f'em_{corr}.npy', d['em'])
            save_npy(sd / f'crps_{corr}.npy', d['crps'])
        if corr == 'dc':
            save_npy(sd / 'drift_bias.npy', b)
    write_json(sd / f'done_{corr}.json', dict(system=system, label=lab, field=field, grid=grid, sampling=samp, members=M,
                                      starts=[str(s) for s in starts], dropped=dropped, verif=str(verif_dir(cfg, tag, field, samp)),
                                      seconds=round(time.time() - t0, 1), rows=len(rows)))
    return f'{S} starts, {len(rows)} rows, {time.time() - t0:.0f}s'


def task_climcheck(cfg, tag, field, samp, n=24):
    """done-criterion of task 4 (+ task 3 for olr): both references load for every field (30 members), climatology-as-forecast has
    CRPSS = 0 against itself (|.| <= 0.01), and ERA5-as-forecast has CRPS 0 (two identical 'members' = ERA5)."""
    cd = mkdir(out_dir(cfg, tag) / 'work' / 'climcheck')
    f = cd / f'{field}_{samp}.json'
    if f.exists():
        return 'cached'
    vmeta = json.loads((verif_dir(cfg, tag, field, samp) / 'meta.json').read_text())
    allst = [date.fromisoformat(s) for s in vmeta['starts']]
    pick = sorted(set(np.linspace(0, len(allst) - 1, n).round().astype(int).tolist()))
    st = [allst[i] for i in pick]
    clim = D.Climatology(cfg['climatology']['stage_dirs'], cfg['climatology']['trend_dirs'],
                         tuple(cfg['climatology'].get('member_years', [1991, 2020])))
    V, _ = load_verif(cfg, tag, field, samp, st, 'g15')
    V = {k: np.asarray(v) for k, v in V.items()}
    ens = [clim.ensembles(field, samp, s) for s in st]
    res = dict(field=field, sampling=samp, starts=[str(s) for s in st], n_members={r: sorted({e['n_members'] for e in ens}) for r in REFS})
    for r in REFS:
        if any(e[r] is None for e in ens):
            res[f'{r}_as_forecast'] = 'reference missing'
            continue
        Fr = np.stack([e[r] for e in ens]).astype(np.float32)
        rows, *_ = Scorer(Fr, V, st, 'g15', clip=False).run(log=lambda *a: None, n_draws=0, do_infl=False, do_maps=False)
        g = {(x['reference'], x['window']): x['value'] for x in rows if x['metric'] == 'crpss' and x['period'] == 'pooled'
             and x['season'] == 'all' and x['region'] == 'global' and x['correction'] == 'raw'}
        res[f'{r}_as_forecast_crpss'] = {f'vs_{rr}': {w: g.get((rr, w)) for w in K.WK} for rr in REFS}
    O = V['O']
    Fo = np.stack([O, O], 1)
    rows, *_ = Scorer(Fo, V, st, 'g15').run(log=lambda *a: None, n_draws=0, do_infl=False, do_maps=False)
    res['era5_as_forecast'] = {x['metric'] + '|' + x['window']: x['value'] for x in rows if x['metric'] in ('crps', 'rmse')
                               and x['period'] == 'pooled' and x['season'] == 'all' and x['region'] == 'global' and x['correction'] == 'raw'}
    self_ok = all(abs(res[f'{r}_as_forecast_crpss'][f'vs_{r}'][w]) <= 0.01 for r in REFS for w in K.WK
                  if isinstance(res.get(f'{r}_as_forecast_crpss'), dict))
    res['pass_self_crpss'] = bool(self_ok) and all(isinstance(res.get(f'{r}_as_forecast_crpss'), dict) for r in REFS)
    res['pass_era5_crps0'] = all(abs(v) < 1e-6 for k, v in res['era5_as_forecast'].items() if k.startswith('crps'))
    res['pass_30_members'] = all(m == [30] for m in res['n_members'].values())
    write_json(f, res)
    return f"self CRPSS ok {res['pass_self_crpss']}, ERA5 CRPS 0 {res['pass_era5_crps0']}, 30 members {res['pass_30_members']}"


def write_perstart(path, per, starts, M, system, field, grid):
    import xarray as xr
    data = {}
    for corr, d in per.items():
        for k, x in d['rm'].items():
            data[f'{k}_{corr}'] = (('start', 'window', 'region'), x.astype(np.float32))
        data[f'rank_hist_{corr}'] = (('start', 'window', 'region', 'rank'), d['rank'].astype(np.float32))
        data[f'reliability_w34_{corr}'] = (('start', 'region', 'reference', 'event', 'prob_bin', 'stat'), d['rel'].astype(np.float32))
        data[f'used_{corr}'] = (('start', 'window'), d['use'].astype(np.int8))
    ds = xr.Dataset(data, coords=dict(start=np.array([np.datetime64(s) for s in starts]), window=K.WK, region=K.RK,
                                      rank=np.arange(M + 1), reference=list(REFS), event=['below_lower_tercile', 'above_upper_tercile'],
                                      prob_bin=np.round(np.linspace(0, 1, 11), 2), stat=['sum_w', 'sum_w_p', 'sum_w_obs']))
    ds.attrs = dict(system=system, field=field, grid=grid, members=M,
                    description='per-start cos-lat region means (crps = fair CRPS; cc_<ref> = fair CRPS of the 30-member reference; '
                                'se = squared error of the ensemble mean; var = member variance ddof 1; rps/rpscl = tercile RPS of the '
                                'forecast / of 1/3-1/3-1/3), area-weighted rank histograms and w34 reliability counts. '
                                'NaN where the start/window is not used. s41eval.bing_eval')
    ds.to_netcdf(str(path) + '.partial', format='NETCDF4')
    os.replace(str(path) + '.partial', path)


def write_maps(path, maps, grid, system, field):
    import xarray as xr
    if not maps:
        return
    subs = [f'{p}|{s}' for p, s in MAP_SUBSETS]
    data = {}
    lat = K.LAT[grid][::-1]
    for corr, mp in maps.items():
        keys = sorted({k for v in mp.values() for k in v})
        for k in keys:
            a = np.full((len(subs), len(K.WK), len(lat), len(K.LON[grid])), np.nan, np.float32)
            for i, (p, s) in enumerate(MAP_SUBSETS):
                if (p, s) in mp:
                    a[i] = mp[(p, s)][k][:, ::-1]
            data[f'{k}_{corr}'] = (('subset', 'window', 'lat', 'lon'), a)
    ds = xr.Dataset(data, coords=dict(subset=subs, window=K.WK, lat=lat.astype(np.float32), lon=K.LON[grid].astype(np.float32)))
    ds.attrs = dict(system=system, field=field, grid=grid, description='per-point CRPSS vs each reference, centred ACC (anomalies vs each '
                    'reference climatology), mean bias, RMSE and spread/error of the ensemble mean; lat 90..-90')
    ds.to_netcdf(str(path) + '.partial', format='NETCDF4')
    os.replace(str(path) + '.partial', path)


# ------------------------------------------------------------------ task lists, run, collect
def make_tasks(cfg, tag, phase, script, cfg_path, est=None):
    od = mkdir(out_dir(cfg, tag))
    mkdir(od / 'work')
    sf = od / 'work' / 'starts.json'
    if not sf.exists():
        write_json(sf, [str(s) for s in all_starts(cfg)])
    py = cfg.get('python', sys.executable)
    pre = f'{py} {script} run --config {cfg_path} --tag {tag} --task '
    tasks = []
    phase = str(phase)
    if phase in ('1', 'reduce', 'verif'):
        for sy in (cfg['run_systems'] if phase != 'verif' else []):
            for s in system_starts(cfg, sy):
                tid = f'reduce:{sy}:{s:%Y%m%d}'
                tasks.append(dict(id=tid.replace(':', '_'), cmd=pre + tid, est_s=(est or {}).get('reduce', 900)))
        for f, smp in (samplings(cfg) if phase != 'reduce' else []):
            tid = f'verif:{f}:{smp}'
            tasks.append(dict(id=tid.replace(':', '_'), cmd=pre + tid, est_s=(est or {}).get('verif', 2400)))
        tasks.sort(key=lambda t: not t['id'].startswith('verif'))           # long verification tasks first
    else:
        for sy in cfg['run_systems']:
            for f in system_fields(cfg, sy):
                for g in cfg.get('grids', ['g15', 'g30']):
                    if g != 'g15' and f not in cfg.get('g30_fields', cfg['fields']):
                        continue
                    for c in ('dc', 'raw'):
                        tid = f'score:{sy}:{f}:{g}:{c}'
                        tasks.append(dict(id=tid.replace(':', '_'), cmd=pre + tid, est_s=(est or {}).get('score', 3000)))
        for f, smp in samplings(cfg):
            tid = f'climcheck:{f}:{smp}'
            tasks.append(dict(id=tid.replace(':', '_'), cmd=pre + tid, est_s=900))
        tasks.sort(key=lambda t: (t['id'].endswith('g30'), t['id'].startswith('climcheck')))
    tf = od / f'tasks_phase{phase}.jsonl'
    tf.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    return tf, len(tasks)


def set_lsm(cfg):
    f = cfg['verification'].get('lsm')
    if f and os.path.exists(f):
        K.set_lsm(np.load(f))
        return True
    return False


def _plog(*a):
    print(f'{datetime.now(timezone.utc):%H:%M:%S}', *a, flush=True)


def run_task(cfg, tag, task, log=_plog):
    set_lsm(cfg)
    kind, *a = task.split(':')
    if kind == 'reduce':
        return task_reduce(cfg, tag, a[0], a[1])
    if kind == 'verif':
        return task_verif(cfg, tag, a[0], a[1])
    if kind == 'score':
        return task_score(cfg, tag, a[0], a[1], a[2], a[3] if len(a) > 3 else 'raw', log=log)
    if kind == 'climcheck':
        return task_climcheck(cfg, tag, a[0], a[1])
    raise ValueError(task)


def collect(cfg, tag, log=print):
    """merge per-task outputs: scores.csv (+ scores.nc), fail_summary.csv, provenance.json; never silent about missing pieces."""
    od = out_dir(cfg, tag)
    fails = []
    # reduce metas
    nfiles = 0
    inputs = []
    for sy in cfg['run_systems']:
        rb = reduce_root(cfg, tag) / sy
        for s in system_starts(cfg, sy):
            mf = rb / 'meta' / f'{s:%Y%m%d}.json'
            if not mf.exists():
                fails.append(dict(system=sy, start=str(s), field='*', problem='not reduced', detail='no meta file (task not run/failed)'))
                continue
            m = json.loads(mf.read_text())
            nfiles += len(m['files'])
            inputs += m['files']
            for p in m['missing']:
                fails.append(dict(system=sy, start=str(s), field='*', problem='member file missing', detail=p))
            for e in m['errors']:
                fails.append(dict(system=sy, start=str(s), field='*', problem='unreadable', detail=f"{e['file']}: {e['error']}"))
            for f in m.get('fields_absent', []):
                fails.append(dict(system=sy, start=str(s), field=f, problem='field absent in some member', detail=''))
    # verification notes
    for f, smp in samplings(cfg):
        vd = verif_dir(cfg, tag, f, smp)
        if not (vd / 'meta.json').exists():
            fails.append(dict(system='ERA5', start='*', field=f, problem='verification not built', detail=smp))
            continue
        vm = json.loads((vd / 'meta.json').read_text())
        for st, n in (vm['notes'].get('missing_days') or {}).items():
            fails.append(dict(system='ERA5', start=st, field=f, problem='ERA5 days missing', detail=f"{smp} leads {n['missing_leads']} "
                              f"(interpolated {n['interpolated']})"))
        for st, n in vm['n_ref_members'].items():
            if n < 30:
                fails.append(dict(system='reference', start=st, field=f, problem=f'{n} reference members', detail=smp))
    # scores
    rows_files = []
    for sy in cfg['run_systems']:
        for f in system_fields(cfg, sy):
            for g in cfg.get('grids', ['g15', 'g30']):
                if g != 'g15' and f not in cfg.get('g30_fields', cfg['fields']):
                    continue
                sd = od / 'score' / sy / f'{f}_{g}'
                for c in ('raw', 'dc'):
                    if not (sd / f'done_{c}.json').exists():
                        fails.append(dict(system=sy, start='*', field=f, problem='not scored', detail=f'{g} {c}'))
                        continue
                    dj = json.loads((sd / f'done_{c}.json').read_text())
                    if c == 'raw' and g == 'g15':
                        for st, why in dj.get('dropped', []):
                            fails.append(dict(system=sy, start=st, field=f, problem='start not scored', detail=why))
                    if (sd / f'rows_{c}.csv').exists():
                        rows_files.append(sd / f'rows_{c}.csv')
    with open(od / 'fail_summary.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['system', 'start', 'field', 'problem', 'detail'])
        w.writeheader()
        w.writerows(fails)
    n = 0
    with open(od / 'scores.csv.partial', 'w') as fo:
        for i, rf in enumerate(rows_files):
            with open(rf) as fi:
                hdr = fi.readline()
                if i == 0:
                    fo.write(hdr)
                for line in fi:
                    fo.write(line)
                    n += 1
    os.replace(od / 'scores.csv.partial', od / 'scores.csv')
    (od / 'inputs_hindcast_files.txt').write_text('\n'.join(inputs) + '\n')
    prov = dict(tag=tag, written=f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}', git_commit=D.git_commit(), config=cfg,
                package='s41eval.bing_eval', python=sys.version.split()[0], numpy=np.__version__,
                hindcast_files_read=nfiles, hindcast_file_list='inputs_hindcast_files.txt',
                era5_store=cfg['verification'].get('store'), climatology=cfg['climatology'], rows=n,
                verification_inputs={f'{f}_{s}': json.loads((verif_dir(cfg, tag, f, s) / 'meta.json').read_text()).get('clim_inputs')
                                     for f, s in samplings(cfg) if (verif_dir(cfg, tag, f, s) / 'meta.json').exists()},
                start_notes=cfg.get('start_notes', {}), fail_summary=dict(n=len(fails), file='fail_summary.csv'))
    write_json(od / 'provenance.json', prov)
    log(f'collect: {n} rows from {len(rows_files)} score dirs, {len(fails)} fail-summary lines')
    return n, len(fails)


def scores_to_nc(od):
    """scores.csv -> scores.nc (one variable 'value' with the row index; columns as coordinates)."""
    import pandas as pd
    df = pd.read_csv(od / 'scores.csv')
    ds = df.to_xarray()
    ds.attrs['description'] = 'long table of s41eval.bing_eval scores (same content as scores.csv)'
    ds.to_netcdf(od / 'scores.nc')
