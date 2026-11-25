#!/usr/bin/env python
"""C01 (exploratory): MJO skill day (bivariate COR = 0.5, 30-member ensemble mean, drift-corrected) by season / initial amplitude /
initial phase group, with block-bootstrap CIs; paired differences UC-S2S - FuXi and SST-anomaly - plain; amplitude and phase error by
lead; winds-only vs full RMM. Inputs: runs/bing/eval/modes/mjo/perstart_mjo_{full,winds}_<system>.nc (modes helper, a75ccbd).
Outputs: v1.5/helperC/mjo/*.csv, *.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval import bing_helperc as HC  # noqa: E402

OUT = HC.OUTC / 'mjo'
LEADS = np.arange(1, 46)
GROUPS = ['all', 'DJF', 'MAM', 'JJA', 'SON', 'active', 'inactive', 'P8-1', 'P2-3', 'P4-5', 'P6-7']


def groups_of(starts, init):
    amp = np.hypot(init[:, 0], init[:, 1])
    ph = HC.rmm_phase(init[:, 0], init[:, 1])
    g = {'all': np.ones(len(starts), bool), 'active': amp >= 1, 'inactive': amp < 1}
    for s in HC.SEASONS:
        g[s] = np.array([HC.season_of(d) == s for d in starts])
    for p, (a, b) in {'P8-1': (8, 1), 'P2-3': (2, 3), 'P4-5': (4, 5), 'P6-7': (6, 7)}.items():
        g[p] = (amp >= 1) & np.isin(ph, [a, b])
    return g


def sd_of(f, o):
    return HC.skill_day(HC.cor_curve(f[:, 1:], o[:, 1:]), LEADS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = {(s, ix): HC.load_mjo(s, ix) for s in HC.SYSTEMS for ix in ('full', 'winds')}
    starts = HC.common_starts([D[(s, 'full')]['starts'] for s in HC.SYSTEMS])
    D = {k: HC.subset(v, starts) for k, v in D.items()}
    init = D[('ucs2s_sstanom', 'full')]['init']            # ERA5 RMM at the init day (same ERA5 for every system file)
    for s in HC.SYSTEMS:
        assert np.allclose(D[(s, 'full')]['obs'][:, 1:30], D[('ucs2s_sstanom', 'full')]['obs'][:, 1:30], equal_nan=True) or s == 'fuxi'
    G = groups_of(starts, init)
    hA, hB = HC.halves(starts)
    rows, prs = [], []
    for ix in ('full', 'winds'):
        for g in GROUPS:
            m = G[g]
            n = int(m.sum())
            sel = np.flatnonzero(m)                      # date order kept
            bidx = HC.mbb_indices(len(sel), 4, 1000, seed=1)
            sd, draws = {}, {}
            for s in HC.SYSTEMS:
                d = D[(s, ix)]
                f, o = d['dc'][sel], d['obs'][sel]
                sd[s] = sd_of(f, o)
                draws[s] = np.array([sd_of(f[b], o[b]) for b in bidx])
                lo, hi = HC.ci_of(draws[s])
                rows.append(dict(index=ix, group=g, system=HC.LABEL[s], n=n, skill_day=sd[s], ci_lo=lo, ci_hi=hi,
                                 skill_day_2019_24=sd_of(f[hA[sel]], o[hA[sel]]) if hA[sel].sum() >= 5 else float('nan'),
                                 skill_day_2025_26=sd_of(f[hB[sel]], o[hB[sel]]) if hB[sel].sum() >= 5 else float('nan')))
            for a, b in (('ucs2s_sstanom', 'fuxi'), ('ucs2s_plain', 'fuxi'), ('ucs2s_sstanom', 'ucs2s_plain')):
                dd = draws[a] - draws[b]
                ci = HC.ci_of(dd)
                ra = [r for r in rows if r['index'] == ix and r['group'] == g]
                ga = {r['system']: r for r in ra}
                ma = ga[HC.LABEL[a]]['skill_day_2019_24'] - ga[HC.LABEL[b]]['skill_day_2019_24']
                mb = ga[HC.LABEL[a]]['skill_day_2025_26'] - ga[HC.LABEL[b]]['skill_day_2025_26']
                prs.append(dict(index=ix, group=g, pair=f'{HC.LABEL[a]} - {HC.LABEL[b]}', n=n, diff=sd[a] - sd[b], ci_lo=ci[0],
                                ci_hi=ci[1], diff_2019_24=ma, diff_2025_26=mb, grade=HC.grade(ci, ma, mb), label='exploratory'))
    # winds-only minus full, per system
    for g in ('all', 'active', 'inactive'):
        sel = np.flatnonzero(G[g])
        bidx = HC.mbb_indices(len(sel), 4, 1000, seed=2)
        for s in HC.SYSTEMS:
            fF, oF = D[(s, 'full')]['dc'][sel], D[(s, 'full')]['obs'][sel]
            fW, oW = D[(s, 'winds')]['dc'][sel], D[(s, 'winds')]['obs'][sel]
            full = sd_of(fW, oW) - sd_of(fF, oF)
            dr = np.array([sd_of(fW[b], oW[b]) - sd_of(fF[b], oF[b]) for b in bidx])
            ci = HC.ci_of(dr)
            a_ = hA[sel]
            b_ = hB[sel]
            ma = sd_of(fW[a_], oW[a_]) - sd_of(fF[a_], oF[a_])
            mb = sd_of(fW[b_], oW[b_]) - sd_of(fF[b_], oF[b_])
            prs.append(dict(index='winds-full', group=g, pair=f'{HC.LABEL[s]}: winds-only - full RMM', n=len(sel), diff=full,
                            ci_lo=ci[0], ci_hi=ci[1], diff_2019_24=ma, diff_2025_26=mb, grade=HC.grade(ci, ma, mb), label='exploratory'))
    HC.write_csv(rows, OUT / 'c01_skill_day_by_group.csv')
    HC.write_csv(prs, OUT / 'c01_skill_day_pairs.csv')

    # amplitude / phase error by lead, active and all; ensemble mean dc and member-mean amplitude ratio (raw members)
    ae = []
    for g in ('all', 'active'):
        sel = np.flatnonzero(G[g])
        bidx = HC.mbb_indices(len(sel), 4, 1000, seed=3)
        for s in HC.SYSTEMS:
            d = D[(s, 'full')]
            f, o, mem = d['dc'][sel], d['obs'][sel], d['members'][sel]
            ampo = np.hypot(o[..., 0], o[..., 1])
            ampf = np.hypot(f[..., 0], f[..., 1])
            ampm = np.hypot(mem[..., 0], mem[..., 1]).mean(1)          # mean member amplitude [S, L]
            cross = o[..., 0] * f[..., 1] - o[..., 1] * f[..., 0]
            pe = np.rad2deg(np.arctan2(cross, (o * f).sum(-1)))       # > 0 = forecast ahead (eastward / too fast)
            crossm = o[:, None, :, 0] * mem[..., 1] - o[:, None, :, 1] * mem[..., 0]
            pem = np.rad2deg(np.arctan2(crossm, (o[:, None] * mem).sum(-1))).mean(1)
            for k in range(1, 46):
                def st(b, k=k):
                    return [np.nanmean(ampf[b, k] - ampo[b, k]), np.nanmean(pe[b, k]), np.nanmean(ampm[b, k]) / np.nanmean(ampo[b, k]),
                            np.nanmean(pem[b, k])]
                full = st(np.arange(len(sel)))
                dr = np.array([st(b) for b in bidx])
                cis = [HC.ci_of(dr[:, j]) for j in range(4)]
                ae.append(dict(group=g, system=HC.LABEL[s], lead=k, n=len(sel), amp_err_em=full[0], amp_err_em_lo=cis[0][0],
                               amp_err_em_hi=cis[0][1], phase_err_em=full[1], phase_err_em_lo=cis[1][0], phase_err_em_hi=cis[1][1],
                               amp_ratio_members_raw=full[2], amp_ratio_lo=cis[2][0], amp_ratio_hi=cis[2][1],
                               phase_err_members_raw=full[3], phase_err_mem_lo=cis[3][0], phase_err_mem_hi=cis[3][1]))
    HC.write_csv(ae, OUT / 'c01_amp_phase_by_lead.csv')
    # COR curves by group (for plotting)
    cc = []
    for g in GROUPS:
        sel = np.flatnonzero(G[g])
        for s in HC.SYSTEMS:
            d = D[(s, 'full')]
            c = HC.cor_curve(d['dc'][sel], d['obs'][sel])
            for k in range(1, 46):
                cc.append(dict(group=g, system=HC.LABEL[s], lead=k, cor=float(c[k])))
    HC.write_csv(cc, OUT / 'c01_cor_by_group.csv')
    meta = dict(starts=len(starts), first=str(starts[0]), last=str(starts[-1]), n_groups={g: int(G[g].sum()) for g in GROUPS},
                source='runs/bing/eval/modes/mjo/perstart_mjo_*_<system>.nc (rmm_em30_dc, rmm_obs, rmm_init_obs, rmm_members_raw)',
                note='init state = ERA5 RMM (calendar-day sampling) at the init day; active = amplitude >= 1; phase groups active only; '
                     'phases depend on the provisional 37.5 deg rotation (same for every system); skill day = first lead with COR < 0.5, '
                     'linear interpolation; bootstrap: moving blocks of 4 starts within the group (date order), 1000 draws; exploratory')
    (OUT / 'c01_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta['n_groups']))


if __name__ == '__main__':
    main()
