#!/usr/bin/env python
"""Helper-C slide figures (S41 v1.5 set). python scripts/helperc_figs.py <fig> ...   (figs: c01 c01b c02 c03 c03b c04 c05 c06 c07)"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s41eval.sci import helperc_style as S  # noqa: E402

OUTC = S.V15 / 'helperC'
SYSL = ['UC-S2S SST-anomaly', 'UC-S2S plain', 'FuXi-S2S']
IFS_NOTE = 'IFS: slot reserved, not yet available.'


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def fmt0(v):
    r = int(round(v))
    return ('+' if r > 0 else '−' if r < 0 else '±') + str(abs(r))


def gshort(g):
    return {'robust': 'robust', 'suggestive': 'sugg.', 'noise': ''}.get(g, g)


# ---------------------------------------------------------------------------------------------------------------- C01
def c01():
    rows = rd(OUTC / 'mjo' / 'c01_skill_day_by_group.csv')
    prs = rd(OUTC / 'mjo' / 'c01_skill_day_pairs.csv')
    groups = ['all', 'DJF', 'MAM', 'JJA', 'SON', 'active', 'inactive', 'P8-1', 'P2-3', 'P4-5', 'P6-7']
    glab = ['all', 'DJF', 'MAM', 'JJA', 'SON', 'active', 'inactive', 'P8–1', 'P2–3', 'P4–5', 'P6–7']
    f, ax = S.fig(1, 1)
    f.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27)
    x = np.arange(len(groups)) + np.array([0, .5, .5, .5, .5, 1, 1, 1.5, 1.5, 1.5, 1.5])
    wd = 0.26
    for j, s in enumerate(SYSL):
        for i, g in enumerate(groups):
            r = next(r for r in rows if r['index'] == 'full' and r['group'] == g and r['system'] == s)
            v, lo, hi = fl(r['skill_day']), fl(r['ci_lo']), fl(r['ci_hi'])
            xx = x[i] + (j - 1) * wd
            ax.bar(xx, v, wd * 0.92, color=S.MODEL[s], label=s if i == 0 else None, alpha=0.9)
            ax.plot([xx, xx], [lo, hi], color='#444444' if s != 'FuXi-S2S' else '#9a9a9a', lw=1.6)
            if v >= 44.99:
                ax.text(xx, v + 0.5, '≥45', ha='center', va='bottom', fontsize=11)
    for i, g in enumerate(groups):
        n = next(r for r in rows if r['group'] == g)['n']
        p = next(r for r in prs if r['index'] == 'full' and r['group'] == g and r['pair'] == 'UC-S2S SST-anomaly - FuXi-S2S')
        q = next(r for r in prs if r['index'] == 'full' and r['group'] == g and r['pair'] == 'UC-S2S plain - FuXi-S2S')
        txt = f"{fmt0(fl(p['diff']))} {gshort(p['grade'])}\n{fmt0(fl(q['diff']))} {gshort(q['grade'])}"
        ax.text(x[i], 47.5, txt, ha='center', va='bottom', fontsize=10.5, color='#333333')
    ax.text(x[0] - 0.75, 47.5, 'UC−FuXi (d):\nSST-anom.\nplain', ha='right', va='bottom', fontsize=10.5, color='#333333')
    ax.set_xticks(x)
    nn = {r['group']: r['n'] for r in rows}
    ax.set_xticklabels([f'{a}\n{nn[g]}' for a, g in zip(glab, groups)], fontsize=14)
    ax.set_ylabel('MJO skill day (COR = 0.5)')
    ax.set_ylim(0, 52)
    ax.set_xlim(x[0] - 1.6, x[-1] + 0.5)
    for xs, t in ((np.mean(x[1:5]), 'season of start'), (np.mean(x[5:7]), 'MJO at init (amp. ≥ 1)'), (np.mean(x[7:]), 'phase at init (active; P4–5 = Maritime Cont.)')):
        ax.text(xs, -7.5, t, ha='center', va='top', fontsize=13, color=S.GREY)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    S.save(f, 'C01', 'mjo_skill_by_init_state',
           footer_text='Exploratory. Full RMM (WH04, ERA5 1979–2018 EOFs), 30-member ensemble mean, drift-corrected (LOYO ±30 d); 402 common starts 2019–26; '
                       'bars 95 % block-bootstrap CI (blocks of 4 starts, 1000 draws).\nNumbers above: skill-day difference UC-S2S − FuXi (days) and grade '
                       '(robust = CI excludes 0 and same sign 2019–24 / 2025–26). ' + S.PROVISIONAL_MJO + '. ' + IFS_NOTE)
    A = {(r['group'], r['system']): r for r in rows if r['index'] == 'full'}
    P = {(r['group'], r['pair']): r for r in prs}
    S.caption('C01', f"MJO skill day by initial state (exploratory): FuXi leads UC-S2S overall ({fl(A[('all','FuXi-S2S')]['skill_day']):.1f} vs "
                     f"{fl(A[('all','UC-S2S SST-anomaly')]['skill_day']):.1f}/{fl(A[('all','UC-S2S plain')]['skill_day']):.1f} d; "
                     f"SST-anom. − FuXi {fl(P[('all','UC-S2S SST-anomaly - FuXi-S2S')]['diff']):+.1f} d, {P[('all','UC-S2S SST-anomaly - FuXi-S2S')]['grade']}); "
                     f"the gap is largest for inactive-MJO starts ({fl(P[('inactive','UC-S2S SST-anomaly - FuXi-S2S')]['diff']):+.1f} d, "
                     f"{P[('inactive','UC-S2S SST-anomaly - FuXi-S2S')]['grade']}) and JJA, ≈ 0 in DJF and for P2–3/P4–5 (noise) · MJO phases provisional "
                     f"(37.5° rotation) · `v1.5/helperC/mjo/c01_skill_day_by_group.csv`, `c01_skill_day_pairs.csv`")
    kn = []
    for g in groups:
        kn.append(f"skill day {g}: " + ', '.join(f"{s} {fl(A[(g, s)]['skill_day']):.1f} [{fl(A[(g, s)]['ci_lo']):.1f}, {fl(A[(g, s)]['ci_hi']):.1f}]"
                                               for s in SYSL) + f" (n={A[(g, SYSL[0])]['n']}) · c01_skill_day_by_group.csv index=full group={g}")
    for g in groups:
        for pr in ('UC-S2S SST-anomaly - FuXi-S2S', 'UC-S2S plain - FuXi-S2S', 'UC-S2S SST-anomaly - UC-S2S plain'):
            r = P[(g, pr)]
            kn.append(f"{pr}, {g}: {fl(r['diff']):+.1f} d [{fl(r['ci_lo']):+.1f}, {fl(r['ci_hi']):+.1f}] {r['grade']} (exploratory) · "
                      f"c01_skill_day_pairs.csv index=full group={g}")
    S.key_numbers('C01', kn)


def c01b():
    rows = rd(OUTC / 'mjo' / 'c01_amp_phase_by_lead.csv')
    prs = rd(OUTC / 'mjo' / 'c01_skill_day_pairs.csv')
    f, axs = S.fig(1, 3)
    f.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, wspace=0.3)
    for s in SYSL:
        rr = [r for r in rows if r['group'] == 'active' and r['system'] == s]
        L = np.array([int(r['lead']) for r in rr])
        for ax, k, lo, hi in ((axs[0], 'amp_err_em', 'amp_err_em_lo', 'amp_err_em_hi'),
                              (axs[1], 'amp_ratio_members_raw', 'amp_ratio_lo', 'amp_ratio_hi'),
                              (axs[2], 'phase_err_members_raw', 'phase_err_mem_lo', 'phase_err_mem_hi')):
            ax.plot(L, [fl(r[k]) for r in rr], color=S.MODEL[s], lw=2.2, label=s)
            ax.fill_between(L, [fl(r[lo]) for r in rr], [fl(r[hi]) for r in rr], color=S.MODEL[s], alpha=0.15, lw=0)
    axs[0].set_title('amplitude error, ensemble mean', fontsize=16)
    axs[0].set_ylabel('|F| − |O| (RMM units)')
    axs[1].set_title('member amplitude / ERA5 amplitude', fontsize=16)
    axs[1].axhline(1, color=S.GREY, lw=1)
    axs[2].set_title('phase error, members (°; < 0 slow)', fontsize=16)
    for ax in axs:
        ax.axhline(0, color=S.GREY, lw=1) if ax is not axs[1] else None
        ax.set_xlabel('lead (days)')
        ax.set_xlim(1, 45)
    axs[0].legend(loc='lower left', frameon=False, fontsize=13)
    S.save(f, 'C01b', 'mjo_amp_phase_error', slide=True,
           footer_text='Exploratory. Starts with an active MJO at init (ERA5 amplitude ≥ 1; 242 of 402 common starts 2019–26). Left: drift-corrected 30-member '
                       'mean; middle/right: raw members, averaged over members and starts.\nShading 95 % block-bootstrap CI. Phase error = signed angle '
                       'ERA5 → forecast in the RMM plane. ' + S.PROVISIONAL_MJO + '. ' + IFS_NOTE)
    A = {(r['system'], int(r['lead'])): r for r in rows if r['group'] == 'active'}
    S.caption('C01b', f"MJO too weak and too slow in UC-S2S (exploratory, active starts): member phase error at day 28 "
                      f"{fl(A[('UC-S2S SST-anomaly', 28)]['phase_err_members_raw']):+.0f}° / {fl(A[('UC-S2S plain', 28)]['phase_err_members_raw']):+.0f}° "
                      f"(FuXi {fl(A[('FuXi-S2S', 28)]['phase_err_members_raw']):+.0f}°); member amplitude ratio at day 7 "
                      f"{fl(A[('UC-S2S SST-anomaly', 7)]['amp_ratio_members_raw']):.2f}/{fl(A[('UC-S2S plain', 7)]['amp_ratio_members_raw']):.2f} "
                      f"(FuXi {fl(A[('FuXi-S2S', 7)]['amp_ratio_members_raw']):.2f}) · phases provisional · `v1.5/helperC/mjo/c01_amp_phase_by_lead.csv`")
    kn = []
    for s in SYSL:
        for k in (7, 14, 21, 28, 35):
            r = A[(s, k)]
            kn.append(f"{s} day {k} (active init): em amp err {fl(r['amp_err_em']):+.2f} [{fl(r['amp_err_em_lo']):+.2f}, {fl(r['amp_err_em_hi']):+.2f}]; "
                      f"member amp ratio {fl(r['amp_ratio_members_raw']):.2f} [{fl(r['amp_ratio_lo']):.2f}, {fl(r['amp_ratio_hi']):.2f}]; member phase err "
                      f"{fl(r['phase_err_members_raw']):+.1f}° [{fl(r['phase_err_mem_lo']):+.1f}, {fl(r['phase_err_mem_hi']):+.1f}] · c01_amp_phase_by_lead.csv group=active lead={k}")
    for s in SYSL:
        r = next(r for r in prs if r['index'] == 'winds-full' and r['group'] == 'all' and r['pair'].startswith(s))
        kn.append(f"{s}: winds-only − full RMM skill day {fl(r['diff']):+.1f} d [{fl(r['ci_lo']):+.1f}, {fl(r['ci_hi']):+.1f}] {r['grade']} · "
                  f"c01_skill_day_pairs.csv index=winds-full group=all")
    S.key_numbers('C01b', kn)


# ---------------------------------------------------------------------------------------------------------------- C02
def c02(groups=('P2-3', 'P4-5'), fid='C02', name='mjo_hovmoller_olr', slide=True):
    Z = np.load(OUTC / 'hov' / 'c02_hov_composites.npz')
    sc = rd(OUTC / 'hov' / 'c02_hov_scores.csv')
    pr = rd(OUTC / 'hov' / 'c02_propagation.csv')
    lon, leads = Z['lon'], Z['leads']
    reg = (lon >= 40) & (lon <= 220)
    cols = [('ERA5', 'era5_00z'), ('UC-S2S SST-anomaly', 'ucs2s_sstanom'), ('UC-S2S plain', 'ucs2s_plain'), ('FuXi-S2S', 'fuxi')]
    f, axs = S.fig(len(groups), 4, sharex=True, sharey=True)
    f.subplots_adjust(left=0.07, right=0.9, top=0.91, bottom=0.27, wspace=0.08, hspace=0.32)
    lev = np.arange(-22, 23, 4)
    for r, g in enumerate(groups):
        for c, (lab, key) in enumerate(cols):
            ax = axs[r, c]
            o = Z[f'{g}_olr_{key}'][:, reg]
            u = Z[f'{g}_u850_{key}'][:, reg]
            cf = ax.contourf(lon[reg], leads, o, levels=lev, cmap=S.DIV, extend='both')
            ax.contour(lon[reg], leads, u, levels=[-3, -2, -1], colors='k', linewidths=0.8, linestyles='--')
            ax.contour(lon[reg], leads, u, levels=[1, 2, 3], colors='k', linewidths=0.8)
            for xx in (100, 150):
                ax.axvline(xx, color=S.GREY, lw=1, ls=':')
            n = next(x['n'] for x in sc if x['group'] == g)
            if c == 0:
                ax.set_title(f'ERA5 · init {g.replace("-", "–")} (n={n})', fontsize=15)
                ax.set_ylabel('lead (days)')
            else:
                q = next(x for x in sc if x['group'] == g and x['system'] == lab and x['field'] == 'olr' and x['window'] == 'w1-4')
                q4 = next(x for x in sc if x['group'] == g and x['system'] == lab and x['field'] == 'olr' and x['window'] == 'w4')
                ax.set_title(f'{lab}\nr = {fl(q["pattern_corr"]):.2f} (wk 4: {fl(q4["pattern_corr"]):.2f})', fontsize=13.5,
                             color=S.MODEL[lab] if lab != 'FuXi-S2S' else 'k')
            ax.set_ylim(0, 28)
            ax.set_xticks([60, 100, 150, 200])
            if r == len(groups) - 1:
                ax.set_xlabel('longitude (°E)')
    cax = f.add_axes([0.915, 0.3, 0.012, 0.55])
    cb = f.colorbar(cf, cax=cax)
    cb.set_label('OLR anomaly (W m⁻²)')
    sp = {x['system']: x for x in pr if x['group'] == 'active'}
    mc = {x['system']: x for x in pr if x['group'].startswith('MC crossing')}
    txt = ('RMM phase speed, active starts, days 1–20 (°/day): ERA5 ' + f"{fl(sp['UC-S2S SST-anomaly']['speed_obs_deg_per_day']):.1f}; " +
           '; '.join(f"{s} {fl(sp[s]['speed_members_deg_per_day']):.1f} ({sp[s]['grade']})" for s in SYSL) +
           '.  MC crossing (init P3–4 reaching P6–7 by day 20): ERA5 ' + f"{100 * fl(mc['UC-S2S SST-anomaly']['speed_obs_deg_per_day']):.0f} %; " +
           '; '.join(f"{s} {100 * fl(mc[s]['speed_members_deg_per_day']):.0f} %" for s in SYSL) + ' of members.')
    S.save(f, fid, name, slide=slide,
           footer_text='Exploratory. 15°S–15°N OLR (shading, blue = convection) and u850 (contours ±1, 2, 3 m/s, dashed easterly) RMM-input anomalies; '
                       'active-MJO starts by init phase; 30-member mean, drift-corrected; dotted: Maritime Continent; r = pattern corr. with ERA5, 40–220°E, days 1–28.\n' + txt + ' ' + S.PROVISIONAL_MJO + '. ' + IFS_NOTE)
    return sp, mc


def c02_caption():
    sp, mc = c02()
    c02(groups=('P6-7', 'P8-1'), fid='C02x', name='mjo_hovmoller_olr_P6-P1', slide=False)
    S.caption('C02', f"UC-S2S MJO propagates too slowly and stalls at the Maritime Continent (exploratory): RMM phase speed days 1–20 "
                     f"{fl(sp['UC-S2S SST-anomaly']['speed_members_deg_per_day']):.1f}/{fl(sp['UC-S2S plain']['speed_members_deg_per_day']):.1f} °/day vs ERA5 "
                     f"{fl(sp['UC-S2S SST-anomaly']['speed_obs_deg_per_day']):.1f} ({sp['UC-S2S SST-anomaly']['grade']}/{sp['UC-S2S plain']['grade']}; FuXi "
                     f"{fl(sp['FuXi-S2S']['speed_members_deg_per_day']):.1f}, {sp['FuXi-S2S']['grade']}); P3–4 starts crossing to P6–7 by day 20: "
                     f"{100 * fl(mc['UC-S2S SST-anomaly']['speed_members_deg_per_day']):.0f}/{100 * fl(mc['UC-S2S plain']['speed_members_deg_per_day']):.0f} % of UC "
                     f"members vs {100 * fl(mc['UC-S2S SST-anomaly']['speed_obs_deg_per_day']):.0f} % in ERA5 (FuXi {100 * fl(mc['FuXi-S2S']['speed_members_deg_per_day']):.0f} %) · "
                     f"phases provisional · `v1.5/helperC/hov/c02_propagation.csv`, `c02_hov_scores.csv`")
    sc = rd(OUTC / 'hov' / 'c02_hov_scores.csv')
    pr = rd(OUTC / 'hov' / 'c02_propagation.csv')
    kn = [f"{x['system']} {x['group']}: phase speed members {fl(x['speed_members_deg_per_day']):.2f} vs ERA5 {fl(x['speed_obs_deg_per_day']):.2f} °/day, diff "
          f"{fl(x['diff']):+.2f} [{fl(x['ci_lo']):+.2f}, {fl(x['ci_hi']):+.2f}] {x['grade']} (n={x['n']}) · c02_propagation.csv" for x in pr if not x['group'].startswith('MC')]
    kn += [f"{x['system']} MC crossing P3–4→P6–7 by day 20: members {fl(x['speed_members_deg_per_day']):.3f} vs ERA5 {fl(x['speed_obs_deg_per_day']):.3f}, diff "
           f"{fl(x['diff']):+.3f} [{fl(x['ci_lo']):+.3f}, {fl(x['ci_hi']):+.3f}] {x.get('grade', '')} (n={x['n']}) · c02_propagation.csv" for x in pr if x['group'].startswith('MC')]
    kn += [f"Hovmöller OLR pattern corr {x['group']} {x['system']} {x['window']}: {fl(x['pattern_corr']):.2f}, amp ratio {fl(x['amp_ratio']):.2f} · c02_hov_scores.csv field=olr"
           for x in sc if x['field'] == 'olr' and x['window'] in ('w1-4', 'w3', 'w4')]
    S.key_numbers('C02', kn)


# ---------------------------------------------------------------------------------------------------------------- C04
def c04(group='DJF', fid='C04', slide=True):
    rows = rd(OUTC / 'tele' / 'c04_tele_cor.csv')
    prs = rd(OUTC / 'tele' / 'c04_tele_pairs.csv')
    f, axs = S.fig(1, 4, sharey=True)
    f.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.25, wspace=0.12)
    wk = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    x = np.arange(1, 7)
    test = {'nao': 'ref. test failed (DJF r 0.88)', 'pna': 'ref. test failed (DJF r 0.88)', 'ao': 'ref. test failed (DJF r 0.91)',
            'sam': 'ref. test passed (DJF r 0.97)'}
    A = {(r['index'], r['system'], r['window']): r for r in rows if r['group'] == group}
    for ax, k in zip(axs, ['nao', 'pna', 'ao', 'sam']):
        for s, col, ls in [(s, S.MODEL[s], '-') for s in SYSL] + [('persistence_00z', S.GREY, '--')]:
            v = [fl(A[(k, s, w)]['value']) for w in wk]
            lo = [fl(A[(k, s, w)]['ci_lo']) for w in wk]
            hi = [fl(A[(k, s, w)]['ci_hi']) for w in wk]
            lab = 'ERA5 persistence' if s.startswith('pers') else s
            ax.plot(x, v, color=col, ls=ls, lw=2.2, marker='o', ms=5, label=lab)
            ax.fill_between(x, lo, hi, color=col, alpha=0.08, lw=0)
        ax.axhline(0.5, color=S.GREY, lw=1, ls=':')
        ax.axhline(0, color=S.GREY, lw=0.8)
        sd = '; '.join(f"{'SST' if 'SST' in s else 'plain' if 'plain' in s else 'FuXi'} {fl(A[(k, s, 'skill_day')]['value']):.0f}" for s in SYSL)
        q = next(r for r in prs if r['group'] == group and r['index'] == k and r['window'] == 'w3' and r['pair'] == 'UC-S2S SST-anomaly - FuXi-S2S')
        ax.set_title(f"{k.upper()} · {test[k]}\nskill day: {sd}\nwk 3 SST-anom. − FuXi {fl(q['diff']):+.2f} ({q['grade']})", fontsize=12.5)
        ax.set_xticks(x)
        ax.set_xlabel('lead week')
        ax.set_ylim(-0.3, 1.02)
    axs[0].set_ylabel(f'COR of weekly index ({group} starts)')
    axs[0].legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='white', fontsize=12)
    n = A[('nao', SYSL[0], 'w1')]['n']
    S.save(f, fid, f'tele_indices_{group}', slide=slide,
           footer_text=f'Exploratory. Weekly-mean z500 indices (CPC-like rotated EOFs of ERA5 1979–2018), 30-member mean, drift-corrected; {group} starts '
                       f'2019–26 (n = {n}); shading 95 % block-bootstrap CI. Persistence = ERA5 index of the 7 days ending at init. Skill day = lead where the weekly COR '
                       f'crosses 0.5 (interpolated between week mid-days; Feng et al. 2021 benchmark 12–16 d from daily indices). NAO/PNA/AO: reference test vs CPC '
                       f'failed (all-month r 0.745/0.54/0.71) — DJF rows only. ' + IFS_NOTE)
    return A


def c04_caption():
    A = c04('DJF', 'C04')
    c04('NDJFM', 'C04x', slide=False)
    prs = rd(OUTC / 'tele' / 'c04_tele_pairs.csv')
    P = {(r['group'], r['index'], r['window'], r['pair']): r for r in prs}
    S.caption('C04', f"Extratropical modes (DJF, exploratory; NAO/PNA/AO reference test failed): UC-S2S weekly NAO/PNA/AO/SAM COR stays above 0.5 to "
                     f"day {fl(A[('nao', 'UC-S2S SST-anomaly', 'skill_day')]['value']):.0f}/{fl(A[('pna', 'UC-S2S SST-anomaly', 'skill_day')]['value']):.0f}/"
                     f"{fl(A[('ao', 'UC-S2S SST-anomaly', 'skill_day')]['value']):.0f}/{fl(A[('sam', 'UC-S2S SST-anomaly', 'skill_day')]['value']):.0f} "
                     f"(FuXi {fl(A[('nao', 'FuXi-S2S', 'skill_day')]['value']):.0f}/{fl(A[('pna', 'FuXi-S2S', 'skill_day')]['value']):.0f}/"
                     f"{fl(A[('ao', 'FuXi-S2S', 'skill_day')]['value']):.0f}/{fl(A[('sam', 'FuXi-S2S', 'skill_day')]['value']):.0f}); week-3 NAO COR "
                     f"{fl(A[('nao', 'UC-S2S SST-anomaly', 'w3')]['value']):.2f} vs persistence {fl(A[('nao', 'persistence_00z', 'w3')]['value']):.2f} "
                     f"({P[('DJF', 'nao', 'w3', 'UC-S2S SST-anomaly - persistence_00z')]['grade']}); UC − FuXi differences noise in DJF · "
                     f"`v1.5/helperC/tele/c04_tele_cor.csv`, `c04_tele_pairs.csv`")
    kn = []
    for g in ('DJF', 'NDJFM'):
        for k in ('nao', 'pna', 'ao', 'sam'):
            rr = [r for r in rd(OUTC / 'tele' / 'c04_tele_cor.csv') if r['group'] == g and r['index'] == k]
            for s in SYSL + ['persistence_00z']:
                a = {r['window']: r for r in rr if r['system'] == s}
                kn.append(f"{g} {k.upper()} {s}: COR w1–w4 " + ' / '.join(f"{fl(a[w]['value']):.2f}" for w in ('w1', 'w2', 'w3', 'w4')) +
                          f"; w34 {fl(a['w34']['value']):.2f} [{fl(a['w34']['ci_lo']):.2f}, {fl(a['w34']['ci_hi']):.2f}]; skill day {fl(a['skill_day']['value']):.1f} "
                          f"[{fl(a['skill_day']['ci_lo']):.1f}, {fl(a['skill_day']['ci_hi']):.1f}] · c04_tele_cor.csv group={g} index={k}")
            for pr in ('UC-S2S SST-anomaly - FuXi-S2S', 'UC-S2S plain - FuXi-S2S', 'UC-S2S SST-anomaly - UC-S2S plain', 'UC-S2S SST-anomaly - persistence_00z'):
                for w in ('w3', 'w34'):
                    r = P[(g, k, w, pr)]
                    kn.append(f"{g} {k.upper()} {w} {pr}: {fl(r['diff']):+.3f} [{fl(r['ci_lo']):+.3f}, {fl(r['ci_hi']):+.3f}] {r['grade']} · c04_tele_pairs.csv")
    S.key_numbers('C04', kn)


# ---------------------------------------------------------------------------------------------------------------- C06
def _damped_from_modes():
    rr = [r for r in rd(S.R / 'runs' / 'bing' / 'eval' / 'modes' / 'enso' / 'perstart_nino34.csv') if r['system_key'] == 'ucs2s_sstanom' and r['correction'] == 'dc']
    out = {}
    for w in ('w1', 'w2', 'w3', 'w4', 'w5', 'w6'):
        d = np.array([fl(r[f'damped_{w}']) for r in rr])
        o = np.array([fl(r[f'obs_{w}']) for r in rr])
        a = np.array([fl(r['init_anom']) for r in rr])
        ok = np.isfinite(d) & np.isfinite(o)
        out[w] = (float(np.sqrt(np.mean((d[ok] - o[ok]) ** 2))), float(np.corrcoef(d[ok] - a[ok], o[ok] - a[ok])[0, 1]))
    return out


def c06():
    rows = rd(OUTC / 'enso' / 'c06_nino34.csv')
    prs = rd(OUTC / 'enso' / 'c06_nino34_pairs.csv')
    cmp_ = rd(OUTC / 'enso' / 'c06_enso_composites.csv')
    A = {(r['system'], r['window'], r['metric']): r for r in rows}
    P = {(r['pair'], r['window'], r['metric']): r for r in prs}
    dm = _damped_from_modes()
    wk = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    x = np.arange(1, 7)
    f, axs = S.fig(1, 3)
    f.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.27, wspace=0.32)
    for ax, met, yl in ((axs[0], 'rmse', 'Niño-3.4 anomaly RMSE (K)'), (axs[1], 'tendency_acc', 'ACC of the anomaly tendency')):
        for sname in SYSL:
            v = [fl(A[(sname, w, met)]['value']) for w in wk]
            ax.plot(x, v, color=S.MODEL[sname], lw=2.2, marker='o', ms=5, label=sname)
            ax.fill_between(x, [fl(A[(sname, w, met)]['ci_lo']) for w in wk], [fl(A[(sname, w, met)]['ci_hi']) for w in wk], color=S.MODEL[sname], alpha=0.12, lw=0)
        ax.plot(x, [dm[w][0 if met == 'rmse' else 1] for w in wk], color=S.GREY, ls='--', lw=2, marker='o', ms=4, label='damped persistence')
        ax.set_xticks(x)
        ax.set_xlabel('lead week')
        ax.set_ylabel(yl)
    q = P[('UC-S2S SST-anomaly - UC-S2S plain', 'w34', 'rmse')]
    q2 = P[('UC-S2S SST-anomaly - UC-S2S plain', 'w34', 'tendency_acc')]
    axs[0].set_title(f"weeks 3–4 SST-anom. − plain: {fl(q['diff']):+.2f} K ({q['grade']})", fontsize=13)
    axs[1].set_title(f"weeks 3–4 SST-anom. − plain: {fl(q2['diff']):+.2f} ({q2['grade']})", fontsize=13)
    axs[0].legend(loc='upper left', frameon=False, fontsize=12)
    ax = axs[2]
    C = {(r['field'], r['group'], r['system']): r for r in cmp_}
    xx = np.arange(2)
    wd = 0.26
    for j, sname in enumerate(SYSL):
        v = [fl(C[('tp24', 'all', sname)]['amp_ratio']), fl(C[('z500', 'all', sname)]['amp_ratio'])]
        r = [fl(C[('tp24', 'all', sname)]['pattern_corr']), fl(C[('z500', 'all', sname)]['pattern_corr'])]
        ax.bar(xx + (j - 1) * wd, v, wd * 0.92, color=S.MODEL[sname])
        for k in range(2):
            ax.text(xx[k] + (j - 1) * wd, v[k] + 0.02, f'r {r[k]:.2f}', ha='center', va='bottom', fontsize=10, rotation=90)
    ax.axhline(1, color=S.GREY, lw=1)
    ax.set_xticks(xx)
    ax.set_xticklabels(['tp24\n30°S–60°N', 'z500\n20–90°N'])
    ax.set_ylim(0, 1.45)
    ax.set_ylabel('El Niño − La Niña amplitude / ERA5')
    ax.set_title('ENSO composite, weeks 3–4', fontsize=13)
    n = C[('tp24', 'all', SYSL[0])]
    S.save(f, 'C06', 'enso_nino34_teleconnections',
           footer_text=f"Exploratory. Package v1.5 (UC-S2S SST land-masked). Niño-3.4 (5°S–5°N, 170–120°W) window-mean SST anomaly vs the fixed 1991–2020 climatology, 30-member mean, drift-corrected, "
                       f"403 common starts; tendency = anomaly minus the ERA5 init-day anomaly; damped persistence (grey) from the modes ENSO part. Right: weeks 3–4 composite "
                       f"difference of starts with El Niño ({n['n_el_nino']}) and La Niña ({n['n_la_nina']}) at init (30-day Niño-3.4 ≥ ±0.5 K); r = pattern correlation with ERA5 "
                       f"same starts. CIs: 95 % block bootstrap. " + IFS_NOTE)
    S.caption('C06', f"ENSO (exploratory): the SST-anomaly model is worse than plain on Niño-3.4 at weeks 3–4 (RMSE {fl(A[('UC-S2S SST-anomaly','w34','rmse')]['value']):.2f} vs "
                     f"{fl(A[('UC-S2S plain','w34','rmse')]['value']):.2f} K, {q['grade']}; tendency ACC {fl(A[('UC-S2S SST-anomaly','w34','tendency_acc')]['value']):.2f} vs "
                     f"{fl(A[('UC-S2S plain','w34','tendency_acc')]['value']):.2f}, {q2['grade']}); FuXi best (0.20 K, {fl(A[('FuXi-S2S','w34','tendency_acc')]['value']):.2f}); "
                     f"UC-S2S reproduces the ENSO rainfall pattern (r {fl(C[('tp24','all','UC-S2S SST-anomaly')]['pattern_corr']):.2f}) but at "
                     f"{100 * fl(C[('tp24','all','UC-S2S SST-anomaly')]['amp_ratio']):.0f} %/{100 * fl(C[('tp24','all','UC-S2S plain')]['amp_ratio']):.0f} % of ERA5's amplitude "
                     f"(FuXi {100 * fl(C[('tp24','all','FuXi-S2S')]['amp_ratio']):.0f} %) · `v1.5/helperC/enso/c06_nino34*.csv`, `c06_enso_composites.csv`")
    kn = []
    for sname in SYSL:
        for w in ('w1', 'w2', 'w34', 'w56'):
            kn.append(f"Niño-3.4 {sname} {w}: RMSE {fl(A[(sname, w, 'rmse')]['value']):.3f} K [{fl(A[(sname, w, 'rmse')]['ci_lo']):.3f}, {fl(A[(sname, w, 'rmse')]['ci_hi']):.3f}], "
                      f"ACC {fl(A[(sname, w, 'acc')]['value']):.3f}, tendency ACC {fl(A[(sname, w, 'tendency_acc')]['value']):.2f} [{fl(A[(sname, w, 'tendency_acc')]['ci_lo']):.2f}, "
                      f"{fl(A[(sname, w, 'tendency_acc')]['ci_hi']):.2f}] · c06_nino34.csv")
    for k_, r in P.items():
        if k_[1] in ('w34', 'w56'):
            kn.append(f"Niño-3.4 {k_[0]} {k_[1]} {k_[2]}: {fl(r['diff']):+.3f} [{fl(r['ci_lo']):+.3f}, {fl(r['ci_hi']):+.3f}] {r['grade']} · c06_nino34_pairs.csv")
    for w in ('w1', 'w2', 'w3', 'w4'):
        kn.append(f"Niño-3.4 damped persistence {w}: RMSE {dm[w][0]:.3f} K, tendency ACC {dm[w][1]:.2f} · modes/enso/perstart_nino34.csv (harmonic clim)")
    for r in cmp_:
        kn.append(f"ENSO composite {r['field']} {r['group']} {r['system']}: pattern r {fl(r['pattern_corr']):.2f}, amplitude ratio {fl(r['amp_ratio']):.2f} "
                  f"(El Niño {r['n_el_nino']}, La Niña {r['n_la_nina']} starts) · c06_enso_composites.csv")
    for r in rd(OUTC / 'enso' / 'c06_soil_tcc.csv'):
        if r['subset'] in ('pooled|all', 'pooled|JJA') and r['window'] in ('w1', 'w2', 'w34', 'w56'):
            kn.append(f"swvl1 TCC land mean {r['system']} {r['correction']} {r['subset']} {r['window']}: {fl(r['land_mean_tcc']):.2f} · c06_soil_tcc.csv")
    S.key_numbers('C06', kn)


def c06b():
    Z = np.load(OUTC / 'enso' / 'c06_enso_composites.npz')
    cmp_ = {(r['field'], r['group'], r['system']): r for r in rd(OUTC / 'enso' / 'c06_enso_composites.csv')}
    lat, lon = Z['lat'], Z['lon']
    lsm = np.load(S.R / 'data' / 'era5_bing_1p5' / 'lsm.npy')
    f, axs = S.fig(2, 4, sharex=True, sharey=True)
    f.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.2, wspace=0.05, hspace=0.25)
    cols = [('ERA5', 'era5_00z'), ('UC-S2S SST-anomaly', 'ucs2s_sstanom'), ('UC-S2S plain', 'ucs2s_plain'), ('FuXi-S2S', 'fuxi')]
    for r, (fld, lev, unit) in enumerate((('tp24', np.arange(-3.0, 3.01, 0.5), 'mm/day'), ('z500', np.arange(-60, 61, 10), 'm'))):
        for c, (lab, key) in enumerate(cols):
            ax = axs[r, c]
            x = Z[f'{fld}_all_{key}']
            if fld == 'z500':
                x = x / 9.80665 if np.nanmax(np.abs(x)) > 300 else x
            cf = ax.contourf(lon, lat, x, levels=lev, cmap=S.DIV if fld == 'z500' else 'RdBu', extend='both')
            ax.contour(np.arange(lsm.shape[1]) * (360 / lsm.shape[1]), np.linspace(-90, 90, lsm.shape[0]), lsm, levels=[0.5], colors='k', linewidths=0.5)
            ax.set_ylim(-30, 80)
            if c == 0:
                ax.set_title(f'ERA5 · {fld}', fontsize=14)
            else:
                q = cmp_[(fld, 'all', lab)]
                ax.set_title(f"{lab}\nr {fl(q['pattern_corr']):.2f}, amp. {fl(q['amp_ratio']):.2f}", fontsize=12.5)
        cax = f.add_axes([0.93, 0.58 - r * 0.38, 0.01, 0.3])
        f.colorbar(cf, cax=cax).set_label(unit)
    S.save(f, 'C06b', 'enso_composite_maps_w34', slide=False,
           footer_text='Exploratory. Weeks 3–4 anomaly composites El Niño minus La Niña starts (all seasons; 30-day Niño-3.4 ≥ +0.5 / ≤ −0.5 K at init), 30-member mean, '
                       'drift-corrected, vs ERA5 same starts (package v1.5). tp24: blue = wetter in El Niño. ' + IFS_NOTE)


# ---------------------------------------------------------------------------------------------------------------- C05
def c05(slide=False):
    hr = rd(OUTC / 'strat' / 'c05_ssw_hit_rate.csv')
    acc = rd(OUTC / 'strat' / 'c05_acc_by_lead.csv')
    l6 = rd(OUTC / 'strat' / 'c05_acc_0p6_lead.csv')
    f, axs = S.fig(1, 2)
    f.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.27, wspace=0.25)
    ax = axs[0]
    bins = ['days 1-7', 'days 8-14', 'days 15-21', 'days 22-28', 'days 29-42']
    x = np.arange(len(bins))
    series = [('UC-S2S SST-anomaly', '10'), ('UC-S2S plain', '10'), ('UC-S2S SST-anomaly', '50'), ('UC-S2S plain', '50'), ('FuXi-S2S', '50')]
    wd = 0.16
    for j, (sname, L) in enumerate(series):
        rr = {r['lead_bin']: r for r in hr if r['system'] == sname and r['level'] == L}
        v = [fl(rr[b]['hit_rate']) if b in rr else np.nan for b in bins]
        ax.bar(x + (j - 2) * wd, v, wd * 0.9, color=S.MODEL[sname], hatch='' if L == '10' else '//', edgecolor='white' if L == '10' else '#dddddd',
               label=f'{sname} {L} hPa')
        for k, b in enumerate(bins):
            if b in rr:
                ax.text(x[k] + (j - 2) * wd, max(v[k], 0) + 0.02, f"{rr[b]['hits']}/{rr[b]['n_cases']}", ha='center', va='bottom', fontsize=8.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(['1–7', '8–14', '15–21', '22–28', '29–42'])
    ax.set_xlabel('lead of the ERA5 SSW onset (days)')
    ax.set_ylabel('hit rate (P(reversal ±3 d) ≥ 0.5)')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', frameon=False, fontsize=10.5)
    ax.set_title('SSW hits, 2019–26 ERA5 events (6 at 10 hPa, 3 at 50 hPa)', fontsize=13)
    ax = axs[1]
    for sname in SYSL:
        for ind, ls in (('U60N10', '-'), ('U60N50', '--'), ('cap100', ':')):
            rr = [r for r in acc if r['system'] == sname and r['index'] == ind]
            if not rr:
                continue
            L = [int(r['lead']) for r in rr]
            ax.plot(L, [fl(r['acc']) for r in rr], color=S.MODEL[sname], ls=ls, lw=2, label=f'{sname} {ind}' if sname != 'UC-S2S plain' else None)
    ax.axhline(0.6, color=S.GREY, lw=1, ls=':')
    ax.set_xlabel('lead (days)')
    ax.set_ylabel('anomaly ACC (NDJFM starts)')
    ax.set_xlim(1, 45)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower left', frameon=False, fontsize=10.5, ncol=1)
    L6 = {(r['system'], r['index']): r for r in l6}
    ax.set_title(f"lead of ACC = 0.6: U60N10 {fl(L6[('UC-S2S SST-anomaly','U60N10')]['lead_acc_0p6']):.0f}/{fl(L6[('UC-S2S plain','U60N10')]['lead_acc_0p6']):.0f} d; "
                 f"U60N50 {fl(L6[('UC-S2S SST-anomaly','U60N50')]['lead_acc_0p6']):.0f}/{fl(L6[('UC-S2S plain','U60N50')]['lead_acc_0p6']):.0f}/{fl(L6[('FuXi-S2S','U60N50')]['lead_acc_0p6']):.0f} d",
                 fontsize=13)
    S.save(f, 'C05', 'stratosphere_ssw_u60', slide=slide,
           footer_text='Exploratory. Left: ERA5 SSWs (Charlton–Polvani U60 reversal) 2019–26 with a start 1–42 d before onset; hit = ≥ 50 % of the 30 members reverse within ±3 d of the '
                       'ERA5 onset; numbers = hits / cases (very small samples). FuXi has no 10 hPa. Right: zonal-mean U at 60°N (10, 50 hPa) and 60–90°N polar-cap height '
                       '(100 hPa, dotted) anomaly ACC, 30-member mean, drift-corrected, 164 NDJFM starts 2019–26 (UC-S2S plain dashed/dotted lines omitted from the legend). ' + IFS_NOTE)
    S.caption('C05', "Stratosphere (exploratory): SSWs are hit only within two weeks (10 hPa: 4/5 and 3/6 cases at days 1–7 / 8–14 for SST-anomaly, 0/23 beyond day 14); "
                     f"U60N10 ACC stays ≥ 0.6 to day {fl(L6[('UC-S2S SST-anomaly','U60N10')]['lead_acc_0p6']):.0f} (CI {fl(L6[('UC-S2S SST-anomaly','U60N10')]['ci_lo']):.0f}–"
                     f"{fl(L6[('UC-S2S SST-anomaly','U60N10')]['ci_hi']):.0f}); at 50 hPa UC-S2S {fl(L6[('UC-S2S SST-anomaly','U60N50')]['lead_acc_0p6']):.0f} d vs FuXi "
                     f"{fl(L6[('FuXi-S2S','U60N50')]['lead_acc_0p6']):.0f} d · `v1.5/helperC/strat/c05_*.csv`", slide=slide)
    kn = [f"SSW hit rate {r['system']} {r['level']} hPa {r['lead_bin']}: {r['hits']}/{r['n_cases']} ({r['n_events']} events), mean P {fl(r['mean_p']):.2f} · c05_ssw_hit_rate.csv" for r in hr]
    kn += [f"{r['system']} {r['index']} lead of ACC 0.6: {fl(r['lead_acc_0p6']):.1f} d [{fl(r['ci_lo']):.1f}, {fl(r['ci_hi']):.1f}] (2019–24 {fl(r['h2019_24']):.1f}, 2025–26 {fl(r['h2025_26']):.1f}) · c05_acc_0p6_lead.csv" for r in l6]
    kn += [f"{r['system']} {r['index']} ACC day {r['lead']}: {fl(r['acc']):.2f} [{fl(r['ci_lo']):.2f}, {fl(r['ci_hi']):.2f}] · c05_acc_by_lead.csv" for r in acc if r['lead'] in ('14', '21', '28') and r['index'] in ('U60N10', 'U60N50', 'cap100', 'cap500')]
    S.key_numbers('C05', kn)


# ---------------------------------------------------------------------------------------------------------------- C07
C07_ROWS = [('MJO active - inactive (NDJFM)', 'MJO active vs inactive (NDJFM)', 'mjo'),
            ('MJO P2-3 - inactive (NDJFM)', '  P2–3 vs inactive', 'mjo'), ('MJO P4-5 - inactive (NDJFM)', '  P4–5 vs inactive', 'mjo'),
            ('MJO P6-7 - inactive (NDJFM)', '  P6–7 vs inactive', 'mjo'), ('MJO P8-1 - inactive (NDJFM)', '  P8–1 vs inactive', 'mjo'),
            ('ENSO active - neutral (all)', 'ENSO active vs neutral (all)', 'enso'), ('ENSO active - neutral (NDJFM)', 'ENSO active vs neutral (NDJFM)', 'enso'),
            ('SSW in previous 30 d - none (NDJFM)', 'SSW in prev. 30 d (NDJFM)', None),
            ('large CONUS soil-moisture anomaly - small (JJA)', 'large CONUS soil moisture (JJA)', None)]
C07_COLS = [('z500', 'nh_extratropics', 'z500, NH extratropics'), ('t2m', 'conus', 't2m, CONUS'), ('tp24', 'tropics', 'tp24, tropics')]


def era5_expect():
    """ERA5 composite-forecast expectation per contrast: mean pattern ACC in A minus B (block bootstrap CI)."""
    p = OUTC / 'c07_era5' / 'era5_expect_perstart.csv'
    if not p.exists():
        return {}
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from s41eval import bing_helperc as HC
    rows = rd(p)
    st = np.load(OUTC / 'windows' / 'c07_init_states.npz', allow_pickle=True)
    starts = [str(x) for x in st['starts']]
    nd = np.array([int(x[5:7]) in (11, 12, 1, 2, 3) for x in starts])
    act = st['amp'] >= 1
    ph = HC.rmm_phase(st['rmm'][:, 0], st['rmm'][:, 1])
    pg = {k: act & np.isin(ph, q) for k, q in {'P8-1': [8, 1], 'P2-3': [2, 3], 'P4-5': [4, 5], 'P6-7': [6, 7]}.items()}
    en = st['enso']
    con = {'MJO active - inactive (NDJFM)': (act & nd, ~act & nd), 'ENSO active - neutral (all)': (np.isin(en, ['el_nino', 'la_nina']), en == 'neutral'),
           'ENSO active - neutral (NDJFM)': (np.isin(en, ['el_nino', 'la_nina']) & nd, (en == 'neutral') & nd)}
    for k in pg:
        con[f'MJO {k} - inactive (NDJFM)'] = (pg[k] & nd, ~act & nd)
    out = {}
    bidx = HC.mbb_indices(len(starts), 4, 1000, seed=12)
    hA = np.array([int(x[:4]) <= 2024 for x in starts])
    for f, rn, _ in C07_COLS:
        for pred in ('mjo', 'enso'):
            rr = {r['start']: fl(r['acc']) for r in rows if r['field'] == f and r['region'] == rn and r['predictor'] == pred}
            a = np.array([rr.get(x, np.nan) for x in starts])
            for cn, (mA, mB) in con.items():
                if (pred == 'mjo') != cn.startswith('MJO'):
                    continue
                full = np.nanmean(a[mA]) - np.nanmean(a[mB])
                dr = [np.nanmean(a[b][mA[b]]) - np.nanmean(a[b][mB[b]]) for b in bidx]
                ci = HC.ci_of(dr)
                out[(cn, f, rn)] = dict(diff=full, ci_lo=ci[0], ci_hi=ci[1], acc_a=np.nanmean(a[mA]), acc_b=np.nanmean(a[mB]),
                                        grade=HC.grade(ci, np.nanmean(a[mA & hA]) - np.nanmean(a[mB & hA]), np.nanmean(a[mA & ~hA]) - np.nanmean(a[mB & ~hA])))
    return out


def c07(metric='pattern_acc', fid='C07', slide=True):
    rows = rd(OUTC / 'windows' / 'c07_windows.csv')
    A = {(r['system'], r['field'], r['region'], r['contrast'], r['metric']): r for r in rows}
    E = era5_expect()
    f, axs = S.fig(1, 3, sharey=True)
    f.subplots_adjust(left=0.24, right=0.99, top=0.88, bottom=0.25, wspace=0.08)
    y = np.arange(len(C07_ROWS))[::-1]
    for ax, (fld, rn, title) in zip(axs, C07_COLS):
        for j, sname in enumerate(SYSL):
            for i, (cn, lab, pred) in enumerate(C07_ROWS):
                r = A[(sname, fld, rn, cn, metric)]
                yy = y[i] + (1 - j) * 0.2
                col = S.MODEL[sname]
                ax.plot([fl(r['ci_lo']), fl(r['ci_hi'])], [yy, yy], color=col, lw=1.8)
                ax.plot(fl(r['diff']), yy, 'o', color=col, ms=6, mfc=col if r['grade'] != 'noise' else 'white', label=sname if i == 0 else None)
        for i, (cn, lab, pred) in enumerate(C07_ROWS):
            e = E.get((cn, fld, rn))
            if e and metric == 'pattern_acc':
                yy = y[i] - 0.38
                ax.plot([e['ci_lo'], e['ci_hi']], [yy, yy], color=S.GREY, lw=1.8)
                ax.plot(e['diff'], yy, 'D', color=S.GREY, ms=6, mfc=S.GREY if e['grade'] != 'noise' else 'white',
                        label='ERA5 composite expectation' if i == 0 else None)
        ax.axvline(0, color=S.GREY, lw=1)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Δ weeks 3–4 pattern ACC (A − B)' if metric == 'pattern_acc' else f'Δ weeks 3–4 {metric}')
        for yy in y:
            ax.axhline(yy - 0.5, color='#eeeeee', lw=0.8, zorder=0)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([lab for _, lab, _ in C07_ROWS], fontsize=13)
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.6, 1.0))
    n = rd(OUTC / 'windows' / 'c07_windows.csv')
    S.save(f, fid, f'windows_of_opportunity_{metric}', slide=slide,
           footer_text='Exploratory. Difference of weeks 3–4 skill between starts in state A and B at init (ERA5): per-start centred pattern ACC of the 30-member drift-corrected '
                       'mean anomaly (package v1.5), 403 starts 2019–26; bars = 95 % block-bootstrap CI (filled = CI excludes 0). Grey diamonds: the same difference for an '
                       'ERA5 composite forecast (weeks 3–4 anomaly composited on the init MJO phase group / ENSO phase, ERA5 1991–2018) = what ERA5 statistics alone lead one to expect. '
                       'n: MJO active/inactive NDJFM 111/53; P2–3 22, P4–5 33, P6–7 34, P8–1 22; ENSO active/neutral 242/161; SSW 29/135; large soil moisture 35/71. '
                       + S.PROVISIONAL_MJO + '. ' + IFS_NOTE)
    return A, E


def c07_caption():
    A, E = c07('pattern_acc', 'C07')
    c07('crpss_fixed', 'C07x', slide=False)
    g = lambda s, f, r, c: A[(s, f, r, c, 'pattern_acc')]  # noqa: E731
    e1 = E.get(('ENSO active - neutral (all)', 'tp24', 'tropics'), {})
    e2 = E.get(('MJO active - inactive (NDJFM)', 'z500', 'nh_extratropics'), {})
    S.caption('C07', "Windows of opportunity (exploratory): ENSO is the only state with a clear weeks 3–4 window, for tropical rainfall "
                     f"(Δ pattern ACC {fl(g('UC-S2S SST-anomaly','tp24','tropics','ENSO active - neutral (all)')['diff']):+.2f} SST-anom., "
                     f"{fl(g('UC-S2S plain','tp24','tropics','ENSO active - neutral (all)')['diff']):+.2f} plain, {fl(g('FuXi-S2S','tp24','tropics','ENSO active - neutral (all)')['diff']):+.2f} FuXi; robust"
                     + (f"; an ERA5 composite forecast gains {e1['diff']:+.2f}, from ACC {e1['acc_b']:.2f} to {e1['acc_a']:.2f}, while UC-S2S goes from 0.35 to 0.41 and FuXi from 0.41 to 0.45" if e1 else '') + "); an active MJO does not raise NH z500 weeks 3–4 skill in any system "
                     f"({fl(g('UC-S2S SST-anomaly','z500','nh_extratropics','MJO active - inactive (NDJFM)')['diff']):+.2f}, noise"
                     + (f"; consistent with ERA5: the 1991–2018 MJO composites give no weeks 3–4 NH z500 skill in 2019–26, Δ {e2['diff']:+.2f}, {e2['grade']}" if e2 else '') + "); SSW and soil-moisture windows not detectable (noise) · "
                     "`v1.5/helperC/windows/c07_windows.csv`, `helperC/c07_era5/era5_expect_perstart.csv`")
    kn = []
    for (sname, fld, rn, cn, met), r in A.items():
        if met in ('pattern_acc', 'crpss_fixed') and (fld, rn) in (('z500', 'nh_extratropics'), ('t2m', 'conus'), ('tp24', 'tropics'), ('t2m', 'global')):
            kn.append(f"{sname} {fld} {rn} {cn} {met}: A {fl(r['value_a']):.3f} (n={r['n_a']}) vs B {fl(r['value_b']):.3f} (n={r['n_b']}), Δ {fl(r['diff']):+.3f} "
                      f"[{fl(r['ci_lo']):+.3f}, {fl(r['ci_hi']):+.3f}] {r['grade']} · c07_windows.csv")
    for (cn, fld, rn), e in E.items():
        kn.append(f"ERA5 composite expectation {fld} {rn} {cn}: ACC A {e['acc_a']:.3f} vs B {e['acc_b']:.3f}, Δ {e['diff']:+.3f} [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] {e['grade']} · era5_expect_perstart.csv")
    S.key_numbers('C07', kn)


# ---------------------------------------------------------------------------------------------------------------- C03
SYSK = {'UC-S2S SST-anomaly': 'ucs2s_sstanom', 'UC-S2S plain': 'ucs2s_plain', 'FuXi-S2S': 'fuxi'}
BANDL = {'mjo': 'MJO', 'kelvin': 'Kelvin', 'er': 'ER', 'mrg': 'MRG'}
LWIN = {'days 1–15': (0, 15), 'days 16–30': (15, 30), 'days 31–45': (30, 45)}


def waves_tables():
    """band ratios and filtered-wave skill by lead window with block-bootstrap CIs -> rows; also writes helperC/waves/c03_*.csv."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from s41eval import bing_helperc as HC
    from s41eval import bing_waves as Wv
    rows, srows = [], []
    for lab, k in SYSK.items():
        Z = np.load(OUTC / 'waves' / f'{k}.npz')
        bands = [str(b) for b in Z['bands']]
        starts = [str(x) for x in Z['starts']]
        hA = np.array([int(x[:4]) <= 2024 for x in starts])
        bidx = HC.mbb_indices(len(starts), 4, 1000, seed=13)
        for v in ('olr', 'u850'):
            F = Z[f'filt_{v}']                                 # [S, band, 5, 45]
            ok = Z[f'ok_{v}']
            for j, b in enumerate(bands):
                for wn, (a, c) in LWIN.items():
                    X = F[:, j, :, a:c].sum(-1)                # [S, 5]
                    X = np.where(ok[:, None], X, np.nan)

                    def st(ix, X=X):
                        s_ = np.nansum(X[ix], 0)
                        return [s_[0] / s_[1], s_[3] / np.sqrt(s_[2] * s_[4])]
                    full = st(np.arange(len(starts)))
                    dr = np.array([st(bb) for bb in bidx])
                    ha, hb = st(np.flatnonzero(hA)), st(np.flatnonzero(~hA))
                    for q, mn in enumerate(('var_ratio', 'skill_cor')):
                        lo, hi = HC.ci_of(dr[:, q])
                        g = HC.grade((lo - 1, hi - 1), ha[q] - 1, hb[q] - 1) if mn == 'var_ratio' else ''
                        rows.append(dict(system=lab, field=v, band=b, window=wn, metric=mn, value=float(full[q]), ci_lo=lo, ci_hi=hi,
                                         h2019_24=float(ha[q]), h2025_26=float(hb[q]),
                                         grade_vs_1=g, n=int(ok.sum())))
                # full-window (days 1-45) spectral band power ratio
                pf, po = Z[f'bandpow_{v}_f_{b}'], Z[f'bandpow_{v}_o_{b}']
                okb = np.isfinite(pf) & np.isfinite(po)
                st2 = lambda ix: np.nansum(pf[ix][okb[ix]]) / np.nansum(po[ix][okb[ix]])  # noqa: E731
                full = st2(np.arange(len(starts)))
                dr = np.array([st2(bb) for bb in bidx])
                lo, hi = HC.ci_of(dr)
                ha, hb = st2(np.flatnonzero(hA)), st2(np.flatnonzero(~hA))
                srows.append(dict(system=lab, field=v, band=b, spectral_power_ratio_d1_45=float(full), ci_lo=lo, ci_hi=hi, h2019_24=float(ha),
                                  h2025_26=float(hb), grade_vs_1=HC.grade((lo - 1, hi - 1), ha - 1, hb - 1), n=int(okb.sum())))
    import csv as _c
    for nm, rr in (('c03_filtered_by_lead.csv', rows), ('c03_spectral_band_ratio.csv', srows)):
        with open(OUTC / 'waves' / nm, 'w', newline='') as fh:
            w = _c.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    return rows, srows


def c03(var='olr', fid='C03', slide=True):
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from s41eval import bing_waves as Wv
    f, axs = S.fig(2, 3, sharex=True, sharey=True)
    f.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.22, wspace=0.08, hspace=0.25)
    kk = np.arange(-15, 16)
    lev = np.linspace(-1, 1, 11)
    for c, (lab, k) in enumerate(SYSK.items()):
        Z = np.load(OUTC / 'waves' / f'{k}.npz')
        freq, ks = Z['freq'], Z['ks']
        sel = (ks >= -15) & (ks <= 15)
        for r, comp in enumerate(('sym', 'anti')):
            ax = axs[r, c]
            Pf, Po = Z[f'spec_{var}_f_{comp}'], Z[f'spec_{var}_o_{comp}']
            # light 1-2-1 smoothing in wavenumber and frequency of both before the ratio (stability at high k)
            def sm(P):
                Q = P.copy()
                Q[:, 1:-1] = 0.25 * P[:, :-2] + 0.5 * P[:, 1:-1] + 0.25 * P[:, 2:]
                R = Q.copy()
                R[1:-1] = 0.25 * Q[:-2] + 0.5 * Q[1:-1] + 0.25 * Q[2:]
                return R
            ratio = np.log2(sm(Pf) / sm(Po))
            cf = ax.contourf(ks[sel], freq[1:], ratio[1:][:, sel], levels=lev, cmap=S.DIV, extend='both')
            kc = np.linspace(-15, 15, 301)
            for h in (12, 25, 50):
                if comp == 'sym':
                    ax.plot(kc[kc > 0], Wv.disp_freq(kc[kc > 0], h, 'kelvin'), color='k', lw=0.8)
                    ax.plot(kc[kc < 0], Wv.disp_freq(kc[kc < 0], h, 'er'), color='k', lw=0.8)
                else:
                    ax.plot(kc, Wv.disp_freq(kc, h, 'mrg'), color='k', lw=0.8)
            for b in (('kelvin', 'mjo', 'er') if comp == 'sym' else ('mrg',)):
                m = Wv.band_mask(freq, ks, b)
                ax.contour(ks[sel], freq[1:], m[1:][:, sel].astype(float), levels=[0.5], colors=S.GREY, linewidths=1.2, linestyles='--')
            ax.axvline(0, color=S.GREY, lw=0.6)
            ax.set_ylim(freq[1], 0.5)
            ax.set_xlim(-15, 15)
            if r == 0:
                ax.set_title(lab, fontsize=15, color=S.MODEL[lab] if lab != 'FuXi-S2S' else 'k')
            if c == 0:
                ax.set_ylabel(f'{"symmetric" if comp == "sym" else "antisymmetric"}\nfrequency (cpd)')
            if r == 1:
                ax.set_xlabel('zonal wavenumber (> 0 east)')
    cax = f.add_axes([0.915, 0.25, 0.012, 0.6])
    cb = f.colorbar(cf, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(['½', '0.71', '1', '1.41', '2'])
    cb.set_label(f'{var.upper()} power, forecast / ERA5')
    _, srows = waves_tables()
    sr = {(r['system'], r['field'], r['band']): r for r in srows}
    txt = '; '.join(f"{BANDL[b]} " + '/'.join(f"{fl(sr[(lab, var, b)]['spectral_power_ratio_d1_45']):.2f}" for lab in SYSK) for b in ('mjo', 'kelvin', 'er', 'mrg'))
    S.save(f, fid, f'wk_spectra_ratio_{var}', slide=slide,
           footer_text=f'Exploratory. Wheeler–Kiladis space–time power of 15°S–15°N {var.upper()} (sym./antisym. about the equator), days 1–45 of every start, 10 members, '
                       f'drift-corrected, mean and trend removed, 10 % taper, vs ERA5 over the same 45-day windows (UC-S2S 402 starts, FuXi 404); 1-2-1 smoothed before the ratio. '
                       f'Black: dispersion curves h = 12, 25, 50 m (Kelvin, ER n=1, MRG); dashed boxes: Kelvin, ER, MRG, MJO bands. Band power ratio (SST-anom./plain/FuXi): {txt}. '
                       + IFS_NOTE)
    return sr


def c03b(fid='C03b', slide=True):
    rows, srows = waves_tables()
    A = {(r['system'], r['field'], r['band'], r['window'], r['metric']): r for r in rows}
    f, axs = S.fig(2, 4, sharex=True)
    f.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.22, wspace=0.28, hspace=0.32)
    x = np.arange(3)
    for r, v in enumerate(('olr', 'u850')):
        for c, b in enumerate(('mjo', 'kelvin', 'er', 'mrg')):
            ax = axs[r, c]
            for j, lab in enumerate(SYSK):
                vals = [fl(A[(lab, v, b, w, 'var_ratio')]['value']) for w in LWIN]
                lo = [fl(A[(lab, v, b, w, 'var_ratio')]['ci_lo']) for w in LWIN]
                hi = [fl(A[(lab, v, b, w, 'var_ratio')]['ci_hi']) for w in LWIN]
                ax.errorbar(x + (j - 1) * 0.15, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], color=S.MODEL[lab], marker='o', ms=6, lw=2, capsize=0,
                            label=lab)
            ax.axhline(1, color=S.GREY, lw=1)
            ax.set_title(f'{v.upper()} · {BANDL[b]}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(['1–15', '16–30', '31–45'])
            if r == 1:
                ax.set_xlabel('lead (days)')
            if c == 0:
                ax.set_ylabel('band variance / ERA5')
    h, l = axs[0, 0].get_legend_handles_labels()
    f.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.53, 0.995))
    S.save(f, fid, 'wave_band_variance_by_lead', slide=slide,
           footer_text='Exploratory. Variance of WK-filtered 15°S–15°N fields (Kelvin, ER, MRG, MJO bands) by lead window, forecast members (10 per start) / ERA5; filtering on '
                       'ERA5 (init − 135 d … init) + forecast days 1–45 + zero padding, and the same for ERA5 truncated at day 45 (identical edge treatment). '
                       '95 % block-bootstrap CI over starts. ' + IFS_NOTE)
    return A


def c03_caption():
    sr = c03('olr', 'C03')
    c03('u850', 'C03x', slide=False)
    A = c03b('C03b')
    c03c()
    g = lambda s, v, b: sr[(s, v, b)]  # noqa: E731
    sb = {(r['system'], r['field'], r['subband']): fl(r['power_ratio']) for r in c03_subbands()}
    hs, kc = 'periods < 3 d, |k| <= 15 (sym)', 'Kelvin core: k 2-8, periods 4-10 d, h 8-90 m (sym)'
    S.caption('C03', "Convectively coupled waves (exploratory): UC-S2S OLR has too much power at periods < 3 d "
                     f"(×{sb[('UC-S2S SST-anomaly','olr',hs)]:.1f}/{sb[('UC-S2S plain','olr',hs)]:.1f}; FuXi ×{sb[('FuXi-S2S','olr',hs)]:.2f}, too smooth) and too little in the "
                     f"Kelvin core (k 2–8, 4–10 d: {sb[('UC-S2S SST-anomaly','olr',kc)]:.2f}/{sb[('UC-S2S plain','olr',kc)]:.2f}; FuXi {sb[('FuXi-S2S','olr',kc)]:.2f}); band power ratios "
                     + '; '.join(f"{BANDL[b]} {fl(g('UC-S2S SST-anomaly','olr',b)['spectral_power_ratio_d1_45']):.2f}/{fl(g('UC-S2S plain','olr',b)['spectral_power_ratio_d1_45']):.2f}/"
                                 f"{fl(g('FuXi-S2S','olr',b)['spectral_power_ratio_d1_45']):.2f}" for b in ('mjo', 'kelvin', 'er', 'mrg'))
                     + " (SST-anom./plain/FuXi; ER and MRG deficits robust in all); u850 wave power 0.5–0.9 of ERA5 in all UC-S2S bands, with no high-frequency excess, so the OLR excess "
                     "may partly be the OLR definition (UC-S2S lead-1 OLR matched the 1-h and the 24-h ERA5 OLR equally well, report §3) · "
                     "`v1.5/helperC/waves/c03_spectral_band_ratio.csv`, `c03_spectral_subbands.csv`")
    S.caption('C03b', "Wave-band variance by lead (exploratory): forecast/ERA5 filtered variance for days 1–15 / 16–30 / 31–45 — see `v1.5/helperC/waves/c03_filtered_by_lead.csv` "
                      + '; '.join(f"OLR {BANDL[b]} SST-anom. " + '/'.join(f"{fl(A[('UC-S2S SST-anomaly','olr',b,w,'var_ratio')]['value']):.2f}" for w in LWIN) +
                                  ', FuXi ' + '/'.join(f"{fl(A[('FuXi-S2S','olr',b,w,'var_ratio')]['value']):.2f}" for w in LWIN) for b in ('mjo', 'kelvin', 'er', 'mrg')))
    kn = [f"spectral sub-band power ratio {k_[0]} {k_[1]} {k_[2]}: {v_:.3f} (all starts, no CI) · c03_spectral_subbands.csv" for k_, v_ in sb.items()]
    kn += [f"spectral band power ratio d1–45 {r['system']} {r['field']} {r['band']}: {fl(r['spectral_power_ratio_d1_45']):.3f} [{fl(r['ci_lo']):.3f}, {fl(r['ci_hi']):.3f}] "
          f"{r['grade_vs_1']} vs 1 · c03_spectral_band_ratio.csv" for r in sr.values()]
    kn += [f"filtered {r['metric']} {r['system']} {r['field']} {r['band']} {r['window']}: {fl(r['value']):.3f} [{fl(r['ci_lo']):.3f}, {fl(r['ci_hi']):.3f}]"
           f"{' ' + r['grade_vs_1'] + ' vs 1' if r['grade_vs_1'] else ''} · c03_filtered_by_lead.csv" for r in A.values()]
    S.key_numbers('C03', kn)


def c03c():
    rows, _ = waves_tables()
    A = {(r['system'], r['field'], r['band'], r['window'], r['metric']): r for r in rows}
    f, axs = S.fig(1, 4, sharey=True)
    f.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.25, wspace=0.1)
    x = np.arange(3)
    for c, b in enumerate(('mjo', 'kelvin', 'er', 'mrg')):
        ax = axs[c]
        for j, lab in enumerate(SYSK):
            for v, ls in (('olr', '-'), ('u850', '--')):
                vals = [fl(A[(lab, v, b, w, 'skill_cor')]['value']) for w in LWIN]
                ax.plot(x, vals, color=S.MODEL[lab], ls=ls, marker='o', lw=2, label=f'{lab} {v}' if c == 0 else None)
        ax.set_title(BANDL[b], fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(['1–15', '16–30', '31–45'])
        ax.set_xlabel('lead (days)')
        ax.axhline(0, color=S.GREY, lw=0.8)
    axs[0].set_ylabel('COR of filtered 30-member mean vs ERA5')
    axs[0].legend(loc='lower left', frameon=False, fontsize=10)
    S.save(f, 'C03c', 'filtered_wave_skill', slide=False,
           footer_text='Exploratory. Uncentred correlation over 15°S–15°N × lead window between the WK-filtered 30-member mean (ERA5 appended before the init, Latos et al. 2026 style) '
                       'and WK-filtered ERA5 (ERA5 through day 135 where available). Solid OLR, dashed u850. ' + IFS_NOTE)


def c06c():
    rr = rd(OUTC / 'enso' / 'c06_soil_tcc.csv')
    f, axs = S.fig(1, 2, sharey=True)
    f.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.25, wspace=0.08)
    wk = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    x = np.arange(1, 7)
    for ax, sub in zip(axs, ('pooled|all', 'pooled|JJA')):
        for sname, corr, col, ls in (('UC-S2S SST-anomaly', 'raw (Bing)', S.MODEL['UC-S2S SST-anomaly'], '-'), ('UC-S2S plain', 'raw (Bing)', S.MODEL['UC-S2S plain'], '-'),
                                     ('UC-S2S SST-anomaly', 'dc', S.MODEL['UC-S2S SST-anomaly'], ':'), ('ERA5 persistence (init-day anomaly)', '-', S.GREY, '--')):
            a = {r['window']: fl(r['land_mean_tcc']) for r in rr if r['system'] == sname and r['correction'] == corr and r['subset'] == sub}
            ax.plot(x, [a[w] for w in wk], color=col, ls=ls, lw=2.2, marker='o', label=f'{sname}, {corr}' if corr != '-' else 'ERA5 persistence of the init-day anomaly')
        ax.set_title('all starts' if sub.endswith('all') else 'JJA starts', fontsize=15)
        ax.set_xticks(x)
        ax.set_xlabel('lead week')
    axs[0].set_ylabel('soil moisture (swvl1) TCC, land mean')
    axs[0].legend(loc='upper right', frameon=False, fontsize=11)
    S.save(f, 'C06c', 'soil_moisture_tcc', slide=False,
           footer_text="Exploratory. Temporal correlation per land point of the weekly 30-member-mean swvl1 anomaly with ERA5 (package v1.5 maps, land-masked; 'raw (Bing)' = fixed "
                       "1991–2020 climatology, no drift removal as in Bing's tcc_soil method; dotted = drift-corrected), cos-lat land mean 75°S–75°N; FuXi has no soil moisture. " + IFS_NOTE)


def c03_subbands():
    """aggregate (all starts) spectral power ratios in sub-bands read off the WK plane (no CI; from the mean spectra)."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from s41eval import bing_waves as Wv
    out = []
    for lab, k in SYSK.items():
        Z = np.load(OUTC / 'waves' / f'{k}.npz')
        freq, ks = Z['freq'], Z['ks']
        F, K = np.meshgrid(freq, ks, indexing='ij')
        per = np.where(F > 0, 1 / np.where(F > 0, F, 1), np.inf)
        kc = Wv.disp_freq(np.where(K > 0, K, np.nan), 8.0, 'kelvin'), Wv.disp_freq(np.where(K > 0, K, np.nan), 90.0, 'kelvin')
        subs = {'periods < 3 d, |k| <= 15 (sym)': ('sym', (per < 3) & (np.abs(K) <= 15)),
                'periods < 3 d, |k| <= 15 (anti)': ('anti', (per < 3) & (np.abs(K) <= 15)),
                'Kelvin core: k 2-8, periods 4-10 d, h 8-90 m (sym)': ('sym', (K >= 2) & (K <= 8) & (per >= 4) & (per <= 10) & (F >= kc[0]) & (F <= kc[1])),
                'periods 3-30 d, |k| <= 15 (sym)': ('sym', (per >= 3) & (per <= 30) & (np.abs(K) <= 15))}
        for v in ('olr', 'u850'):
            for nm, (comp, m) in subs.items():
                out.append(dict(system=lab, field=v, subband=nm, power_ratio=float(Z[f'spec_{v}_f_{comp}'][m].sum() / Z[f'spec_{v}_o_{comp}'][m].sum())))
    import csv as _c
    with open(OUTC / 'waves' / 'c03_spectral_subbands.csv', 'w', newline='') as fh:
        w = _c.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


if __name__ == '__main__':
    for a in sys.argv[1:]:
        globals()[a]()
