"""
MJO RMM phase space plot using field-bias-corrected RMM files.
4-panel figure: one panel per model (UC-S2S V2, UC-S2S SSTanom, ECMWF, raw V2 for comparison).
Each panel shows all ensemble members (faint) + ensemble mean (bold) + observation (black).
Usage: python plot_phase_space_fbcorr.py [YYYYMMDD]
Default init date: 20210711
"""

import numpy as np
import xarray as xr
import glob, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

INIT_DATE = sys.argv[1] if len(sys.argv) > 1 else '20210711'

# (rmm_dir, has_day0, label, n_members_max)
# (title, rmm_dir, has_day0, color, sign_flip)
# sign_flip=True means RMM1/RMM2 stored with opposite EOF orientation — negate when loading
PANELS = [
    ('UC-S2S V2 (raw)',
     '/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output',       True,  'tab:red',   False),
    ('UC-S2S V2 (field-BC)',
     '/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output_fbcorr', True,  'tab:blue',  False),
    ('UC-S2S SSTanom (raw)',
     '/glade/campaign/univ/uchi0018/bing/mjo/uc_s2s_sstanom/chi200_1p0deg/rmm_output',       True, 'tab:green', True),
    ('UC-S2S SSTanom (field-BC)',
     '/glade/campaign/univ/uchi0018/bing/mjo/uc_s2s_sstanom/chi200_1p0deg/rmm_output_fbcorr', True, 'tab:olive', True),
]

def draw_phase_diagram(ax):
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    for ang in [45, 135]:
        r = 4.5
        ax.plot([-r*np.cos(np.radians(ang)),  r*np.cos(np.radians(ang))],
                [-r*np.sin(np.radians(ang)),  r*np.sin(np.radians(ang))],
                color='k', lw=0.5)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='k', lw=0.8))
    labels = [
        (1, 'W. Hem\n& Africa',    (-3.0, -1.8)),
        (2, 'Indian\nOcean',       (-1.8, -3.0)),
        (3, 'Indian\nOcean',       ( 1.8, -3.0)),
        (4, 'Maritime\nContinent', ( 3.0, -1.8)),
        (5, 'Maritime\nContinent', ( 3.0,  1.8)),
        (6, 'W. Pacific',          ( 1.8,  3.0)),
        (7, 'W. Pacific',          (-1.8,  3.0)),
        (8, 'W. Hem\n& Africa',    (-3.0,  1.8)),
    ]
    for ph, reg, (rx, ry) in labels:
        ax.text(rx*0.85, ry*0.85, str(ph), fontsize=13, color='0.65',
                ha='center', va='center', fontweight='bold')
        ax.text(rx*1.38, ry*1.38, reg, fontsize=6.5, color='0.55', ha='center', va='center')
    ax.text( 0,  4.3, 'Pacific Ocean',       ha='center', va='bottom',  fontsize=9, color='0.5')
    ax.text( 0, -4.3, 'Indian Ocean',        ha='center', va='top',     fontsize=9, color='0.5')
    ax.text(-4.3, 0,  'W. Hem.\n& Africa',   ha='center', va='center',  fontsize=9, color='0.5', rotation=90)
    ax.text( 4.3, 0,  'Maritime\nContinent', ha='center', va='center',  fontsize=9, color='0.5', rotation=-90)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel('RMM1', fontsize=11); ax.set_ylabel('RMM2', fontsize=11)
    ax.set_aspect('equal'); ax.tick_params(labelsize=9)

# Canonical obs source: always load from V2 raw file so all panels share the same ERA5 obs
OBS_REF_DIR = '/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output'

def load_obs(init_date):
    """Load shared ERA5 obs RMM from V2 raw file (has_day0=True)."""
    fp = os.path.join(OBS_REF_DIR, f'MJO_chi200_Forecast_Init_{init_date}.nc')
    if not os.path.exists(fp):
        return None, None
    ds = xr.open_dataset(fp)
    r1o = ds['RMM1_obs'].values
    r2o = ds['RMM2_obs'].values
    ds.close()
    return r1o, r2o

def load_rmm(rmm_dir, init_date, has_day0, r1o_shared, r2o_shared, sign_flip=False):
    fp = os.path.join(rmm_dir, f'MJO_chi200_Forecast_Init_{init_date}.nc')
    if not os.path.exists(fp):
        return None
    ds = xr.open_dataset(fp)
    r1e  = ds['RMM1_emean'].values   # (time,)
    r2e  = ds['RMM2_emean'].values
    r1m  = ds['RMM1'].values         # (time, nmem)
    r2m  = ds['RMM2'].values
    nmem = r1m.shape[1]
    ds.close()
    if sign_flip:
        r1e = -r1e; r2e = -r2e
        r1m = -r1m; r2m = -r2m
    if not has_day0:
        r1e = np.concatenate([[np.nan], r1e])
        r2e = np.concatenate([[np.nan], r2e])
        r1m = np.concatenate([np.full((1, nmem), np.nan), r1m], axis=0)
        r2m = np.concatenate([np.full((1, nmem), np.nan), r2m], axis=0)
    return dict(r1e=r1e, r2e=r2e, r1o=r1o_shared, r2o=r2o_shared, r1m=r1m, r2m=r2m, nmem=nmem)

init_str = f'{INIT_DATE[:4]}-{INIT_DATE[4:6]}-{INIT_DATE[6:8]}'

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
fig.suptitle(f'MJO RMM Phase Space — Field Bias Corrected\nInit: {init_str}', fontsize=15)

# Load shared obs once
r1o_shared, r2o_shared = load_obs(INIT_DATE)

for ax, (title, rmm_dir, has_day0, color, sign_flip) in zip(axes, PANELS):
    draw_phase_diagram(ax)
    ax.set_title(title, fontsize=11, pad=6)

    data = load_rmm(rmm_dir, INIT_DATE, has_day0, r1o_shared, r2o_shared, sign_flip=sign_flip)
    if data is None:
        ax.text(0, 0, f'No file for {INIT_DATE}', ha='center', va='center', fontsize=10)
        continue

    nmem = data['nmem']

    # ensemble members (faint)
    for m in range(nmem):
        ax.plot(data['r1m'][:, m], data['r2m'][:, m],
                color=color, lw=0.5, alpha=0.25, zorder=2)

    # ensemble mean (bold)
    ax.plot(data['r1e'], data['r2e'], color=color, lw=2.5, zorder=4,
            label=f'{title.split("(")[1][:-1]} (n={nmem})')
    ax.plot(data['r1e'][0] if not np.isnan(data['r1e'][0]) else data['r1e'][1],
            data['r2e'][0] if not np.isnan(data['r2e'][0]) else data['r2e'][1],
            '*', ms=12, color=color, markeredgecolor='k', zorder=6)

    # observation (black)
    ax.plot(data['r1o'], data['r2o'], color='k', lw=2.0, zorder=5)
    ax.plot(data['r1o'][0] if not np.isnan(data['r1o'][0]) else data['r1o'][1],
            data['r2o'][0] if not np.isnan(data['r2o'][0]) else data['r2o'][1],
            'k*', ms=14, zorder=7)

    ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
    # dummy for obs
    ax.plot([], [], 'k-', lw=2, label='Observation')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.85)

plt.tight_layout()
out = f'/glade/work/bgong/MJO_results/plots/phase_space_fbcorr_{INIT_DATE}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
