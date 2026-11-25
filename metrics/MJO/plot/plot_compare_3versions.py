"""
Compare 3 versions of MJO skill for UC-S2S V2 and UC-S2S SSTanom:
  - Raw (no correction)
  - RMM bias corrected (leave-one-year-out on RMM indices)
  - Field bias corrected (leave-one-year-out on normalized fields before projection)

2 models × 4 amplitude groups = 2-row × 4-col figure, one panel per group.
Each panel shows 3 lines (raw/RMM-BC/field-BC) per model (6 lines total).
"""
import numpy as np
import xarray as xr
import glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
sys.path.insert(0, '/glade/work/bgong')
from model_colors import model_color, model_style

MAXLEAD = 46
N_BOOT  = 2000
SEED    = 42
YEARS   = {'2019', '2020', '2021'}

# (rmm_dir, has_day0)
VERSIONS = {
    'UC-S2S V2': {
        'raw':        ('/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output',        True),
        'field-BC':   ('/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output_fbcorr', True),
    },
    'UC-S2S SSTanom': {
        'raw':        ('/glade/campaign/univ/uchi0018/bing/mjo/uc_s2s_sstanom/chi200_1p0deg/rmm_output',        True),
        'field-BC':   ('/glade/campaign/univ/uchi0018/bing/mjo/uc_s2s_sstanom/chi200_1p0deg/rmm_output_fbcorr', True),
    },
}
MODELS  = ['UC-S2S V2', 'UC-S2S SSTanom']

# shared dates: ECMWF ∩ SSTanom, same-month for V2
def _files(d):
    return sorted(glob.glob(os.path.join(d, 'MJO_chi200_Forecast_Init_????????.nc')))
def _date(fp):
    return re.search(r'\d{8}', os.path.basename(fp)).group()

ec_files = sorted(glob.glob('/glade/campaign/univ/uchi0018/bing/mjo/ecmwf_s2s/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc'))
ec_d  = {_date(f) for f in ec_files if _date(f)[:4] in YEARS}
ss_d  = {_date(f) for f in _files(VERSIONS['UC-S2S SSTanom']['raw'][0]) if _date(f)[:4] in YEARS}
v2_d  = {_date(f) for f in _files(VERSIONS['UC-S2S V2']['raw'][0])      if _date(f)[:4] in YEARS}

shared         = sorted(ec_d & ss_d)
shared_months  = {d[:6] for d in shared}
dates_map = {
    'UC-S2S SSTanom': set(shared),
    'UC-S2S V2':      {d for d in v2_d if d[:6] in shared_months},
}

def load(rmm_dir, has_day0, model):
    date_set = dates_map[model]
    r1e, r2e, r1o, r2o, amps, inits = [], [], [], [], [], []
    for fp in _files(rmm_dir):
        dt = _date(fp)
        if dt not in date_set: continue
        ds = xr.open_dataset(fp)
        a = ds['RMM1_emean'].values; b = ds['RMM2_emean'].values
        c = ds['RMM1_obs'].values;   e = ds['RMM2_obs'].values
        ds.close()
        if not has_day0:
            a = np.concatenate([[np.nan], a]); b = np.concatenate([[np.nan], b])
            c = np.concatenate([[np.nan], c]); e = np.concatenate([[np.nan], e])
        a, b, c, e = a[:MAXLEAD], b[:MAXLEAD], c[:MAXLEAD], e[:MAXLEAD]
        pad = lambda x: x if len(x)==MAXLEAD else np.concatenate([x, np.full(MAXLEAD-len(x), np.nan)])
        r1e.append(pad(a)); r2e.append(pad(b)); r1o.append(pad(c)); r2o.append(pad(e))
        amp_val = np.sqrt(c[1]**2 + e[1]**2) if len(c)>1 and not np.isnan(c[1]) else np.nan
        amps.append(amp_val)
        inits.append(dt)
    return np.stack(r1e), np.stack(r2e), np.stack(r1o), np.stack(r2o), np.array(amps), inits

def bias_correct_rmm(r1e, r2e, r1o, r2o, inits):
    yrs = np.array([d[:4] for d in inits])
    r1e_c, r2e_c = r1e.copy(), r2e.copy()
    for y in np.unique(yrs):
        m = yrs == y; nm = ~m
        if nm.sum() == 0: continue
        r1e_c[m] -= np.nanmean(r1e[nm] - r1o[nm], axis=0)
        r2e_c[m] -= np.nanmean(r2e[nm] - r2o[nm], axis=0)
    return r1e_c, r2e_c

def bivariate_acc(r1f, r2f, r1o, r2o):
    num   = np.nansum(r1f*r1o + r2f*r2o, axis=0)
    denom = np.sqrt(np.nansum(r1f**2+r2f**2, axis=0)*np.nansum(r1o**2+r2o**2, axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom>0, num/denom, np.nan)

def skilful_day(acc):
    for i, v in enumerate(acc):
        if not np.isnan(v) and v < 0.5: return i
    return None

def bootstrap_sd(r1f, r2f, r1o, r2o):
    n = len(r1o)
    rng = np.random.default_rng(SEED)
    boot_sd = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        a = bivariate_acc(r1f[idx], r2f[idx], r1o[idx], r2o[idx])
        sd = skilful_day(a)
        boot_sd[i] = sd if sd is not None else MAXLEAD
    return int(np.round(np.percentile(boot_sd, 2.5))), int(np.round(np.percentile(boot_sd, 97.5)))

groups = [
    ('a) All dates',           lambda a: np.ones(len(a), dtype=bool)),
    ('b) Strong (amp ≥ 1.5)',  lambda a: a >= 1.5),
    ('c) Moderate (1.0–1.5)', lambda a: (a >= 1.0) & (a < 1.5)),
    ('d) Weak (amp < 1.0)',    lambda a: a < 1.0),
]

# Line styles for 3 versions
VER_STYLE = {
    'raw':      dict(ls='-',  lw=2.0, alpha=1.0,  label_sfx='raw'),
    'RMM-BC':   dict(ls='--', lw=2.0, alpha=1.0,  label_sfx='RMM-BC'),
    'field-BC': dict(ls=':',  lw=2.5, alpha=1.0,  label_sfx='field-BC'),
}

print("Loading data ...")
all_data = {}  # all_data[model][version] = (r1e, r2e, r1o, r2o, amps)
for model in MODELS:
    all_data[model] = {}
    for ver, (rmm_dir, has_day0) in VERSIONS[model].items():
        r1e, r2e, r1o, r2o, amps, inits = load(rmm_dir, has_day0, model)
        all_data[model][ver] = (r1e, r2e, r1o, r2o, amps, inits)
        # also compute RMM-BC from raw
        if ver == 'raw':
            r1e_bc, r2e_bc = bias_correct_rmm(r1e, r2e, r1o, r2o, inits)
            all_data[model]['RMM-BC'] = (r1e_bc, r2e_bc, r1o, r2o, amps, inits)
        print(f"  {model} [{ver}]: n={len(inits)}")

leads = np.arange(MAXLEAD)
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
fig.suptitle('MJO forecast skill — raw vs RMM bias-corrected vs field bias-corrected\nUC-S2S V2 and UC-S2S SSTanom  |  2019–2021', fontsize=14)

for ax, (gtitle, mfn) in zip(axes, groups):
    ax.set_title(gtitle, fontsize=13, loc='left')
    ax.axhline(0.5, color='k', lw=1.0, ls='--', zorder=1)
    ax.axhline(0.6, color='gray', lw=0.7, ls=':', zorder=1)

    for model in MODELS:
        clr = model_color(model)
        for ver, vstyle in VER_STYLE.items():
            r1e, r2e, r1o, r2o, amps, inits = all_data[model][ver]
            mask = mfn(amps)
            n = mask.sum()
            if n < 3: continue
            acc = bivariate_acc(r1e[mask], r2e[mask], r1o[mask], r2o[mask])
            sd  = skilful_day(acc)
            sd_str = f'day {sd}' if sd is not None else '>45'
            label = f'{model} {vstyle["label_sfx"]} ({sd_str})'
            ax.plot(leads, acc, color=clr, ls=vstyle['ls'], lw=vstyle['lw'],
                    alpha=vstyle['alpha'], label=label)

    ax.set_xlim(0, 45); ax.set_ylim(-0.1, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Lead time (days)', fontsize=13)
    ax.grid(True, alpha=0.3, lw=0.5)
    if ax is axes[0]:
        ax.set_ylabel('RMM COR', fontsize=14)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)

plt.tight_layout()
out = '/glade/work/bgong/MJO_results/plots/paper_mjo_compare_3versions.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

# print skill-day table
print(f"\n{'Model':<20} {'Version':<12} {'All':>6} {'Strong':>8} {'Mod':>8} {'Weak':>6}")
print('-'*68)
for model in MODELS:
    for ver in ['raw', 'RMM-BC', 'field-BC']:
        r1e, r2e, r1o, r2o, amps, inits = all_data[model][ver]
        row = []
        for _, mfn in groups:
            mask = mfn(amps)
            if mask.sum() < 3: row.append('—'); continue
            acc = bivariate_acc(r1e[mask], r2e[mask], r1o[mask], r2o[mask])
            sd = skilful_day(acc)
            row.append(f'day {sd}' if sd is not None else '>45')
        print(f'{model:<20} {ver:<12} {row[0]:>6} {row[1]:>8} {row[2]:>8} {row[3]:>6}')
    print()
