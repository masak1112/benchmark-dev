"""
MJO bivariate ACC skill plot: FuXi (jhayron), UC-S2S (S41u/jhayron), ECMWF hindcast.
All models with field-level LOYO bias correction (drift-removed).

FuXi/S2S: OLR-based RMM (olr_1p0deg), LOYO bias using 2019-2022 obs.
ECMWF:    chi200-based RMM (chi200_1p0deg), LOYO bias using all hindcast years.

Same-month date matching across models.
Bootstrap CI: 2000 iterations, seed=42, 95%.
"""

import numpy as np
import xarray as xr
import glob, os, re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.insert(0, '/glade/work/bgong')
from model_colors import MODEL_COLORS, MODEL_LINES

# ── Directories ───────────────────────────────────────────────────────────────
DIRS = {
    'FuXi (jhayron)':  {
        'rmm_dir':  '/glade/campaign/univ/uchi0018/bing/mjo/fuxi_jhayron/olr_1p0deg/rmm_output_fbcorr',
        'pattern':  'MJO_olr_Forecast_Init_????????.nc',
        'has_day0': True,
    },
    'UC-S2S (S41u)': {
        'rmm_dir':  '/glade/campaign/univ/uchi0018/bing/mjo/s2s_jhayron/olr_1p0deg/rmm_output_fbcorr',
        'pattern':  'MJO_olr_Forecast_Init_????????.nc',
        'has_day0': False,
    },
    'ECMWF-hc': {
        'rmm_dir':  '/glade/campaign/univ/uchi0018/bing/mjo/ecmwf_s2s/chi200_1p0deg/rmm_output_fbcorr',
        'pattern':  'MJO_chi200_Forecast_Init_????????.nc',
        'has_day0': True,
    },
}
OUT_FILE = '/glade/work/bgong/MJO_results/plots/paper_mjo_fuxi_s2s_jhayron_fieldbcorr.png'
MAXLEAD  = 46
N_BOOT   = 2000
SEED     = 42
SKILL_THRESH = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────
def _date(fp): return re.search(r'\d{8}', os.path.basename(fp)).group()

def load_model(label, cfg):
    rmm_dir  = cfg['rmm_dir']
    pattern  = cfg['pattern']
    has_day0 = cfg['has_day0']
    files = sorted(glob.glob(os.path.join(rmm_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files in {rmm_dir}")
    lead_offset = 0 if has_day0 else 1
    dates, r1fc, r2fc, r1ob, r2ob = [], [], [], [], []
    for fp in files:
        ds = xr.open_dataset(fp)
        n_t = ds.sizes['time']
        leads = min(n_t, MAXLEAD - lead_offset)
        # ensemble mean
        r1 = ds['RMM1_emean'].values       # (time,) or need to compute
        r2 = ds['RMM2_emean'].values
        o1 = ds['RMM1_obs'].values
        o2 = ds['RMM2_obs'].values
        # Pad to MAXLEAD
        r1_full = np.full(MAXLEAD, np.nan)
        r2_full = np.full(MAXLEAD, np.nan)
        o1_full = np.full(MAXLEAD, np.nan)
        o2_full = np.full(MAXLEAD, np.nan)
        r1_full[lead_offset:lead_offset+leads] = r1[:leads]
        r2_full[lead_offset:lead_offset+leads] = r2[:leads]
        o1_full[lead_offset:lead_offset+leads] = o1[:leads]
        o2_full[lead_offset:lead_offset+leads] = o2[:leads]
        ds.close()
        dates.append(_date(fp))
        r1fc.append(r1_full); r2fc.append(r2_full)
        r1ob.append(o1_full); r2ob.append(o2_full)
    return {
        'dates': np.array(dates),
        'r1fc':  np.array(r1fc),  # (n, MAXLEAD)
        'r2fc':  np.array(r2fc),
        'r1ob':  np.array(r1ob),
        'r2ob':  np.array(r2ob),
        'has_day0': has_day0,
    }

def bivariate_acc(r1fc, r2fc, r1ob, r2ob):
    """Bivariate ACC over first axis (dates). Returns (MAXLEAD,)."""
    num = np.nansum(r1fc*r1ob + r2fc*r2ob, axis=0)
    den = np.sqrt(np.nansum(r1fc**2 + r2fc**2, axis=0) *
                  np.nansum(r1ob**2 + r2ob**2, axis=0))
    return np.where(den > 0, num / den, np.nan)

def bootstrap_acc(r1fc, r2fc, r1ob, r2ob, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = r1fc.shape[0]
    boot = np.empty((n_boot, MAXLEAD))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = bivariate_acc(r1fc[idx], r2fc[idx], r1ob[idx], r2ob[idx])
    ci_lo = np.nanpercentile(boot, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot, 97.5, axis=0)
    return ci_lo, ci_hi

def skill_day(acc_curve, thresh=SKILL_THRESH):
    leads = np.arange(MAXLEAD)
    for l in range(len(acc_curve)):
        if acc_curve[l] < thresh:
            if l == 0: return 0
            # linear interpolation
            x0, x1 = leads[l-1], leads[l]
            y0, y1 = acc_curve[l-1], acc_curve[l]
            return x0 + (thresh - y0)/(y1 - y0)*(x1 - x0)
    return float(MAXLEAD)

# ── Load all models ───────────────────────────────────────────────────────────
print("Loading models ...")
data = {}
for label, cfg in DIRS.items():
    print(f"  {label} ...", flush=True)
    try:
        data[label] = load_model(label, cfg)
        print(f"    {len(data[label]['dates'])} dates")
    except FileNotFoundError as e:
        print(f"    SKIP: {e}")

# ── Same-month date matching ───────────────────────────────────────────────────
# Use FuXi and S2S as primary (they should share common dates from the array job)
# ECMWF: match by month-year
labels = list(data.keys())
if len(labels) < 2:
    raise RuntimeError("Need at least 2 models loaded")

# Build month-year sets for each model
def month_set(dates): return set(d[:6] for d in dates)

# If both jhayron models present, use their common dates as anchor
anchor_labels = [l for l in labels if 'jhayron' in l or 'FuXi' in l or 'S41u' in l]
if len(anchor_labels) >= 2:
    anchor_months = month_set(data[anchor_labels[0]]['dates']) & month_set(data[anchor_labels[1]]['dates'])
else:
    anchor_months = month_set(data[labels[0]]['dates'])

print(f"\nAnchor months: {len(anchor_months)}")

def filter_by_months(d, months):
    mask = np.array([dt[:6] in months for dt in d['dates']])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[0]==len(d['dates']) else v)
            for k, v in d.items()}

for label in labels:
    data[label] = filter_by_months(data[label], anchor_months)
    print(f"  {label}: {len(data[label]['dates'])} dates after month-filter")

# ── Compute ACC + bootstrap ───────────────────────────────────────────────────
leads_days = np.arange(MAXLEAD) + 1   # 1-indexed lead days

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(SKILL_THRESH, color='k', lw=0.8, ls='--', alpha=0.5)

colors = {
    'FuXi (jhayron)': MODEL_COLORS.get('FuXi', 'tab:orange'),
    'UC-S2S (S41u)':  MODEL_COLORS.get('UC-S2S V2', 'tab:blue'),
    'ECMWF-hc':       MODEL_COLORS.get('ECMWF', 'tab:green'),
}
# Fallback colors if not in model_colors
_fallbacks = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']
for i, lbl in enumerate(labels):
    if lbl not in colors:
        colors[lbl] = _fallbacks[i % len(_fallbacks)]

for label in labels:
    d = data[label]
    acc  = bivariate_acc(d['r1fc'], d['r2fc'], d['r1ob'], d['r2ob'])
    ci_lo, ci_hi = bootstrap_acc(d['r1fc'], d['r2fc'], d['r1ob'], d['r2ob'])
    sd   = skill_day(acc)
    lead_offset = 0 if d['has_day0'] else 1
    plot_leads = leads_days[lead_offset:]
    c = colors[label]
    n = len(d['dates'])
    ax.plot(plot_leads, acc[lead_offset:], lw=2, color=c,
            label=f'{label}  (n={n}, skill≈day {sd:.0f})')
    ax.fill_between(plot_leads, ci_lo[lead_offset:], ci_hi[lead_offset:],
                    color=c, alpha=0.15)

ax.set_xlabel('Forecast lead (days)')
ax.set_ylabel('Bivariate ACC')
ax.set_title('MJO skill — field-level bias-corrected\n(FuXi, UC-S2S S41u, ECMWF hindcast)')
ax.set_xlim(1, MAXLEAD)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, lw=0.5)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_FILE}")
plt.close(fig)
