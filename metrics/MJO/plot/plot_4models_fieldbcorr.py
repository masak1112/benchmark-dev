"""
MJO skill plot: UC-S2S V2, UC-S2S SSTanom, ECMWF, NGCM — all field-BC.
4-panel ACC plot (all / strong / moderate / weak) with skill-day table.
SSTanom sign-flip applied for consistent EOF orientation.
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

# (rmm_dir, has_day0, sign_flip)
DIRS = {
    'ECMWF':          ('/glade/campaign/univ/uchi0018/bing/mjo/ecmwf_s2s/chi200_1p0deg/rmm_output_fbcorr',      True,  False),
    'UC-S2S SSTanom': ('/glade/campaign/univ/uchi0018/bing/mjo/uc_s2s_sstanom/chi200_1p0deg/rmm_output_fbcorr', True,  True),
    'UC-S2S V2':      ('/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output_fbcorr',     True,  False),
    'NGCM':           ('/glade/campaign/univ/uchi0018/bing/mjo/ngcm/chi200_1p0deg/rmm_output_fbcorr',           False, False),
}
MODELS = ['ECMWF', 'UC-S2S SSTanom', 'UC-S2S V2', 'NGCM']

def _files(d):
    return sorted(glob.glob(os.path.join(d, 'MJO_chi200_Forecast_Init_????????.nc')))
def _date(fp):
    return re.search(r'\d{8}', os.path.basename(fp)).group()
def avail(label):
    d, _, _ = DIRS[label]
    return {_date(f) for f in _files(d) if _date(f)[:4] in YEARS}

# shared dates: ECMWF ∩ SSTanom, same-month matching for V2 and NGCM
ec_d = avail('ECMWF')
ss_d = avail('UC-S2S SSTanom')
v2_d = avail('UC-S2S V2')
ng_d = avail('NGCM')

shared_ec_ss   = sorted(ec_d & ss_d)
shared_months  = {d[:6] for d in shared_ec_ss}
dates_map = {
    'ECMWF':          shared_ec_ss,
    'UC-S2S SSTanom': shared_ec_ss,
    'UC-S2S V2':      sorted(d for d in v2_d if d[:6] in shared_months),
    'NGCM':           sorted(d for d in ng_d if d[:6] in shared_months),
}
for lbl, dts in dates_map.items():
    print(f"{lbl}: {len(dts)} dates")

def load(label):
    d, has_day0, sign_flip = DIRS[label]
    date_set = set(dates_map[label])
    r1e, r2e, r1o, r2o, amps, inits = [], [], [], [], [], []
    for fp in _files(d):
        dt = _date(fp)
        if dt not in date_set: continue
        ds = xr.open_dataset(fp)
        a = ds['RMM1_emean'].values; b = ds['RMM2_emean'].values
        c = ds['RMM1_obs'].values;   e = ds['RMM2_obs'].values
        ds.close()
        if sign_flip:
            a = -a; b = -b; c = -c; e = -e
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

def bivariate_acc(r1f, r2f, r1o, r2o):
    num   = np.nansum(r1f*r1o + r2f*r2o, axis=0)
    denom = np.sqrt(np.nansum(r1f**2+r2f**2, axis=0)*np.nansum(r1o**2+r2o**2, axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom>0, num/denom, np.nan)

def skilful_day(acc):
    for i, v in enumerate(acc):
        if not np.isnan(v) and v < 0.5: return i
    return None

def bootstrap_acc_and_sd(r1f, r2f, r1o, r2o):
    n = len(r1o)
    rng = np.random.default_rng(SEED)
    boot_acc = np.zeros((N_BOOT, MAXLEAD))
    boot_sd  = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        a   = bivariate_acc(r1f[idx], r2f[idx], r1o[idx], r2o[idx])
        boot_acc[i] = a
        sd = skilful_day(a)
        boot_sd[i]  = sd if sd is not None else MAXLEAD
    return (np.nanpercentile(boot_acc, 2.5, axis=0),
            np.nanpercentile(boot_acc, 97.5, axis=0),
            np.percentile(boot_sd, 2.5),
            np.percentile(boot_sd, 97.5))

print("Loading ...")
data = {}
for lbl in MODELS:
    r1e, r2e, r1o, r2o, amps, inits = load(lbl)
    data[lbl] = (r1e, r2e, r1o, r2o, amps)
    print(f"  {lbl}: n={len(inits)}, strong={np.sum(amps>=1.5)}, mod={np.sum((amps>=1.0)&(amps<1.5))}, weak={np.sum(amps<1.0)}")

leads = np.arange(MAXLEAD)
groups = [
    ('a) All dates',           lambda a: np.ones(len(a), dtype=bool)),
    ('b) Strong (amp ≥ 1.5)',  lambda a: a >= 1.5),
    ('c) Moderate (1.0–1.5)', lambda a: (a >= 1.0) & (a < 1.5)),
    ('d) Weak (amp < 1.0)',    lambda a: a < 1.0),
]

results = {}
for lbl in MODELS:
    results[lbl] = {}
    r1e, r2e, r1o, r2o, amps = data[lbl]
    for gtitle, mfn in groups:
        mask = mfn(amps)
        n = mask.sum()
        if n < 3: results[lbl][gtitle] = None; continue
        acc = bivariate_acc(r1e[mask], r2e[mask], r1o[mask], r2o[mask])
        ci_lo, ci_hi, sd_lo, sd_hi = bootstrap_acc_and_sd(r1e[mask], r2e[mask], r1o[mask], r2o[mask])
        sd = skilful_day(acc)
        results[lbl][gtitle] = dict(acc=acc, ci_lo=ci_lo, ci_hi=ci_hi,
                                    sd=sd, sd_lo=sd_lo, sd_hi=sd_hi, n=int(n))

fig = plt.figure(figsize=(22, 9))
ax_height = 0.58
axes = [fig.add_axes([0.04 + i*0.245, 0.32, 0.225, ax_height]) for i in range(4)]
fig.suptitle('MJO forecast skill  |  Field bias corrected (lead-dependent drift removed)', fontsize=16)

for ax, (gtitle, mfn) in zip(axes, groups):
    ax.set_title(gtitle, fontsize=15, loc='left')
    ax.axhline(0.5, color='k', lw=1.0, ls='--', zorder=1)
    ax.axhline(0.6, color='gray', lw=0.7, ls=':', zorder=1)
    for lbl in MODELS:
        r = results[lbl][gtitle]
        if r is None: continue
        sd_str = f'day {r["sd"]}' if r["sd"] is not None else f'>{MAXLEAD}'
        clr = model_color(lbl); st = model_style(lbl)
        ax.plot(leads, r['acc'], color=clr, ls=st['linestyle'], lw=2.5,
                label=f'{lbl} ({sd_str}, n={r["n"]})')
        ax.fill_between(leads, r['ci_lo'], r['ci_hi'], color=clr, alpha=0.15, lw=0)
    ax.set_xlim(0, 45); ax.set_ylim(-0.1, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
    ax.tick_params(labelsize=13)
    ax.set_xlabel('Lead time (days)', fontsize=14)
    ax.grid(True, alpha=0.3, lw=0.5)
    if ax is axes[0]: ax.set_ylabel('RMM COR', fontsize=16)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.85)

# skill-day table
group_names = [g[0].split(')')[1].strip() for g in groups]
col_labels  = ['Model'] + group_names
table_data  = []
for lbl in MODELS:
    row = [lbl]
    for gtitle, _ in groups:
        r = results[lbl][gtitle]
        if r is None:
            row.append('—')
        else:
            sd_str = str(r['sd']) if r['sd'] is not None else f'>{MAXLEAD}'
            lo = int(np.round(r['sd_lo'])); hi = int(np.round(r['sd_hi']))
            row.append(f'{sd_str}  [{lo}–{hi}]')
    table_data.append(row)

ax_tab = fig.add_axes([0.02, 0.01, 0.96, 0.26])
ax_tab.axis('off')
tbl = ax_tab.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.2)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i, lbl in enumerate(MODELS):
    clr = model_color(lbl)
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor(clr + '22')
        if j == 0: tbl[i+1, j].set_text_props(fontweight='bold')

out = '/glade/work/bgong/MJO_results/plots/paper_mjo_4models_fieldbcorr.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
