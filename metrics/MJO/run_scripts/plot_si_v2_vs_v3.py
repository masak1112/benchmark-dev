"""
MJO Bivariate ACC: ours_si_v2 (May-Jul) vs ours_si_v3 (May-Jul and all months).
"""
import sys, os, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.step5_acc_plot import bivariate_acc, _skilful_day
sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_style, model_color

MAY_JUL = {'05', '06', '07'}

def load_files(pattern, month_filter=None):
    files = sorted(glob.glob(pattern))
    if month_filter:
        files = [f for f in files if re.search(r'Init_(\d{8})', f) and
                 re.search(r'Init_(\d{8})', f).group(1)[4:6] in month_filter]
    good = []
    for f in files:
        try:
            ds = xr.open_dataset(f); ds['RMM1_emean']; ds.close(); good.append(f)
        except: pass
    if not good:
        return None, None, None, None, None, 0
    r1_em, r2_em, r1_ob, r2_ob = [], [], [], []
    max_lead = None
    for fp in good:
        ds = xr.open_dataset(fp)
        r1e = ds['RMM1_emean'].values; r2e = ds['RMM2_emean'].values
        r1o = ds['RMM1_obs'].values;   r2o = ds['RMM2_obs'].values
        if max_lead is None: max_lead = len(r1e)
        n = max_lead
        pad = lambda a: a[:n] if len(a) >= n else np.concatenate([a, np.full(n - len(a), np.nan)])
        r1_em.append(pad(r1e)); r2_em.append(pad(r2e))
        r1_ob.append(pad(r1o)); r2_ob.append(pad(r2o))
        ds.close()
    return np.stack(r1_em), np.stack(r2_em), np.stack(r1_ob), np.stack(r2_ob), max_lead, len(good)

V2  = '/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v2/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc'
V3  = '/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc'

datasets = [
    ('UC-S2S V2 — May-Jul',    V3, MAY_JUL,  model_color('UC-S2S V2'), 'o', '--'),
    ('UC-S2S V2 — all months', V3, None,      model_color('UC-S2S V2'), 's', '-'),
]

fig, ax = plt.subplots(figsize=(13, 5))

for label, pattern, mf, color, marker, ls in datasets:
    r1e, r2e, r1o, r2o, ml, n = load_files(pattern, month_filter=mf)
    if r1e is None:
        print(f'No data: {label}'); continue
    leads = np.arange(1, ml + 1)
    acc = bivariate_acc(r1e, r2e, r1o, r2o)
    sd  = _skilful_day(acc, leads)
    print(f'{label}: n={n}, skill day={sd}')
    ax.plot(leads, acc, color=color, marker=marker, ms=4, lw=2, ls=ls,
            label=f'{label} (n={n})  →day {sd}')

ax.axhline(0.5, color='grey', ls=':', lw=1.5, label='ACC = 0.5')
ax.axhline(0.0, color='k',    ls=':', lw=0.8)
ax.set(xlabel='Lead time (days)', ylabel='Bivariate ACC',
       title='MJO Bivariate ACC — UC-S2S V2  (chi200)',
       xlim=(1, None), ylim=(-0.15, 1.05))
ax.legend(fontsize=10, loc='upper right', framealpha=0.85)
ax.grid(alpha=0.3)
fig.tight_layout()

outpath = '/glade/work/bgong/MJO_results/plots/acc_si_v2_vs_v3.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {outpath}')
