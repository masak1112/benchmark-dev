"""
MJO Bivariate ACC: FuXi — all May-Jul (n=130) vs. 33 GenCast common dates.
Uses existing 1.5deg chi200 rmm_output from chi250_eval.
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

gencast_dates = set(
    re.search(r'Init_(\d{8})', f).group(1)
    for f in glob.glob('/glade/campaign/univ/uchi0018/bing/mjo/gencast/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc')
    if re.search(r'Init_(\d{8})', f)
)

PATTERN = '/glade/work/bgong/MJO/chi250_eval/Fuxi_S2S_may_aug/chi200_1p5deg_rmm_output/MJO_fuxi_chi200_1p5deg_may_aug_Forecast_Init_????????.nc'
MAY_JUL = {'05', '06', '07'}

def load_files(pattern, date_filter=None, month_filter=None):
    files = sorted(glob.glob(pattern))
    if month_filter:
        files = [f for f in files if re.search(r'Init_(\d{8})', f) and
                 re.search(r'Init_(\d{8})', f).group(1)[4:6] in month_filter]
    if date_filter is not None:
        files = [f for f in files if re.search(r'Init_(\d{8})', f) and
                 re.search(r'Init_(\d{8})', f).group(1) in date_filter]
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

r1e_all, r2e_all, r1o_all, r2o_all, ml_all, n_all = load_files(PATTERN, month_filter=MAY_JUL)
r1e_33,  r2e_33,  r1o_33,  r2o_33,  ml_33,  n_33  = load_files(PATTERN, month_filter=MAY_JUL, date_filter=gencast_dates)

print(f'FuXi May-Jul all: n={n_all}')
print(f'FuXi May-Jul + GenCast common: n={n_33}')

acc_all = bivariate_acc(r1e_all, r2e_all, r1o_all, r2o_all)
acc_33  = bivariate_acc(r1e_33,  r2e_33,  r1o_33,  r2o_33)
leads_all = np.arange(1, ml_all + 1)
leads_33  = np.arange(1, ml_33  + 1)
sd_all = _skilful_day(acc_all, leads_all)
sd_33  = _skilful_day(acc_33,  leads_33)
print(f'Skill day (all May-Jul): {sd_all}')
print(f'Skill day (33 GenCast): {sd_33}')

color = model_color('FuXi')
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(leads_all, acc_all, color=color, marker='o', ms=4, lw=2, ls='-',
        label=f'FuXi — all May-Jul (n={n_all})  →day {sd_all}')
ax.plot(leads_33,  acc_33,  color=color, marker='s', ms=4, lw=2, ls='--',
        label=f'FuXi — 33 GenCast dates (n={n_33})  →day {sd_33}')
ax.axhline(0.5, color='grey', ls=':', lw=1.5, label='ACC = 0.5')
ax.axhline(0.0, color='k',    ls=':', lw=0.8)
ax.set(xlabel='Lead time (days)', ylabel='Bivariate ACC',
       title='MJO Bivariate ACC — FuXi  (chi200)',
       xlim=(1, None), ylim=(-0.15, 1.05))
ax.legend(fontsize=10, loc='upper right', framealpha=0.85)
ax.grid(alpha=0.3)
fig.tight_layout()

outpath = '/glade/work/bgong/MJO_results/plots/acc_fuxi_33vsall.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {outpath}')
