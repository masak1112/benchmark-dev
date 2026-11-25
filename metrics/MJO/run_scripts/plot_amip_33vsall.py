"""
MJO Bivariate ACC: amip_s2s — all May-Jul (n=71) vs. 33 GenCast common dates.
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
import sys as _sys; _sys.path.insert(0, '/glade/work/bgong/MJO_modular/run_scripts')
import mjo_rmm_io as _io   # lead alignment, one RMM sign convention, ERA5 init state

gencast_dates = set(
    re.search(r'Init_(\d{8})', f).group(1)
    for f in glob.glob('/glade/campaign/univ/uchi0018/bing/mjo/gencast/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc')
    if re.search(r'Init_(\d{8})', f)
)

PATTERN = '/glade/campaign/univ/uchi0018/bing/mjo/amip_s2s/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc'


def _trim(d):
    """drop trailing all-NaN leads; returns arrays and the last lead with data"""
    ok = np.where(np.isfinite(d["r1e"]).any(0))[0]
    ml = int(ok[-1]) + 1 if len(ok) else 0
    return [d[k][:, :ml] for k in ("r1e", "r2e", "r1o", "r2o")], ml

def load_files(pattern, date_filter=None, year_filter=None, month_filter=None):
    d = _io.load(pattern, date_filter=date_filter, year_filter=year_filter,
                 month_filter=None if month_filter is None else {int(m) for m in month_filter})
    if d["n"] == 0:
        return None, None, None, None, None, 0
    a, ml = _trim(d)
    return (*a, ml, d["n"])

r1e_all, r2e_all, r1o_all, r2o_all, ml_all, n_all = load_files(PATTERN)
r1e_33,  r2e_33,  r1o_33,  r2o_33,  ml_33,  n_33  = load_files(PATTERN, date_filter=gencast_dates)

print(f'amip_s2s all: n={n_all}')
print(f'amip_s2s GenCast common: n={n_33}')

acc_all = bivariate_acc(r1e_all, r2e_all, r1o_all, r2o_all)
acc_33  = bivariate_acc(r1e_33,  r2e_33,  r1o_33,  r2o_33)
leads_all = np.arange(1, ml_all + 1)
leads_33  = np.arange(1, ml_33  + 1)
sd_all = _skilful_day(acc_all, leads_all)
sd_33  = _skilful_day(acc_33,  leads_33)
print(f'Skill day (all): {sd_all}')
print(f'Skill day (33 GenCast): {sd_33}')

color = model_color('SI-S2S')
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(leads_all, acc_all, color=color, marker='o', ms=4, lw=2, ls='-',
        label=f'SI-S2S — all May-Jul (n={n_all})  →day {sd_all}')
ax.plot(leads_33,  acc_33,  color=color, marker='s', ms=4, lw=2, ls='--',
        label=f'SI-S2S — 33 GenCast dates (n={n_33})  →day {sd_33}')
ax.axhline(0.5, color='grey', ls=':', lw=1.5, label='ACC = 0.5')
ax.axhline(0.0, color='k',    ls=':', lw=0.8)
ax.set(xlabel='Lead time (days)', ylabel='Bivariate ACC',
       title='MJO Bivariate ACC — SI-S2S  (chi200)',
       xlim=(1, None), ylim=(-0.15, 1.05))
ax.legend(fontsize=10, loc='upper right', framealpha=0.85)
ax.grid(alpha=0.3)
fig.tight_layout()

outpath = '/glade/work/bgong/MJO_results/plots/acc_amip_s2s_33vsall.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {outpath}')
