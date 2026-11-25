"""
Generate multi-model MJO bivariate ACC and RMSE comparison plots.
Usage: /glade/work/bgong/mjocast_env/bin/python run_scripts/plot_comparison.py
"""
import sys, os, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.step5_acc_plot import bivariate_acc, bivariate_rmse, _skilful_day
sys.path.insert(0, "/glade/work/bgong")
from model_colors import model_style, model_color
import sys as _sys; _sys.path.insert(0, '/glade/work/bgong/MJO_modular/run_scripts')
import mjo_rmm_io as _io   # lead alignment, one RMM sign convention, ERA5 init state

gencast_dates = set(
    re.search(r'Init_(\d{8})', f).group(1)
    for f in glob.glob('/glade/campaign/univ/uchi0018/bing/mjo/gencast/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc')
    if re.search(r'Init_(\d{8})', f)
)

common33_fuxi = set([
    '20190509','20190516','20190523','20190530','20190606','20190613','20190620','20190627',
    '20190704','20190711','20190718','20190725','20200509','20200516','20200523','20200606',
    '20200613','20200620','20200704','20200711','20200718','20200725','20210509','20210516',
    '20210523','20210606','20210613','20210620','20210627','20210704','20210711','20210718',
    '20210725',
])


def _trim(d):
    """drop trailing all-NaN leads; returns arrays and the last lead with data"""
    ok = np.where(np.isfinite(d["r1e"]).any(0))[0]
    ml = int(ok[-1]) + 1 if len(ok) else 0
    return [d[k][:, :ml] for k in ("r1e", "r2e", "r1o", "r2o")], ml

def load_dir(pattern, date_filter=None, year_filter=None, month_filter=None):
    d = _io.load(pattern, date_filter=date_filter, year_filter=year_filter,
                 month_filter=None if month_filter is None else {int(m) for m in month_filter})
    if d["n"] == 0:
        return None, None, None, None, None, 0
    a, ml = _trim(d)
    return (*a, ml, d["n"])

FUXI_15 = '/glade/work/bgong/MJO/chi250_eval/Fuxi_S2S_may_aug/chi200_1p5deg_rmm_output/MJO_fuxi_chi200_1p5deg_may_aug_Forecast_Init_????????.nc'
FUXI_10 = '/glade/campaign/univ/uchi0018/bing/mjo/fuxi_s2s/chi200_1p5deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc'

# (label, pattern, date_filter, year_filter, model, marker, ls); ls=None -> model_style's
# ours_si_v3 dir holds predictions_0903 output = UC-S2S V2 (dataset registry)
datasets = [
    ('FuXi 1.5° (n=33)',   FUXI_15, common33_fuxi, None, 'FuXi', 'o', '-'),
    ('FuXi 1.0° (n=33)',   FUXI_10, None,           None, 'FuXi', 'o', '--'),
    ('UC-S2S V2 (n=33)',
     '/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc',
     gencast_dates, None, 'UC-S2S V2', 's', None),
    ('GenCast (n=33)',
     '/glade/campaign/univ/uchi0018/bing/mjo/gencast/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc',
     None, None, 'GenCast', '^', None),
    ('NeuralGCM (n=33)',
     '/glade/campaign/univ/uchi0018/bing/mjo/ngcm_campaign/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc',
     gencast_dates, None, 'NeuralGCM', 'D', None),
    ('SI-S2S (n=33)',
     '/glade/campaign/univ/uchi0018/bing/mjo/amip_s2s/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc',
     gencast_dates, None, 'SI-S2S', 'v', None),
]

outdir = '/glade/work/bgong/MJO_results/plots'
os.makedirs(outdir, exist_ok=True)

for suffix in ['acc', 'rmse']:
    fig, ax = plt.subplots(figsize=(13, 5))
    for (label, pattern, df, yf, model, marker, ls) in datasets:
        r1e, r2e, r1o, r2o, max_lead, n = load_dir(pattern, date_filter=df, year_filter=yf)
        if r1e is None: continue
        leads = np.arange(1, max_lead+1)
        acc  = bivariate_acc(r1e, r2e, r1o, r2o)
        rmse = bivariate_rmse(r1e, r2e, r1o, r2o)
        metric = acc if suffix == 'acc' else rmse
        sd = _skilful_day(acc, leads)
        lbl = f'{label}  →day {sd}' if sd else label
        st = model_style(model)
        if ls: st['linestyle'] = ls
        ax.plot(leads, metric, marker=marker, ms=4, label=lbl, **st)
        if suffix == 'acc':
            print(f'{label}: {n} dates, skill day {sd}')

    if suffix == 'acc':
        ax.axhline(0.5, color='grey', ls=':', lw=1.5, label='ACC = 0.5')
        ax.axhline(0.0, color='k', ls=':', lw=0.8)
        ax.set(xlabel='Lead time (days)', ylabel='Bivariate ACC',
               title='MJO Bivariate ACC — chi200',
               xlim=(1, None), ylim=(-0.15, 1.05))
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    else:
        ax.axhline(np.sqrt(2), color='grey', ls=':', lw=1.5, label='RMSE=sqrt(2) (random)')
        ax.set(xlabel='Lead time (days)', ylabel='Bivariate RMSE',
               title='MJO Bivariate RMSE — chi200', xlim=(1, None))
        ax.legend(fontsize=8, loc='upper left', framealpha=0.85)

    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, f'{suffix}_4model_alldates.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
