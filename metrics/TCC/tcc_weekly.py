"""
Weekly Temporal Correlation Coefficient (TCC) of the ensemble mean vs ERA5.

TCC at each grid point = Pearson correlation, across init dates, between the
forecast weekly-mean anomaly and the ERA5 weekly-mean anomaly (both relative to
the same ERA5 1979-2018 DOY climatology).
Week k = lead days 7k-6 .. 7k (week 1 = days 1-7), k = 1..6; plus weeks 3-4 and 5-6.

Stage 1 (--stage extract): per init date, ensemble-mean weekly means of the
forecast and the matching ERA5 truth -> cache/{model}/{init}.npz
Stage 2 (--stage aggregate): subtract weekly climatology, correlate across
inits -> results/tcc_weekly_{model}_{tag}.nc (+ regional means in .pkl)

Truth/climatology per model:
  ucs2s_v2 : instantaneous 00Z output -> ERA5 00Z,         clim_00z
  fuxi     : daily-mean output        -> ERA5 daily mean,  clim_dailymean

Usage:
  python tcc_weekly.py --model fuxi --stage extract --workers 32
  python tcc_weekly.py --model fuxi --stage aggregate
"""

import argparse
import os
import pickle
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool

import netCDF4
import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
ERA5_DIR  = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_ROOT = '/glade/work/bgong/benchmark-dev/metrics/ACC'
CACHE_DIR = '/glade/derecho/scratch/bgong/tcc_cache'
# Optional cap on ensemble size (--n-members): readers keep the first N members, so ensemble-mean
# skill is not inflated for models with more members. None = use every member.
N_MEMBERS = None
# Optional detrending (--detrend, FuXi-S2S paper method, observation side): subtract a linear-trend
# ERA5 climatology fitted over 2002-2016 (compute_trend_clim.py) instead of the fixed 1979-2018 one.
DETREND = False
TREND_DIR = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_trend'
# Optional (--monthly-mean): after the climatology, subtract the sample's own mean anomaly for each
# calendar month of the init date (all years pooled), separately for forecast and ERA5, at every
# grid point / lead week. Removes seasonal residuals of the fixed climatology (e.g. warming since
# 1979-2018) and each model's season-dependent drift.
MONTHLY_MEAN = False
# Optional (--yearly-mean): subtract the sample's own mean anomaly for each init year (forecast, ERA5 separately),
# so pooling windows from different years scores only within-year variability, not year-to-year offsets
YEARLY_MEAN = False
RES_DIR   = os.path.join(HERE, 'results')
TMP_DIR   = '/glade/derecho/scratch/bgong/tmp'

N_WEEKS = 6
WEEK_NAMES = [f'week{k}' for k in range(1, N_WEEKS + 1)] + ['week3-4', 'week5-6']

# label, ERA5 var / subdir, level hPa, UC-S2S var, FuXi channel, FuXi scale
VARIABLES = [
    ('t2m',  '2m_temperature',           None, '2m_temperature',           't2m',  1.0),
    ('z500', 'geopotential',             500,  'geopotential',             'z500', 1.0),
    # FuXi tp is a mean rate in mm/h; ERA5 total_precipitation_24hr is m/day
    ('tp',   'total_precipitation_24hr', None, 'total_precipitation_24hr', 'tp',   0.024),
    ('u10',  '10m_u_component_of_wind',  None, '10m_u_component_of_wind',  '10u',  1.0),
    ('v10',  '10m_v_component_of_wind',  None, '10m_v_component_of_wind',  '10v',  1.0),
    ('u850', 'u_component_of_wind',      850,  'u_component_of_wind',      'u850', 1.0),
    ('v850', 'v_component_of_wind',      850,  'v_component_of_wind',      'v850', 1.0),
    # OLR: ERA5 and FuXi 'ttr' are both W m-2, negative upward (checked: global means -240 vs -241)
    ('olr',  'mean_top_net_long_wave_radiation_flux', None, 'mean_top_net_long_wave_radiation_flux', 'ttr', 1.0),
]
LABELS = [v[0] for v in VARIABLES]

MODELS = {
    'ucs2s_v2': {
        'name': 'UC-S2S V2',
        'members': 20,
        'dir': '/glade/derecho/scratch/bgong/ours_si_postprocess_v2/predictions_0903',
        'daily_mean': False,
        'clim': ('clim_00z', 'clim_00z_{}.npy'),
        # OLR-free model: its mean_top_net_long_wave_radiation_flux is written as all zeros
        'missing': {'olr'},
    },
    'fuxi': {
        'name': 'FuXi',
        'members': 51,
        'dir': '/glade/derecho/scratch/bgong/fuxi_s2s',
        'daily_mean': True,
        'clim': ('clim_dailymean', 'clim_dailymean_{}.npy'),
        # tp of lead day L is the total over day L (00->24 UTC), i.e. ERA5 tp24 at 00 UTC of day L+1
        # (checked: pattern corr with that truth beats the daily-mean truth of day L)
        'truth': {'tp': ('00z', 1)},
    },
    'si_s2s': {
        'name': 'SI-S2S',
        'members': 56,
        'dir': '/glade/derecho/scratch/bgong/amip_s2s_train_scratch',
        'daily_mean': False,       # instantaneous 00Z output (passes the day-0 check vs ERA5 00Z)
        'clim': ('clim_00z', 'clim_00z_{}.npy'),
        'missing': {'olr'},        # no OLR in the output
    },
    'ngcm': {
        'name': 'NeuralGCM',
        'members': 25,
        'dir': '/glade/derecho/scratch/bgong/ngcm/climatology',
        'daily_mean': True,        # *_avg fields and daily leads are daily means
        'clim': ('clim_dailymean', 'clim_dailymean_{}.npy'),
        'missing': {'t2m', 'u10', 'v10', 'olr'},   # not in the stores
        # tp of lead day L is the total over day L (00->24 UTC), i.e. ERA5 tp24 at 00 UTC of day L+1
        # (checked: pattern corr with that truth beats the daily-mean truth of day L)
        'truth': {'tp': ('00z', 1)},
    },
    'ecmwf': {
        'name': 'ECMWF',
        'members': 51,             # control (cf) + 50 perturbed (pf)
        'dir': '/glade/derecho/scratch/bgong/ec_s2s_model_2025',
        # output type differs by variable: t2m is a daily mean (stepType avg); z500 / u850 / v850 are
        # instantaneous 00 UTC fields; tp is taken as the 24 h total ENDING at 00 UTC of the valid day,
        # which is exactly ERA5 total_precipitation_24hr at 00 UTC (so tp also uses the 00Z truth)
        'daily_mean': True,
        'daily_mean_vars': {'t2m'},
        'clim': ('clim_dailymean', 'clim_dailymean_{}.npy'),   # used for daily_mean_vars; 00Z clim otherwise
        'missing': {'u10', 'v10', 'olr'},  # not downloaded
    },
}


# ECMWF reforecasts (hindcasts) of the 2025 model version: one file per model date (every other day of 2025),
# each holding that calendar day of hindcast years 2019-2021; 11 members (control + 10 perturbed); same steps,
# units and truth conventions as the real-time ECMWF set. Groups: t2m, tp, pl500_z (gh), uv200_850 (u, v).
MODELS['ecmwf_rf'] = {**MODELS['ecmwf'], 'name': 'ECMWF hindcast', 'members': 11,
                      'dir': '/glade/derecho/scratch/bgong/ec_s2s_model_2025_hindcast'}
HC_YEARS = (2019, 2020, 2021)
HC_GROUPS = ('t2m', 'tp', 'pl500_z', 'uv200_850')


def hc_file(group, ens, init):
    return f"{MODELS['ecmwf_rf']['dir']}/ecmwf_rf_{group}_{ens}_2025{init[4:]}_h2019-2021.grib"


def truth_spec(model, lbl):
    """(daily, offset): verify variable lbl of lead day L against the ERA5 daily mean (daily=True) or the
    00 UTC value (daily=False) of day L + offset."""
    cfg = MODELS[model]
    if lbl in cfg.get('truth', {}):
        kind, offset = cfg['truth'][lbl]
        return kind == 'dailymean', offset
    daily = lbl in cfg['daily_mean_vars'] if 'daily_mean_vars' in cfg else cfg['daily_mean']
    return daily, 0


def truth_is_daily(model, lbl):
    """True if variable lbl of this model is verified against the ERA5 daily mean (else ERA5 00 UTC)."""
    return truth_spec(model, lbl)[0]


def clim_file(model, lbl):
    if truth_is_daily(model, lbl):
        return os.path.join(CLIM_ROOT, 'clim_dailymean', f'clim_dailymean_{lbl}.npy')
    return os.path.join(CLIM_ROOT, 'clim_00z', f'clim_00z_{lbl}.npy')

lat = np.linspace(-89.5, 89.5, 180)
W2D = np.outer(np.cos(np.deg2rad(lat)), np.ones(360))
REGIONS = {'global': (-90, 90), 'tropics': (-30, 30), 'NH': (30, 90), 'SH': (-90, -30)}
MASKS = {r: np.outer((lat >= a) & (lat <= b), np.ones(360, dtype=bool)) for r, (a, b) in REGIONS.items()}
# land / ocean: ERA5 land-sea mask >= 0.5 (same 1-deg grid, lat -89.5..89.5, lon 0..359)
_LAND = np.asarray(netCDF4.Dataset('/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata/'
                                   'land_sea_mask_180x360.nc').variables['land_sea_mask'][:]) >= 0.5
MASKS.update({'land': _LAND, 'ocean': ~_LAND})
# extratropics: both hemispheres poleward of 30 deg (bootstrap / plots only; same cos-lat weighting)
MASKS['extratropics'] = np.outer(np.abs(lat) >= 30, np.ones(360, dtype=bool))
# land_interior: land with lsm >= 0.99 whose 8 neighbours are all land (lsm >= 0.5; lon wraps, edge rows repeated at the poles):
# drops coastal / mixed cells where 00Z-vs-daily and land/ocean mismatches inflate T2M errors. Keeps ~64% of land area.
_LSM = np.asarray(netCDF4.Dataset('/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata/'
                                  'land_sea_mask_180x360.nc').variables['land_sea_mask'][:]).squeeze()
_LI = _LSM >= 0.99
_L5 = np.pad(_LSM >= 0.5, ((1, 1), (0, 0)), mode='edge')          # poles: repeat edge row (no wrap in latitude)
for _dy in (-1, 0, 1):
    for _dx in (-1, 0, 1):
        _LI &= np.roll(_L5, _dx, axis=1)[1 + _dy:181 + _dy]        # longitude wraps
MASKS['land_interior'] = _LI


def era5_dates(tvar):
    """Decode ERA5 time from its own units: 1979-2018 t2m/u10/v10 files use
    'hours since 1959-01-01', everything else 'hours since 1900-01-01'."""
    return [datetime(d.year, d.month, d.day, d.hour) for d in
            netCDF4.num2date(tvar[:], tvar.units, only_use_cftime_datetimes=False,
                             only_use_python_datetimes=True)]


def cache_dir(model):
    """Per-init cache; a separate directory when the ensemble is truncated below its native size."""
    base = MODELS[model].get('cache', model)      # alias keys share another key's cache
    if N_MEMBERS is not None and N_MEMBERS < MODELS[model]['members']:
        return os.path.join(CACHE_DIR, f'{base}_m{N_MEMBERS}')
    return os.path.join(CACHE_DIR, base)


def _member_num(path, sep):
    return int(os.path.basename(path).split(sep)[1].split('.')[0])


def doy_index(d):
    if d.month == 2 and d.day == 29:
        return 59
    return d.timetuple().tm_yday - 1


# ── Model init-date listing ───────────────────────────────────────────────────

UC_PREFIX = 'pangu_plasim_pp_si_infer_v2_24h_45step_'
# UC-S2S V2 with a forecast-bias correction applied to all variables: same files / variables / grid as V2
MODELS['ucs2s_v2_fbcorr'] = {**MODELS['ucs2s_v2'], 'name': 'UC-S2S V2-fbcorr',
                             'dir': '/glade/derecho/scratch/bgong/pp_si_fbcorr_allvars',
                             'prefix': 'pangu_plasim_pp_si_infer_v2_fbcorr_allvars_24h_45step_'}
# UC-S2S D (pangu-only base, deterministic latent): ONE member (ens_0), 00Z output like V2; files hold only
# t2m / msl / z / tp, so the other variables are NaN in the cache
MODELS['ucs2s_d'] = {**MODELS['ucs2s_v2'], 'name': 'UC-S2S D', 'members': 1,
                     'dir': '/glade/derecho/scratch/bgong/pp_si_infer_D_pangu_only/predictions',
                     'prefix': 'pangu_plasim_pp_si_infer_D_pangu_only_24h_45step_',
                     'missing': {'u10', 'v10', 'u850', 'v850', 'olr'}}
# UC-D_ft: same as UC-S2S D but the fine-tuned (ft-only) run; 1 member, t2m / msl / z / tp only
MODELS['ucs2s_d_ft'] = {**MODELS['ucs2s_d'], 'name': 'UC-D_ft',
                        'dir': '/glade/derecho/scratch/bgong/pp_si_infer_D_ft_only/predictions',
                        'prefix': 'pangu_plasim_pp_si_infer_D_ft_only_24h_45step_'}
# UC-S2S V2 member 0 only (single-member counterpart of UC-S2S D / UC-D_ft); extract with --n-members 1
MODELS['ucs2s_v2_e0'] = {**MODELS['ucs2s_v2'], 'name': 'UC-S2S V2 (ens_0)', 'members': 1}
# UC-S2S L_latent_only: base only with the SI latent (base_latent=si, mc_dropout), no SI post-processing; 4 members,
# t2m / msl / z / tp only
MODELS['ucs2s_l'] = {**MODELS['ucs2s_d'], 'name': 'UC-S2S L_latent_only', 'members': 4,
                     'dir': '/glade/derecho/scratch/bgong/pp_si_infer_L_latent_only/predictions',
                     'prefix': 'pangu_plasim_pp_si_infer_L_latent_only_24h_45step_'}
# UC-S2S SST-anomaly conditioning (sst_cond_anom, Midway run S2S_SSTANOM, 2019 inits); 4 members, t2m / z / tp only
MODELS['ucs2s_sstanom'] = {**MODELS['ucs2s_d'], 'name': 'UC-S2S SSTanom', 'members': 4,
                           'dir': '/glade/derecho/scratch/bgong/pp_si_infer_sstanom_2019/predictions',
                           'prefix': 'pangu_plasim_pp_si_infer_sstanom_2019_24h_45step_'}
# UC-S2S V2 first 4 members (to match the 4-member runs); same cache as `--model ucs2s_v2 --n-members 4` (ucs2s_v2_m4)
MODELS['ucs2s_v2_m4'] = {**MODELS['ucs2s_v2'], 'members': 4}
# UC-S2S SSTanom-evolving: as SSTanom but with an evolving SST anomaly (run pp_si_infer_sstanom_2019_evolving); 4 members
# UC-S2S multicond: V2 configuration with multiple conditioning fields (run pp_si_infer_multicond_2019); 4 members
MODELS['ucs2s_multicond'] = {**MODELS['ucs2s_sstanom'], 'name': 'UC-S2S multicond',
                             'dir': '/glade/derecho/scratch/bgong/pp_si_infer_multicond_2019/predictions',
                             'prefix': 'pangu_plasim_pp_si_infer_multicond_2019_24h_45step_'}
# multicond rerun (2026-10-27) has 20 members; separate key so the 4-member cache 'ucs2s_multicond' stays valid
MODELS['ucs2s_multicond20'] = {**MODELS['ucs2s_multicond'], 'members': 20}
# multicond 2020 inits (same dir, prefix ..._multicond_2020_); 20 members, full variable set like V2
MODELS['ucs2s_multicond2020'] = {**MODELS['ucs2s_multicond20'], 'prefix': 'pangu_plasim_pp_si_infer_multicond_2020_24h_45step_',
                                 'missing': MODELS['ucs2s_v2'].get('missing', set())}
# alias of UC-S2S V2 (same files and cache) so V2 can appear twice in one figure, e.g. 2019 and 2020 columns
MODELS['ucs2s_v2_y2020'] = {**MODELS['ucs2s_v2'], 'cache': 'ucs2s_v2'}
# multicond pooled over the 2019 and 2020 runs (20 members): its cache dir holds symlinks to the
# ucs2s_multicond20 (2019) and ucs2s_multicond2020 caches; inits are listed from that cache
MODELS['ucs2s_multicond1920'] = {**MODELS['ucs2s_multicond20'], 'cache': 'ucs2s_multicond1920', 'from_cache': True}
# V2 paired init-by-init with ucs2s_multicond1920: the 2020 multicond inits are FuXi dates, so each is matched to
# V2's nearest Mon/Thu init (<=1 day). date_map: label (multicond date) -> real V2 init used for cache + climatology
MODELS['ucs2s_v2_mcpair'] = {**MODELS['ucs2s_v2'], 'cache': 'ucs2s_v2', 'base': 'ucs2s_v2', 'date_map': {
    '20200103': '20200102', '20200110': '20200109', '20200117': '20200116', '20200124': '20200123',
    '20200131': '20200130'}}
# multicond 2019 + 2020 pooled (plot-only): cache dir ucs2s_multicond_1920 holds symlinks to the ucs2s_multicond20 (2019)
# and ucs2s_multicond2020 caches; do not --stage extract this key (the two years have different file prefixes)
MODELS['ucs2s_multicond_1920'] = {**MODELS['ucs2s_multicond2020'], 'cache': 'ucs2s_multicond_1920'}
# multicond 2019 rerun (2026-10-28): FuXi init dates, 20 members, full variable set; same prefix as the old
# 2019 run, whose leftover Monday files (3 variables) share the folder -> always pass an explicit dates file
MODELS['ucs2s_multicond2019n'] = {**MODELS['ucs2s_multicond2020'], 'prefix': 'pangu_plasim_pp_si_infer_multicond_2019_24h_45step_'}
# pooled new 2019 + 2020 multicond: symlinks to the ucs2s_multicond2019n / ucs2s_multicond2020 caches
MODELS['ucs2s_multicond1920n'] = {**MODELS['ucs2s_multicond2020'], 'cache': 'ucs2s_multicond1920n', 'from_cache': True}
# V2 paired with ucs2s_multicond1920n (FuXi dates): each init -> V2's nearest Mon/Thu init (<=1 day), from pairs_v2_mc1920n.txt
_PAIRS = os.path.join(HERE, 'pairs_v2_mc1920n.txt')
MODELS['ucs2s_v2_mcpair2'] = {**MODELS['ucs2s_v2'], 'cache': 'ucs2s_v2', 'base': 'ucs2s_v2', 'date_map': dict(
    l.split() for l in open(_PAIRS) if l.strip()) if os.path.exists(_PAIRS) else {}}
# multicond 2021 inits (prefix ..._multicond_2021_), FuXi dates, 20 members, full variable set
MODELS['ucs2s_multicond2021'] = {**MODELS['ucs2s_multicond2020'], 'prefix': 'pangu_plasim_pp_si_infer_multicond_2021_24h_45step_'}
# pooled multicond 2019 + 2020 + 2021 (rerun 2026-10-29): symlinks to the three per-year caches
MODELS['ucs2s_multicond_all'] = {**MODELS['ucs2s_multicond2020'], 'cache': 'ucs2s_multicond_all', 'from_cache': True}
# V2 paired with ucs2s_multicond_all (FuXi dates, 2019-2021): nearest V2 init (<=1 day), from pairs_v2_mcall.txt
_PAIRS3 = os.path.join(HERE, 'pairs_v2_mcall.txt')
MODELS['ucs2s_v2_mcpair3'] = {**MODELS['ucs2s_v2'], 'cache': 'ucs2s_v2', 'base': 'ucs2s_v2', 'date_map': dict(
    l.split() for l in open(_PAIRS3) if l.strip()) if os.path.exists(_PAIRS3) else {}}
# multicond, new full-variable run only (files of 2026-10-29/30), all years: /glade/derecho/scratch/bgong/multicond_all_links
# holds symlinks (uniform prefix) to the complete 20-member inits; the old 3-variable 2019 files of 10-27 are excluded
MODELS['ucs2s_multicond_new'] = {**MODELS['ucs2s_multicond2020'], 'dir': '/glade/derecho/scratch/bgong/multicond_all_links',
                                'prefix': 'pangu_plasim_pp_si_infer_multicond_all_24h_45step_'}
# multicond with the earlier fine-tuned rollout checkpoint (rollout_ckpt_1050 instead of rollout_last); Jan 2019/2020, 4 members
MODELS['ucs2s_mc_pp12800'] = {**MODELS['ucs2s_multicond2020'], 'name': 'UC-S2S multicond pp12800', 'members': 4,
                              'dir': '/glade/derecho/scratch/bgong/pp_si_infer_multicond_pp12800_jan4m/predictions',
                              'prefix': 'pangu_plasim_pp_si_infer_multicond_pp12800_jan4m_24h_45step_'}
# full multicond rerun with postprocess SI pp_si_ckpt_12800 for every year: symlinks (one run per init, 20 members)
# built by job_scripts/submit_tcc_pp12800_m20.sh from pp_si_infer_multicond_pp12800/predictions
MODELS['ucs2s_mc_pp12800f'] = {**MODELS['ucs2s_multicond2020'], 'name': 'UC-S2S multicond', 'cache': 'ucs2s_mc_pp12800f',
                               'dir': '/glade/derecho/scratch/bgong/multicond_pp12800_links',
                               'prefix': 'pangu_plasim_pp_si_infer_multicond_pp12800f_24h_45step_'}
# V2 paired with ucs2s_mc_pp12800f: nearest V2 init (one-to-one, <=1 day), from pairs_v2_pp12800.txt (make_pairs_v2.py)
_PAIRS4 = os.path.join(HERE, 'pairs_v2_pp12800.txt')
MODELS['ucs2s_v2_pp12800pair'] = {**MODELS['ucs2s_v2'], 'cache': 'ucs2s_v2', 'base': 'ucs2s_v2', 'date_map': dict(
    l.split() for l in open(_PAIRS4) if l.strip()) if os.path.exists(_PAIRS4) else {}}
# FuXi paired with ucs2s_mc_pp12800f: nearest FuXi init (one-to-one, <=2 days), from pairs_fuxi_pp12800.txt
_PAIRS5 = os.path.join(HERE, 'pairs_fuxi_pp12800.txt')
MODELS['fuxi_pp12800pair'] = {**MODELS['fuxi'], 'cache': 'fuxi', 'base': 'fuxi', 'date_map': dict(
    l.split() for l in open(_PAIRS5) if l.strip()) if os.path.exists(_PAIRS5) else {}}
# NeuralGCM paired with ucs2s_mc_pp12800f: nearest NGCM init (one-to-one, <=2 days), from pairs_ngcm_pp12800.txt
_PAIRS6 = os.path.join(HERE, 'pairs_ngcm_pp12800.txt')
MODELS['ngcm_pp12800pair'] = {**MODELS['ngcm'], 'cache': 'ngcm', 'base': 'ngcm', 'date_map': dict(
    l.split() for l in open(_PAIRS6) if l.strip()) if os.path.exists(_PAIRS6) else {}}
# 2026-11-02 regenerated multicond pp12800 (ucs2s_mc_pp12800f, 3 years): partners paired by nearest init
# (one-to-one, <=2 days; make_pairs_v2.py --have-file) -> pairs_{ucs2s_v2,si_s2s,ngcm}_mc3y.txt
def _pairs(fn):
    f = os.path.join(HERE, fn)
    return dict(l.split() for l in open(f) if l.strip()) if os.path.exists(f) else {}
MODELS['ucs2s_v2_mc3ypair'] = {**MODELS['ucs2s_v2'], 'cache': 'ucs2s_v2', 'base': 'ucs2s_v2', 'date_map': _pairs('pairs_ucs2s_v2_mc3y.txt')}
MODELS['si_s2s_mc3ypair'] = {**MODELS['si_s2s'], 'cache': 'si_s2s', 'base': 'si_s2s', 'date_map': _pairs('pairs_si_s2s_mc3y.txt')}
MODELS['fuxi_mc3ypair'] = {**MODELS['fuxi'], 'cache': 'fuxi', 'base': 'fuxi', 'date_map': _pairs('pairs_fuxi_mc3y.txt')}
MODELS['ngcm_mc3ypair'] = {**MODELS['ngcm'], 'cache': 'ngcm', 'base': 'ngcm', 'date_map': _pairs('pairs_ngcm_mc3y.txt')}
# UC-S2S SSTanom 3-year rerun (2026-11-04, pp_si_infer_sstanom/predictions, per-year prefix sstanom_{YYYY}); 20 members,
# full variable set; symlinks under one prefix built by job_scripts/submit_tcc_sst3y.sh
MODELS['ucs2s_sstanom3y'] = {**MODELS['ucs2s_v2'], 'name': 'UC-S2S SSTanom', 'cache': 'ucs2s_sstanom3y',
                             'dir': '/glade/derecho/scratch/bgong/sstanom3y_links',
                             'prefix': 'pangu_plasim_pp_si_infer_sstanom3y_24h_45step_'}
# SSTanom 3-year rerun, member 0 only (extract with --n-members 1; own cache, readable at any --n-members)
MODELS['ucs2s_sstanom3y_e0'] = {**MODELS['ucs2s_sstanom3y'], 'name': 'UC-S2S SSTanom (ens_0)', 'members': 1, 'cache': 'ucs2s_sstanom3y_e0'}
# first 4 members of ucs2s_multicond_new (cache from `--model ucs2s_multicond_new --n-members 4`), for 4-vs-4 comparisons
MODELS['ucs2s_multicond_new_m4'] = {**MODELS['ucs2s_multicond_new'], 'members': 4, 'cache': 'ucs2s_multicond_new_m4'}
MODELS['ucs2s_sstanom_evo'] = {**MODELS['ucs2s_sstanom'], 'name': 'UC-S2S SSTanom-evolving',
                               'dir': '/glade/derecho/scratch/bgong/pp_si_infer_sstanom_2019_envolving/predictions',
                               'prefix': 'pangu_plasim_pp_si_infer_sstanom_2019_evolving_24h_45step_'}
UC_MODELS = ('ucs2s_v2', 'ucs2s_v2_fbcorr', 'ucs2s_d', 'ucs2s_d_ft', 'ucs2s_v2_e0', 'ucs2s_l', 'ucs2s_sstanom', 'ucs2s_v2_m4', 'ucs2s_sstanom_evo', 'ucs2s_multicond', 'ucs2s_multicond20', 'ucs2s_multicond2020', 'ucs2s_v2_y2020',
             'ucs2s_multicond1920', 'ucs2s_v2_mcpair',
             'ucs2s_multicond2019n', 'ucs2s_multicond1920n', 'ucs2s_v2_mcpair2',
             'ucs2s_multicond2021', 'ucs2s_multicond_all', 'ucs2s_v2_mcpair3', 'ucs2s_multicond_new', 'ucs2s_mc_pp12800', 'ucs2s_multicond_new_m4', 'ucs2s_mc_pp12800f', 'ucs2s_v2_pp12800pair', 'ucs2s_v2_mc3ypair', 'ucs2s_sstanom3y', 'ucs2s_sstanom3y_e0')


def uc_prefix(model):
    return MODELS[model].get('prefix', UC_PREFIX)


def list_dates(model):
    cfg = MODELS[model]
    if cfg.get('from_cache'):
        return sorted(f[:8] for f in os.listdir(cache_dir(model)) if f.endswith('.npz'))
    if 'date_map' in cfg:
        inv = {v: k for k, v in cfg['date_map'].items()}
        return sorted(inv.get(d, d) for d in list_dates(cfg['base']))
    d = MODELS[model]['dir']
    if model in UC_MODELS:
        pre = uc_prefix(model)
        return sorted(f[len(pre):len(pre) + 8] for f in os.listdir(d)
                      if f.startswith(pre) and f.endswith('_ens_0.nc'))
    if model == 'ecmwf_rf':
        # an init (YYYY in HC_YEARS, MMDD of a 2025 model date) counts once all groups exist for cf and pf
        pre = lambda g, e: f'ecmwf_rf_{g}_{e}_2025'
        have = [{f[len(pre(g, e)):len(pre(g, e)) + 4] for f in os.listdir(d)
                 if f.startswith(pre(g, e)) and f.endswith('_h2019-2021.grib')} for g in HC_GROUPS for e in ('cf', 'pf')]
        return sorted(f'{y}{md}' for y in HC_YEARS for md in set.intersection(*have))
    if model == 'ecmwf':
        # an init counts only if every variable group needed here exists for control and perturbed members
        groups = ('t2m', 'tp', 'pl500_z_u', 'uv200_850')
        have = [{f.rsplit('_', 1)[1][:8] for f in os.listdir(d)
                 if f.startswith(f'ecmwf_{g}_{e}_') and f.endswith('.grib')} for g in groups for e in ('cf', 'pf')]
        return sorted(set.intersection(*have))
    if model == 'ngcm':
        return sorted(f[:10].replace('-', '') for y in os.listdir(d) if y.isdigit()
                      for f in os.listdir(os.path.join(d, y)) if f.endswith('T00.zarr'))
    if model == 'si_s2s':
        return sorted(f for f in os.listdir(d) if f.isdigit() and os.path.isdir(os.path.join(d, f)))
    return sorted(f[:8] for f in os.listdir(d) if f.endswith('.7z') and f[:8].isdigit())


# ── ERA5 truth ────────────────────────────────────────────────────────────────

class ERA5:
    """Caches open yearly files; returns the 00Z field or the daily mean."""

    def __init__(self, daily_mean):
        self.daily_mean = daily_mean
        self.h = {}

    def _open(self, subdir, year):
        key = (subdir, year)
        if key not in self.h:
            nc = netCDF4.Dataset(f'{ERA5_DIR}/{subdir}/{year}_180x360.nc')
            t = era5_dates(nc.variables['time'])
            self.h[key] = (nc, {d: i for i, d in enumerate(t)})
        return self.h[key]

    def get(self, subdir, level, date):
        nc, idx = self._open(subdir, date.year)
        times = [date + timedelta(hours=h) for h in ((0, 6, 12, 18) if self.daily_mean else (0,))]
        ii = [idx[t] for t in times if t in idx]
        if len(ii) != len(times):
            return None
        v = nc.variables[subdir]
        if level is not None:
            li = int(np.where(nc.variables['level'][:] == level)[0][0])
            x = v[ii, li, :, :]
        else:
            x = v[ii, :, :]
        return np.asarray(x, dtype=np.float64).mean(axis=0)

    def close(self):
        for nc, _ in self.h.values():
            nc.close()


# ── Forecast readers: return {label: (n_leads, 180, 360)} ensemble mean, leads 1..42 ──

def read_ucs2s(init, model='ucs2s_v2'):
    d = MODELS[model]['dir']
    files = sorted((os.path.join(d, f) for f in os.listdir(d)
                    if f.startswith(f'{uc_prefix(model)}{init}') and f.endswith('.nc')),
                   key=lambda f: _member_num(f, '_ens_'))[:N_MEMBERS]
    n_lead = 7 * N_WEEKS
    acc = {lbl: np.zeros((n_lead, 180, 360)) for lbl in LABELS}
    n = 0
    for fp in files:
        with netCDF4.Dataset(fp) as nc:
            t = np.asarray(nc.variables['time'][:]).astype(int)
            # lead day L (1..42) -> time index where t == L
            ti = [int(np.where(t == L)[0][0]) for L in range(1, n_lead + 1)]
            levs = nc.variables['level'][:]
            for lbl, _, level, ucvar, _, _ in VARIABLES:
                if ucvar not in nc.variables:          # e.g. UC-S2S D has no winds / OLR
                    acc[lbl][:] = np.nan
                    continue
                v = nc.variables[ucvar]
                if level is not None:
                    li = int(np.where(levs == level)[0][0])
                    acc[lbl] += np.asarray(v[ti, li, :, :], dtype=np.float64)
                else:
                    acc[lbl] += np.asarray(v[ti, :, :], dtype=np.float64)
        n += 1
    if n == 0:
        return None, 0
    return {k: a / n for k, a in acc.items()}, n


# SI-S2S variable names: (var, level in Pa, scale). PRATEsfc_24h is kg m-2 s-1 -> x86.4 = m/day.
SI_VARS = {
    't2m':  ('2m_temperature',          None,    1.0),
    'z500': ('geopotential',            50000.0, 1.0),
    'tp':   ('PRATEsfc_24h',            None,    86.4),
    'u10':  ('10m_u_component_of_wind', None,    1.0),
    'v10':  ('10m_v_component_of_wind', None,    1.0),
    'u850': ('u_component_of_wind',     85000.0, 1.0),
    'v850': ('v_component_of_wind',     85000.0, 1.0),
}


def read_si_s2s(init):
    d = os.path.join(MODELS['si_s2s']['dir'], init)
    files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.startswith(f'ensemble_{init}_seed') and f.endswith('.nc')),
                   key=lambda f: _member_num(f, '_seed'))
    n_lead = 7 * N_WEEKS
    acc = {lbl: np.zeros((n_lead, 180, 360)) for lbl in LABELS}
    t0 = datetime.strptime(init, '%Y%m%d')
    n = 0
    for fp in files:
        with netCDF4.Dataset(fp) as nc:
            tv = nc.variables['time']
            vt = era5_dates(tv)
            assert vt[0] == t0, f'{fp}: first time {vt[0]} != init {t0}'
            ti = [vt.index(t0 + timedelta(days=L)) for L in range(1, n_lead + 1)]
            plev = nc.variables['plev'][:]
            n_file = nc.dimensions['member'].size
            take = n_file if N_MEMBERS is None else min(n_file, N_MEMBERS - n)
            if take <= 0:
                break
            for lbl, (var, level, scale) in SI_VARS.items():
                v = nc.variables[var]
                if level is not None:
                    li = int(np.where(plev == level)[0][0])
                    x = np.asarray(v[:take, ti, li, :, :], dtype=np.float64)
                else:
                    x = np.asarray(v[:take, ti, :, :], dtype=np.float64)
                acc[lbl] += x.sum(axis=0) * scale          # sum over the members used from this seed file
            n += take
    if n == 0:
        return None, 0
    return {k: a / n for k, a in acc.items()}, n


# NeuralGCM store variables (level baked into the name); tp is m per day like ERA5
NGCM_VARS = {
    'z500': 'geopotential_500_avg',
    'tp':   'total_precipitation_24hr',
    'u850': 'u_component_of_wind_850_avg',
    'v850': 'v_component_of_wind_850_avg',
}


def read_ngcm(init):
    import zarr
    from scipy.interpolate import RegularGridInterpolator
    store = os.path.join(MODELS['ngcm']['dir'], init[:4], f'{init[:4]}-{init[4:6]}-{init[6:]}T00.zarr')
    z = zarr.open(store, mode='r')
    n_lead = 7 * N_WEEKS
    leads = np.asarray(z['prediction_timedelta_daily'][:]).astype(int)
    li = [int(np.where(leads == L)[0][0]) for L in range(1, n_lead + 1)]
    src_lat = np.asarray(z['latitude'][:])            # Gaussian, south -> north
    src_lon = np.append(np.asarray(z['longitude'][:]), 360.0)
    assert src_lat[0] < src_lat[-1]
    # ERA5 target; rows poleward of the outermost Gaussian latitude (+-87.86) take that row's value
    LA, LO = np.meshgrid(np.clip(lat, src_lat[0], src_lat[-1]), np.arange(0.0, 360.0, 1.0), indexing='ij')
    pts = np.stack([LA.ravel(), LO.ravel()], axis=-1)
    fc = {}
    n_mem = z['number'].shape[0] if N_MEMBERS is None else min(N_MEMBERS, z['number'].shape[0])
    for lbl in LABELS:
        if lbl not in NGCM_VARS:
            fc[lbl] = np.full((n_lead, 180, 360), np.nan)
            continue
        a = z[NGCM_VARS[lbl]]
        x = np.asarray(a[0, :n_mem], dtype=np.float64)    # (number, lead, [surface,] lon, lat)
        if x.ndim == 5:
            x = x[:, :, 0]
        x = x[:, li].mean(axis=0).transpose(0, 2, 1)      # ensemble mean -> (lead, lat, lon)
        x = np.concatenate([x, x[..., :1]], axis=-1)      # periodic longitude
        out = np.empty((n_lead, 180, 360))
        for k in range(n_lead):
            out[k] = RegularGridInterpolator((src_lat, src_lon), x[k])(pts).reshape(180, 360)
        fc[lbl] = out
    return fc, n_mem


def read_ecmwf(init, hindcast=False):
    """ECMWF S2S (1.5 deg, 46 daily steps). Lead day L (valid date init+L):
         t2m  : stepType avg, step 24(L+1) h = mean over that day
         tp   : accumulated from init (kg m-2 = mm) -> step 24L - step 24(L-1) = 24 h total ending at
                00 UTC of day L (matches ERA5 tp24 at 00 UTC), /1000 -> m/day
         z500 : gh at 500 hPa (gpm), instantaneous at step 24L, x g -> geopotential
         u850, v850 : instantaneous at step 24L
       Members: control first, then perturbed 1, 2, ...; capped at N_MEMBERS.
       hindcast=True reads the 2025-model reforecasts: the init's year is selected from the file's time axis."""
    import xarray as xr
    d = MODELS['ecmwf']['dir']
    n_lead = 7 * N_WEEKS
    n_pf = None if N_MEMBERS is None else N_MEMBERS - 1
    hours = lambda steps: [np.timedelta64(int(h), 'h') for h in steps]

    def load(group, var, steps, level=None):
        out = []
        for e in ('cf', 'pf'):
            path = hc_file(group, e, init) if hindcast else f'{d}/ecmwf_{group}_{e}_{init}.grib'
            ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
            a = ds[var]
            if hindcast:
                a = a.sel(time=np.datetime64(f'{init[:4]}-{init[4:6]}-{init[6:]}'))
            a = a.sel(step=hours(steps))
            if level is not None:
                a = a.sel(isobaricInhPa=level)
            if e == 'cf':
                a = a.expand_dims(number=[0])
            else:
                a = a.sortby('number').isel(number=slice(0, n_pf))
            out.append(np.asarray(a.values, dtype=np.float64))   # (member, lead, lat, lon)
            ds.close()
        x = np.concatenate(out, axis=0)
        return x, x.shape[0]

    L = np.arange(1, n_lead + 1)
    t2m, n_mem = load('t2m', 't2m', 24 * (L + 1))
    tp_hi, _ = load('tp', 'tp', 24 * L)
    tp_lo, _ = load('tp', 'tp', 24 * (L - 1))
    gh, _ = load('pl500_z' if hindcast else 'pl500_z_u', 'gh', 24 * L)
    u, _ = load('uv200_850', 'u', 24 * L, level=850)
    v, _ = load('uv200_850', 'v', 24 * L, level=850)
    fields = {'t2m': t2m, 'tp': (tp_hi - tp_lo) / 1000.0, 'z500': gh * 9.80665, 'u850': u, 'v850': v}
    src_lat = np.linspace(90.0, -90.0, 121)
    src_lon = np.arange(0.0, 360.0, 1.5)
    mean = np.stack([fields[lbl].mean(axis=0) for lbl in fields])    # ensemble mean first (interp is linear)
    reg = _regrid_fuxi(mean, src_lat, src_lon)
    fc = {lbl: reg[i] for i, lbl in enumerate(fields)}
    for lbl in LABELS:
        fc.setdefault(lbl, np.full((n_lead, 180, 360), np.nan))
    return fc, n_mem


def _extract_7z(archive, dest):
    if shutil.which('7zr'):
        r = subprocess.run(['7zr', 'x', archive, '-o' + dest, '-y', '-aoa'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
    else:
        import py7zr
        with py7zr.SevenZipFile(archive, mode='r') as z:
            z.extractall(path=dest)


def _regrid_fuxi(x, src_lat, src_lon):
    """(..., 121, 240) native 1.5deg (lat N->S) -> ERA5 1deg (180, 360), bilinear, periodic lon."""
    from scipy.interpolate import RegularGridInterpolator
    if src_lat[0] > src_lat[-1]:
        src_lat, x = src_lat[::-1], x[..., ::-1, :]
    src_lon = np.append(src_lon, 360.0)
    x = np.concatenate([x, x[..., :1]], axis=-1)
    LA, LO = np.meshgrid(lat, np.arange(0.0, 360.0, 1.0), indexing='ij')
    pts = np.stack([LA.ravel(), LO.ravel()], axis=-1)
    out = np.empty(x.shape[:-2] + (180, 360))
    for idx in np.ndindex(x.shape[:-2]):
        f = RegularGridInterpolator((src_lat, src_lon), x[idx], method='linear')
        out[idx] = f(pts).reshape(180, 360)
    return out


def read_fuxi(init):
    archive = os.path.join(MODELS['fuxi']['dir'], f'{init}.7z')
    tmp = tempfile.mkdtemp(prefix=f'tcc_fuxi_{init}_', dir=TMP_DIR)
    n_lead = 7 * N_WEEKS
    try:
        _extract_7z(archive, tmp)
        # 2002/2017 archives nest under {YYYY}/{date}/member; 2019-2021 use {date}/member
        base = next(p for p in (os.path.join(tmp, init[:4], init, 'member'),
                                os.path.join(tmp, init, 'member')) if os.path.isdir(p))
        members = sorted((m for m in os.listdir(base) if os.path.isdir(os.path.join(base, m))), key=int)
        acc = None
        n = 0
        for m in members:
            if N_MEMBERS is not None and n >= N_MEMBERS:
                break
            mdir = os.path.join(base, m)
            if not all(os.path.exists(os.path.join(mdir, f'{L:02d}.nc')) for L in range(1, n_lead + 1)):
                print(f'  {init} member {m}: incomplete leads, skipped', flush=True)
                continue
            for L in range(1, n_lead + 1):
                with netCDF4.Dataset(os.path.join(mdir, f'{L:02d}.nc')) as nc:
                    if acc is None:
                        chans = [str(c) for c in nc.variables['channel'][:]]
                        ci = [chans.index(v[4]) for v in VARIABLES]
                        src_lat = np.asarray(nc.variables['lat'][:])
                        src_lon = np.asarray(nc.variables['lon'][:])
                        acc = np.zeros((len(VARIABLES), n_lead) + (len(src_lat), len(src_lon)))
                    lt = int(nc.variables['lead_time'][0])
                    assert lt == L, f'{init} m{m}: file {L:02d} has lead_time {lt}'
                    x = nc.variables['__xarray_dataarray_variable__'][0, 0, :, :, :]
                    acc[:, L - 1] += np.asarray(x[ci], dtype=np.float64)
            n += 1
        if n == 0:
            return None, 0
        acc = _regrid_fuxi(acc / n, src_lat, src_lon)
        return {lbl: acc[i] * VARIABLES[i][5] for i, lbl in enumerate(LABELS)}, n
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def process_init(args):
    global N_MEMBERS
    model, init, N_MEMBERS = args
    out = os.path.join(cache_dir(model), f'{init}.npz')
    if os.path.exists(out):
        return init, 'cached'
    try:
        fc, n_mem = {'ucs2s_v2': read_ucs2s, 'ucs2s_v2_fbcorr': lambda i: read_ucs2s(i, 'ucs2s_v2_fbcorr'),
                      'ucs2s_d': lambda i: read_ucs2s(i, 'ucs2s_d'),
                      'ucs2s_d_ft': lambda i: read_ucs2s(i, 'ucs2s_d_ft'),
                      'ucs2s_v2_e0': lambda i: read_ucs2s(i, 'ucs2s_v2_e0'),
                      'ucs2s_l': lambda i: read_ucs2s(i, 'ucs2s_l'),
                      'ucs2s_sstanom': lambda i: read_ucs2s(i, 'ucs2s_sstanom'),
                      'ucs2s_v2_m4': lambda i: read_ucs2s(i, 'ucs2s_v2_m4'),
                      'ucs2s_sstanom_evo': lambda i: read_ucs2s(i, 'ucs2s_sstanom_evo'),
                      'ucs2s_multicond': lambda i: read_ucs2s(i, 'ucs2s_multicond'),
                      'ucs2s_multicond20': lambda i: read_ucs2s(i, 'ucs2s_multicond20'),
                      'ucs2s_multicond2020': lambda i: read_ucs2s(i, 'ucs2s_multicond2020'),
                      'ucs2s_multicond_new': lambda i: read_ucs2s(i, 'ucs2s_multicond_new'),
                      'ucs2s_mc_pp12800': lambda i: read_ucs2s(i, 'ucs2s_mc_pp12800'),
                      'ucs2s_mc_pp12800f': lambda i: read_ucs2s(i, 'ucs2s_mc_pp12800f'),
                      'ucs2s_sstanom3y': lambda i: read_ucs2s(i, 'ucs2s_sstanom3y'),
                      'ucs2s_sstanom3y_e0': lambda i: read_ucs2s(i, 'ucs2s_sstanom3y_e0'),
                      'ucs2s_multicond2019n': lambda i: read_ucs2s(i, 'ucs2s_multicond2019n'),
                      'ucs2s_multicond2021': lambda i: read_ucs2s(i, 'ucs2s_multicond2021'),
                      'fuxi': read_fuxi, 'si_s2s': read_si_s2s,
                      'ngcm': read_ngcm, 'ecmwf': read_ecmwf,
                      'ecmwf_rf': lambda i: read_ecmwf(i, hindcast=True)}[model](init)
        if fc is None:
            return init, 'no forecast data'
        if N_MEMBERS is not None and n_mem < N_MEMBERS:
            return init, f'SKIPPED: only {n_mem} members (< {N_MEMBERS})'
        t0 = datetime.strptime(init, '%Y%m%d')
        era5s = {True: ERA5(True), False: ERA5(False)}   # daily-mean / 00Z truth, chosen per variable
        f_wk = np.full((len(LABELS), N_WEEKS, 180, 360), np.nan, dtype=np.float32)
        o_wk = np.full_like(f_wk, np.nan)
        try:
            for i, (lbl, subdir, level, *_) in enumerate(VARIABLES):
                obs = []
                for L in range(1, 7 * N_WEEKS + 1):
                    daily, off = truth_spec(model, lbl)
                    o = era5s[daily].get(subdir, level, t0 + timedelta(days=L + off))
                    if o is None:
                        break
                    obs.append(o)
                for k in range(N_WEEKS):
                    if len(obs) >= 7 * (k + 1):
                        sl = slice(7 * k, 7 * (k + 1))
                        f_wk[i, k] = fc[lbl][sl].mean(axis=0)
                        o_wk[i, k] = np.mean(obs[sl], axis=0)
        finally:
            for e in era5s.values():
                e.close()
        np.savez(out + '.tmp.npz', fc=f_wk, obs=o_wk, n_members=n_mem, labels=LABELS)
        os.replace(out + '.tmp.npz', out)
        return init, f'ok ({n_mem} members)'
    except Exception as e:
        return init, f'ERROR {type(e).__name__}: {e}'


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def weekly_clim(clim, init, offset=0):
    """(N_WEEKS, 180, 360) weekly mean of the DOY climatology for this init (truth days shifted by offset)."""
    t0 = datetime.strptime(init, '%Y%m%d')
    doys = [doy_index(t0 + timedelta(days=L + offset)) for L in range(1, 7 * N_WEEKS + 1)]
    return np.stack([clim[doys[7 * k:7 * (k + 1)]].mean(axis=0) for k in range(N_WEEKS)])


def weekly_trend_clim(tc, init, offset=0):
    """(N_WEEKS, 180, 360) weekly mean of the 2002-2016 linear-trend climatology, evaluated at each
    valid day's year (extrapolated beyond 2016)."""
    t0 = datetime.strptime(init, '%Y%m%d')
    days = [t0 + timedelta(days=L + offset) for L in range(1, 7 * N_WEEKS + 1)]
    out = []
    for k in range(N_WEEKS):
        vals = [tc['ybar'][doy_index(d)] + tc['slope'][doy_index(d)] * (d.year - tc['xbar'][doy_index(d)])
                for d in days[7 * k:7 * (k + 1)]]
        out.append(np.mean(vals, axis=0))
    return np.stack(out)


def pearson(a, b):
    """Correlation along axis 0, NaN where either series is constant."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (a * b).sum(axis=0) / den, np.nan)


def load_anomalies(model, dates):
    """Weekly anomalies vs the model's ERA5 climatology for every cached init.
    Returns (used_dates, fa, oa); fa/oa are (N, var, week, lat, lon) with WEEK_NAMES weeks."""
    if DETREND:
        clims = {lbl: dict(np.load(os.path.join(
            TREND_DIR, f"{'dailymean' if truth_is_daily(model, lbl) else '00z'}_{lbl}.npz"))) for lbl in LABELS}
        clim_fn = weekly_trend_clim
    else:
        # same climatology type as the truth for each variable (daily mean or 00Z)
        clims = {lbl: np.load(clim_file(model, lbl)) for lbl in LABELS}
        clim_fn = weekly_clim

    used, fa, oa = [], [], []
    for init in dates:
        real = MODELS[model].get('date_map', {}).get(init, init)   # cache file / climatology date
        p = os.path.join(cache_dir(model), f'{real}.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p)
        assert list(z['labels']) == LABELS, f'{p}: cached variables {list(z["labels"])} != {LABELS}; delete the cache'
        f, o = z['fc'].astype(np.float64), z['obs'].astype(np.float64)
        if np.isnan(f).all() or np.isnan(o).any():
            print(f'  skip {init}: incomplete weeks')
            continue
        c = np.stack([clim_fn(clims[lbl], real, truth_spec(model, lbl)[1]) for lbl in LABELS])
        fa.append(f - c)
        oa.append(o - c)
        used.append(init)
    fa, oa = np.stack(fa), np.stack(oa)          # (N, var, week, lat, lon)
    if MONTHLY_MEAN:
        months = np.array([int(d[4:6]) for d in used])
        for mo in np.unique(months):
            idx = np.where(months == mo)[0]
            if len(idx) < 2:
                print(f'  WARNING: month {mo:02d} has only {len(idx)} init; its anomaly becomes 0')
            fa[idx] -= fa[idx].mean(axis=0)
            oa[idx] -= oa[idx].mean(axis=0)
    if YEARLY_MEAN:
        years = np.array([d[:4] for d in used])
        for yr in np.unique(years):
            idx = np.where(years == yr)[0]
            fa[idx] -= fa[idx].mean(axis=0)
            oa[idx] -= oa[idx].mean(axis=0)
    # weeks 3-4 / 5-6 = mean of weekly means (lead days 15-28 / 29-42)
    fa = np.concatenate([fa, fa[:, :, 2:4].mean(axis=2, keepdims=True), fa[:, :, 4:6].mean(axis=2, keepdims=True)], axis=2)
    oa = np.concatenate([oa, oa[:, :, 2:4].mean(axis=2, keepdims=True), oa[:, :, 4:6].mean(axis=2, keepdims=True)], axis=2)
    return used, fa, oa


def aggregate(model, dates, tag):
    used, fa, oa = load_anomalies(model, dates)
    n = len(used)
    print(f'{MODELS[model]["name"]}: {n} init dates used')

    tcc = pearson(fa, oa).astype(np.float32)     # (var, week, lat, lon)
    for lbl in MODELS[model].get('missing', ()):
        tcc[LABELS.index(lbl)] = np.nan
    regional = {lbl: {r: np.array([np.nan if np.isnan(tcc[i, k][m]).all() else
                                   np.nansum(W2D[m] * tcc[i, k][m]) / np.sum(W2D[m][~np.isnan(tcc[i, k][m])])
                                   for k in range(len(WEEK_NAMES))])
                      for r, m in MASKS.items()}
                for i, lbl in enumerate(LABELS)}

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f'tcc_weekly_{model}_{tag}.nc')
    with netCDF4.Dataset(out, 'w') as nc:
        nc.createDimension('var', len(LABELS))
        nc.createDimension('week', len(WEEK_NAMES))
        nc.createDimension('lat', 180)
        nc.createDimension('lon', 360)
        nc.createVariable('var', str, ('var',))[:] = np.array(LABELS, dtype=object)
        nc.createVariable('week', str, ('week',))[:] = np.array(WEEK_NAMES, dtype=object)
        nc.createVariable('lat', 'f4', ('lat',))[:] = lat
        nc.createVariable('lon', 'f4', ('lon',))[:] = np.arange(0.0, 360.0, 1.0)
        v = nc.createVariable('tcc', 'f4', ('var', 'week', 'lat', 'lon'), zlib=True)
        v[:] = tcc
        v.long_name = 'temporal correlation of weekly-mean ensemble-mean anomaly vs ERA5'
        nc.model = MODELS[model]['name']
        nc.n_init = n
        nc.init_dates = ','.join(used)
        nc.truth = ', '.join(f"{lbl}: ERA5 {'daily mean' if truth_spec(model, lbl)[0] else '00Z'}"
                             f"{' +' + str(truth_spec(model, lbl)[1]) + 'd' if truth_spec(model, lbl)[1] else ''}"
                             for lbl in LABELS)
    with open(out.replace('.nc', '.pkl'), 'wb') as f:
        pickle.dump({'regional': regional, 'weeks': WEEK_NAMES, 'n_init': n,
                     'init_dates': used, 'model': MODELS[model]['name']}, f)
    print(f'Saved {out}')

    print(f'\n{"":6s}{"region":9s}' + ''.join(f'{w:>9s}' for w in WEEK_NAMES))
    for lbl in LABELS:
        for r in REGIONS:
            print(f'{lbl:6s}{r:9s}' + ''.join(f'{x:9.3f}' for x in regional[lbl][r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--stage', required=True, choices=['extract', 'aggregate'])
    ap.add_argument('--dates-file', default=os.path.join(HERE, 'common_dates_ucs2sv2_fuxi.txt'),
                    help="YYYYMMDD per line; 'all' = every init the model has")
    ap.add_argument('--tag', default='common86')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n-members', type=int, default=None,
                    help='use only the first N ensemble members (separate cache when N < native size)')
    ap.add_argument('--monthly-mean', action='store_true',
                    help="subtract the sample's own mean for each init calendar month (forecast, ERA5 separately)")
    ap.add_argument('--yearly-mean', action='store_true',
                    help="subtract the sample's own mean for each init year (forecast, ERA5 separately)")
    ap.add_argument('--detrend', action='store_true',
                    help='anomalies vs the 2002-2016 linear-trend ERA5 climatology (aggregate stage)')
    a = ap.parse_args()
    global N_MEMBERS, DETREND, MONTHLY_MEAN, YEARLY_MEAN
    N_MEMBERS = a.n_members
    DETREND = a.detrend
    MONTHLY_MEAN = a.monthly_mean
    YEARLY_MEAN = a.yearly_mean

    avail = list_dates(a.model)
    if a.dates_file == 'all':
        dates = avail
    else:
        want = {l.strip() for l in open(a.dates_file) if l.strip()}
        dates = sorted(want & set(avail))
        missing = want - set(avail)
        if missing:
            print(f'WARNING: {len(missing)} requested dates not in {a.model}: {sorted(missing)[:5]}...')
    print(f'{a.model}: {len(dates)} init dates, stage={a.stage}', flush=True)

    if a.stage == 'extract':
        os.makedirs(cache_dir(a.model), exist_ok=True)
        with Pool(a.workers) as p:
            jobs = [(a.model, d, N_MEMBERS) for d in dates]
            for k, (init, msg) in enumerate(p.imap_unordered(process_init, jobs), 1):
                print(f'[{k}/{len(dates)}] {init}: {msg}', flush=True)
    else:
        aggregate(a.model, dates, a.tag)


if __name__ == '__main__':
    main()
