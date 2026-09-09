"""
Compute ACC for a single ensemble member (ens_0) of ours_si_s2s_sampling_temperature_15_tg_01.
File layout: pangu_plasim_finetune_rollout_crps_24h_45step_{YYYYMMDDHHH}_ens_0.nc
  - shape (46_leads, [level,] 180, 360)
  - latitude: south-to-north -> flipped to north-to-south to match ERA5/clim
  - level in hPa
  - time units: days since init date
"""

import numpy as np
import netCDF4
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==================== Configuration ====================
PRED_DIR    = '/glade/derecho/scratch/bgong/ours_si_s2s_sampling_temperature_15_tg_01'
ERA5_DIR    = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_DIR    = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_00z/'
RESULTS_DIR = '/glade/work/bgong/benchmark-dev/metrics/ACC/results_ours_si_1member/'
MAX_LEAD    = 45
MEMBER      = 0   # which ens member to use

# (nc_var_name, level_hPa_or_None, short_label, era5_subdir)
VARIABLES = [
    ('2m_temperature',           None, 't2m',  '2m_temperature'),
    ('geopotential',             500,  'z500', 'geopotential'),
    ('total_precipitation_24hr', None, 'tp',   'total_precipitation_24hr'),
    ('10m_u_component_of_wind',  None, 'u10',  '10m_u_component_of_wind'),
    ('10m_v_component_of_wind',  None, 'v10',  '10m_v_component_of_wind'),
]

REGION_CONFIG = {
    'global':  {'lat_min': -90, 'lat_max':  90},
    'tropics': {'lat_min': -30, 'lat_max':  30},
    'NH':      {'lat_min':  30, 'lat_max':  90},
    'SH':      {'lat_min': -90, 'lat_max': -30},
}

# ==================== Grid & Weights ====================
# Data, ERA5, and clim are all south-to-north (-89.5 ... 89.5)
lat = np.linspace(-89.5, 89.5, 180)
W2D = np.outer(np.cos(np.deg2rad(lat)), np.ones(360))

def regional_mask(lat_min, lat_max):
    return np.outer((lat >= lat_min) & (lat <= lat_max), np.ones(360, dtype=bool))

MASKS = {r: regional_mask(cfg['lat_min'], cfg['lat_max'])
         for r, cfg in REGION_CONFIG.items()}

def weighted_acc(pred_anom, gt_anom, mask):
    a = pred_anom[mask]
    b = gt_anom[mask]
    wm = W2D[mask]
    num = np.sum(wm * a * b)
    den = np.sqrt(np.sum(wm * a**2) * np.sum(wm * b**2))
    return float(num / den) if den > 0 else np.nan

# ==================== Climatology ====================
print("Loading 00Z climatology...")
CLIM = {}
VAR_CLIM_MAP = {
    '2m_temperature':           't2m',
    'geopotential':             'z500',
    'total_precipitation_24hr': 'tp',
    '10m_u_component_of_wind':  'u10',
    '10m_v_component_of_wind':  'v10',
}
for var_name, lbl in VAR_CLIM_MAP.items():
    fpath = os.path.join(CLIM_DIR, f'clim_00z_{lbl}.npy')
    CLIM[var_name] = np.load(fpath)   # (366, 180, 360) south-to-north
    print(f"  Loaded {lbl}: {CLIM[var_name].shape}")

def doy_index(date):
    if date.month == 2 and date.day == 29:
        return 59
    return date.timetuple().tm_yday - 1

def get_clim(var_name, date):
    return CLIM[var_name][doy_index(date)]

# ==================== Init Dates ====================
def list_init_dates():
    suffix = f'_ens_{MEMBER}.nc'
    prefix = 'pangu_plasim_finetune_rollout_crps_24h_45step_'
    files = [f for f in os.listdir(PRED_DIR) if f.startswith(prefix) and f.endswith(suffix)]
    dates = sorted(set(f.replace(prefix, '').replace(suffix, '') for f in files))
    return dates  # e.g. '2019050200'

def member_file(init_str):
    fname = f'pangu_plasim_finetune_rollout_crps_24h_45step_{init_str}_ens_{MEMBER}.nc'
    return os.path.join(PRED_DIR, fname)

# ==================== Worker ====================
def process_init_date(init_str):
    result = {label: {r: {} for r in REGION_CONFIG}
              for _, _, label, _ in VARIABLES}

    era5_handles = {}

    def get_era5_gt(var_name, level_hpa, era5_subdir, date):
        year = date.year
        fpath = f'{ERA5_DIR}/{era5_subdir}/{year}_180x360.nc'
        if not os.path.exists(fpath):
            return None
        key = (fpath, var_name)
        if key not in era5_handles:
            nc = netCDF4.Dataset(fpath)
            t_hrs = nc.variables['time'][:]
            base  = datetime(1900, 1, 1)
            t_dates = [base + timedelta(hours=float(h)) for h in t_hrs]
            era5_handles[key] = (nc, t_dates)
        nc, t_dates = era5_handles[key]
        indices = [i for i, d in enumerate(t_dates) if d == date]
        if not indices:
            return None
        if level_hpa is not None:
            levs = nc.variables['level'][:]
            li   = int(np.where(levs == level_hpa)[0][0])
            data = nc.variables[var_name][indices[0], li, :, :]
        else:
            data = nc.variables[var_name][indices[0], :, :]
        return np.array(data).astype(np.float64)  # south-to-north

    try:
        fp = member_file(init_str)
        if not os.path.exists(fp):
            return result

        with netCDF4.Dataset(fp) as nc:
            t_vals  = nc.variables['time'][:]        # days since init
            t_units = nc.variables['time'].units     # "days since YYYY-MM-DD ..."
            base    = datetime.strptime(t_units.split('since')[1].strip()[:10], '%Y-%m-%d')
            lead_dates = [base + timedelta(days=int(d)) for d in t_vals]

            for var_name, level_hpa, label, era5_subdir in VARIABLES:
                if level_hpa is not None:
                    levs = nc.variables['level'][:]
                    li   = int(np.where(levs == level_hpa)[0][0])
                    data = nc.variables[var_name][:, li, :, :].astype(np.float64)
                else:
                    data = nc.variables[var_name][:].astype(np.float64)

                for li_t, date in enumerate(lead_dates):
                    if li_t > MAX_LEAD:
                        break
                    gt_val = get_era5_gt(var_name, level_hpa, era5_subdir, date)
                    if gt_val is None:
                        continue
                    clim = get_clim(var_name, date)
                    pred_anom = data[li_t] - clim
                    gt_anom   = gt_val - clim
                    for region, mask in MASKS.items():
                        result[label][region][li_t] = weighted_acc(pred_anom, gt_anom, mask)
    finally:
        for (nc, _) in era5_handles.values():
            nc.close()

    return result

# ==================== Main ====================
os.makedirs(RESULTS_DIR, exist_ok=True)

init_dates = list_init_dates()
print(f"\nFound {len(init_dates)} init dates: {init_dates[0]} ... {init_dates[-1]}\n")

N_WORKERS = min(len(init_dates), cpu_count())
print(f"Using {N_WORKERS} parallel workers\n")

acc_data = {label: {r: defaultdict(list) for r in REGION_CONFIG}
            for _, _, label, _ in VARIABLES}

with Pool(N_WORKERS) as pool:
    for init_result in tqdm(
            pool.imap_unordered(process_init_date, init_dates),
            total=len(init_dates), desc='Init dates'):
        for label in init_result:
            for region in init_result[label]:
                for lead_day, acc in init_result[label][region].items():
                    acc_data[label][region][lead_day].append(acc)

# ==================== Aggregate & Save ====================
acc_summary = {}
for label in acc_data:
    acc_summary[label] = {}
    for region in REGION_CONFIG:
        arr = np.full(MAX_LEAD + 1, np.nan)
        for ld in range(0, MAX_LEAD + 1):
            vals = acc_data[label][region][ld]
            if vals:
                arr[ld] = np.nanmean(vals)
        acc_summary[label][region] = arr

out_file = os.path.join(RESULTS_DIR, 'acc_summary.pkl')
with open(out_file, 'wb') as f:
    pickle.dump({'acc_summary': acc_summary, 'acc_data': acc_data}, f)
print(f"\nSaved: {out_file}")

# ==================== Print Summary ====================
for label in acc_summary:
    print(f"\n=== {label} ===")
    for region in REGION_CONFIG:
        arr = acc_summary[label][region]
        skill_days = np.where(arr >= 0.6)[0]
        threshold_day = int(skill_days[-1]) + 1 if len(skill_days) > 0 else 0
        print(f"  {region:8s}: ACC@7d={arr[6]:.3f}  ACC@14d={arr[13]:.3f}  "
              f"days>=0.6: {threshold_day}")
