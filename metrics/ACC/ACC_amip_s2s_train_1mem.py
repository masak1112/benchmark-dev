"""
Compute ACC for member 0 of seed 0 of amip_s2s_train_scratch vs ERA5.
Restricted to the 32 init dates matching ours_si_s2s_sampling_temperature_15_tg_01.
Output: results_amip_s2s_train_1mem/acc_summary.pkl
"""

import numpy as np
import netCDF4
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PRED_DIR  = '/glade/derecho/scratch/bgong/amip_s2s_train_scratch'
OURS_DIR  = '/glade/derecho/scratch/bgong/ours_si_s2s_sampling_temperature_15_tg_01'
ERA5_DIR  = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_DIR  = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_00z/'
RESULTS_DIR = '/glade/work/bgong/benchmark-dev/metrics/ACC/results_amip_s2s_train_1mem/'
MAX_LEAD  = 45

VARIABLES = [
    ('2m_temperature',           None,  't2m',  '2m_temperature',           '2m_temperature'),
    ('geopotential',             50000, 'z500', 'geopotential',             'geopotential'),
    ('PRATEsfc_24h',             None,  'tp',   'total_precipitation_24hr', 'total_precipitation_24hr'),
    ('10m_u_component_of_wind',  None,  'u10',  '10m_u_component_of_wind',  '10m_u_component_of_wind'),
    ('10m_v_component_of_wind',  None,  'v10',  '10m_v_component_of_wind',  '10m_v_component_of_wind'),
]

REGION_CONFIG = {
    'global':  {'lat_min': -90, 'lat_max':  90},
    'tropics': {'lat_min': -30, 'lat_max':  30},
    'NH':      {'lat_min':  30, 'lat_max':  90},
    'SH':      {'lat_min': -90, 'lat_max': -30},
}

# Data, ERA5, and clim are all south-to-north (-89.5 ... 89.5)
lat = np.linspace(-89.5, 89.5, 180)
W2D = np.outer(np.cos(np.deg2rad(lat)), np.ones(360))

def regional_mask(lat_min, lat_max):
    return np.outer((lat >= lat_min) & (lat <= lat_max), np.ones(360, dtype=bool))

MASKS = {r: regional_mask(cfg['lat_min'], cfg['lat_max']) for r, cfg in REGION_CONFIG.items()}

def weighted_acc(pred_anom, gt_anom, mask):
    a, b, wm = pred_anom[mask], gt_anom[mask], W2D[mask]
    num = np.sum(wm * a * b)
    den = np.sqrt(np.sum(wm * a**2) * np.sum(wm * b**2))
    return float(num / den) if den > 0 else np.nan

print("Loading climatology...")
CLIM = {}
for var_name, lbl in [('2m_temperature','t2m'),('geopotential','z500'),
                       ('PRATEsfc_24h','tp'),('10m_u_component_of_wind','u10'),
                       ('10m_v_component_of_wind','v10')]:
    CLIM[var_name] = np.load(os.path.join(CLIM_DIR, f'clim_00z_{lbl}.npy'))
    print(f"  Loaded {lbl}: {CLIM[var_name].shape}")

def doy_index(date):
    if date.month == 2 and date.day == 29:
        return 59
    return date.timetuple().tm_yday - 1

def list_init_dates():
    ours_prefix = 'pangu_plasim_finetune_rollout_crps_24h_45step_'
    ours_suffix = '_ens_0.nc'
    ours_dates = set()
    for f in os.listdir(OURS_DIR):
        if f.startswith(ours_prefix) and f.endswith(ours_suffix):
            ours_dates.add(f.replace(ours_prefix, '').replace(ours_suffix, '')[:8])
    return sorted([d for d in os.listdir(PRED_DIR)
                   if os.path.isdir(os.path.join(PRED_DIR, d))
                   and len(d) == 8 and d.isdigit() and d in ours_dates])

def process_init_date(init_str):
    result = {label: {r: {} for r in REGION_CONFIG} for _, _, label, _, _ in VARIABLES}
    era5_handles = {}

    def get_era5_gt(era5_var, level_hpa, era5_subdir, date):
        fpath = f'{ERA5_DIR}/{era5_subdir}/{date.year}_180x360.nc'
        if not os.path.exists(fpath):
            return None
        key = (fpath, era5_var)
        if key not in era5_handles:
            nc = netCDF4.Dataset(fpath)
            t_hrs = nc.variables['time'][:]
            t_dates = [datetime(1900,1,1) + timedelta(hours=float(h)) for h in t_hrs]
            era5_handles[key] = (nc, t_dates)
        nc, t_dates = era5_handles[key]
        indices = [i for i, d in enumerate(t_dates) if d == date]
        if not indices:
            return None
        if level_hpa is not None:
            levs = nc.variables['level'][:]
            li = int(np.where(levs == level_hpa)[0][0])
            data = nc.variables[era5_var][indices[0], li, :, :]
        else:
            data = nc.variables[era5_var][indices[0], :, :]
        return np.array(data).astype(np.float64)

    try:
        # Use member 0 of seed 0 only
        seed0_file = os.path.join(PRED_DIR, init_str, f'ensemble_{init_str}_seed0.nc')
        if not os.path.exists(seed0_file):
            return result

        with netCDF4.Dataset(seed0_file) as nc:
            t_vals = nc.variables['time'][:]
            lead_dates = [datetime(1850,1,1) + timedelta(days=int(d)) for d in t_vals]

            for var_name, level_pa, label, era5_subdir, era5_var in VARIABLES:
                if level_pa is not None:
                    plev = nc.variables['plev'][:]
                    li = int(np.where(plev == level_pa)[0][0])
                    data = nc.variables[var_name][0, :, li, :, :].astype(np.float64)  # member 0
                else:
                    data = nc.variables[var_name][0, :, :, :].astype(np.float64)  # member 0
                # shape: (n_leads, 180, 360), lat already south-to-north

                level_hpa = int(level_pa / 100) if level_pa is not None else None
                for li_t, date in enumerate(lead_dates):
                    if li_t > MAX_LEAD:
                        break
                    gt_val = get_era5_gt(era5_var, level_hpa, era5_subdir, date)
                    if gt_val is None:
                        continue
                    clim = CLIM[var_name][doy_index(date)]
                    pred_anom = data[li_t] - clim
                    gt_anom   = gt_val - clim
                    for region, mask in MASKS.items():
                        result[label][region][li_t] = weighted_acc(pred_anom, gt_anom, mask)
    finally:
        for (nc, _) in era5_handles.values():
            nc.close()

    return result

os.makedirs(RESULTS_DIR, exist_ok=True)
init_dates = list_init_dates()
print(f"\nFound {len(init_dates)} matched init dates: {init_dates[0]} ... {init_dates[-1]}\n")

N_WORKERS = min(len(init_dates), cpu_count())
print(f"Using {N_WORKERS} parallel workers\n")

acc_data = {label: {r: defaultdict(list) for r in REGION_CONFIG} for _, _, label, _, _ in VARIABLES}

with Pool(N_WORKERS) as pool:
    for init_result in tqdm(pool.imap_unordered(process_init_date, init_dates),
                            total=len(init_dates), desc='Init dates'):
        for label in init_result:
            for region in init_result[label]:
                for ld, acc in init_result[label][region].items():
                    acc_data[label][region][ld].append(acc)

acc_summary = {}
for label in acc_data:
    acc_summary[label] = {}
    for region in REGION_CONFIG:
        arr = np.full(MAX_LEAD + 1, np.nan)
        for ld in range(MAX_LEAD + 1):
            vals = acc_data[label][region][ld]
            if vals:
                arr[ld] = np.nanmean(vals)
        acc_summary[label][region] = arr

out_file = os.path.join(RESULTS_DIR, 'acc_summary.pkl')
with open(out_file, 'wb') as f:
    pickle.dump({'acc_summary': acc_summary, 'acc_data': acc_data}, f)
print(f"\nSaved: {out_file}")

for label in acc_summary:
    print(f"\n=== {label} ===")
    for region in REGION_CONFIG:
        arr = acc_summary[label][region]
        skill_days = np.where(arr >= 0.6)[0]
        threshold_day = int(skill_days[-1]) + 1 if len(skill_days) > 0 else 0
        print(f"  {region:8s}: ACC@7d={arr[6]:.3f}  ACC@14d={arr[13]:.3f}  days>=0.6: {threshold_day}")
