"""
Compute ACC for ours_si_postprocess_v2 ensemble mean vs ERA5.
File layout: ours_si_postprocess_v2/predictions/pangu_plasim_pp_si_infer_v2_24h_45step_{YYYYMMDDHHH}_ens_N.nc
"""

import numpy as np
import netCDF4
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PRED_DIR    = '/glade/derecho/scratch/bgong/ours_si_postprocess_v2/predictions'
ERA5_DIR    = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_DIR    = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_00z/'
RESULTS_DIR = '/glade/work/bgong/benchmark-dev/metrics/ACC/results_ours_si_postprocess_v2/'
MAX_LEAD    = 45

VARIABLES = [
    ('2m_temperature',           None, 't2m',  '2m_temperature',           '2m_temperature'),
    ('geopotential',             500,  'z500', 'geopotential',             'geopotential'),
    ('total_precipitation_24hr', None, 'tp',   'total_precipitation_24hr', 'total_precipitation_24hr'),
    ('10m_u_component_of_wind',  None, 'u10',  '10m_u_component_of_wind',  '10m_u_component_of_wind'),
    ('10m_v_component_of_wind',  None, 'v10',  '10m_v_component_of_wind',  '10m_v_component_of_wind'),
]

REGION_CONFIG = {
    'global':  {'lat_min': -90, 'lat_max':  90},
    'tropics': {'lat_min': -30, 'lat_max':  30},
    'NH':      {'lat_min':  30, 'lat_max':  90},
    'SH':      {'lat_min': -90, 'lat_max': -30},
}

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
for var_name, lbl in [('2m_temperature','t2m'), ('geopotential','z500'), ('total_precipitation_24hr','tp'),
                      ('10m_u_component_of_wind','u10'), ('10m_v_component_of_wind','v10')]:
    CLIM[var_name] = np.load(os.path.join(CLIM_DIR, f'clim_00z_{lbl}.npy'))
    print(f"  Loaded {lbl}: {CLIM[var_name].shape}")

def doy_index(date):
    if date.month == 2 and date.day == 29:
        return 59
    return date.timetuple().tm_yday - 1

def list_init_dates():
    prefix = 'pangu_plasim_pp_si_infer_v2_24h_45step_'
    suffix = '_ens_0.nc'
    return sorted(set(
        f.replace(prefix, '').replace(suffix, '')
        for f in os.listdir(PRED_DIR)
        if f.startswith(prefix) and f.endswith(suffix)
    ))

def ens_files_for_init(init_str):
    prefix = f'pangu_plasim_pp_si_infer_v2_24h_45step_{init_str}_ens_'
    return sorted(
        [os.path.join(PRED_DIR, f) for f in os.listdir(PRED_DIR)
         if f.startswith(prefix) and f.endswith('.nc')],
        key=lambda x: int(x.split('_ens_')[1].replace('.nc', ''))
    )

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
            li   = int(np.where(levs == level_hpa)[0][0])
            data = nc.variables[era5_var][indices[0], li, :, :]
        else:
            data = nc.variables[era5_var][indices[0], :, :]
        return np.array(data).astype(np.float64)

    try:
        files = ens_files_for_init(init_str)
        if not files:
            return result

        for var_name, level_hpa, label, era5_subdir, era5_var in VARIABLES:
            accum = None
            count = 0
            lead_dates = None

            for fp in files:
                try:
                    with netCDF4.Dataset(fp) as nc:
                        if lead_dates is None:
                            t_vals  = nc.variables['time'][:]
                            t_units = nc.variables['time'].units
                            base    = datetime.strptime(t_units.split('since')[1].strip()[:10], '%Y-%m-%d')
                            lead_dates = [base + timedelta(days=int(d)) for d in t_vals]
                        if level_hpa is not None:
                            levs = nc.variables['level'][:]
                            li   = int(np.where(levs == level_hpa)[0][0])
                            data = nc.variables[var_name][:, li, :, :].astype(np.float64)
                        else:
                            data = nc.variables[var_name][:].astype(np.float64)
                        accum = data if accum is None else accum + data
                        count += 1
                except Exception as e:
                    print(f"  Warning: skipping {fp} var={var_name}: {e}")

            if accum is None or count == 0 or lead_dates is None:
                continue

            ens_mean = accum / count

            for li, date in enumerate(lead_dates):
                if li > MAX_LEAD:
                    break
                gt_val = get_era5_gt(era5_var, level_hpa, era5_subdir, date)
                if gt_val is None:
                    continue
                clim = CLIM[var_name][doy_index(date)]
                pred_anom = ens_mean[li] - clim
                gt_anom   = gt_val - clim
                for region, mask in MASKS.items():
                    result[label][region][li] = weighted_acc(pred_anom, gt_anom, mask)
    finally:
        for (nc, _) in era5_handles.values():
            nc.close()

    return result

os.makedirs(RESULTS_DIR, exist_ok=True)
init_dates = list_init_dates()
print(f"\nFound {len(init_dates)} init dates: {init_dates[0]} ... {init_dates[-1]}\n")

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
