"""
Compute ACC for amip_s2s_train_scratch predictions vs ERA5.
Restricted to init dates matching ours_si_s2s_sampling_temperature_15_tg_01.
Produces two outputs:
  - results_amip_s2s_train/acc_summary.pkl      (ensemble mean across all seeds/members)
  - results_amip_s2s_train_1mem/acc_summary.pkl (member 0 of seed 0 only)

File layout: amip_s2s_train_scratch/YYYYMMDD/ensemble_YYYYMMDD_seedN.nc
  - shape (4_members, 46_leads, [level,] 180, 360)
  - lat: south-to-north (-89.5 ... 89.5), same as ERA5 and clim
  - plev in Pa (500 hPa = 50000 Pa)
  - precip var: PRATEsfc_24h
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
PRED_DIR     = '/glade/derecho/scratch/bgong/amip_s2s_train_scratch'
OURS_DIR     = '/glade/derecho/scratch/bgong/ours_si_s2s_sampling_temperature_15_tg_01'
ERA5_DIR     = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_DIR     = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_00z/'
RESULTS_ENSMEAN = '/glade/work/bgong/benchmark-dev/metrics/ACC/results_amip_s2s_train/'
RESULTS_1MEM    = '/glade/work/bgong/benchmark-dev/metrics/ACC/results_amip_s2s_train_1mem/'
MAX_LEAD     = 45

# (nc_var_name_in_pred, level_Pa_or_None, short_label, era5_subdir, era5_var_name)
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
    'PRATEsfc_24h':             'tp',
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
    """Return AMIP init dates (YYYYMMDD) that match ours init dates."""
    ours_prefix = 'pangu_plasim_finetune_rollout_crps_24h_45step_'
    ours_suffix = '_ens_0.nc'
    ours_dates = set()
    for f in os.listdir(OURS_DIR):
        if f.startswith(ours_prefix) and f.endswith(ours_suffix):
            init_str = f.replace(ours_prefix, '').replace(ours_suffix, '')
            ours_dates.add(init_str[:8])  # YYYYMMDD (drop HH)

    return sorted([d for d in os.listdir(PRED_DIR)
                   if os.path.isdir(os.path.join(PRED_DIR, d))
                   and len(d) == 8 and d.isdigit()
                   and d in ours_dates])

def seed_files_for_init(init_str):
    init_dir = os.path.join(PRED_DIR, init_str)
    prefix = f'ensemble_{init_str}_seed'
    return sorted(
        [os.path.join(init_dir, f) for f in os.listdir(init_dir)
         if f.startswith(prefix) and f.endswith('.nc')],
        key=lambda x: int(x.split('_seed')[1].replace('.nc', ''))
    )

# ==================== Worker ====================
def process_init_date(init_str):
    """Returns (ensmean_result, onemem_result) dicts."""
    empty = lambda: {label: {r: {} for r in REGION_CONFIG}
                     for _, _, label, _, _ in VARIABLES}
    res_mean = empty()
    res_1mem = empty()

    era5_handles = {}

    def get_era5_gt(era5_var, level_hpa, era5_subdir, date):
        year = date.year
        fpath = f'{ERA5_DIR}/{era5_subdir}/{year}_180x360.nc'
        if not os.path.exists(fpath):
            return None
        key = (fpath, era5_var)
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
            data = nc.variables[era5_var][indices[0], li, :, :]
        else:
            data = nc.variables[era5_var][indices[0], :, :]
        return np.array(data).astype(np.float64)

    try:
        files = seed_files_for_init(init_str)
        if not files:
            return res_mean, res_1mem

        for var_name, level_pa, label, era5_subdir, era5_var in VARIABLES:
            accum = None
            count = 0
            lead_dates = None
            one_mem_data = None  # member 0 of seed 0

            for fi, fp in enumerate(files):
                try:
                    with netCDF4.Dataset(fp) as nc:
                        if lead_dates is None:
                            t_vals = nc.variables['time'][:]  # days since 1850-01-01
                            base = datetime(1850, 1, 1)
                            lead_dates = [base + timedelta(days=int(d)) for d in t_vals]

                        if level_pa is not None:
                            plev = nc.variables['plev'][:]
                            li   = int(np.where(plev == level_pa)[0][0])
                            data = nc.variables[var_name][:, :, li, :, :].astype(np.float64)
                        else:
                            data = nc.variables[var_name][:, :, :, :].astype(np.float64)
                        # data shape: (n_members, n_leads, 180, 360)

                        if fi == 0:
                            # Save member 0 from seed 0 for the 1-member output
                            one_mem_data = data[0]  # (n_leads, 180, 360)

                        data_mean = data.mean(axis=0)  # (n_leads, 180, 360)
                        accum = data_mean if accum is None else accum + data_mean
                        count += 1
                except Exception as e:
                    print(f"  Warning: skipping {fp} var={var_name}: {e}")

            if accum is None or count == 0 or lead_dates is None:
                continue

            ens_mean = accum / count  # (n_leads, 180, 360)
            level_hpa = int(level_pa / 100) if level_pa is not None else None

            for li, date in enumerate(lead_dates):
                if li > MAX_LEAD:
                    break
                gt_val = get_era5_gt(era5_var, level_hpa, era5_subdir, date)
                if gt_val is None:
                    continue
                clim = get_clim(var_name, date)
                gt_anom = gt_val - clim

                pred_anom_mean = ens_mean[li] - clim
                for region, mask in MASKS.items():
                    res_mean[label][region][li] = weighted_acc(pred_anom_mean, gt_anom, mask)

                if one_mem_data is not None:
                    pred_anom_1m = one_mem_data[li] - clim
                    for region, mask in MASKS.items():
                        res_1mem[label][region][li] = weighted_acc(pred_anom_1m, gt_anom, mask)
    finally:
        for (nc, _) in era5_handles.values():
            nc.close()

    return res_mean, res_1mem

# ==================== Main ====================
os.makedirs(RESULTS_ENSMEAN, exist_ok=True)
os.makedirs(RESULTS_1MEM, exist_ok=True)

init_dates = list_init_dates()
print(f"\nFound {len(init_dates)} matched init dates: {init_dates[0]} ... {init_dates[-1]}\n")

N_WORKERS = min(len(init_dates), cpu_count())
print(f"Using {N_WORKERS} parallel workers\n")

def empty_acc_data():
    return {label: {r: defaultdict(list) for r in REGION_CONFIG}
            for _, _, label, _, _ in VARIABLES}

acc_data_mean = empty_acc_data()
acc_data_1mem = empty_acc_data()

with Pool(N_WORKERS) as pool:
    for res_mean, res_1mem in tqdm(
            pool.imap_unordered(process_init_date, init_dates),
            total=len(init_dates), desc='Init dates'):
        for label in res_mean:
            for region in res_mean[label]:
                for ld, acc in res_mean[label][region].items():
                    acc_data_mean[label][region][ld].append(acc)
        for label in res_1mem:
            for region in res_1mem[label]:
                for ld, acc in res_1mem[label][region].items():
                    acc_data_1mem[label][region][ld].append(acc)

# ==================== Aggregate & Save ====================
def summarize_and_save(acc_data, out_dir):
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
    out_file = os.path.join(out_dir, 'acc_summary.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump({'acc_summary': acc_summary, 'acc_data': acc_data}, f)
    print(f"Saved: {out_file}")
    for label in acc_summary:
        print(f"\n=== {label} ===")
        for region in REGION_CONFIG:
            arr = acc_summary[label][region]
            skill_days = np.where(arr >= 0.6)[0]
            threshold_day = int(skill_days[-1]) + 1 if len(skill_days) > 0 else 0
            print(f"  {region:8s}: ACC@7d={arr[6]:.3f}  ACC@14d={arr[13]:.3f}  "
                  f"days>=0.6: {threshold_day}")

print("\n===== ENSEMBLE MEAN =====")
summarize_and_save(acc_data_mean, RESULTS_ENSMEAN)

print("\n===== 1 MEMBER (seed0, member0) =====")
summarize_and_save(acc_data_1mem, RESULTS_1MEM)
