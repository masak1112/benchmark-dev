"""
Compute regional ACC metrics for GenCast ensemble mean vs ERA5 at 00Z.
Variables: 2m_temperature, geopotential@500hPa, total_precipitation (12hr->24hr sum),
           u/v wind at 10m not available; skipped.
Grid: GenCast 0.25deg (721x1440) regridded to 1deg (180x360) to match ERA5/climatology.
Init dates matched to: /glade/derecho/scratch/bgong/ours_si_s2s_sampling_temperature_15_tg_01
"""

import numpy as np
import zarr
import netCDF4
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm

# ==================== Configuration ====================
GENCAST_DIR = '/glade/campaign/univ/uchi0018/bing/gencast'
ERA5_DIR    = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_DIR    = './clim_00z/'
RESULTS_DIR = './results_gencast/'
MAX_LEAD    = 45  # days

# Init dates to match ours_si_s2s (same 32 init dates)
PRED_INIT_DATES = [
    '2019050200','2019050600','2019050900','2019051300','2019051600',
    '2019052000','2019052300','2019052700','2019053000','2019060200',
    '2019060600','2019060900','2019061300','2019061600','2019062000',
    '2019062300','2020050200','2020050600','2020050900','2020051300',
    '2020051600','2020052000','2020052300','2020052700','2020053000',
    '2020060200','2020060600','2020060900','2020061300','2020061600',
    '2020062000','2020062300',
]

# Variables: (zarr_var, level_hPa_or_None, short_label, era5_subdir, era5_var)
# Note: GenCast has no 10m u/v wind; precipitation is 12-hourly (sum pairs for 24hr)
VARIABLES = [
    ('2m_temperature',        None, 't2m',  '2m_temperature',           '2m_temperature'),
    ('temperature',           500,  'z500', 'geopotential',             'geopotential'),   # use ERA5 geopotential as GT; GenCast has temperature not geopot at 500
    ('total_precipitation_12hr', None, 'tp', 'total_precipitation_24hr', 'total_precipitation_24hr'),
]
# NOTE: GenCast stores temperature at pressure levels, not geopotential.
# For z500 ACC we use GenCast temperature@500hPa vs ERA5 geopotential@500hPa — units differ.
# Better: skip z500 and only do t2m and tp for GenCast.
# We'll handle geopotential by checking if GenCast has it; otherwise skip.

VARIABLES = [
    ('2m_temperature',           None, 't2m', '2m_temperature',           '2m_temperature'),
    ('total_precipitation_12hr', None, 'tp',  'total_precipitation_24hr', 'total_precipitation_24hr'),
]

REGION_CONFIG = {
    'global':  {'lat_min': -90, 'lat_max':  90},
    'tropics': {'lat_min': -30, 'lat_max':  30},
    'NH':      {'lat_min':  30, 'lat_max':  90},
    'SH':      {'lat_min': -90, 'lat_max': -30},
}

# ==================== 1-deg Grid & Weights ====================
# S->N to match ERA5 yearly files and climatology (both stored S->N)
lat1 = np.linspace(-89.5, 89.5, 180)
lon1 = np.linspace(0.5, 359.5, 360)
W2D  = np.outer(np.cos(np.deg2rad(lat1)), np.ones(360))

def regional_mask(lat_min, lat_max):
    return np.outer((lat1 >= lat_min) & (lat1 <= lat_max), np.ones(360, dtype=bool))

MASKS = {r: regional_mask(cfg['lat_min'], cfg['lat_max'])
         for r, cfg in REGION_CONFIG.items()}

def weighted_acc(pred_anom, gt_anom, mask):
    w = W2D * mask
    a, b, wm = pred_anom[mask], gt_anom[mask], W2D[mask]
    num = np.sum(wm * a * b)
    den = np.sqrt(np.sum(wm * a**2) * np.sum(wm * b**2))
    return float(num / den) if den > 0 else np.nan

# ==================== Regrid 0.25->1deg ====================
# GenCast lat: -90 to 90 (S->N), lon: 0 to 359.75
# ERA5/clim lat: 89.5 to -89.5 (N->S), lon: 0.5 to 359.5
# Simple block averaging: each 1-deg cell = mean of 4x4 0.25-deg cells

def regrid_025_to_1deg(data_025):
    """
    Regrid (721, 1440) 0.25-deg S->N grid to (180, 360) 1-deg N->S grid.
    Flips lat direction and does 4x4 block mean (ignoring the poles row).
    """
    # data_025: shape (721, 1440), lat from -90 to 90 S->N
    # Drop pole rows to get 720 rows, then reshape to (180,4,360,4) and average
    data_no_poles = data_025[1:-1, :]   # (719, 1440) — still not cleanly divisible
    # Use nearest-neighbor sampling at 1-deg centers instead
    # 1-deg centers N->S: 89.5, 88.5, ..., -89.5
    # 0.25-deg centers S->N: -89.875, -89.625, ..., 89.875 (720 values, no poles)
    lat025 = np.linspace(-89.875, 89.875, 720)   # S->N
    lon025 = np.linspace(0.125, 359.875, 1440)

    # For each 1-deg cell, average 0.25-deg cells within ±0.5 deg
    out = np.empty((180, 360), dtype=np.float64)
    data_nopole = data_025[1:-1, :]  # drop exact pole rows -> (719... no, 721-2=719)
    # Actually lat025 spacing is regular 0.25, so use 720 rows (drop both poles)
    # lat025 has 720 elements matching data[1:-1] only if 721-2=719 -- mismatch
    # Simpler: just slice at 4-row intervals after flipping
    data_flip = data_025[::-1, :]    # flip to N->S: (721, 1440)
    # lat flipped: 90, 89.75, ..., -89.75, -90
    # Take every 4th row starting at index 0 (90.0) won't align to 89.5
    # Best: pick rows closest to 89.5, 88.5, ..., -89.5
    lat_flip = np.linspace(90, -90, 721)  # N->S
    row_idx = np.array([np.argmin(np.abs(lat_flip - c)) for c in lat1])   # lat1 = 89.5...-89.5
    col_idx = np.array([np.argmin(np.abs(np.linspace(0, 359.75, 1440) - c)) for c in lon1])
    out = data_flip[np.ix_(row_idx, col_idx)]
    return out.astype(np.float64)

# Precompute regrid indices once
# GenCast lat: S->N (-90 to 90, 721 points); we keep S->N to match ERA5/clim
_lat025 = np.linspace(-90, 90, 721)
_lon025  = np.linspace(0, 359.75, 1440)
_row_idx = np.array([np.argmin(np.abs(_lat025 - c)) for c in lat1])
_col_idx = np.array([np.argmin(np.abs(_lon025 - c)) for c in lon1])

def regrid(data_025):
    """Nearest-neighbor regrid (721,1440) S->N to (180,360) S->N 1-deg."""
    return data_025[np.ix_(_row_idx, _col_idx)].astype(np.float64)

# ==================== Climatology ====================
print("Loading 00Z climatology...")
CLIM = {}
VAR_CLIM_MAP = {'2m_temperature': 't2m', 'total_precipitation_12hr': 'tp'}
for var_name, label in VAR_CLIM_MAP.items():
    fpath = os.path.join(CLIM_DIR, f'clim_00z_{label}.npy')
    CLIM[var_name] = np.load(fpath)
    print(f"  Loaded {label}: {CLIM[var_name].shape}")

def doy_index(date):
    if date.month == 2 and date.day == 29:
        return 59
    return date.timetuple().tm_yday - 1

def get_clim(var_name, date):
    return CLIM[var_name][doy_index(date)]

# ==================== ERA5 Ground Truth ====================
_era5_cache = {}

def load_era5_gt(era5_subdir, era5_var, level_hpa, date):
    year = date.year
    fpath = f'{ERA5_DIR}/{era5_subdir}/{year}_180x360.nc'
    if not os.path.exists(fpath):
        return None
    key = (fpath, era5_var)
    if key not in _era5_cache:
        nc = netCDF4.Dataset(fpath)
        t_hrs = nc.variables['time'][:]
        base  = datetime(1900, 1, 1)
        t_dates = [base + timedelta(hours=float(h)) for h in t_hrs]
        _era5_cache[key] = (nc, t_dates)
    nc, t_dates = _era5_cache[key]
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

def close_era5_cache():
    for (nc, _) in _era5_cache.values():
        nc.close()
    _era5_cache.clear()

# ==================== GenCast Data Loading ====================
_zarr_cache = {}

def get_zarr(year):
    if year not in _zarr_cache:
        _zarr_cache[year] = zarr.open(
            f'{GENCAST_DIR}/gencast_{year}.zarr', mode='r')
    return _zarr_cache[year]

def get_init_time_index(z, init_dt):
    """Find the time index in the zarr for this init date."""
    units = z['time'].attrs.get('units', '')
    base_str = units.replace('days since ', '').strip()[:10]
    base = datetime.strptime(base_str, '%Y-%m-%d')
    t_raw = z['time'][:]
    # time[0] is garbled (nanoseconds encoding artifact) — treat as base date (day 0)
    # time[1:] are integer days since base
    for i, v in enumerate(t_raw):
        if i == 0:
            candidate = base
        else:
            candidate = base + timedelta(days=int(v))
        if candidate.date() == init_dt.date():
            return i
    return None

def load_gencast_ens_mean(init_str, zarr_var, step_indices):
    """
    Load ensemble mean for given variable at the needed step indices.
    Returns dict {step_idx: (180,360) array} or None.

    Chunk layout is (4, 1, 90, 721, 1440) — all 90 steps are in one chunk.
    We load 4 samples at a time (one chunk), average over samples and keep
    only the needed steps. This caps peak memory at ~1.4 GB per chunk read.
    """
    init_dt = datetime.strptime(init_str, '%Y%m%d%H')
    year = init_dt.year
    z = get_zarr(year)

    t_idx = get_init_time_index(z, init_dt)
    if t_idx is None:
        return None

    n_samples = z[zarr_var].shape[0]
    chunk_size = 4   # matches zarr chunk along sample axis
    step_set = set(step_indices)

    # accum[si] accumulates sum over all samples at that step
    accum = {si: np.zeros((721, 1440), dtype=np.float64) for si in step_indices}

    for s_start in range(0, n_samples, chunk_size):
        s_end = min(s_start + chunk_size, n_samples)
        # Load (chunk, 90, 721, 1440) in float32 — ~1.4 GB per chunk
        block = z[zarr_var][s_start:s_end, t_idx, :, :, :]  # float32
        # Immediately extract only needed steps and accumulate in float64
        for si in step_indices:
            accum[si] += block[:, si, :, :].astype(np.float64).sum(axis=0)
        del block

    result = {si: regrid(accum[si] / n_samples) for si in step_indices}
    return result

# ==================== Main Computation ====================
os.makedirs(RESULTS_DIR, exist_ok=True)

# step indices for 00Z forecasts: step=24h->idx1, 48h->idx3, ..., 1080h->idx89
step_hours = list(zarr.open(f'{GENCAST_DIR}/gencast_2019.zarr', mode='r')['step'][:])
lead_day_to_step_idx = {}   # lead_day (1..45) -> step index in zarr
for ld in range(1, MAX_LEAD + 1):
    h = ld * 24
    if h in step_hours:
        lead_day_to_step_idx[ld] = step_hours.index(h)

# For precipitation: 24hr total = sum of two consecutive 12hr steps
# step 24h = step_idx 1 (12-24hr accumulation), step 12h = step_idx 0 (0-12hr)
# So 24hr precip for lead day 1 = step_idx 0 + step_idx 1
precip_lead_to_step_pair = {}
for ld in range(1, MAX_LEAD + 1):
    h1 = ld * 24 - 12   # 12hr step ending at 12Z
    h2 = ld * 24        # 12hr step ending at 00Z next day
    if h1 in step_hours and h2 in step_hours:
        precip_lead_to_step_pair[ld] = (step_hours.index(h1), step_hours.index(h2))

print(f"Lead day -> step index (first 5): {list(lead_day_to_step_idx.items())[:5]}")
print(f"Precip step pairs (first 3): {list(precip_lead_to_step_pair.items())[:3]}")

acc_data = {label: {r: defaultdict(list) for r in REGION_CONFIG}
            for _, _, label, _, _ in VARIABLES}

for init_str in tqdm(PRED_INIT_DATES, desc='Init dates'):
    init_dt = datetime.strptime(init_str, '%Y%m%d%H')

    for zarr_var, level_hpa, label, era5_subdir, era5_var in VARIABLES:

        if label == 'tp':
            # Load all needed step pairs for precipitation
            all_step_idxs = set()
            for s1, s2 in precip_lead_to_step_pair.values():
                all_step_idxs.update([s1, s2])
            step_data = load_gencast_ens_mean(init_str, zarr_var, sorted(all_step_idxs))
            if step_data is None:
                continue

            for ld in range(1, MAX_LEAD + 1):  # GenCast 00Z starts at lead day 1
                if ld not in precip_lead_to_step_pair:
                    continue
                s1, s2 = precip_lead_to_step_pair[ld]
                if s1 not in step_data or s2 not in step_data:
                    continue
                pred_24hr = step_data[s1] + step_data[s2]   # sum two 12hr accumulations

                date = init_dt + timedelta(days=ld)
                gt_val = load_era5_gt(era5_subdir, era5_var, None, date)
                if gt_val is None:
                    continue
                clim = get_clim(zarr_var, date)
                pred_anom = pred_24hr - clim
                gt_anom   = gt_val - clim
                for region, mask in MASKS.items():
                    acc_data[label][region][ld].append(weighted_acc(pred_anom, gt_anom, mask))

        else:
            # t2m: only load 00Z steps (24h, 48h, ...)
            step_idxs = [lead_day_to_step_idx[ld]
                         for ld in range(1, MAX_LEAD + 1)
                         if ld in lead_day_to_step_idx]
            step_data = load_gencast_ens_mean(init_str, zarr_var, step_idxs)
            if step_data is None:
                continue

            for ld in range(1, MAX_LEAD + 1):  # GenCast 00Z starts at lead day 1
                if ld not in lead_day_to_step_idx:
                    continue
                si = lead_day_to_step_idx[ld]
                if si not in step_data:
                    continue

                date = init_dt + timedelta(days=ld)
                gt_val = load_era5_gt(era5_subdir, era5_var, level_hpa, date)
                if gt_val is None:
                    continue
                clim = get_clim(zarr_var, date)
                pred_anom = step_data[si] - clim
                gt_anom   = gt_val - clim
                for region, mask in MASKS.items():
                    acc_data[label][region][ld].append(weighted_acc(pred_anom, gt_anom, mask))

close_era5_cache()

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

# Print summary
leads = np.arange(0, MAX_LEAD + 1)
for label in acc_summary:
    print(f"\n=== {label} ===")
    for region in REGION_CONFIG:
        arr = acc_summary[label][region]
        above = np.where(arr >= 0.6)[0]
        threshold_day = int(above[-1]) if len(above) > 0 else 0
        print(f"  {region:8s}: ACC@0d={arr[0]:.3f}  ACC@7d={arr[7]:.3f}  "
              f"ACC@14d={arr[14]:.3f}  days>=0.6: {threshold_day}")
