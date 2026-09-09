"""
Compute ACC for GraphCast ReForecastNet predictions vs ERA5.
Data: /glade/work/bgong/ReForecastNet/graphcast/output/YYYY-MM-DDT00.zarr
  - 2m_temperature: (1, 180_leads_6hrly, 721, 1440) at 0.25 deg, lat S->N
  - 00Z snapshots extracted at indices 3,7,11,... (lead days 1,2,...,45)
  - Regridded from 0.25 deg to 1 deg to match ERA5/clim
Variables: t2m only (z500 not available per lead)
"""

import numpy as np
import zarr
import netCDF4
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.ndimage import map_coordinates

PRED_DIR    = '/glade/work/bgong/ReForecastNet/graphcast/output'
ERA5_DIR    = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_DIR    = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_00z/'
RESULTS_DIR = '/glade/work/bgong/benchmark-dev/metrics/ACC/results_graphcast/'
MAX_LEAD    = 45

REGION_CONFIG = {
    'global':  {'lat_min': -90, 'lat_max':  90},
    'tropics': {'lat_min': -30, 'lat_max':  30},
    'NH':      {'lat_min':  30, 'lat_max':  90},
    'SH':      {'lat_min': -90, 'lat_max': -30},
}

# ==================== 1-deg grid (S->N) ====================
lat1 = np.linspace(-89.5, 89.5, 180)
lon1 = np.linspace(0.5, 359.5, 360)
W2D  = np.outer(np.cos(np.deg2rad(lat1)), np.ones(360))

def regional_mask(lat_min, lat_max):
    return np.outer((lat1 >= lat_min) & (lat1 <= lat_max), np.ones(360, dtype=bool))

MASKS = {r: regional_mask(cfg['lat_min'], cfg['lat_max']) for r, cfg in REGION_CONFIG.items()}

def weighted_acc(pred_anom, gt_anom, mask):
    a, b, wm = pred_anom[mask], gt_anom[mask], W2D[mask]
    num = np.sum(wm * a * b)
    den = np.sqrt(np.sum(wm * a**2) * np.sum(wm * b**2))
    return float(num / den) if den > 0 else np.nan

# ==================== GraphCast grid ====================
# GC grid: lat 721 pts (-90..90, step 0.25), lon 1440 pts (0..359.75, step 0.25)
# Precompute target coordinates in GC index space once
_gc_lat_step = 0.25
_gc_lon_step = 0.25
_gc_lat0     = -90.0
_gc_lon0     =   0.0

# lat1/lon1 in GC fractional index space
_row_coords = (lat1 - _gc_lat0) / _gc_lat_step   # (180,) in [0, 720]
_col_coords = (lon1 - _gc_lon0) / _gc_lon_step   # (360,) in [0, 1439]
_rows = np.tile(_row_coords[:, None], (1, 360)).ravel()   # (64800,)
_cols = np.tile(_col_coords[None, :], (180, 1)).ravel()   # (64800,)

def regrid_gc(field):
    """Regrid (721, 1440) 0.25-deg field to (180, 360) 1-deg S->N using map_coordinates."""
    result = map_coordinates(field.astype(np.float64), [_rows, _cols],
                             order=1, mode='wrap')
    return result.reshape(180, 360)

# ==================== Climatology ====================
print("Loading climatology...")
clim_t2m = np.load(os.path.join(CLIM_DIR, 'clim_00z_t2m.npy'))  # (366, 180, 360) S->N
print(f"  t2m clim: {clim_t2m.shape}")

def doy_index(date):
    if date.month == 2 and date.day == 29:
        return 59
    return date.timetuple().tm_yday - 1

# ==================== ERA5 ====================
_era5_cache = {}

def get_era5_t2m(date):
    fpath = f'{ERA5_DIR}/2m_temperature/{date.year}_180x360.nc'
    if not os.path.exists(fpath):
        return None
    key = fpath
    if key not in _era5_cache:
        nc = netCDF4.Dataset(fpath)
        t_hrs = nc.variables['time'][:]
        t_dates = [datetime(1900,1,1) + timedelta(hours=float(h)) for h in t_hrs]
        _era5_cache[key] = (nc, t_dates)
    nc, t_dates = _era5_cache[key]
    indices = [i for i, d in enumerate(t_dates) if d == date]
    if not indices:
        return None
    return np.array(nc.variables['2m_temperature'][indices[0], :, :]).astype(np.float64)

# ==================== Init dates ====================
def list_init_dates():
    dates = []
    for f in sorted(os.listdir(PRED_DIR)):
        if f.endswith('.zarr'):
            dt = datetime.strptime(f.replace('.zarr', ''), '%Y-%m-%dT%H')
            dates.append(dt)
    return dates

# ==================== Main ====================
os.makedirs(RESULTS_DIR, exist_ok=True)
init_dates = list_init_dates()
print(f"\nFound {len(init_dates)} init dates: {[d.strftime('%Y-%m-%d') for d in init_dates]}\n")

acc_data = {'t2m': {r: defaultdict(list) for r in REGION_CONFIG}}

for init_dt in init_dates:
    fname = init_dt.strftime('%Y-%m-%dT%H') + '.zarr'
    fpath = os.path.join(PRED_DIR, fname)
    z = zarr.open(fpath, mode='r')

    td = np.array(z['prediction_timedelta'][:])  # seconds
    # 00Z indices: where timedelta is exactly N*86400 seconds, N=1..45
    daily_idx = np.where(td % 86400 == 0)[0]  # indices 3,7,11,...
    lead_days  = (td[daily_idx] / 86400).astype(int)  # 1,2,...,45

    t2m_var = z['2m_temperature']  # lazy zarr array (1, 180, 721, 1440)

    for arr_idx, ld in zip(daily_idx, lead_days):
        if ld > MAX_LEAD:
            break
        date = init_dt + timedelta(days=int(ld))

        gt_val = get_era5_t2m(date)
        if gt_val is None:
            continue

        field = np.array(t2m_var[0, arr_idx, :, :])  # (721, 1440), one step at a time
        pred_1deg = regrid_gc(field)  # (180, 360)
        clim = clim_t2m[doy_index(date)]
        pred_anom = pred_1deg - clim
        gt_anom   = gt_val   - clim

        for region, mask in MASKS.items():
            acc_data['t2m'][region][ld].append(weighted_acc(pred_anom, gt_anom, mask))

for nc, _ in _era5_cache.values():
    nc.close()

# ==================== Aggregate & Save ====================
acc_summary = {'t2m': {}}
for region in REGION_CONFIG:
    arr = np.full(MAX_LEAD + 1, np.nan)
    for ld in range(1, MAX_LEAD + 1):
        vals = acc_data['t2m'][region][ld]
        if vals:
            arr[ld] = np.nanmean(vals)
    acc_summary['t2m'][region] = arr

out_file = os.path.join(RESULTS_DIR, 'acc_summary.pkl')
with open(out_file, 'wb') as f:
    pickle.dump({'acc_summary': acc_summary, 'acc_data': acc_data}, f)
print(f"Saved: {out_file}")

print("\n=== t2m ===")
for region in REGION_CONFIG:
    arr = acc_summary['t2m'][region]
    skill_days = np.where(arr >= 0.6)[0]
    threshold_day = int(skill_days[-1]) + 1 if len(skill_days) > 0 else 0
    print(f"  {region:8s}: ACC@7d={arr[7]:.3f}  ACC@14d={arr[14]:.3f}  days>=0.6: {threshold_day}")
