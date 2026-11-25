"""
Compute regional ACC metrics for NGCM ensemble mean vs ERA5 at 00Z.
Variables: geopotential@500hPa, total_precipitation_24hr
           (no 2m_temperature or 10m winds available)
Grid: NGCM T63 Gaussian (128x64, lon x lat) -> regrid to ERA5 1-deg (180x360, lat x lon S->N)
"""

import numpy as np
import zarr
import netCDF4
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

# ==================== Configuration ====================
NGCM_DIR    = '/glade/derecho/scratch/bgong/ngcm'
ERA5_DIR    = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_FILE   = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata/1979-2018_mean_climatology.nc'
RESULTS_DIR = './results_ngcm/'
MAX_LEAD    = 45  # days

REGION_CONFIG = {
    'global':  {'lat_min': -90, 'lat_max':  90},
    'tropics': {'lat_min': -30, 'lat_max':  30},
    'NH':      {'lat_min':  30, 'lat_max':  90},
}

# Variables: (ngcm_var, era5_subdir, era5_var, level_hpa, short_label, clim_label)
VARIABLES = [
    ('geopotential_500_avg',   'geopotential',           'geopotential',           500,  'z500', 'z500'),
    ('total_precipitation_24hr', 'total_precipitation_24hr', 'total_precipitation_24hr', None, 'tp',   'tp'),
]

# ==================== Target 1-deg S->N Grid ====================
lat1 = np.linspace(-89.5, 89.5, 180)   # S->N, matches ERA5 yearly files
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

# ==================== Regrid NGCM -> 1-deg ====================
# NGCM: dims are (lon=128, lat=64), Gaussian lat S->N, lon 0->357.1875
# Output: (180, 360) lat S->N, lon 0.5->359.5

def make_ngcm_interpolator(ngcm_lat, ngcm_lon):
    """Build a reusable interpolator for the NGCM Gaussian grid."""
    # data on NGCM is (lon, lat) so we need to transpose to (lat, lon) for interpolation
    # ngcm_lat is already S->N
    return ngcm_lat, ngcm_lon

# Precompute NGCM grid once from first file
_z = zarr.open(f'{NGCM_DIR}/2019/2019-05-01T00.zarr', mode='r')
_ngcm_lat = _z['latitude'][:]   # 64 values, S->N
_ngcm_lon = _z['longitude'][:]  # 128 values, 0->357.1875

def regrid_ngcm(data_lonlat):
    """
    Regrid NGCM field from (128 lon, 64 lat) to (180 lat, 360 lon) 1-deg S->N.
    Uses bilinear interpolation via RegularGridInterpolator.
    """
    # data_lonlat: (128, 64), lon x lat
    data_latlon = data_lonlat.T   # (64, 128), lat x lon — lat is S->N

    # Wrap longitude: add a repeated column at lon=360 for periodicity
    lon_ext = np.append(_ngcm_lon, 360.0)
    data_ext = np.hstack([data_latlon, data_latlon[:, :1]])  # (64, 129)

    interp = RegularGridInterpolator(
        (_ngcm_lat, lon_ext), data_ext,
        method='linear', bounds_error=False, fill_value=None
    )
    pts = np.array([[la, lo] for la in lat1 for lo in lon1])
    result = interp(pts).reshape(180, 360)
    return result.astype(np.float64)

# ==================== Climatology ====================
print("Loading climatology from 1979-2018_mean_climatology.nc ...")
CLIM = {}
with netCDF4.Dataset(CLIM_FILE) as nc:
    plev = nc.variables['plev'][:]
    li500 = int(np.where(plev == 500)[0][0])
    CLIM['z500'] = nc.variables['geopotential'][:, li500, :, :]  # (366,180,360) S->N
    CLIM['tp']   = nc.variables['total_precipitation_24hr'][:]   # (366,180,360) S->N
for label, arr in CLIM.items():
    print(f"  Loaded {label}: {arr.shape}")

def doy_index(date):
    if date.month == 2 and date.day == 29:
        return 59
    return date.timetuple().tm_yday - 1

def get_clim(label, date):
    return CLIM[label][doy_index(date)]

# ==================== ERA5 Ground Truth ====================
_era5_cache = {}

def load_era5_gt(era5_subdir, era5_var, level_hpa, date):
    """Load ERA5 daily mean (avg of 00/06/12/18Z) to match NGCM _avg variables."""
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
    # All 6-hourly steps for this calendar day -> daily mean
    indices = [i for i, d in enumerate(t_dates) if d.date() == date.date()]
    if not indices:
        return None
    if level_hpa is not None:
        levs = nc.variables['level'][:]
        li   = int(np.where(levs == level_hpa)[0][0])
        data = np.mean(nc.variables[era5_var][indices, li, :, :], axis=0)
    else:
        data = np.mean(nc.variables[era5_var][indices, :, :], axis=0)
    return np.array(data).astype(np.float64)

def close_era5_cache():
    for (nc, _) in _era5_cache.values():
        nc.close()
    _era5_cache.clear()

# ==================== NGCM File Helpers ====================
def list_ngcm_init_dates():
    """Return all available init date strings YYYYMMDDHH."""
    dates = []
    for year_dir in sorted(os.listdir(NGCM_DIR)):
        year_path = os.path.join(NGCM_DIR, year_dir)
        if not os.path.isdir(year_path):
            continue
        for f in sorted(os.listdir(year_path)):
            if f.endswith('.zarr'):
                # format: 2019-05-01T00.zarr
                dt = datetime.strptime(f.replace('.zarr', ''), '%Y-%m-%dT%H')
                dates.append(dt.strftime('%Y%m%d%H'))
    return dates

def load_ngcm_ens_mean(init_str, ngcm_var):
    """
    Load ensemble mean for all lead days.
    Returns (46, 128, 64) array or None.
    Chunks are (1,1,46,128,64) so each member read pulls all leads at once.
    """
    init_dt = datetime.strptime(init_str, '%Y%m%d%H')
    year = str(init_dt.year)
    fname = init_dt.strftime('%Y-%m-%dT%H') + '.zarr'
    fpath = os.path.join(NGCM_DIR, year, fname)
    if not os.path.exists(fpath):
        return None

    z = zarr.open(fpath, mode='r')
    if ngcm_var not in z:
        return None

    arr = z[ngcm_var]  # (1, 25, 46, 128, 64) or (1,25,46,1,128,64) for precip
    n_members = arr.shape[1]
    accum = None
    for m in range(n_members):
        if arr.ndim == 6:   # precip has extra surface dim
            data = arr[0, m, :, 0, :, :].astype(np.float64)  # (46, 128, 64)
        else:
            data = arr[0, m, :, :, :].astype(np.float64)      # (46, 128, 64)
        accum = data if accum is None else accum + data
    return accum / n_members   # (46, 128, 64)

# ==================== Main Computation ====================
os.makedirs(RESULTS_DIR, exist_ok=True)

init_dates = list_ngcm_init_dates()
print(f"\nFound {len(init_dates)} NGCM init dates: {init_dates[0]} ... {init_dates[-1]}\n")

acc_data = {label: {r: defaultdict(list) for r in REGION_CONFIG}
            for _, _, _, _, label, _ in VARIABLES}

for init_str in tqdm(init_dates, desc='Init dates'):
    init_dt = datetime.strptime(init_str, '%Y%m%d%H')

    for ngcm_var, era5_subdir, era5_var, level_hpa, label, _ in VARIABLES:
        ens_mean = load_ngcm_ens_mean(init_str, ngcm_var)  # (46, 128, 64)
        if ens_mean is None:
            continue

        for ld in range(0, MAX_LEAD + 1):
            if ld >= ens_mean.shape[0]:
                break
            date = init_dt + timedelta(days=ld)

            gt_val = load_era5_gt(era5_subdir, era5_var, level_hpa, date)
            if gt_val is None:
                continue

            # Regrid NGCM field from (128,64) to (180,360)
            pred_1deg = regrid_ngcm(ens_mean[ld])   # (180, 360)

            clim = get_clim(label, date)
            pred_anom = pred_1deg - clim
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

for label in acc_summary:
    print(f"\n=== {label} ===")
    for region in REGION_CONFIG:
        arr = acc_summary[label][region]
        above = np.where(arr >= 0.6)[0]
        threshold_day = int(above[-1]) if len(above) > 0 else 0
        print(f"  {region:8s}: ACC@0d={arr[0]:.3f}  ACC@7d={arr[7]:.3f}  "
              f"ACC@14d={arr[14]:.3f}  days>=0.6: {threshold_day}")
