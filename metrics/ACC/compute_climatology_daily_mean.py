"""
Compute DOY climatology as daily mean (avg of 00/06/12/18Z) from ERA5 yearly files (1979-2018).
Used for NGCM which outputs daily-averaged fields.
Saves clim_daily_mean/clim_daily_<label>.npy, shape (366, 180, 360).
"""

import numpy as np
import netCDF4
import os
from datetime import datetime, timedelta
from tqdm import tqdm

ERA5_DIR   = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_OUT   = './clim_daily_mean/'
CLIM_YEARS = range(1979, 2019)

os.makedirs(CLIM_OUT, exist_ok=True)

VARIABLES = [
    ('geopotential',             500,  'z500', 'geopotential'),
    ('total_precipitation_24hr', None, 'tp',   'total_precipitation_24hr'),
]

def doy_index(date):
    if date.month == 2 and date.day == 29:
        return 59
    return date.timetuple().tm_yday - 1

for var_name, level_hpa, label, subdir in VARIABLES:
    out_file = os.path.join(CLIM_OUT, f'clim_daily_{label}.npy')
    if os.path.exists(out_file):
        print(f"Already exists, skipping: {out_file}")
        continue

    print(f"\nComputing daily-mean climatology for {label} ...")
    accum = np.zeros((366, 180, 360), dtype=np.float64)
    count = np.zeros(366, dtype=np.int32)

    for year in tqdm(CLIM_YEARS, desc=label):
        fpath = f'{ERA5_DIR}/{subdir}/{year}_180x360.nc'
        if not os.path.exists(fpath):
            continue
        with netCDF4.Dataset(fpath) as nc:
            t_hrs = nc.variables['time'][:]
            base  = datetime(1900, 1, 1)
            t_dates = [base + timedelta(hours=float(h)) for h in t_hrs]
            # Group by calendar date and average
            from collections import defaultdict
            day_indices = defaultdict(list)
            for i, d in enumerate(t_dates):
                day_indices[d.date()].append(i)

            for date, idxs in day_indices.items():
                doy = doy_index(datetime(date.year, date.month, date.day))
                if level_hpa is not None:
                    levs = nc.variables['level'][:]
                    li   = int(np.where(levs == level_hpa)[0][0])
                    data = np.mean(nc.variables[var_name][idxs, li, :, :], axis=0)
                else:
                    data = np.mean(nc.variables[var_name][idxs, :, :], axis=0)
                accum[doy] += data.astype(np.float64)
                count[doy] += 1

    clim = np.full((366, 180, 360), np.nan, dtype=np.float32)
    for doy in range(366):
        if count[doy] > 0:
            clim[doy] = accum[doy] / count[doy]

    np.save(out_file, clim)
    print(f"  Saved: {out_file}  (counts min={count.min()}, max={count.max()})")

print("\nDone.")
