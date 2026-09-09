"""
Compute DOY climatology at 00Z from ERA5 yearly files (1979-2018).
Saves one file per variable: clim_00z_<label>.npy, shape (366, 180, 360).
DOY index: 0 = Jan 1, 1 = Jan 2, ..., 365 = Dec 31 (leap day = Feb 29 = index 59).
"""

import numpy as np
import netCDF4
import os
from datetime import datetime, timedelta
from tqdm import tqdm

ERA5_DIR  = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
CLIM_OUT  = './clim_00z/'
CLIM_YEARS = range(1979, 2019)   # 1979-2018 inclusive

os.makedirs(CLIM_OUT, exist_ok=True)

# (nc_var_name, level_hPa_or_None, short_label, era5_subdir)
VARIABLES = [
    ('2m_temperature',           None, 't2m',  '2m_temperature'),
    ('geopotential',             500,  'z500', 'geopotential'),
    ('total_precipitation_24hr', None, 'tp',   'total_precipitation_24hr'),
    ('10m_u_component_of_wind',  None, 'u10',  '10m_u_component_of_wind'),
    ('10m_v_component_of_wind',  None, 'v10',  '10m_v_component_of_wind'),
]

def doy_index(date):
    """0-based DOY index consistent across years; leap day (Feb 29) = 59."""
    if date.month > 2:
        return date.timetuple().tm_yday - 1   # same as non-leap
    elif date.month == 2 and date.day == 29:
        return 59
    else:
        return date.timetuple().tm_yday - 1

for var_name, level_hpa, label, subdir in VARIABLES:
    out_file = os.path.join(CLIM_OUT, f'clim_00z_{label}.npy')
    if os.path.exists(out_file):
        print(f"Already exists, skipping: {out_file}")
        continue

    print(f"\nComputing climatology for {label} ...")

    # accum[doy] = sum of values, count[doy] = number of samples
    accum = np.zeros((366, 180, 360), dtype=np.float64)
    count = np.zeros(366, dtype=np.int32)

    for year in tqdm(CLIM_YEARS, desc=label):
        fpath = f'{ERA5_DIR}/{subdir}/{year}_180x360.nc'
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath}")
            continue

        with netCDF4.Dataset(fpath) as nc:
            t_hrs = nc.variables['time'][:]
            base  = datetime(1900, 1, 1)
            t_dates = [base + timedelta(hours=float(h)) for h in t_hrs]

            # Get indices of 00Z steps only
            idx_00z = [i for i, d in enumerate(t_dates) if d.hour == 0]

            if level_hpa is not None:
                levs = nc.variables['level'][:]
                li   = int(np.where(levs == level_hpa)[0][0])
                data = nc.variables[var_name][idx_00z, li, :, :].astype(np.float64)
            else:
                data = nc.variables[var_name][idx_00z, :, :].astype(np.float64)

            for k, i in enumerate(idx_00z):
                doy = doy_index(t_dates[i])
                accum[doy] += data[k]
                count[doy] += 1

    # Average
    clim = np.full((366, 180, 360), np.nan, dtype=np.float32)
    for doy in range(366):
        if count[doy] > 0:
            clim[doy] = accum[doy] / count[doy]

    np.save(out_file, clim)
    print(f"  Saved: {out_file}  (counts min={count.min()}, max={count.max()})")

print("\nDone.")
