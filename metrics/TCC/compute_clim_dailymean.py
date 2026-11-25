"""
Per-DOY daily-mean ERA5 climatology (avg of 00/06/12/18Z), 1979-2018, for the
variables not yet covered in ACC/clim_dailymean (u850/v850 already exist there).
Needed for FuXi, whose fields are daily means.
Output: ACC/clim_dailymean/clim_dailymean_<label>.npy  shape (366, 180, 360)
DOY index convention matches clim_00z: 0 = Jan 1, Feb 29 -> 59.
"""

import os
import numpy as np
import netCDF4
from datetime import datetime, timedelta
from multiprocessing import Pool

ERA5_DIR = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata'
OUT_DIR  = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_dailymean'
YEARS    = range(1979, 2019)

# (nc_var, level_hPa or None, label)
VARIABLES = [
    ('2m_temperature',           None, 't2m'),
    ('geopotential',             500,  'z500'),
    ('total_precipitation_24hr', None, 'tp'),
    ('10m_u_component_of_wind',  None, 'u10'),
    ('10m_v_component_of_wind',  None, 'v10'),
    ('u_component_of_wind',      850,  'u850'),
    ('v_component_of_wind',      850,  'v850'),
    ('mean_top_net_long_wave_radiation_flux', None, 'olr'),
]
# labels that also lack a 00Z climatology in ACC/clim_00z (UC-S2S is verified at 00Z)
NEED_00Z = {'olr'}
OUT_00Z  = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_00z'


def era5_dates(tvar):
    """Decode ERA5 time from its own units: 1979-2018 t2m/u10/v10 files use
    'hours since 1959-01-01', everything else 'hours since 1900-01-01'."""
    return [datetime(d.year, d.month, d.day, d.hour) for d in
            netCDF4.num2date(tvar[:], tvar.units, only_use_cftime_datetimes=False,
                             only_use_python_datetimes=True)]


def doy_index(d):
    if d.month == 2 and d.day == 29:
        return 59
    return d.timetuple().tm_yday - 1


def compute(args):
    vname, level, label = args
    out = os.path.join(OUT_DIR, f'clim_dailymean_{label}.npy')
    out00 = os.path.join(OUT_00Z, f'clim_00z_{label}.npy') if label in NEED_00Z else None
    if os.path.exists(out) and (out00 is None or os.path.exists(out00)):
        print(f'{label}: exists, skipping', flush=True)
        return
    accum = np.zeros((366, 180, 360), dtype=np.float64)
    count = np.zeros(366, dtype=np.int32)
    accum00 = np.zeros((366, 180, 360), dtype=np.float64)
    count00 = np.zeros(366, dtype=np.int32)
    for yr in YEARS:
        fpath = f'{ERA5_DIR}/{vname}/{yr}_180x360.nc'
        if not os.path.exists(fpath):
            print(f'  MISSING: {fpath}', flush=True)
            continue
        with netCDF4.Dataset(fpath) as nc:
            dates = era5_dates(nc.variables['time'])
            if level is not None:
                li = int(np.where(nc.variables['level'][:] == level)[0][0])
                data = np.asarray(nc.variables[vname][:, li, :, :], dtype=np.float64)
            else:
                data = np.asarray(nc.variables[vname][:], dtype=np.float64)
        days = np.array([d.toordinal() for d in dates])
        for od in np.unique(days):
            d = datetime.fromordinal(int(od))
            accum[doy_index(d)] += data[days == od].mean(axis=0)
            count[doy_index(d)] += 1
        if out00 is not None:
            for i, d in enumerate(dates):
                if d.hour == 0:
                    accum00[doy_index(d)] += data[i]
                    count00[doy_index(d)] += 1
        print(f'  {label} {yr}', flush=True)
    accum /= np.maximum(count, 1)[:, None, None]
    np.save(out, accum.astype(np.float32))
    if out00 is not None:
        accum00 /= np.maximum(count00, 1)[:, None, None]
        np.save(out00, accum00.astype(np.float32))
        print(f'Saved {out00}  count00[0]={count00[0]}', flush=True)
    print(f'Saved {out}  count[0]={count[0]} count[59]={count[59]}', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    with Pool(len(VARIABLES)) as p:
        p.map(compute, VARIABLES)
