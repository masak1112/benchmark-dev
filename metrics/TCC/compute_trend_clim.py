"""
Linear-trend ERA5 climatology for detrended TCC (FuXi-S2S paper method, observation side).

For every day of year d and grid point, fit by least squares over TREND_YEARS (2002-2016,
the paper's hindcast period)
    y(d, year) = ybar(d) + slope(d) * (year - xbar)
The trend value for any date is then ybar(doy) + slope(doy) * (year - xbar). Because OLS is
linear in y, averaging these per-day fits over a 7-day week equals the fit on weekly means,
so this reproduces the paper's week-of-year regression.

Two versions, matching the two truth types: '00z' (00 UTC value) and 'dailymean'.
Output: ACC/clim_trend/{mode}_{label}.npz with ybar (366,180,360), slope (366,180,360), xbar.
Feb 29 shares DOY 59 with Mar 1 (same convention as the other climatologies).
"""

import os
from multiprocessing import Pool

import netCDF4
import numpy as np

from tcc_weekly import ERA5_DIR, VARIABLES, doy_index, era5_dates

OUT_DIR = '/glade/work/bgong/benchmark-dev/metrics/ACC/clim_trend'
TREND_YEARS = range(2002, 2017)


def compute(args):
    label, subdir, level = args
    outs = {m: os.path.join(OUT_DIR, f'{m}_{label}.npz') for m in ('00z', 'dailymean')}
    if all(os.path.exists(p) for p in outs.values()):
        print(f'{label}: exists, skipping', flush=True)
        return
    # running sums per DOY for OLS in x = year
    sums = {m: {k: np.zeros((366, 180, 360)) for k in ('y', 'xy')} for m in outs}
    sx = {m: np.zeros(366) for m in outs}
    sxx = {m: np.zeros(366) for m in outs}
    n = {m: np.zeros(366) for m in outs}
    for yr in TREND_YEARS:
        with netCDF4.Dataset(f'{ERA5_DIR}/{subdir}/{yr}_180x360.nc') as nc:
            dates = era5_dates(nc.variables['time'])
            if level is not None:
                li = int(np.where(nc.variables['level'][:] == level)[0][0])
                data = np.asarray(nc.variables[subdir][:, li, :, :], dtype=np.float64)
            else:
                data = np.asarray(nc.variables[subdir][:], dtype=np.float64)
        days = np.array([d.toordinal() for d in dates])
        hours = np.array([d.hour for d in dates])
        for od in np.unique(days):
            sel = days == od
            d = dates[int(np.argmax(sel))]
            k = doy_index(d)
            vals = {'dailymean': data[sel].mean(axis=0)}
            z = sel & (hours == 0)
            if z.any():
                vals['00z'] = data[z][0]
            for m, v in vals.items():
                sums[m]['y'][k] += v
                sums[m]['xy'][k] += yr * v
                sx[m][k] += yr
                sxx[m][k] += yr * yr
                n[m][k] += 1
        print(f'  {label} {yr}', flush=True)
    for m, p in outs.items():
        nn = n[m][:, None, None]
        xbar = sx[m] / n[m]
        ybar = sums[m]['y'] / nn
        sxx_c = sxx[m] - n[m] * xbar ** 2                      # sum (x - xbar)^2
        sxy_c = sums[m]['xy'] - xbar[:, None, None] * sums[m]['y']
        slope = sxy_c / sxx_c[:, None, None]
        np.savez(p, ybar=ybar.astype(np.float32), slope=slope.astype(np.float32), xbar=xbar)
        print(f'Saved {p}  n[0]={n[m][0]:.0f} n[59]={n[m][59]:.0f}', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [(lbl, subdir, level) for lbl, subdir, level, *_ in VARIABLES]
    with Pool(len(jobs)) as p:
        p.map(compute, jobs)
