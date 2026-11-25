"""
ERA5 DJF North Atlantic Oscillation (NAO) pattern and daily index.

NAO pattern = leading EOF of ERA5 monthly-mean Z500 anomalies over the North Atlantic box
(20-80N, 90W-40E), DJF months (Dec/Jan/Feb) of 1979-2018, sqrt(cos lat) area weighting.
The daily index is the projection of daily-mean anomalies onto that pattern.
Anomalies are relative to the same 1979-2018 DOY climatology the TCC/ACC pipelines use
(ACC/clim_dailymean/clim_dailymean_z500.npy). Sign: NAO+ = low over Iceland.

The raw index of any anomaly field A(lat, lon) is  pc = sum(sqrt(cos) * A * eof)  over the box.
It is normalised by the std of the 7-day-mean ERA5 index (DJF 1979-2018), so weekly-mean
forecasts are scored in units of the observed weekly NAO std (climatology forecast RMSE ~ 1).

Output: results/nao_eof_djf.npz
  eof (nlat, nlon) unit vector in weighted space, lat_idx / lon_idx (into the 180x360 grid),
  wgt (nlat, nlon) sqrt(cos), evr (first 5 explained-variance fractions), sd_daily, sd_weekly,
  dates (YYYYMMDD, 1979-2024 every day), pc_daily (daily-mean ERA5) and pc_00z (00Z ERA5 vs
  clim_00z; truth for the 00Z models), both raw / unnormalised.

Usage:  python nao_eof.py --workers 16
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TCC'))
from tcc_weekly import CLIM_ROOT, ERA5_DIR, doy_index, era5_dates, lat  # noqa: E402

RES_DIR = os.path.join(HERE, 'results')
FIT_YEARS = (1979, 2018)
ALL_YEARS = (1979, 2024)
# NAO_SEASON=ndjfm fits the pattern on Nov-Mar months instead of DJF (separate output file)
SEASON = os.environ.get('NAO_SEASON', 'djf')
DJF = {'djf': (12, 1, 2), 'ndjfm': (11, 12, 1, 2, 3)}[SEASON]

# North Atlantic box on the ERA5 1deg grid (lat -89.5..89.5 S->N, lon 0..359 E)
LAT_IDX = np.where((lat >= 20) & (lat <= 80))[0]
LON_IDX = np.concatenate([np.arange(270, 360), np.arange(0, 41)])     # 90W..1W, 0..40E
WGT = np.sqrt(np.cos(np.deg2rad(lat[LAT_IDX])))[:, None] * np.ones(len(LON_IDX))
ICELAND = (65.5, 338)       # (lat, lon E) used only to fix the sign


def read_year(year):
    """Z500 anomalies over the box for every day of the year -> (dates, daily mean (n, nlat, nlon),
    00Z (n, nlat, nlon)); daily mean vs clim_dailymean, 00Z vs clim_00z."""
    clim = np.load(os.path.join(CLIM_ROOT, 'clim_dailymean', 'clim_dailymean_z500.npy'), mmap_mode='r')
    clim0 = np.load(os.path.join(CLIM_ROOT, 'clim_00z', 'clim_00z_z500.npy'), mmap_mode='r')
    with netCDF4.Dataset(f'{ERA5_DIR}/geopotential/{year}_180x360.nc') as nc:
        t = era5_dates(nc.variables['time'])
        li = int(np.where(nc.variables['level'][:] == 500)[0][0])
        v = nc.variables['geopotential']
        idx = {d: i for i, d in enumerate(t)}
        dates, out, out0 = [], [], []
        d = datetime(year, 1, 1)
        while d.year == year:
            ii = [idx.get(d + timedelta(hours=h)) for h in (0, 6, 12, 18)]
            if None not in ii:
                x4 = np.asarray(v[ii, li, LAT_IDX[0]:LAT_IDX[-1] + 1, :], dtype=np.float64)[:, :, LON_IDX]
                k = doy_index(d)
                out.append(x4.mean(axis=0) - np.asarray(clim[k][LAT_IDX][:, LON_IDX], dtype=np.float64))
                out0.append(x4[0] - np.asarray(clim0[k][LAT_IDX][:, LON_IDX], dtype=np.float64))
                dates.append(d.strftime('%Y%m%d'))
            d += timedelta(days=1)
    print(f'  {year}: {len(dates)} days', flush=True)
    return dates, np.stack(out).astype(np.float32), np.stack(out0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save-anom', help='also save the daily box anomalies (npz) for diagnostics')
    a = p.parse_args()

    with Pool(a.workers) as pool:
        res = pool.map(read_year, range(ALL_YEARS[0], ALL_YEARS[1] + 1))
    dates = np.array([d for r in res for d in r[0]])
    anom = np.concatenate([r[1] for r in res])                       # (n_day, nlat, nlon)
    anom0 = np.concatenate([r[2] for r in res])                      # 00Z
    if a.save_anom:
        np.savez(a.save_anom, dates=dates, anom=anom)
        print(f'saved anomalies to {a.save_anom}')
    ndays = np.array([(datetime.strptime(d, '%Y%m%d') - datetime(1979, 1, 1)).days for d in dates])
    assert (np.diff(ndays) == 1).all(), 'ERA5 daily series has gaps'

    month = np.array([int(d[4:6]) for d in dates])
    year = np.array([int(d[:4]) for d in dates])
    fit = np.isin(month, DJF) & (year >= FIT_YEARS[0]) & (year <= FIT_YEARS[1])
    # EOF fitted on DJF monthly means (Hurrell-style): the daily-data EOF1 mixes in the East
    # Atlantic pattern (southern centre over Europe, r=0.77 with the CPC daily NAO vs 0.83 here)
    months = sorted(set(zip(year[fit], month[fit])))
    X = np.stack([(anom[(year == y) & (month == m)].astype(np.float64) * WGT).mean(axis=0).ravel()
                  for y, m in months])
    X -= X.mean(axis=0)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    evr = s ** 2 / (s ** 2).sum()
    eof = vt[0].reshape(WGT.shape)
    ii = int(np.argmin(np.abs(lat[LAT_IDX] - ICELAND[0])))
    jj = int(np.where(LON_IDX == ICELAND[1])[0][0])
    if eof[ii, jj] > 0:          # NAO+ = negative height anomaly over Iceland
        eof = -eof

    pc = np.einsum('tij,ij->t', anom.astype(np.float64) * WGT, eof)
    pc_00z = np.einsum('tij,ij->t', anom0.astype(np.float64) * WGT, eof)
    sd_daily = pc[fit].std()
    # 7-day trailing means whose 7 days all fall in the DJF fitting window
    roll = np.convolve(pc, np.ones(7) / 7, mode='valid')             # roll[k] = mean(pc[k:k+7])
    ok = np.array([fit[k:k + 7].all() for k in range(len(roll))])
    sd_weekly = roll[ok].std()

    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(RES_DIR, f'nao_eof_{SEASON}.npz')
    np.savez(out, eof=eof, lat_idx=LAT_IDX, lon_idx=LON_IDX, wgt=WGT, evr=evr[:5],
             sd_daily=sd_daily, sd_weekly=sd_weekly, dates=dates, pc_daily=pc, pc_00z=pc_00z,
             fit_years=np.array(FIT_YEARS))
    print(f'EOF1 explains {evr[0]:.1%} (EOF2 {evr[1]:.1%}, EOF3 {evr[2]:.1%}) of {SEASON.upper()} monthly Z500 variance ({len(months)} months)')
    print(f'corr(daily-mean, 00Z index) = {np.corrcoef(pc, pc_00z)[0, 1]:.4f}')
    print(f'sd daily {sd_daily:.1f}, sd weekly {sd_weekly:.1f} (ratio {sd_weekly / sd_daily:.2f})')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
