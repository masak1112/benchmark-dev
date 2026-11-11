"""
Per-member daily NAO index of every DJF init (Phase B extraction, Z500 over the NAO box only).

For each member and lead day 0..42 the forecast Z500 is put on the ERA5 1deg box (FuXi 1.5deg and
NeuralGCM Gaussian grids are bilinearly interpolated to the box points), the model's ERA5 DOY
climatology of the valid day is removed (00Z models: clim_00z, daily-mean models: clim_dailymean,
as in tcc_weekly.py) and the anomaly is projected onto the ERA5 DJF NAO EOF (nao_eof.py).

Output: /glade/derecho/scratch/bgong/nao_cache/{model}/{init}.npz
  idx (n_mem, 43) raw (unnormalised) index, leads 0..42; lead 0 is NaN where the model has no
  initial-time output (FuXi, NeuralGCM). n_members.

Members: the first 20 (sorted by member / seed number); SI-S2S inits with 16 members are kept
(Dec 2020: seed 3 missing), anything with fewer is skipped.

Usage:  python nao_extract.py --model fuxi --workers 32 [--dates-file F]
        python nao_extract.py --model ecmwf_rf2018 --months 11 12 01 02 03   (needs eccodes: .venv python)
        python nao_extract.py --model ecmwf_rt --workers 8                   (real-time 2019-, .venv python)
"""

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool

import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'TCC'))
from tcc_weekly import (CLIM_ROOT, MODELS, TMP_DIR, UC_PREFIX, _extract_7z, _member_num,  # noqa: E402
                        doy_index, era5_dates, lat, list_dates)

# NAO_SEASON=ndjfm: Nov-Mar NAO pattern (nao_eof_ndjfm.npz), Nov-Mar inits, separate cache nao_cache_ndjfm
SEASON = os.environ.get('NAO_SEASON', 'djf')
EOF_FILE = os.path.join(HERE, 'results', f'nao_eof_{SEASON}.npz')
CACHE = '/glade/derecho/scratch/bgong/nao_cache' + ('' if SEASON == 'djf' else f'_{SEASON}')
N_LEAD = 42
N_MEMBERS = 20
MIN_MEMBERS = {'si_s2s': 16, 'ucs2s_mc12800': 1, 'ucs2s_sstanom': 1}     # keep partially written inits
DJF = {'djf': ('12', '01', '02'), 'ndjfm': ('11', '12', '01', '02', '03')}[SEASON]   # init months

E = np.load(EOF_FILE)
LAT_IDX, LON_IDX, WGT, EOF = E['lat_idx'], E['lon_idx'], E['wgt'], E['eof']
BOX_LAT = lat[LAT_IDX]
BOX_LON = LON_IDX.astype(float)
J0, J1 = int(LAT_IDX[0]), int(LAT_IDX[-1]) + 1


def project(box_anom):
    """(..., nlat, nlon) box anomaly -> raw index (...)."""
    return np.einsum('...ij,ij->...', box_anom * WGT, EOF)


def interp_to_box(x, src_lat, src_lon):
    """(..., nlat_src, nlon_src) on a regular-in-index grid (lat S->N, lon 0..360) -> (..., box)."""
    src_lon = np.append(src_lon, 360.0)
    x = np.concatenate([x, x[..., :1]], axis=-1)
    LA, LO = np.meshgrid(np.clip(BOX_LAT, src_lat[0], src_lat[-1]), BOX_LON, indexing='ij')
    pts = np.stack([LA.ravel(), LO.ravel()], axis=-1)
    out = np.empty(x.shape[:-2] + LA.shape)
    for i in np.ndindex(x.shape[:-2]):
        out[i] = RegularGridInterpolator((src_lat, src_lon), x[i])(pts).reshape(LA.shape)
    return out


# ── readers: return (n_mem, N_LEAD + 1, nlat_box, nlon_box) Z500, NaN where a lead is missing ──

# UC-S2S variants with the UC-S2S V2 file layout: key -> (directory, filename prefix or tuple of prefixes)
UC_VARIANTS = {
    'ucs2s_v2': (MODELS['ucs2s_v2']['dir'], UC_PREFIX),
    'ucs2s_v2_fbcorr': ('/glade/derecho/scratch/bgong/pp_si_fbcorr_allvars',
                        'pangu_plasim_pp_si_infer_v2_fbcorr_allvars_24h_45step_'),
}
MODELS['ucs2s_v2_fbcorr'] = {'name': 'UC-S2S V2-fbcorr', 'daily_mean': False, 'clim': ('clim_00z', 'clim_00z_{}.npy')}
# multi-condition run: files for 2019 inits use prefix ..._multicond_2019_, 2020 inits ..._multicond_2020_
UC_VARIANTS['ucs2s_multicond'] = ('/glade/derecho/scratch/bgong/pp_si_infer_multicond_2019/predictions',
                                  tuple(f'pangu_plasim_pp_si_infer_multicond_{y}_24h_45step_' for y in range(2019, 2024)))
MODELS['ucs2s_multicond'] = {'name': 'UC-S2S multicond', 'daily_mean': False, 'clim': ('clim_00z', 'clim_00z_{}.npy')}
UC_VARIANTS['ucs2s_mc12800'] = ('/glade/derecho/scratch/bgong/pp_si_infer_multicond_pp12800/predictions',
                                tuple(f'pangu_plasim_pp_si_infer_multicond12800_{y}_24h_45step_' for y in range(2019, 2024)))
MODELS['ucs2s_mc12800'] = {'name': 'UC-S2S multicond pp12800', 'daily_mean': False, 'clim': ('clim_00z', 'clim_00z_{}.npy')}
UC_VARIANTS['ucs2s_sstanom'] = ('/glade/derecho/scratch/bgong/pp_si_infer_sstanom/predictions',
                                tuple(f'pangu_plasim_pp_si_infer_sstanom_{y}_24h_45step_' for y in range(2019, 2024)))
MODELS['ucs2s_sstanom'] = {'name': 'UC-S2S SSTanom', 'daily_mean': False, 'clim': ('clim_00z', 'clim_00z_{}.npy')}


def _prefixes(model):
    p = UC_VARIANTS[model][1]
    return p if isinstance(p, tuple) else (p,)


def uc_dates(model):
    d = UC_VARIANTS[model][0]
    return sorted({f[len(p):len(p) + 8] for f in os.listdir(d) for p in _prefixes(model)
                   if f.startswith(p) and '_ens_' in f and f.endswith('.nc')})   # any member (ens_0 may be missing)


def read_ucs2s(init, model='ucs2s_v2'):
    d = UC_VARIANTS[model][0]
    files = sorted((os.path.join(d, f) for f in os.listdir(d)
                    if any(f.startswith(f'{p}{init}') for p in _prefixes(model)) and f.endswith('.nc')),
                   key=lambda f: _member_num(f, '_ens_'))[:N_MEMBERS]
    out = []
    for fp in files:
        with netCDF4.Dataset(fp) as nc:
            t = np.asarray(nc.variables['time'][:]).astype(int)
            ti = [int(np.where(t == L)[0][0]) for L in range(N_LEAD + 1)]
            li = int(np.where(nc.variables['level'][:] == 500)[0][0])
            assert nc.variables['latitude'][0] < nc.variables['latitude'][-1]
            x = np.asarray(nc.variables['geopotential'][ti, li, J0:J1, :], dtype=np.float64)
            if not np.isfinite(x).all():                # e.g. a file still being written
                raise ValueError(f'non-finite Z500 in {os.path.basename(fp)}')
            out.append(x[..., LON_IDX])
    return np.stack(out) if out else None


def read_si_s2s(init):
    d = os.path.join(MODELS['si_s2s']['dir'], init)
    files = sorted((os.path.join(d, f) for f in os.listdir(d)
                    if f.startswith(f'ensemble_{init}_seed') and f.endswith('.nc')),
                   key=lambda f: _member_num(f, '_seed'))
    t0 = datetime.strptime(init, '%Y%m%d')
    out = []
    for fp in files:
        if len(out) >= N_MEMBERS:
            break
        with netCDF4.Dataset(fp) as nc:
            vt = era5_dates(nc.variables['time'])
            assert vt[0] == t0, f'{fp}: first time {vt[0]} != init {t0}'
            ti = [vt.index(t0 + timedelta(days=L)) for L in range(N_LEAD + 1)]
            li = int(np.where(nc.variables['plev'][:] == 50000.0)[0][0])
            assert nc.variables['lat'][0] < nc.variables['lat'][-1]
            take = min(nc.dimensions['member'].size, N_MEMBERS - len(out))
            x = np.asarray(nc.variables['geopotential'][:take, ti, li, J0:J1, :], dtype=np.float64)
            out.extend(x[..., LON_IDX])
    return np.stack(out) if out else None


def read_ngcm(init):
    import zarr
    store = os.path.join(MODELS['ngcm']['dir'], init[:4], f'{init[:4]}-{init[4:6]}-{init[6:]}T00.zarr')
    z = zarr.open(store, mode='r')
    leads = np.asarray(z['prediction_timedelta_daily'][:]).astype(int)
    src_lat = np.asarray(z['latitude'][:])
    src_lon = np.asarray(z['longitude'][:])
    assert src_lat[0] < src_lat[-1]
    n_mem = min(N_MEMBERS, z['number'].shape[0])
    x = np.asarray(z['geopotential_500_avg'][0, :n_mem], dtype=np.float64)   # (number, lead, [1,] lon, lat)
    if x.ndim == 5:
        x = x[:, :, 0]
    x = x.transpose(0, 1, 3, 2)                                                # -> (number, lead, lat, lon)
    out = np.full((n_mem, N_LEAD + 1, len(BOX_LAT), len(BOX_LON)), np.nan)
    for L in range(N_LEAD + 1):
        hit = np.where(leads == L)[0]
        if len(hit):
            out[:, L] = interp_to_box(x[:, int(hit[0])], src_lat, src_lon)
    return out


def _extract_fuxi_members(archive, dest, n):
    """Unpack only members 00..n-1 (the 51-member solid archive is disk-bound to unpack whole)."""
    import subprocess
    inc = [f'-i!*/member/{m:02d}/*' for m in range(n)]
    r = subprocess.run(['7zr', 'x', archive, '-o' + dest, '-y', '-aoa', '-r'] + inc, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())


def read_fuxi(init, n_unpack=N_MEMBERS):
    archive = os.path.join(MODELS['fuxi']['dir'], f'{init}.7z')
    tmp = tempfile.mkdtemp(prefix=f'nao_fuxi_{init}_', dir=TMP_DIR)
    try:
        if n_unpack and shutil.which('7zr'):
            _extract_fuxi_members(archive, tmp, n_unpack)
        else:
            _extract_7z(archive, tmp)
        base = next(p for p in (os.path.join(tmp, init[:4], init, 'member'),
                                os.path.join(tmp, init, 'member')) if os.path.isdir(p))
        members = sorted((m for m in os.listdir(base) if os.path.isdir(os.path.join(base, m))), key=int)
        out = []
        for m in members:
            if len(out) >= N_MEMBERS:
                break
            mdir = os.path.join(base, m)
            if not all(os.path.exists(os.path.join(mdir, f'{L:02d}.nc')) for L in range(1, N_LEAD + 1)):
                print(f'  {init} member {m}: incomplete leads, skipped', flush=True)
                continue
            raw = None
            for L in range(1, N_LEAD + 1):
                with netCDF4.Dataset(os.path.join(mdir, f'{L:02d}.nc')) as nc:
                    if raw is None:
                        ci = [str(c) for c in nc.variables['channel'][:]].index('z500')
                        src_lat = np.asarray(nc.variables['lat'][:], dtype=np.float64)
                        src_lon = np.asarray(nc.variables['lon'][:], dtype=np.float64)
                        raw = np.full((N_LEAD + 1, len(src_lat), len(src_lon)), np.nan)
                    assert int(nc.variables['lead_time'][0]) == L
                    raw[L] = np.asarray(nc.variables['__xarray_dataarray_variable__'][0, 0, ci], dtype=np.float64)
            if src_lat[0] > src_lat[-1]:
                src_lat, raw = src_lat[::-1], raw[:, ::-1]
            box = np.full((N_LEAD + 1, len(BOX_LAT), len(BOX_LON)), np.nan)
            box[1:] = interp_to_box(raw[1:], src_lat, src_lon)
            out.append(box)
        if n_unpack and len(out) < N_MEMBERS:                  # some of 00..n-1 incomplete: unpack everything
            shutil.rmtree(tmp, ignore_errors=True)
            return read_fuxi(init, n_unpack=None)
        return np.stack(out) if out else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


READERS = {'ucs2s_v2': read_ucs2s, 'si_s2s': read_si_s2s, 'ngcm': read_ngcm, 'fuxi': read_fuxi}


# ── ECMWF S2S reforecasts (ECMWF Data Store GRIB, download_ecmwf_s2s_reforecast.py) ──
# One cf + one pf file per model-version date, each holding 20 hindcast dates (same day/month,
# 20 earlier years). Every hindcast date becomes one cache entry keyed by its own init date.
# gh is in gpm (x g -> geopotential), instantaneous 00 UTC -> ERA5 clim_00z, like UC-S2S.
EC_MODELS = {
    # model year 2018 (CY45r1/43r3): hindcasts 1998-2017, NAO box only (lon 261 -> 409.5)
    'ecmwf_rf2018': {'dir': '/glade/derecho/scratch/bgong/ec_s2s_model_2018',
                     'file': 'ecmwf_rf_gh500_{tag}_{date}_h1998-2017.grib', 'year': '2018', 'members': 11},
    # model year 2025: hindcasts 2019-2021 and 2022-2024 (3 years per file), start dates every other day, global 1.5deg
    'ecmwf_rf2025': {'dir': '/glade/derecho/scratch/bgong/ec_s2s_model_2025_hindcast',
                     'file': ('ecmwf_rf_pl500_z_{tag}_{date}_h2019-2021.grib',
                              'ecmwf_rf_pl500_z_{tag}_{date}_h2022-2024.grib'), 'year': '2025', 'members': 11},
}
G = 9.80665


def _ec_files(cfg):
    f = cfg['file']
    return f if isinstance(f, tuple) else (f,)


def read_ecmwf_rf_pair(cfg, model_date, fpat):
    """cf + pf GRIB (file pattern fpat) of one model-version date -> {hindcast date: (11, N_LEAD + 1, box) Z500}."""
    import eccodes
    fields = {}                                         # (hdate, member) -> {step_day: 2-D}
    grid = None
    for tag in ('cf', 'pf'):
        path = os.path.join(cfg['dir'], fpat.format(tag=tag, date=model_date))
        with open(path, 'rb') as fh:
            while (gid := eccodes.codes_grib_new_from_file(fh)) is not None:
                try:
                    step = int(eccodes.codes_get(gid, 'step'))
                    if step % 24 or step // 24 > N_LEAD or eccodes.codes_get(gid, 'shortName') != 'gh':
                        continue
                    hdate = str(eccodes.codes_get(gid, 'dataDate'))
                    num = int(eccodes.codes_get(gid, 'number'))
                    ni, nj = eccodes.codes_get(gid, 'Ni'), eccodes.codes_get(gid, 'Nj')
                    if grid is None:
                        la0 = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
                        la1 = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
                        lo0 = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
                        dlo = eccodes.codes_get(gid, 'iDirectionIncrementInDegrees')
                        grid = (np.linspace(la0, la1, nj), lo0 + dlo * np.arange(ni))
                    x = eccodes.codes_get_values(gid).reshape(nj, ni) * G
                    fields.setdefault((hdate, (tag, num)), {})[step // 24] = x
                finally:
                    eccodes.codes_release(gid)
    src_lat, src_lon = grid                             # lat N->S, lon continuous (may exceed 360)
    order = np.argsort(src_lat)
    src_lat = src_lat[order]
    dlo = src_lon[1] - src_lon[0]
    periodic = abs(len(src_lon) * dlo - 360.0) < 1e-6   # global grid: wrap longitude
    if periodic:
        src_lon = np.append(src_lon, src_lon[0] + 360.0)
    # box longitudes in the same continuous frame as the source grid
    box_lon = np.where(BOX_LON < src_lon[0], BOX_LON + 360.0, BOX_LON)
    LA, LO = np.meshgrid(BOX_LAT, box_lon, indexing='ij')
    pts = np.stack([LA.ravel(), LO.ravel()], axis=-1)
    out = {}
    for hdate in sorted({k[0] for k in fields}):
        mems = sorted(k[1] for k in fields if k[0] == hdate)   # ('cf', 0) first, then ('pf', 1..10)
        arr = np.full((len(mems), N_LEAD + 1, len(BOX_LAT), len(BOX_LON)), np.nan)
        for i, mm in enumerate(mems):
            for L, x in fields[(hdate, mm)].items():
                xs = x[order]
                if periodic:
                    xs = np.concatenate([xs, xs[:, :1]], axis=1)
                f = RegularGridInterpolator((src_lat, src_lon), xs)
                arr[i, L] = f(pts).reshape(LA.shape)
        out[hdate] = arr
    return out


def process_ecmwf(args):
    """One model-version date -> one cache file per hindcast date."""
    model, (fpat, model_date) = args
    cfg = EC_MODELS[model]
    try:
        clim = np.load(os.path.join(CLIM_ROOT, 'clim_00z', 'clim_00z_z500.npy'), mmap_mode='r')
        msgs = []
        for hdate, x in read_ecmwf_rf_pair(cfg, model_date, fpat).items():
            out = os.path.join(CACHE, model, f'{hdate}.npz')
            if os.path.exists(out):
                continue
            n = x.shape[0]
            if n < cfg['members'] or np.isnan(x[:, 1:]).any():
                msgs.append(f'{hdate} SKIPPED ({n} members, missing leads {int(np.isnan(x[:, 1:]).any())})')
                continue
            t0 = datetime.strptime(hdate, '%Y%m%d')
            c = np.stack([np.asarray(clim[doy_index(t0 + timedelta(days=L))][J0:J1][:, LON_IDX], dtype=np.float64)
                          for L in range(N_LEAD + 1)])
            idx = project(x - c)
            np.savez(out + '.tmp.npz', idx=idx, n_members=n, model_date=model_date)
            os.replace(out + '.tmp.npz', out)
        return model_date, 'ok' + (f'; {msgs}' if msgs else '')
    except Exception as e:
        return model_date, f'ERROR {type(e).__name__}: {e}'


# ── ECMWF S2S real-time forecasts (download_ecmwf_s2s_mjo.py "other vars": ecmwf_pl500_z_u_{cf,pf}_{date}.grib) ──
# Global 1.5deg (lat N->S, lon 0..358.5), gh [gpm] + u at 500 hPa, cf + 50 pf, steps 0..1104 h.
# Uses cf + pf 1..(N_MEMBERS-1) so the ensemble size matches the ML models; 00 UTC -> clim_00z.
EC_RT_DIR = '/glade/derecho/scratch/bgong/ec_s2s_model_2025'
MODELS['ecmwf_rt'] = {'name': 'ECMWF', 'daily_mean': False, 'clim': ('clim_00z', 'clim_00z_{}.npy')}


def read_ecmwf_rt(init):
    import eccodes
    fields = {}                                         # member number -> {lead day: 2-D}
    grid = None
    for tag in ('cf', 'pf'):
        with open(os.path.join(EC_RT_DIR, f'ecmwf_pl500_z_u_{tag}_{init}.grib'), 'rb') as fh:
            while (gid := eccodes.codes_grib_new_from_file(fh)) is not None:
                try:
                    if eccodes.codes_get(gid, 'shortName') != 'gh':
                        continue
                    step, num = int(eccodes.codes_get(gid, 'step')), int(eccodes.codes_get(gid, 'number'))
                    if step % 24 or step // 24 > N_LEAD or num >= N_MEMBERS:
                        continue
                    assert str(eccodes.codes_get(gid, 'dataDate')) == init
                    ni, nj = eccodes.codes_get(gid, 'Ni'), eccodes.codes_get(gid, 'Nj')
                    if grid is None:
                        la0 = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
                        la1 = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
                        lo0 = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
                        dlo = eccodes.codes_get(gid, 'iDirectionIncrementInDegrees')
                        grid = (np.linspace(la0, la1, nj), lo0 + dlo * np.arange(ni))
                    fields.setdefault(num, {})[step // 24] = eccodes.codes_get_values(gid).reshape(nj, ni) * G
                finally:
                    eccodes.codes_release(gid)
    if not fields:
        return None
    src_lat, src_lon = grid
    assert src_lon[0] == 0 and abs(src_lon[-1] + (src_lon[1] - src_lon[0]) - 360) < 1e-6, 'expected a global grid'
    order = np.argsort(src_lat)
    out = np.full((len(fields), N_LEAD + 1, len(BOX_LAT), len(BOX_LON)), np.nan)
    for i, num in enumerate(sorted(fields)):            # 0 = control, then perturbed 1..
        leads = sorted(fields[num])
        x = np.stack([fields[num][L][order] for L in leads])
        out[i, leads] = interp_to_box(x, src_lat[order], src_lon)
    return out


READERS['ecmwf_rt'] = read_ecmwf_rt
READERS['ucs2s_v2_fbcorr'] = lambda init: read_ucs2s(init, 'ucs2s_v2_fbcorr')
READERS['ucs2s_multicond'] = lambda init: read_ucs2s(init, 'ucs2s_multicond')
READERS['ucs2s_mc12800'] = lambda init: read_ucs2s(init, 'ucs2s_mc12800')
READERS['ucs2s_sstanom'] = lambda init: read_ucs2s(init, 'ucs2s_sstanom')


def ecmwf_rt_dates():
    """Init dates with both cf and pf z500 files."""
    have = {}
    for f in os.listdir(EC_RT_DIR):
        if f.startswith('ecmwf_pl500_z_u_') and f.endswith('.grib'):
            p = f[:-5].split('_')                       # ecmwf pl500 z u {cf|pf} {date}
            have.setdefault(p[5], set()).add(p[4])
    return sorted(d for d, t in have.items() if t == {'cf', 'pf'})


def ecmwf_model_dates(model, months):
    """(file pattern, model-version date) with BOTH cf and pf files present, init month in `months`."""
    import re
    cfg = EC_MODELS[model]
    files = os.listdir(cfg['dir'])
    out = []
    for fpat in _ec_files(cfg):
        pat = re.compile(re.escape(fpat).replace(r'\{tag\}', '(cf|pf)').replace(r'\{date\}', r'(\d{8})') + '$')
        have = {}
        for f in files:
            m = pat.match(f)
            if m and m.group(2)[:4] == cfg['year']:
                have.setdefault(m.group(2), set()).add(m.group(1))
        out += [(fpat, d) for d, t in sorted(have.items()) if t == {'cf', 'pf'} and d[4:6] in months]
    return out


def process(args):
    model, init = args
    out = os.path.join(CACHE, model, f'{init}.npz')
    if os.path.exists(out):
        return init, 'cached'
    try:
        x = READERS[model](init)
        n = 0 if x is None else x.shape[0]
        if n < MIN_MEMBERS.get(model, N_MEMBERS):
            return init, f'SKIPPED: {n} members'
        cdir, cpat = MODELS[model]['clim']
        clim = np.load(os.path.join(CLIM_ROOT, cdir, cpat.format('z500')), mmap_mode='r')
        t0 = datetime.strptime(init, '%Y%m%d')
        c = np.stack([np.asarray(clim[doy_index(t0 + timedelta(days=L))][J0:J1][:, LON_IDX], dtype=np.float64)
                      for L in range(N_LEAD + 1)])
        idx = project(x - c)                                       # (n_mem, N_LEAD + 1)
        np.savez(out + '.tmp.npz', idx=idx, n_members=n)
        os.replace(out + '.tmp.npz', out)
        return init, f'ok ({n} members)'
    except Exception as e:
        return init, f'ERROR {type(e).__name__}: {e}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=sorted(set(READERS) | set(EC_MODELS)))
    p.add_argument('--months', nargs='+', default=list(DJF),
                   help='ECMWF reforecasts only: init months to extract (MM), default DJF')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--dates-file', help='YYYYMMDD per line (default: every DJF init of the model)')
    a = p.parse_args()
    if a.model in EC_MODELS:
        mdates = ecmwf_model_dates(a.model, set(a.months))
        os.makedirs(os.path.join(CACHE, a.model), exist_ok=True)
        print(f'{a.model}: {len(mdates)} model-version dates with cf+pf, months {a.months}', flush=True)
        with Pool(a.workers) as pool:
            for k, (d, msg) in enumerate(pool.imap_unordered(process_ecmwf, [(a.model, fd) for fd in mdates]), 1):
                print(f'[{k}/{len(mdates)}] {d}: {msg}', flush=True)
        return
    lister = {'ecmwf_rt': ecmwf_rt_dates, 'ucs2s_v2_fbcorr': lambda: uc_dates('ucs2s_v2_fbcorr'),
              'ucs2s_multicond': lambda: uc_dates('ucs2s_multicond'),
              'ucs2s_mc12800': lambda: uc_dates('ucs2s_mc12800'),
              'ucs2s_sstanom': lambda: uc_dates('ucs2s_sstanom')}
    dates = [d for d in lister.get(a.model, lambda: list_dates(a.model))() if d[4:6] in DJF]
    if a.dates_file:
        want = {l.strip() for l in open(a.dates_file) if l.strip()}
        dates = [d for d in dates if d in want]
    os.makedirs(os.path.join(CACHE, a.model), exist_ok=True)
    print(f'{a.model}: {len(dates)} DJF inits', flush=True)
    with Pool(a.workers) as pool:
        for k, (init, msg) in enumerate(pool.imap_unordered(process, [(a.model, d) for d in dates]), 1):
            print(f'[{k}/{len(dates)}] {init}: {msg}', flush=True)


if __name__ == '__main__':
    main()
