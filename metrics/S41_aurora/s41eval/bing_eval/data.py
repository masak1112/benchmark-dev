"""Readers (one per system file format), the ERA5 verification store and the two climatological references.

Conventions (every system is converted to these on read):
  * grid g15 (``core``), lat ascending; window means of daily values by lead day (lead 0 = start day, week 1 = days 1-7).
  * OLR positive up, W m-2. UC-S2S and FuXi files hold net TOA long-wave (ERA5 sign: negative = upward) -> olr = -file value;
    the reader refuses a file whose OLR is not negative on average (e.g. the zero OLR of the clipped upstream writer before d7cede4).
    IFS ``ttr`` is accumulated J m-2 from step 0 -> olr = -(ttr(24k) - ttr(24(k-1))) / 86400.
  * tp24 in mm/day (files: m per 24 h).
  * u200 for UC-S2S / FuXi (no 200 hPa in the h25 files) = log-p interpolation of u150 and u250 (Bing's weights), flagged.
Verification sampling (per system time convention, config ``time``):
  ``00z``  states = ERA5 00 UTC of the valid date D; tp24 / olr = 24 h ending 00 UTC of D ('prev') — UC-S2S, IFS;
  ``day``  daily mean (00/06/12/18 UTC) of the calendar date D; tp24 / olr = 24 h from 00 UTC of D — FuXi (daily-mean model);
  per-variable overrides in the config (IFS 2t/sst are daily means of the 24 h ending at the step -> 'prev').
References (both 30-member same-calendar-date ERA5 ensembles of the window means, members = years 1991-2020):
  ``fixed``  x_y unshifted; its member mean is the fixed 1991-2020 climatology used for anomalies;
  ``trend``  x_y + b(d) (Y - y), b = 1991-2025 linear-trend slope (``clim_trend`` fit, smoothed by 6 harmonics), Y = start year
             (S33 convention, as ``bing_scores`` v0); anomalies use the fitted trend climatology at the valid year.
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from s41eval.bing_eval import core as K

FORMATS = {}


def register(name):
    def deco(f):
        FORMATS[name] = f
        return f
    return deco


class ReadError(Exception):
    pass


# ------------------------------------------------------------------ system readers
def sampling(sys_cfg: dict, var: str) -> str:
    ov = (sys_cfg.get('sampling_override') or {}).get(var)
    if ov:
        return ov
    if sys_cfg.get('time', '00z') == 'day':
        return 'day'
    return 'prev' if K.FIELDS[var].get('flux') else '00z'


def member_paths(sys_cfg: dict, init: date):
    """-> list of (member, path or None) for members 0..n-1 (None = missing/unreadable file)."""
    out = []
    tpl = sys_cfg['path']
    for m in range(int(sys_cfg['members'])):
        p = tpl.format(date=f'{init:%Y%m%d}', member=m, year=init.year)
        out.append((m, p if os.path.exists(p) else None))
    return out


NATIVE_MASKS = os.environ.get('BING_NATIVE_MASKS', '/lus/flare/projects/S2Storm/s2storm/data/era5_bing_1p5/ucs2s_1deg_masks.npz')
_NM = {}
# UC-S2S writes surface fields everywhere: upstream fills land SST / sea stl1 with 270 K and sea swvl1 with 0 (_fill_mask), and the
# model keeps them there. Remapping 1 deg -> 1.5 deg conservatively then mixes the fill into every coastal cell (lead-0 global-ocean
# SST -0.36 K vs ERA5 on 2025-01-06; -0.0003 K with the mask). Fix 2026-11-12 (v1.5): NaN outside each field's ERA5 domain first.
NATIVE_DOMAIN = {'sst': 'sea', 'swvl1': 'land', 'stl1': 'land'}


def native_mask(kind):
    if kind not in _NM:
        z = np.load(NATIVE_MASKS)
        _NM.update(sea=z['sea'].astype(bool), land=z['land'].astype(bool))
    return _NM[kind]


def read_h25_daily(path, init, fields, olr_sign=-1.0, check_olr=True, regrid=True):
    """UC-S2S / FuXi h25 member file -> ({field: daily [T, 121, 240] in output units (olr positive up)}, lead_days [T], info).
    Shared with the modes code (daily fields)."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        names = set(ds.variables)
        if 'lead_day' in names:
            ld = np.asarray(ds['lead_day'][:], int)
        else:
            t = ds['time']
            ld = np.asarray(np.round(netCDF4.date2num(netCDF4.num2date(t[:], t.units, t.calendar), f'days since {init:%Y-%m-%d}')), int)
        flip = ds['lat'][0] > ds['lat'][-1]

        def get(v):
            x = np.asarray(ds[v][:], np.float32)
            return x[..., ::-1, :] if flip else x
        out, info = {}, {'u200_src': None, 'lead_days': [int(ld[0]), int(ld[-1])]}
        for v in fields:
            if v == 'u200' and 'u200' not in names:
                if 'u150' in names and 'u250' in names:
                    x = K.W150 * get('u150') + (1 - K.W150) * get('u250')
                    info['u200_src'] = 'u150/u250 log-p'
                else:
                    continue
            elif v in names:
                x = get(v)
            else:
                continue
            if v == 'olr':
                gm = float(np.nanmean(x[ld >= 1]))
                if check_olr and not gm < -50.0:
                    raise ReadError(f'olr mean {gm:.1f} W m-2: expected net TOA LW < 0 (zero = upstream clip before d7cede4)')
                x = olr_sign * x
            x = x * K.FIELDS[v].get('scale', 1.0)
            if v in NATIVE_DOMAIN and x.shape[-2:] == (180, 360):
                x = np.where(native_mask(NATIVE_DOMAIN[v]), x, np.nan).astype(np.float32)
                info['native_mask'] = NATIVE_MASKS
            out[v] = K.to_g15(x) if regrid else x
        return out, ld, info


def _h25(path, init, fields, olr_sign=-1.0, check_olr=True):
    """-> ({field: window means [W, 121, 240]}, info). Window means are taken on the native grid, then remapped (linear, so the
    same as remapping the daily fields except at cells whose NaN pattern changes in time)."""
    daily, ld, info = read_h25_daily(path, init, fields, olr_sign, check_olr, regrid=False)
    return {v: K.to_g15(K.window_means(x, ld)) for v, x in daily.items()}, info


@register('ucs2s_h25')
def read_ucs2s_h25(path, init, fields, sys_cfg=None):
    """UC-S2S ``save_hindcast`` 27-field file (1 deg native or 1.5 deg), lead_day 0..45 at 00 UTC."""
    return _h25(path, init, fields)


@register('fuxi_h25')
def read_fuxi_h25(path, init, fields, sys_cfg=None):
    """FuXi-S2S file in the same layout (1.5 deg, daily means labelled 00 UTC, lead_day 0 = start-day ERA5 mean; olr = ttr/3600)."""
    return _h25(path, init, fields)


@register('netcdf_generic')
def read_netcdf_generic(path, init, fields, sys_cfg=None):
    """Pluggable reader for other NetCDF layouts (e.g. Bing's IFS when it arrives): config keys ``var_map`` {field: file var},
    ``scale`` {field: factor to output units}, ``olr_sign`` (+1 if the file already holds positive-up OLR), ``lead`` ('lead_day'
    variable or 'time' in days since the start). One member per file, dims [lead, lat, lon] on g15 or 1 deg."""
    import netCDF4
    cfg = sys_cfg or {}
    vm, sc = cfg.get('var_map', {}), cfg.get('scale', {})
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        if 'lead_day' in ds.variables:
            ld = np.asarray(ds['lead_day'][:], int)
        else:
            t = ds['time']
            ld = np.asarray(np.round(netCDF4.date2num(netCDF4.num2date(t[:], t.units), f'days since {init:%Y-%m-%d}')), int)
        latn = cfg.get('lat_name', 'lat' if 'lat' in ds.variables else 'latitude')
        flip = ds[latn][0] > ds[latn][-1]
        out = {}
        for v in fields:
            fv = vm.get(v)
            if fv is None or fv not in ds.variables:
                continue
            x = np.asarray(ds[fv][:], np.float32)
            x = x[..., ::-1, :] if flip else x
            if v == 'olr':
                x = float(cfg.get('olr_sign', -1.0)) * x
            x = x * float(sc.get(v, K.FIELDS[v].get('scale', 1.0)))
            out[v] = K.to_g15(K.window_means(x, ld))
    return out, {'lead_days': [int(ld[0]), int(ld[-1])]}


@register('ifs_ecds_grib')
def read_ifs_start(day_dir, init, fields, sys_cfg=None):
    """IFS ENS extended range from ECDS (``runs/bing/ifs/data/<YYYYMMDD>/<part>.grib``, 1.5 deg; split by ``ifs_ext.py``):
    all members of one start -> list of per-member dicts (window means, output units). Reuses ``bing_scores.read_ifs``
    (olr = -d(ttr)/86400 positive up; tp m; 2t/sst daily means at lead k = (k-1, k] days)."""
    from s41eval import bing_scores as B
    vs = [v for v in fields if v in B.VARS]
    recs = B.read_ifs(Path(day_dir), init, vs, mjo=False)
    return [{v: r[v] for v in vs if v in r} for r in recs], {'members': len(recs)}


# ------------------------------------------------------------------ ERA5 verification store
class Era5:
    """``<store>/<var>/<year>.npy`` [days of year incl. Feb 29, 4 slots, 121, 240] (``scripts/bing_eval_stage.py``)."""

    def __init__(self, store):
        self.store = Path(store)
        self.missing = []

    @lru_cache(maxsize=64)
    def _year(self, var, year):
        f = self.store / var / f'{year}.npy'
        return np.load(f, mmap_mode='r') if f.exists() else None

    def files(self, var):
        return sorted(str(p) for p in (self.store / var).glob('*.npy'))

    def day(self, var, samp, d: date):
        """field [121, 240] in file units for valid label date d, or None."""
        flux = K.FIELDS[var].get('flux')
        if samp == 'prev':
            return self.day(var, 'day', d - timedelta(days=1))
        A = self._year(var, d.year)
        if A is None:
            return None
        i = (d - date(d.year, 1, 1)).days
        if flux:
            x = np.asarray(A[i, 0])
        elif samp == '00z':
            x = np.asarray(A[i, 0])
        else:
            s = np.asarray(A[i])
            if not all(np.isfinite(s[h]).any() for h in range(4)):
                return None
            x = s.mean(0)
        return x if np.isfinite(x).any() else None

    def windows(self, var, samp, starts, log=None):
        """ERA5 window means [S, W, 121, 240] (output units) + notes (days interpolated / missing per start)."""
        S = len(starts)
        O = np.full((S, len(K.WINDOWS), 121, 240), np.nan, np.float32)
        notes = {}
        sc = K.FIELDS[var].get('scale', 1.0)
        for i, st in enumerate(starts):
            od = np.full((K.NLEAD, 121, 240), np.nan, np.float32)
            miss = []
            for k in range(1, K.NLEAD):
                x = self.day(var, samp, st + timedelta(days=k))
                if x is None:
                    miss.append(k)
                else:
                    od[k] = x
            n = K.fill_gaps(od)
            if miss:
                notes[str(st)] = dict(missing_leads=miss, interpolated=n)
            O[i] = K.window_means(od) * sc
        return O, notes


class Climatology:
    """stage arrays (``data/clim_trend*/stage/<var>_<year>.npy`` [365, 4, 240 lon, 121 lat]) and trend fits
    (``<dir>/<var>/coef_a.npy, coef_b.npy, clim_<Y>.npy``)."""

    def __init__(self, stage_dirs, trend_dirs, member_years=(1991, 2020)):
        self.stage_dirs = [Path(p) for p in stage_dirs]
        self.trend_dirs = [Path(p) for p in trend_dirs]
        self.years = list(range(member_years[0], member_years[1] + 1))
        self.used = set()

    def stage_file(self, var, year):
        for d in self.stage_dirs:
            f = d / f'{var}_{year}.npy'
            if f.exists():
                return f
        return None

    def trend_dir(self, var):
        for d in self.trend_dirs:
            if (d / var / 'coef_b.npy').exists():
                return d / var
        return None

    @staticmethod
    def hours(var, samp):
        return [0] if (samp == '00z' or K.FIELDS[var].get('flux')) else [0, 1, 2, 3]

    @lru_cache(maxsize=40)
    def member_year(self, var, hrs, year):
        """[365, 121, 240] mean over the hour slots ``hrs`` of the stage array (None if missing)."""
        f = self.stage_file(var, year)
        if f is None:
            return None
        self.used.add(str(f))
        A = np.load(f, mmap_mode='r')
        return np.swapaxes(np.asarray(A[:, list(hrs)], np.float32).mean(1), 1, 2).copy()

    @lru_cache(maxsize=8)
    def slope(self, var, hrs):
        """trend slope b(d) [365, 121, 240] per year, mean over hour slots (smoothed with the fit's harmonics)."""
        from s41eval.sci.clim_trend import harmonics
        d = self.trend_dir(var)
        if d is None:
            return None
        self.used.add(str(d / 'coef_b.npy'))
        cb = np.load(d / 'coef_b.npy')[:, list(hrs)].mean(1)           # [13, 240, 121]
        return np.swapaxes(np.einsum('dk,k...->d...', harmonics(), cb), 1, 2).astype(np.float32)

    @lru_cache(maxsize=16)
    def trend_mean(self, var, hrs, Y):
        """fitted trend climatology at year Y, [365, 121, 240] mean over the hour slots (clim_<Y>.npy if present, else from the
        coefficients with the NaN mask of clim_2025)."""
        from s41eval.sci.clim_trend import Y0, harmonics
        d = self.trend_dir(var)
        if d is None:
            return None
        f = d / f'clim_{Y}.npy'
        if f.exists():
            self.used.add(str(f))
            A = np.load(f, mmap_mode='r')
            return np.swapaxes(np.asarray(A[:, list(hrs)], np.float32).mean(1), 1, 2).copy()
        ca = np.load(d / 'coef_a.npy')[:, list(hrs)].mean(1)
        cb = np.load(d / 'coef_b.npy')[:, list(hrs)].mean(1)
        c = np.einsum('dk,k...->d...', harmonics(), ca + cb * (Y - Y0))
        if var == 'tp24':
            c = np.maximum(c, 0)
        m = np.load(d / 'clim_2025.npy', mmap_mode='r')
        bad = ~np.isfinite(np.asarray(m[0, 0]))
        c[:, bad] = np.nan
        self.used.update({str(d / 'coef_a.npy'), str(d / 'coef_b.npy')})
        return np.swapaxes(c, 1, 2).astype(np.float32)

    def ensembles(self, var, samp, st: date):
        """-> dict ref -> [n_years, W, 121, 240] window means (output units) for start st, plus 'trend_mean' / 'fixed_mean'
        [W, 121, 240]. Members missing a stage year are dropped (listed in self.dropped)."""
        hrs = tuple(self.hours(var, samp))
        shift = 1 if samp == 'prev' else 0
        sc = K.FIELDS[var].get('scale', 1.0)
        dates = [st + timedelta(days=k - shift) for k in range(1, K.NLEAD)]
        offs = [d.year - st.year for d in dates]
        js = np.array([K.doy365(d) for d in dates])
        b = self.slope(var, hrs)
        fx, tr = [], []
        for y in self.years:
            parts = {o: self.member_year(var, hrs, y + o) for o in set(offs)}
            if any(v is None for v in parts.values()):
                continue
            daily = np.full((K.NLEAD, 121, 240), np.nan, np.float32)
            for o, A in parts.items():
                sel = [k for k, oo in enumerate(offs) if oo == o]
                daily[np.array(sel) + 1] = A[js[sel]]
            dt = daily.copy()
            if b is not None:
                dt[1:] = daily[1:] + b[js] * (st.year - y)
            if var == 'tp24':
                daily, dt = np.clip(daily, 0, None), np.clip(dt, 0, None)
            fx.append(K.window_means(daily) * sc)
            tr.append(K.window_means(dt) * sc)
        out = {'fixed': np.stack(fx) if fx else None, 'trend': np.stack(tr) if (tr and b is not None) else None}
        # anomaly climatologies: fixed = member mean; trend = fitted trend climatology at the valid year
        out['fixed_mean'] = out['fixed'].mean(0) if fx else None
        cd = np.full((K.NLEAD, 121, 240), np.nan, np.float32)
        ok = True
        for k, d in enumerate(dates, start=1):
            C = self.trend_mean(var, hrs, d.year)
            if C is None:
                ok = False
                break
            cd[k] = C[K.doy365(d)]
        out['trend_mean'] = K.window_means(cd) * sc if ok else None
        out['n_members'] = len(fx)
        return out


# ------------------------------------------------------------------ v0 adapter (regression test only)
class V0Verif:
    """the verification and trend reference of ``bing_scores`` v0 (era5_verif00 / era5_verif / era5_bing_obs, 1 deg remapped)."""

    def __init__(self):
        from s41eval import bing_scores as B
        self.B = B

    def windows_and_trend(self, var, samp, starts):
        B = self.B
        O, C, used = B.obs_windows(var, samp, starts)
        CE = B.clim_ensemble(var, samp, starts)
        return O, C, CE, used


# ------------------------------------------------------------------ provenance helpers
def git_commit(path=None):
    import subprocess
    p = Path(path or Path(__file__).resolve().parents[3])
    try:
        h = subprocess.run(['git', '-C', str(p), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(p), 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True,
                               timeout=60).stdout.strip()
        return h + ('-dirty' if dirty else '')
    except Exception:  # noqa: BLE001
        return 'unknown'


def list_starts(csv_path, first='2019-01-01', last='2026-12-31'):
    import csv
    with open(csv_path) as f:
        return sorted({date.fromisoformat(r['start']) for r in csv.DictReader(f) if first <= r['start'] <= last})
