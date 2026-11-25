"""Modes and patterns evaluation of the S41 hindcasts (plan 2026-11-11 section 3 B): MJO (WH04 RMM and winds-only RMM), NAO / PNA /
AO / SAM, MJO -> extratropics, stratosphere (U60N10, polar-cap height, SSWs), ENSO / SST vs damped persistence, land, QBO drift.

Conventions (recorded choices; deviations from the plan are marked DEVIATION)
  Grid      everything on the WB2 1.5-degree grid (121 x 240, lat -90..90 ascending internally, lon 0..358.5); UC-S2S 1-degree fields are
            remapped conservatively (``bing_scores.to15``); FuXi is native 1.5 degree (lat flipped).
  Leads     lead day k = valid date init + k; windows as ``bing_scores.WINDOWS`` (w1 = days 1-7 ... w6, w34 = 15-28, w56 = 29-42).
  Sampling  ERA5 is sampled like each system's output: '00z' (UC-S2S, IFS): states at 00 UTC of D, fluxes (olr, tp24) = calendar
            day D-1 (24 h ending 00 UTC of D); 'day' (FuXi, daily means labelled D): calendar-day means of D (stage hours 00/06/12/18;
            where only 00 UTC is staged, (00 UTC D + 00 UTC D+1)/2), fluxes = day D.
  ERA5      ``data/clim_trend/stage`` (z500, t2m, tp24, msl, swvl1; 1979/1991-2025), ``data/clim_trend_bing/stage`` (olr, u850, u200,
            sst ...; 1991-2025), ``data/era5_modes/stage`` (olr, u850, u200 1979-1990, this module's staging), 2026 from
            ``data/era5_verif00`` / ``era5_verif`` / ``era5_bing_obs``. Cached as daily memmaps in ``data/era5_modes/daily``.
  Anomalies fixed climatology = mean + 4 annual harmonics fitted per grid point over 1991-2020 (365-day calendar, Feb 29 = Feb 28),
            per sampling. MJO uses its own 1979-2018 3-harmonic climatology (WH04).
  Drift     'dc' = lead-dependent mean error removed leave-one-year-out within +-30 calendar days (``biascorr.training_sets``), per
            system, applied to the index (RMM, NAO, ...) or field window means (linear quantities: same as correcting the fields).
  Ensembles ensemble-mean scores at 30 members and at 11 members (mean of the score over 10 random 11-member draws, rng seed 11);
            field-based 11-member scores (SST/land maps) use members 0-10 (members are exchangeable).

MJO (WH04)  OLR (positive up, W m-2), u850, u200 averaged 15S-15N (cos-lat), anomalies vs the 1979-2018 mean + 3 harmonics, minus the
            mean of the PREVIOUS 120 days (days d-120..d-1; WH04, not MJOcast's window that includes day d), each field divided by
            sqrt(zonal mean of its 1979-2018 variance), covariance EOFs of the combined 1979-2018 'day' series, PCs scaled to unit
            variance over 1979-2018. ENSO is not regressed out (BoM did this before 2014 only). Sign: RMM1 > 0 = negative OLR over
            100-150E (phases 4-5 = Maritime Continent), RMM2 lags RMM1 (eastward propagation; phases 6-7 = western Pacific); then
            checked against the BoM RMM (blocking test). Winds-only RMM: same with (u850, u200), its order/sign aligned to the full RMM.
            Forecasts: the 120-day window uses ERA5 (same sampling) for days up to and including the init day and the forecast for
            leads >= 1 (lead 0 of every system is replaced by ERA5). UC-S2S and FuXi have no 200 hPa in the member files:
            u200 = log-p interpolation of u150/u250 (Bing's weight 0.437 on 150 hPa) for both, recorded in ``u200_src``.
            Scores: bivariate COR (uncentred, Rashid et al. 2011) and RMSE of the ensemble-mean RMM vs ERA5 RMM per lead day;
            skill day = first lead where COR < 0.5 (linear interpolation between lead days); amplitude error mean(|F| - |O|);
            phase error mean of the signed angle from O to F (degrees, > 0 = forecast ahead / too fast). By init state = ERA5 RMM at
            the init day: active (amplitude >= 1) / inactive, phase pairs 8-1, 2-3, 4-5, 6-7 (active only).
NAO / PNA   CPC-like rotated PCA: for each calendar month m, monthly-mean z500 anomalies of months m-1, m, m+1 1979-2018 over 20-90N,
            sqrt(cos lat) weighted (covariance; not standardised per grid point, see TelePatterns), 10 leading EOFs varimax-rotated; NAO and PNA = the rotated modes that
            best match the Azores-Iceland dipole / Wallace-Gutzler PNA point pattern; index = least-squares fit of the standardised
            anomaly on all 10 rotated loadings. AO / SAM: EOF1 of the same anomalies over 20-90N / 20-90S. Before projection the
            cos-lat domain mean of the anomaly is removed (a uniform z500 rise from warming would otherwise project on AO / SAM).
            Weekly / window indices: the window-mean anomaly projected with the pattern set of the calendar month of the window's
            middle day, divided by the std of the ERA5 index of 7-day means 1979-2018 in that month window.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from s41eval import bing_scores as BS

R = BS.R
DATA = BS.DATA
MD = DATA / 'era5_modes'
DAILY = MD / 'daily'
IDX = DATA / 'indices_modes'
OUT = R / 'runs' / 'bing' / 'eval' / 'modes'
STAGE_DIRS = [DATA / 'clim_trend' / 'stage', DATA / 'clim_trend_bing' / 'stage', MD / 'stage']
LAT15, LON15 = BS.LAT15, BS.LON15
NLEAD = BS.NLEAD
WINDOWS = BS.WINDOWS
WK = list(WINDOWS)
W150 = BS.W150
BAND = 15.0
FIRST = {'default': date(1991, 1, 1), 'mjo': date(1979, 1, 1), 'z500': date(1979, 1, 1)}
CLIM_BASE = (1991, 2020)
EOF_YEARS = (1979, 2018)
SAMPLING = {'ucs2s_sstanom': '00z', 'ucs2s_plain': '00z', 'fuxi': 'day', 'ifs': 'ifs'}   # 'ifs': states 00 UTC, 2t/sst/fluxes (D-1, D]
SYSTEM_LABEL = {'ucs2s_sstanom': 'UC-S2S SST-anomaly', 'ucs2s_plain': 'UC-S2S plain', 'fuxi': 'FuXi-S2S', 'ifs': 'IFS'}
NINO34 = (-5.0, 5.0, 190.0, 240.0)
SEASONS = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}
RNG_SEED = 11
N_DRAWS = 10


def doy365(d: date) -> int:
    return BS.doy365(d)


def daterange(a: date, b: date):
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


def cosw_lat(lat=LAT15):
    return np.cos(np.deg2rad(lat))


def harm(doy, nh):
    t = 2 * np.pi * np.asarray(doy, float) / 365.0
    cols = [np.ones_like(t)]
    for k in range(1, nh + 1):
        cols += [np.cos(k * t), np.sin(k * t)]
    return np.stack(cols, -1)


# ====================================================================== ERA5 daily cache
@lru_cache(maxsize=64)
def stage_arr(var, year):
    for d in STAGE_DIRS:
        f = d / f'{var}_{year}.npy'
        if f.exists() and BS.readable(f):
            return np.load(f, mmap_mode='r')
    return None


def _stage_day(var, d: date, hours):
    A = stage_arr(var, d.year)
    if A is None:
        return None
    x = np.asarray(A[doy365(d), hours], np.float32)                   # [h, 240, 121]
    if not np.isfinite(x).any(axis=(1, 2)).all():
        return None
    return x.mean(0).T


def _late(var, d: date, kind):
    """2026 (and any day missing in the stage) from the verification stores; [121, 240] lat ascending or None."""
    if kind == 'flux':
        if var == 'olr':
            return BS._get(BS.VEXTRA, d, 'olr_day')
        return BS._get(BS.VDAY, d, var)
    if kind == '00':
        x = BS._get(BS.V00, d, var)
        return x if x is not None else BS._get(BS.VEXTRA, d, f'{var}_00')
    x = BS._get(BS.VDAY, d, var) if var != 'sst' else None
    return x if x is not None else BS._get(BS.VEXTRA, d, f'{var}_day')


def era5_raw(var, d: date, kind):
    """kind '00' (00 UTC state), 'mean4' (calendar-day mean of 00/06/12/18 UTC), 'flux' (daily value of day d: olr, tp24)."""
    if kind == 'flux':
        x = _stage_day(var, d, [0])
        return x if x is not None else _late(var, d, 'flux')
    if kind == '00':
        x = _stage_day(var, d, [0])
        return x if x is not None else _late(var, d, '00')
    x = _stage_day(var, d, [0, 1, 2, 3])
    if x is not None:
        return x
    x = _late(var, d, 'mean4') if d.year >= 2026 or stage_arr(var, d.year) is None else None
    if x is not None:
        return x
    a, b = era5_raw(var, d, '00'), era5_raw(var, d + timedelta(days=1), '00')
    if a is None or b is None:
        return None
    return 0.5 * (a + b)


DAILY_SPEC = {   # name -> (var, kind, first date, band only)
    'z500_00': ('z500', '00', FIRST['z500'], False), 'z500_mean4': ('z500', 'mean4', FIRST['z500'], False),
    't2m_00': ('t2m', '00', FIRST['default'], False), 't2m_mean4': ('t2m', 'mean4', FIRST['default'], False),
    'tp24_flux': ('tp24', 'flux', FIRST['default'], False),
    'sst_00': ('sst', '00', FIRST['default'], False), 'sst_mean4': ('sst', 'mean4', FIRST['default'], False),
    'swvl1_00': ('swvl1', '00', FIRST['default'], False), 'stl1_00': ('stl1', '00', FIRST['default'], False), 'swvl1_mean4': ('swvl1', 'mean4', FIRST['default'], False),
    'olr_flux_band': ('olr', 'flux', FIRST['mjo'], True), 'u850_00_band': ('u850', '00', FIRST['mjo'], True),
    'u200_00_band': ('u200', '00', FIRST['mjo'], True),
}


def band_mean(x, lat=LAT15, band=BAND):
    """[..., lat, lon] -> cos-lat mean over |lat| <= band -> [..., lon]."""
    m = np.abs(lat) <= band + 1e-6
    w = cosw_lat(lat[m])
    return (np.asarray(x)[..., m, :] * w[:, None]).sum(-2) / w.sum()


def build_daily(name, last: date, log=print):
    """write DAILY/<name>.npy ([n, 121, 240] or [n, 240] float32, NaN where missing) + <name>.json (first, last, n_missing)."""
    var, kind, first, band = DAILY_SPEC[name]
    BS.mkdir_striped(DAILY)
    days = daterange(first, last)
    shape = (len(days), 240) if band else (len(days), 121, 240)
    tmp = DAILY / f'{name}.partial.npy'
    A = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=shape)
    miss = []
    for i, d in enumerate(days):
        x = era5_raw(var, d, kind)
        if x is None:
            A[i] = np.nan
            miss.append(d.isoformat())
            continue
        A[i] = band_mean(x) if band else x
        if log and d.month == 1 and d.day == 1:
            log(f'{name} {d.year}')
    A.flush()
    del A
    os.replace(tmp, DAILY / f'{name}.npy')
    meta = dict(name=name, var=var, kind=kind, first=first.isoformat(), last=last.isoformat(), n=len(days), n_missing=len(miss),
                missing=miss[:400], grid='WB2 1.5 deg, lat -90..90' + (', 15S-15N cos-lat band mean' if band else ''))
    (DAILY / f'{name}.json').write_text(json.dumps(meta, indent=1))
    return meta


class Daily:
    """read access to the daily ERA5 caches by date (explicit sequential reads: memmap page faults on Lustre are very slow)."""

    def __init__(self, name):
        self.name = name
        self.meta = json.loads((DAILY / f'{name}.json').read_text())
        self.first = date.fromisoformat(self.meta['first'])
        self.last = date.fromisoformat(self.meta['last'])
        self.path = DAILY / f'{name}.npy'
        mm = np.load(self.path, mmap_mode='r')
        self.shape, self.dtype, self.offset = mm.shape, mm.dtype, mm.offset
        self.rowbytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        del mm
        self._full = None

    @property
    def A(self):
        """the whole array in memory (read once)."""
        if self._full is None:
            self._full = self.slab(0, self.shape[0])
        return self._full

    def slab(self, i0, i1):
        i0, i1 = max(i0, 0), min(i1, self.shape[0])
        if i1 <= i0:
            return np.empty((0,) + self.shape[1:], self.dtype)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            chunks, pos, n = [], self.offset + i0 * self.rowbytes, (i1 - i0) * self.rowbytes
            while n > 0:                                     # os.pread: one large read (buffered readinto is ~50x slower here)
                b = os.pread(fd, min(n, 1 << 30), pos)
                if not b:
                    raise IOError(f'short read {self.path}')
                chunks.append(b)
                pos += len(b)
                n -= len(b)
        finally:
            os.close(fd)
        buf = chunks[0] if len(chunks) == 1 else b''.join(chunks)
        return np.frombuffer(buf, self.dtype).reshape((i1 - i0,) + self.shape[1:]).copy()

    def idx(self, d):
        return (d - self.first).days

    def get(self, days):
        """list of dates -> array [n, ...] (NaN outside the record)."""
        out = np.full((len(days),) + self.shape[1:], np.nan, np.float32)
        ii = np.array([self.idx(d) for d in days])
        ok = (ii >= 0) & (ii < self.shape[0])
        if not ok.any():
            return out
        if self._full is not None:
            out[ok] = self._full[ii[ok]]
            return out
        a, b = int(ii[ok].min()), int(ii[ok].max()) + 1
        if b - a <= 4 * ok.sum() + 64:
            blk = self.slab(a, b)
            out[ok] = blk[ii[ok] - a]
        else:
            for j in np.flatnonzero(ok):
                out[j] = self.slab(ii[j], ii[j] + 1)[0]
        return out

    def dates(self):
        return daterange(self.first, self.last)


@lru_cache(maxsize=32)
def daily(name):
    return Daily(name)


def state_samp(samp):
    """sampling of instantaneous states / of the MJO and strat series ('ifs' -> '00z')."""
    return '00z' if samp == 'ifs' else samp


def obs_name(var, samp):
    """cache name and day shift for ERA5 matching a system's sampling (see module docstring)."""
    if samp == 'ifs':
        if var in ('t2m', 'sst'):
            return f'{var}_mean4', -1
        samp = '00z'
    if var in ('tp24', 'olr'):
        return (f'{var}_flux' + ('_band' if var == 'olr' else '')), (-1 if samp == '00z' else 0)
    return f'{var}_{"00" if samp == "00z" else "mean4"}', 0


def obs_days(var, samp, days):
    """ERA5 [n, ...] for valid label dates ``days`` in sampling samp."""
    name, sh = obs_name(var, samp)
    return daily(name).get([d + timedelta(days=sh) for d in days])


# ====================================================================== climatology (harmonic fit)
class HarmClim:
    """per grid point mean + nh annual harmonics over base years (365-day calendar)."""

    def __init__(self, X, days, base=CLIM_BASE, nh=4):
        tr = np.array([base[0] <= d.year <= base[1] for d in days])
        ok = tr & np.isfinite(X.reshape(len(days), -1)).any(1)
        H = harm([doy365(d) for d in days], nh)
        Xf = X[ok].reshape(ok.sum(), -1).astype(np.float64)
        good = np.isfinite(Xf).all(0)
        coef = np.full((H.shape[1], Xf.shape[1]), np.nan)
        coef[:, good] = np.linalg.lstsq(H[ok], Xf[:, good], rcond=None)[0]
        self.coef = coef.reshape((H.shape[1],) + X.shape[1:])
        self.nh = nh
        self.base = base

    def __call__(self, days):
        H = harm([doy365(d) for d in days], self.nh)
        return np.tensordot(H, self.coef, axes=(1, 0)).astype(np.float32)


def clim_path(name, base, nh):
    return MD / 'clim' / f'{name}_{base[0]}-{base[1]}_h{nh}.npy'


def fixed_clim(name, base=CLIM_BASE, nh=4):
    """HarmClim of a daily cache, cached on disk (coefficients)."""
    f = clim_path(name, base, nh)
    if f.exists():
        c = HarmClim.__new__(HarmClim)
        c.coef, c.nh, c.base = np.load(f), nh, base
        return c
    D = daily(name)
    days = D.dates()
    sel = [i for i, d in enumerate(days) if base[0] <= d.year <= base[1]]
    X = D.slab(sel[0], sel[-1] + 1)
    c = HarmClim(X, days[sel[0]:sel[-1] + 1], base, nh)
    if not f.parent.exists():
        BS.mkdir_striped(f.parent)
    tmp = f.with_name(f'{f.stem}.{os.getpid()}.partial.npy')
    np.save(tmp, c.coef)
    os.replace(tmp, f)
    return c


def obs_clim(var, samp, days, base=CLIM_BASE):
    name, sh = obs_name(var, samp)
    return fixed_clim(name, base)([d + timedelta(days=sh) for d in days])


# ====================================================================== window helpers
def window_mean_daily(x, lead_axis=0):
    """x [46, ...] by lead day 0..45 -> [W, ...] (NaN if any day is missing)."""
    x = np.moveaxis(np.asarray(x, np.float32), lead_axis, 0)
    out = []
    for a, b in WINDOWS.values():
        seg = x[a:b + 1]
        out.append(np.where(np.isfinite(seg).all(0), seg.mean(0), np.nan) if seg.shape[0] == b - a + 1 else np.full(x.shape[1:], np.nan))
    return np.moveaxis(np.stack(out, 0), 0, lead_axis)


def valid_dates(init):
    return [init + timedelta(days=k) for k in range(NLEAD)]


def window_mid(init, w):
    a, b = WINDOWS[w]
    return init + timedelta(days=(a + b) // 2)


def loyo_bias(err, starts):
    """err [S, ...] forecast - obs -> LOYO +-30 d mean error per start [S, ...] (NaN-aware; NaN if no training start)."""
    from s41eval.biascorr import training_sets
    tr = training_sets(list(starts))
    out = np.full(err.shape, np.nan, np.float64)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for s, idx in enumerate(tr):
            if idx:
                out[s] = np.nanmean(err[idx], 0)
    return out


def draws11(M, n=N_DRAWS, k=11, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    if M <= k:
        return [np.arange(M)]
    return [np.sort(rng.choice(M, k, replace=False)) for _ in range(n)]


# ====================================================================== MJO
def rmm_phase(r1, r2):
    """WH04 phase 1..8 (phase 1: angle in [-180, -135) deg, counter-clockwise)."""
    ang = np.arctan2(r2, r1)
    return (np.floor((ang + np.pi) / (np.pi / 4)).astype(int) % 8) + 1


def prev120(a, ok_days=None):
    """a [n, L] daily (consecutive days) -> a - mean of the previous 120 days (d-120..d-1); NaN unless all 120 are finite."""
    a = np.asarray(a, np.float64)
    fin = np.isfinite(a).all(1)
    z = np.where(fin[:, None], a, 0.0)
    cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(z, 0)], 0)
    cn = np.concatenate([[0], np.cumsum(fin)])
    out = np.full_like(a, np.nan)
    n = a.shape[0]
    i = np.arange(120, n)
    full = (cn[i] - cn[i - 120]) == 120
    ii = i[full & fin[i]]
    out[ii] = a[ii] - (cs[ii] - cs[ii - 120]) / 120.0
    return out


class MJOIndex:
    """WH04 RMM (fields olr, u850, u200) or winds-only RMM (u850, u200); see module docstring."""

    def __init__(self, series_fit, days, fields=('olr', 'u850', 'u200'), eof_years=EOF_YEARS, ref=None):
        self.fields = tuple(fields)
        self.eof_years = eof_years
        self.days = list(days)
        tr = np.array([eof_years[0] <= d.year <= eof_years[1] for d in days])
        self.clim_coef = {}
        F = {}
        for v in self.fields:
            c, A = self._anom(series_fit[v], days, tr)
            self.clim_coef[('fit', v)] = c
            F[v] = prev120(A)
        ok = tr & np.all([np.isfinite(F[v]).all(1) for v in self.fields], 0)
        self.n_fit = int(ok.sum())
        self.norm = {v: float(np.sqrt(np.nanvar(F[v][ok], 0).mean())) for v in self.fields}
        Z = np.concatenate([F[v][ok] / self.norm[v] for v in self.fields], 1)
        Zc = Z - Z.mean(0)
        _, sv, Vt = np.linalg.svd(Zc, full_matrices=False)
        eof = Vt[:2].copy()
        self.explained = (sv[:2] ** 2 / (sv ** 2).sum()).tolist()
        pcs = Z @ eof.T
        sd = pcs.std(0)
        eof = eof / sd[:, None]
        pcs = pcs / sd
        okd = np.flatnonzero(ok)
        if ref is None:
            mc = (LON15 >= 100) & (LON15 <= 150)
            if 'olr' in self.fields:
                j0 = self.fields.index('olr') * 240
                if eof[0, j0:j0 + 240][mc].mean() > 0:
                    eof[0] *= -1
                    pcs[:, 0] *= -1
            lag = 10
            if np.corrcoef(pcs[:-lag, 0], pcs[lag:, 1])[0, 1] < 0:          # RMM2 lags RMM1 (eastward propagation)
                eof[1] *= -1
                pcs[:, 1] *= -1
        else:
            Q = ref.rmm_fit[okd]
            good = np.isfinite(Q).all(1)
            Cm = np.corrcoef(pcs[good].T, Q[good].T)[:2, 2:]
            if abs(Cm[1, 0]) + abs(Cm[0, 1]) > abs(Cm[0, 0]) + abs(Cm[1, 1]):
                eof, pcs, Cm = eof[::-1].copy(), pcs[:, ::-1].copy(), Cm[::-1].copy()
            for j in range(2):
                if Cm[j, j] < 0:
                    eof[j] *= -1
                    pcs[:, j] *= -1
            self.ref_corr = [float(abs(Cm[0, 0])), float(abs(Cm[1, 1]))]
        self.eof = eof                                                     # [2, 240 * nfields], includes 1/sd
        full = np.concatenate([F[v] / self.norm[v] for v in self.fields], 1)
        self.rmm_fit = full @ eof.T                                       # [n, 2] on the fit series (NaN where filter incomplete)
        self.series = {}

    @staticmethod
    def _anom(x, days, tr, nh=3):
        H = harm([doy365(d) for d in days], nh)
        ok = tr & np.isfinite(x).all(1)
        c = np.linalg.lstsq(H[ok], np.asarray(x, np.float64)[ok], rcond=None)[0]
        return c, np.asarray(x, np.float64) - H @ c

    def align_rotation(self, samp, ref, years):
        """rotate (and, if needed, reflect) the EOF pair within its plane so that the RMM of series ``samp`` best matches ``ref``
        ({date: (rmm1, rmm2, ...)}) over ``years`` (complex correlation). The EOF1/EOF2 eigenvalues are nearly equal, so their
        orientation within the plane is not fixed by the data; this sets the WH04/BoM phase convention. Returns the angle (deg)."""
        S = self.series[samp]
        dd = [d for d in S['days'] if years[0] <= d.year <= years[1] and d in ref]
        O = self.obs(samp, dd)
        B = np.array([ref[d][:2] for d in dd])
        ok = np.isfinite(O).all(1) & np.isfinite(B).all(1)
        zo, zb = O[ok, 0] + 1j * O[ok, 1], B[ok, 0] + 1j * B[ok, 1]
        c = (zo * np.conj(zb)).sum()
        cr = (np.conj(zo) * np.conj(zb)).sum()
        refl = abs(cr) > abs(c)
        if refl:
            self.eof[1] *= -1
            c = cr
        th = -float(np.angle(c))
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        self.eof = Rm @ self.eof
        self.rotation = dict(angle_deg=float(np.rad2deg(th)), reflected=bool(refl), fit_years=list(years), fit_sampling=samp,
                             n_days=int(ok.sum()))
        self.rmm_fit = (Rm @ (self.rmm_fit * ([1, -1] if refl else [1, 1])).T).T
        for k, S2 in self.series.items():
            Fz = np.concatenate([prev120(S2['anom'][v]) / self.norm[v] for v in self.fields], 1)
            S2['rmm'] = Fz @ self.eof.T
        return self.rotation['angle_deg']

    def add_series(self, samp, series, days):
        """register an ERA5 series of another sampling: own 1979-2018 climatology, filtered anomalies, RMM."""
        tr = np.array([self.eof_years[0] <= d.year <= self.eof_years[1] for d in days])
        S = dict(days=list(days), ix={d: i for i, d in enumerate(days)}, anom={}, coef={})
        for v in self.fields:
            c, A = self._anom(series[v], days, tr)
            S['coef'][v], S['anom'][v] = c, A
        Fz = np.concatenate([prev120(S['anom'][v]) / self.norm[v] for v in self.fields], 1)
        S['rmm'] = Fz @ self.eof.T
        self.series[samp] = S
        return S['rmm']

    def obs(self, samp, days):
        S = self.series[samp]
        out = np.full((len(days), 2), np.nan)
        for j, d in enumerate(days):
            i = S['ix'].get(d)
            if i is not None:
                out[j] = S['rmm'][i]
        return out

    def forecast(self, samp, init: date, band):
        """band {var: [M, 46, 240]} raw 15S-15N means by lead day 0..45 -> RMM [M, 46, 2] (lead 0 = ERA5; NaN where incomplete)."""
        S = self.series[samp]
        i0 = S['ix'].get(init)
        M = next(iter(band.values())).shape[0]
        out = np.full((M, NLEAD, 2), np.nan)
        if i0 is None or i0 < 120:
            return out
        H = harm([doy365(d) for d in valid_dates(init)], 3)
        Z = []
        for v in self.fields:
            fa = np.asarray(band[v], np.float64) - (H @ S['coef'][v])[None]            # [M, 46, 240]
            past = S['anom'][v][i0 - 120:i0 + 1]                                         # days init-120 .. init (ERA5)
            seq = np.concatenate([np.broadcast_to(past[None], (M,) + past.shape), fa[:, 1:]], 1)   # index 120 = lead 0
            f = np.full((M, NLEAD, 240), np.nan)
            for k in range(NLEAD):
                win = seq[:, k:k + 120]                                                  # days lead k-120 .. k-1
                cur = seq[:, 120 + k]
                okm = np.isfinite(win).all((1, 2)) & np.isfinite(cur).all(1)
                f[okm, k] = cur[okm] - win[okm].mean(1)
            Z.append(f / self.norm[v])
        Z = np.concatenate(Z, -1)
        return Z @ self.eof.T

    def to_json(self):
        return dict(fields=self.fields, eof_years=self.eof_years, n_fit=self.n_fit, norm=self.norm, explained=self.explained,
                    ref_corr=getattr(self, 'ref_corr', None), rotation=getattr(self, 'rotation', None))


def mjo_series(samp, last: date, first=FIRST['mjo']):
    """ERA5 15S-15N series for sampling samp ('00z' or 'day') -> ({olr, u850, u200: [n, 240]}, days)."""
    days = daterange(first, last)
    olr = daily('olr_flux_band').get([d + timedelta(days=-1 if samp == '00z' else 0) for d in days])
    out = {'olr': olr}
    for v in ('u850', 'u200'):
        D = daily(f'{v}_00_band')
        a = D.get(days)
        out[v] = a if samp == '00z' else 0.5 * (a + D.get([d + timedelta(days=1) for d in days]))
    return out, days


def read_bom_rmm(path=IDX / 'rmm.74toRealtime.txt'):
    """BoM RMM table -> {date: (rmm1, rmm2, phase, amplitude)} (missing values dropped)."""
    out = {}
    for line in Path(path).read_text().splitlines()[2:]:
        t = line.split()
        if len(t) < 7:
            continue
        try:
            y, m, d = int(t[0]), int(t[1]), int(t[2])
            r1, r2, ph, am = float(t[3]), float(t[4]), int(t[5]), float(t[6])
        except ValueError:
            continue
        if abs(r1) > 1e10 or abs(r2) > 1e10 or am > 1e10 or r1 == 999:
            continue
        out[date(y, m, d)] = (r1, r2, ph, am)
    return out


def bivar(F, O):
    """F, O [n, 2] -> (COR, RMSE, n) over rows finite in both."""
    ok = np.isfinite(F).all(-1) & np.isfinite(O).all(-1)
    f, o = F[ok], O[ok]
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    cor = float((f * o).sum() / np.sqrt((f ** 2).sum() * (o ** 2).sum()))
    rmse = float(np.sqrt(((f - o) ** 2).sum(-1).mean()))
    return cor, rmse, int(ok.sum())


def amp_phase_err(F, O):
    ok = np.isfinite(F).all(-1) & np.isfinite(O).all(-1)
    f, o = F[ok], O[ok]
    if ok.sum() < 3:
        return np.nan, np.nan
    ae = float((np.hypot(f[:, 0], f[:, 1]) - np.hypot(o[:, 0], o[:, 1])).mean())
    cross = o[:, 0] * f[:, 1] - o[:, 1] * f[:, 0]
    dot = (o * f).sum(-1)
    pe = float(np.rad2deg(np.arctan2(cross, dot)).mean())
    return ae, pe


def skill_day(cor, leads, thr=0.5):
    """first lead where COR drops below thr, linearly interpolated; NaN if never above, max lead if never below."""
    cor = np.asarray(cor, float)
    leads = np.asarray(leads, float)
    for j in range(len(cor)):
        if not np.isfinite(cor[j]):
            continue
        if cor[j] < thr:
            if j == 0 or not np.isfinite(cor[j - 1]):
                return float(leads[j]) if j else float('nan')
            a, b = cor[j - 1], cor[j]
            return float(leads[j - 1] + (a - thr) / (a - b) * (leads[j] - leads[j - 1]))
    return float(leads[np.isfinite(cor)][-1]) if np.isfinite(cor).any() else float('nan')


def mjo_cor_from_components(f, o, leads=None):
    """f, o [S, L, 2] (per-start ensemble-mean and ERA5 RMM by lead) -> COR per lead [L] (for bootstrap resampling of starts)."""
    ok = np.isfinite(f).all(-1) & np.isfinite(o).all(-1)
    ff = np.where(ok, (f ** 2).sum(-1), 0).sum(0)
    oo = np.where(ok, (o ** 2).sum(-1), 0).sum(0)
    num = np.where(ok, (f * o).sum(-1), 0).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / np.sqrt(ff * oo)


# ====================================================================== teleconnection patterns (z500)
def varimax(L, gamma=1.0, q=100, tol=1e-7):
    p, k = L.shape
    Rm = np.eye(k)
    d = 0
    for _ in range(q):
        Lam = L @ Rm
        u, s, vh = np.linalg.svd(L.T @ (Lam ** 3 - (gamma / p) * Lam @ np.diag((Lam ** 2).sum(0))))
        Rm = u @ vh
        dn = s.sum()
        if d and dn / d < 1 + tol:
            break
        d = dn
    return L @ Rm


def _pt(lat, lon):
    return int(np.argmin(np.abs(LAT15 - lat))), int(np.argmin(np.abs(LON15 - lon % 360)))


def nao_template():
    """Azores (38N, 28W) minus Iceland (65N, 22W) point dipole (positive NAO = low over Iceland)."""
    t = np.zeros((121, 240))
    for (la, lo), s in (((38, -28), 1), ((65, -22), -1)):
        i, j = _pt(la, lo)
        t[i - 3:i + 4, j - 4:j + 5] += s
    return t


def pna_template():
    """Wallace-Gutzler PNA: + (20N, 160W) - (45N, 165W) + (55N, 115W) - (30N, 85W)."""
    t = np.zeros((121, 240))
    for (la, lo), s in (((20, -160), 1), ((45, -165), -1), ((55, -115), 1), ((30, -85), -1)):
        i, j = _pt(la, lo)
        t[i - 3:i + 4, j - 4:j + 5] += s
    return t


class TelePatterns:
    """per calendar month m (1..12): rotated loadings (NH, 10), NAO/PNA mode ids, AO / SAM EOF1; built from monthly z500 anomalies."""

    def __init__(self, monthly, months, std_by_m=None, n_rot=10, standardize=False):
        """monthly [n, 121, 240] anomalies (base 1979-2018), months: list of (year, month). standardize=False (covariance RPCA): the
        per-grid-point standardised (correlation) version matched CPC worse (monthly NAO r 0.46 vs 0.74, 1979-2018; 11-11 test)."""
        self.nh = LAT15 >= 20
        self.sh = LAT15 <= -20
        self.n_rot = n_rot
        self.sets = {}
        mo = np.array([m for _, m in months])
        for m in range(1, 13):
            sel = np.isin(mo, [(m - 2) % 12 + 1, m, m % 12 + 1])
            X = monthly[sel].astype(np.float64)
            st = {}
            for hemi, mk in (('nh', self.nh), ('sh', self.sh)):
                Y = X[:, mk]                                                   # [n, ny, 240]
                Y = Y - self._dmean(Y, LAT15[mk])[:, None, None]
                sd = Y.std(0) if standardize else np.ones(Y.shape[1:])
                sd[sd == 0] = 1
                w = np.sqrt(np.cos(np.deg2rad(LAT15[mk])))[:, None] * np.ones((1, 240))
                Zs = (Y / sd) * w
                Zf = Zs.reshape(len(Zs), -1)
                Zf = Zf - Zf.mean(0)
                U, S, Vt = np.linalg.svd(Zf, full_matrices=False)
                eof1 = Vt[0].reshape(w.shape)
                ent = dict(sd=sd.astype(np.float32), w=w.astype(np.float32), eof1=eof1.astype(np.float32))
                pc1 = Zf @ Vt[0]
                if hemi == 'nh':
                    Lr = varimax((Vt[:n_rot].T * S[:n_rot]))                       # loadings [p, k]
                    Lr = Lr / np.linalg.norm(Lr, axis=0)
                    ent['rot'] = Lr.T.reshape((n_rot,) + w.shape).astype(np.float32)
                    ent['pinv'] = np.linalg.pinv(ent['rot'].reshape(n_rot, -1).astype(np.float64))     # [p, k]
                    for nm, tmpl in (('nao', nao_template()), ('pna', pna_template())):
                        tt = (tmpl[mk] * w).ravel()
                        c = Lr.T @ tt
                        k = int(np.argmax(np.abs(c)))
                        ent[f'{nm}_k'] = k
                        ent[f'{nm}_sign'] = float(np.sign(c[k]))
                    # AO sign: positive = low over the pole
                    pole = LAT15[mk] >= 70
                    s = -np.sign((eof1[pole[:]] * w[pole]).mean()) if pole.any() else 1.0
                else:
                    pole = LAT15[mk] <= -70
                    s = -np.sign((eof1[pole] * w[pole]).mean())
                ent['eof1'] = (ent['eof1'] * s).astype(np.float32)
                st[hemi] = ent
            self.sets[m] = st
        self.norm = {}                                                       # (index, month) -> std of 7-day-mean ERA5 index

    @staticmethod
    def _dmean(Y, lat):
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, Y.shape[-1]))
        return (np.where(np.isfinite(Y), Y, 0) * w).sum((-2, -1)) / (np.isfinite(Y) * w).sum((-2, -1))

    def raw_indices(self, anom, m):
        """anom [..., 121, 240] (z500 anomaly) with pattern set of month m -> dict nao, pna, ao, sam [...] (unnormalised)."""
        st = self.sets[m]
        out = {}
        a = np.asarray(anom, np.float64)
        for hemi, mk in (('nh', self.nh), ('sh', self.sh)):
            Y = a[..., mk, :]
            Y = Y - self._dmean(Y, LAT15[mk])[..., None, None]
            e = st[hemi]
            Z = (Y / e['sd']) * e['w']
            Zf = np.nan_to_num(Z.reshape(Z.shape[:-2] + (-1,)))
            if hemi == 'nh':
                coef = Zf @ e['pinv']                                                 # least squares on all rotated loadings
                out['nao'] = coef[..., e['nao_k']] * e['nao_sign']
                out['pna'] = coef[..., e['pna_k']] * e['pna_sign']
                out['ao'] = Zf @ e['eof1'].ravel()
            else:
                out['sam'] = Zf @ e['eof1'].ravel()
        return out


def read_cpc_table(path):
    """CPC monthly table (year + 12 values) -> {(year, month): value}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if len(t) >= 2 and t[0].isdigit() and len(t[0]) == 4:
            for m, v in enumerate(t[1:13], 1):
                try:
                    x = float(v)
                except ValueError:
                    continue
                if abs(x) < 90:
                    out[(int(t[0]), m)] = x
    return out


# ====================================================================== member-file extraction
EXTRACT_VARS = ['olr', 'u850', 'u150', 'u250', 'u200', 'z500', 't2m', 'tp24', 'sst', 'swvl1', 'stl1', 'z10', 'z50', 'z100', 'z250']
EM_FIELDS = ['sst', 't2m', 'tp24', 'swvl1', 'stl1', 'z500']
LAND_FIELDS = ['t2m', 'tp24', 'swvl1']
CAP_LEVELS = [10, 50, 100, 250, 500]


@lru_cache(maxsize=1)
def land15():
    """land mask [121, 240]: ERA5 SST missing (land; also inland water without SST) on 2020-07-01."""
    A = stage_arr('sst', 2020)
    x = np.asarray(A[doy365(date(2020, 7, 1)), 0]).T
    return ~np.isfinite(x)


def _zm_rows(lat_zm, x, lat_target, half=0.0):
    """zonal-mean array [..., lat] -> value at lat_target (|dlat| <= half: cos-weighted mean, else linear interp)."""
    lat = np.asarray(lat_zm, float)
    o = np.argsort(lat)
    lat, x = lat[o], np.asarray(x)[..., o]
    if half > 0:
        m = np.abs(lat - lat_target) <= half + 1e-6
        w = np.cos(np.deg2rad(lat[m]))
        return (x[..., m] * w).sum(-1) / w.sum()
    j = np.searchsorted(lat, lat_target)
    a, b = lat[j - 1], lat[j]
    f = (lat_target - a) / (b - a)
    return (1 - f) * x[..., j - 1] + f * x[..., j]


@lru_cache(maxsize=1)
def _w15():
    from s41eval.hindcast_grid import conservative_1p5_weights
    Wlat, Wlon = conservative_1p5_weights()[:2]
    full = Wlat.sum(1)[:, None] * Wlon.sum(1)[None, :]
    return Wlat.astype(np.float32), np.ascontiguousarray(Wlon.T.astype(np.float32)), full.astype(np.float32)


def fast15(x):
    """[..., 180, 360] (1 deg, lat ascending) -> [..., 121, 240] conservative (same weights as ``ucs2s_hindcast.to_1p5``, float32
    matmuls; NaN-aware like it); 1.5-degree input returned as float32."""
    x = np.asarray(x, np.float32)
    if x.shape[-2:] == (121, 240):
        return x
    Wlat, WlonT, full = _w15()
    m = np.isfinite(x)
    if m.all():
        return (Wlat @ x @ WlonT) / full
    num = Wlat @ np.where(m, x, 0.0).astype(np.float32) @ WlonT
    den = Wlat @ m.astype(np.float32) @ WlonT
    out = num / np.where(den > 0, den, 1.0)
    out[den < 0.5 * full] = np.nan
    return out


def _lead_days_nc(ds, init):
    if 'lead_day' in ds.variables:
        return np.asarray(ds['lead_day'][:], int)
    import netCDF4
    t = netCDF4.num2date(ds['time'][:], ds['time'].units, only_use_cftime_datetimes=False)
    return np.array([(date(x.year, x.month, x.day) - init).days for x in t])


def read_member(path, init: date):
    """one h25 member file -> dict of compact arrays (see ``extract_start``). OLR is returned positive up (= -net TOA LW)."""
    import netCDF4
    with open(path, 'rb') as fh:
        buf = fh.read()                                       # one sequential Lustre read, then decompress from memory
    out = {}
    with netCDF4.Dataset('mem', memory=buf) as ds:
        ds.set_auto_mask(False)
        ld = _lead_days_nc(ds, init)
        assert len(ld) == NLEAD and (ld == np.arange(NLEAD)).all(), f'lead days {ld[:3]}..{ld[-1]}'
        lat = np.asarray(ds['lat'][:], float)
        flip = lat[0] > lat[-1]
        latn = lat[::-1] if flip else lat
        names = set(ds.variables)

        def get(v):
            x = np.asarray(ds[v][:], np.float32)
            fv = getattr(ds[v], '_FillValue', None)
            if fv is not None:
                x = np.where(x == fv, np.nan, x)
            return x[..., ::-1, :] if flip else x
        f15 = {}
        for v in EXTRACT_VARS:
            if v == 'u200' or v.startswith('z') and v != 'z500':
                continue
            if v in names:
                f15[v] = fast15(get(v))
        if 'olr' in f15:
            f15['olr'] = -f15['olr']
        if 'u150' in f15 and 'u250' in f15:
            f15['u200'], out['u200_src'] = W150 * f15['u150'] + (1 - W150) * f15['u250'], 'u150/u250 log-p'
        elif 'u200' in names:
            f15['u200'], out['u200_src'] = fast15(get('u200')), 'u200'
        for v in ('olr', 'u850', 'u200'):
            out[f'band_{v}'] = band_mean(f15[v]) if v in f15 else np.full((NLEAD, 240), np.nan, np.float32)
        for v in EM_FIELDS:
            if v in f15:
                out[f'w_{v}'] = window_mean_daily(f15[v] * (1000.0 if v == 'tp24' else 1.0))
        if 'sst' in f15:
            s, n, w, e = NINO34
            mk = (LAT15[:, None] >= s) & (LAT15[:, None] <= n) & (LON15[None] >= w) & (LON15[None] <= e)
            wt = cosw_lat()[:, None] * mk
            x = f15['sst']
            ok = np.isfinite(x)
            out['nino34'] = (np.where(ok, x, 0) * wt).sum((1, 2)) / (ok * wt).sum((1, 2))
        cap = np.full((NLEAD, len(CAP_LEVELS)), np.nan, np.float32)
        m60 = latn >= 60
        wt60 = np.cos(np.deg2rad(latn[m60]))
        for j, L in enumerate(CAP_LEVELS):
            v = f'z{L}'
            if v in names:
                x = get(v)
                cap[:, j] = (x[:, m60].mean(-1) * wt60).sum(-1) / wt60.sum()
        out['cap_z'] = cap
        if 'zm_u' in names:
            zl = np.asarray(ds['level_zm'][:], float)
            latz = np.asarray(ds['lat_zm'][:], float)
            zu = np.asarray(ds['zm_u'][:], np.float32)                                        # [46, lev, lat]
            out['zm_levels'] = zl
            out['zmu60'] = _zm_rows(latz, zu, 60.0)
            out['zmueq'] = _zm_rows(latz, zu, 0.0, half=5.0)
    return out


def extract_start(system, init: date, files, out_path: Path, log=print):
    """read all member files of one start -> npz with
    band_{olr,u850,u200} [M, 46, 240]; w_z500 [M, W, 121, 240]; em30_<v>, em11_<v> [W, 121, 240] (v in EM_FIELDS, members 0-10);
    land_<v> [M, W, nland] (LAND_FIELDS, land15 points); nino34 [M, 46]; cap_z [M, 46, 5]; zmu60, zmueq [M, 46, nlev]; zm_levels."""
    import time
    t0 = time.time()
    mem = []
    for f in files:
        mem.append(read_member(f, init))
    M = len(mem)
    lm = land15()
    out = dict(members=np.int32(M), init=np.str_(init.isoformat()), system=np.str_(system),
               u200_src=np.str_(mem[0].get('u200_src', 'missing')), files=np.array([os.path.basename(f) for f in files]))
    for v in ('olr', 'u850', 'u200'):
        out[f'band_{v}'] = np.stack([m[f'band_{v}'] for m in mem]).astype(np.float32)
    out['w_z500'] = np.stack([m['w_z500'] for m in mem]).astype(np.float32)
    for v in EM_FIELDS:
        if f'w_{v}' in mem[0]:
            X = np.stack([m[f'w_{v}'] for m in mem])
            out[f'em30_{v}'] = X.mean(0).astype(np.float32)
            out[f'em11_{v}'] = X[:11].mean(0).astype(np.float32)
            out[f'sd30_{v}'] = X.std(0, ddof=1).astype(np.float32)
            if v in LAND_FIELDS:
                out[f'land_{v}'] = X[..., lm].astype(np.float32)
    if 'nino34' in mem[0]:
        out['nino34'] = np.stack([m['nino34'] for m in mem]).astype(np.float32)
    out['cap_z'] = np.stack([m['cap_z'] for m in mem]).astype(np.float32)
    if 'zmu60' in mem[0]:
        out['zm_levels'] = mem[0]['zm_levels']
        out['zmu60'] = np.stack([m['zmu60'] for m in mem]).astype(np.float32)
        out['zmueq'] = np.stack([m['zmueq'] for m in mem]).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.partial.npz')
    np.savez(tmp, **out)
    os.replace(tmp, out_path)
    if log:
        log(f'{system} {init} {M} members {time.time() - t0:.0f}s')
    return M


def member_files(system, init: date):
    """sorted member h25 files of a start (UC-S2S per-member dirs, FuXi per-start dir)."""
    import glob
    import re
    if system == 'fuxi':
        return BS.h25_members(BS.HC / 'fuxi', init)
    sub, pre = {'ucs2s_sstanom': ('sstanom/S2S_SSTANOM', 'sstanom'), 'ucs2s_plain': ('plain/S2S', 'plain')}[system]
    fs = glob.glob(str(BS.HC / sub / f'S41u_{pre}_{init:%Y%m%d}' / 'predictions' / f'*_{init:%Y%m%d}00_ens_*_h25.nc'))
    key = lambda f: int(re.search(r'_ens_(\d+)_h25\.nc$', f).group(1))  # noqa: E731
    return sorted([f for f in fs if BS.readable(f)], key=key)


def start_dir(system, init):
    if system == 'ifs':
        return BS.IFS_DIR / f'{init:%Y%m%d}'
    if system == 'fuxi':
        return BS.HC / 'fuxi' / f'{init:%Y%m%d}'
    sub, pre = {'ucs2s_sstanom': ('sstanom/S2S_SSTANOM', 'sstanom'), 'ucs2s_plain': ('plain/S2S', 'plain')}[system]
    return BS.HC / sub / f'S41u_{pre}_{init:%Y%m%d}' / 'predictions'


def extract_path(system, init):
    return OUT / 'extract' / system / f'{init:%Y%m%d}.npz'


def list_starts(system=None):
    st = BS.list_starts(first='2019-01-01', last='2026-12-31')
    return st


# ====================================================================== scoring helpers
PERIODS = {'2019-26': (2019, 2026), '2019-24': (2019, 2024), '2025-26': (2025, 2026), '2020-26': (2020, 2026)}


def period_mask(starts, per):
    a, b = PERIODS[per]
    return np.array([a <= d.year <= b for d in starts])


def season_of(d: date):
    for k, ms in SEASONS.items():
        if d.month in ms:
            return k
    return None


def load_extract(system, starts, keys):
    """stack extraction arrays over starts -> (dict key -> [S, ...] (NaN for missing starts), list of starts present)."""
    out, have = {}, []
    for i, d in enumerate(starts):
        f = extract_path(system, d)
        if not f.exists():
            continue
        z = np.load(f)
        have.append(i)
        for k in keys:
            if k not in z.files:
                continue
            x = z[k]
            if k not in out:
                out[k] = np.full((len(starts),) + x.shape, np.nan, np.float32 if x.dtype.kind == 'f' else x.dtype)
            if x.shape == out[k].shape[1:]:
                out[k][i] = x
    return out, have


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fair_crps_1d(ens, obs):
    """ens [S, M], obs [S] -> fair CRPS [S]."""
    return BS.fair_crps(np.moveaxis(np.asarray(ens, np.float64), 1, 0), np.asarray(obs, np.float64))


# ----------------------------------------------------------------- MJO scores
def mjo_scores(F, O, starts, init_state, label, leads=range(1, NLEAD)):
    """F [S, M, 46, 2] member RMM, O [S, 46, 2] ERA5 RMM, init_state [S, 2] ERA5 RMM at the init day.
    -> rows (dicts) and per-start arrays {'em30_raw','em30_dc': [S,46,2], 'em11_raw','em11_dc': [D,S,46,2]}."""
    S, M = F.shape[:2]
    leads = list(leads)
    ems = {'30': [F.mean(1)]} if M >= 30 else {}
    ems['11'] = [F[:, ix].mean(1) for ix in draws11(M)]
    amp0 = np.hypot(init_state[:, 0], init_state[:, 1])
    ph0 = rmm_phase(init_state[:, 0], init_state[:, 1])
    groups = {'all': np.ones(S, bool), 'active': amp0 >= 1, 'inactive': amp0 < 1}
    for p, (a, b) in {'P8-1': (8, 1), 'P2-3': (2, 3), 'P4-5': (4, 5), 'P6-7': (6, 7)}.items():
        groups[p] = (amp0 >= 1) & np.isin(ph0, [a, b])
    for sn in SEASONS:
        groups[sn] = np.array([season_of(d) == sn for d in starts])
    groups['NDJFM'] = np.array([d.month in (11, 12, 1, 2, 3) for d in starts])
    rows, per = [], {}
    for msz, lst in ems.items():
        for corr in ('raw', 'dc'):
            fl = []
            for Fm in lst:
                if corr == 'dc':
                    Fm = Fm - loyo_bias(Fm - O, starts)
                fl.append(Fm)
            per[f'em{msz}_{corr}'] = np.stack(fl) if msz == '11' else fl[0]
            for per_name in PERIODS:
                pm = period_mask(starts, per_name)
                for g, gm in groups.items():
                    sel = pm & gm
                    if sel.sum() < 5:
                        continue
                    res = {k: [] for k in ('cor', 'rmse', 'amp_err', 'phase_err', 'n')}
                    for Fm in fl:
                        c_, r_, n_, a_, p_ = [], [], [], [], []
                        for k in leads:
                            c, r, n = bivar(Fm[sel, k], O[sel, k])
                            ae, pe = amp_phase_err(Fm[sel, k], O[sel, k])
                            c_.append(c), r_.append(r), n_.append(n), a_.append(ae), p_.append(pe)
                        for key, v in zip(('cor', 'rmse', 'n', 'amp_err', 'phase_err'), (c_, r_, n_, a_, p_)):
                            res[key].append(v)
                    cor = np.nanmean(np.array(res['cor'], float), 0)
                    sd = skill_day(cor, leads)
                    sd_draws = [skill_day(c, leads) for c in res['cor']]
                    base = dict(system=label, members=int(msz), correction=corr, period=per_name, group=g)
                    for j, k in enumerate(leads):
                        rows.append(dict(base, lead=k, cor=float(cor[j]), rmse=float(np.nanmean(np.array(res['rmse'])[:, j])),
                                         amp_err=float(np.nanmean(np.array(res['amp_err'])[:, j])),
                                         phase_err=float(np.nanmean(np.array(res['phase_err'])[:, j])),
                                         n=int(np.array(res['n'])[0, j])))
                    # weeks 3-4: (a) mean of the daily COR over leads 15-28; (b) COR of the RMM averaged over days 15-28
                    j34 = [leads.index(k) for k in range(15, 29)]
                    cor_b = []
                    for Fm in fl:
                        c, _, _ = bivar(Fm[sel, 15:29].mean(1), O[sel, 15:29].mean(1))
                        cor_b.append(c)
                    rows.append(dict(base, lead='w34', cor=float(np.nanmean(cor[j34])), cor_w34_meanrmm=float(np.nanmean(cor_b)),
                                     rmse=float(np.nanmean(np.array(res['rmse'])[:, j34])), skill_day=sd,
                                     skill_day_mean_of_draws=float(np.nanmean(sd_draws)),
                                     n=int(np.isfinite(fl[0][sel, 15:29]).all((1, 2)).sum())))
    return rows, per


# ----------------------------------------------------------------- teleconnection indices
def tele_window_index(TP, anom_w, init, windows=WK, norm=True):
    """anom_w [..., W, 121, 240] z500 window-mean anomalies of one start -> {nao, pna, ao, sam: [..., W]}."""
    out = {k: np.full(anom_w.shape[:-3] + (len(windows),), np.nan) for k in ('nao', 'pna', 'ao', 'sam')}
    for j, w in enumerate(windows):
        m = window_mid(init, w).month
        r = TP.raw_indices(anom_w[..., j, :, :], m)
        for k, v in r.items():
            out[k][..., j] = v / (TP.norm[(k, m)] if norm else 1.0)
    return out


def era5_daily_tele(TP, name, clim, first=date(1991, 1, 1), last=None):
    """ERA5 daily raw (unnormalised) indices with every month's pattern set: {k: [n, 12]} + days."""
    D = daily(name)
    last = last or D.last
    days = daterange(first, last)
    out = {k: np.full((len(days), 12), np.nan) for k in ('nao', 'pna', 'ao', 'sam')}
    for i0 in range(0, len(days), 366):
        dd = days[i0:i0 + 366]
        X = D.get(dd) - clim(dd)
        for m in range(1, 13):
            r = TP.raw_indices(X, m)
            for k, v in r.items():
                out[k][i0:i0 + len(dd), m - 1] = v
    return out, days


def window_from_daily(series, days_ix, init, w, m):
    """mean of a daily [n, 12] series over the days of window w after init with pattern set m."""
    a, b = WINDOWS[w]
    ii = [days_ix.get(init + timedelta(days=k)) for k in range(a, b + 1)]
    if any(i is None for i in ii):
        return np.nan
    v = series[ii, m - 1]
    return float(v.mean()) if np.isfinite(v).all() else np.nan


# ====================================================================== IFS (ECDS s2s-forecasts GRIB, runs/bing/ifs/data/<YYYYMMDD>/)
IFS_FILES = {   # variable -> (candidate part files, shortNames, level)   (parts of runs/bing/ifs/scripts/ifs_ext.py, README 11-11)
    'olr': (('p1i', 'p1sfc'), ('ttr',), 0), 'tp24': (('p1i', 'p1sfc'), ('tp',), 0),
    't2m': (('p1d', 'p1sfc'), ('2t', 'avg_2t', 'mean2t'), 0), 'sst': (('p2sfc',), ('sst', 'avg_sst'), 0),
    'z500': (('p1z', 'p1pl'), ('gh',), 500), 'u850': (('p1u', 'p1pl'), ('u',), 850), 'u200': (('p1u', 'p1pl'), ('u',), 200),
    'z10': (('p2pla',), ('gh',), 10), 'z50': (('p2pla',), ('gh',), 50), 'z100': (('p2pla',), ('gh',), 100),
    'u10': (('p2pla',), ('u',), 10), 'u50': (('p2pla',), ('u',), 50)}


def ifs_daily(day_dir: Path, v):
    """one IFS variable -> (member numbers, [M, 46, 121, 240] by lead day (lat ascending, our units: olr positive up W m-2 =
    -d(ttr)/86400, tp m per 24 h, gh -> geopotential m2 s-2), kind) or None. Fluxes and daily means at lead k cover (k-1, k] days."""
    parts, sn, lev = IFS_FILES[v]
    f = next((Path(day_dir) / f'{p}.grib' for p in parts if (Path(day_dir) / f'{p}.grib').exists()), None)
    if f is None or not BS.readable(f):
        return None
    raw, units = BS._grib_var(f, sn, lev)
    if not raw:
        return None
    mem = sorted(raw)
    arr = np.full((len(mem), NLEAD, 121, 240), np.nan, np.float32)
    kinds = {k for d in raw.values() for (k, _) in d}
    kind = 'mean' if 'mean' in kinds and v in ('t2m', 'sst') else 'inst'
    for i, n in enumerate(mem):
        d = {e: x for (k, e), x in raw[n].items() if k == kind}
        for k in range(1, NLEAD):
            e = 24 * k
            if v in ('tp24', 'olr'):
                if e in d:
                    dl = d[e] - d.get(e - 24, 0.0 if k == 1 else np.nan)
                    arr[i, k] = np.clip(dl / 1000.0 if 'kg' in units else dl, 0, None) if v == 'tp24' else -dl / 86400.0
            elif e in d:
                arr[i, k] = d[e] * (9.80665 if sn == ('gh',) else 1.0)
        if v not in ('tp24', 'olr') and kind == 'inst' and 0 in d:
            arr[i, 0] = d[0] * (9.80665 if sn == ('gh',) else 1.0)
    return mem, arr, kind


def extract_ifs_start(init: date, day_dir: Path, out_path: Path, log=print):
    """IFS start -> npz in the ``extract_start`` layout (members = all; ensemble-mean scores use members 0-10)."""
    import time
    t0 = time.time()
    lm = land15()
    out = dict(init=np.str_(init.isoformat()), system=np.str_('ifs'), u200_src=np.str_('u200 (true 200 hPa)'))
    M = None
    for v in ('olr', 'u850', 'u200'):
        r = ifs_daily(day_dir, v)
        if r is None:
            continue
        M = len(r[0]) if M is None else M
        out[f'band_{v}'] = band_mean(r[1][:M]).astype(np.float32)
    for v in ('z500', 'sst', 't2m', 'tp24'):
        r = ifs_daily(day_dir, v)
        if r is None:
            continue
        X = window_mean_daily(r[1] * (1000.0 if v == 'tp24' else 1.0), lead_axis=1)          # [M, W, 121, 240]
        M = X.shape[0] if M is None else M
        if v == 'z500':
            out['w_z500'] = X.astype(np.float32)
        out[f'em30_{v}'] = X.mean(0).astype(np.float32)
        out[f'em11_{v}'] = X[:11].mean(0).astype(np.float32)
        if v in LAND_FIELDS:
            out[f'land_{v}'] = X[..., lm].astype(np.float32)
        if v == 'sst':
            s, n, w, e = NINO34
            mk = (LAT15[:, None] >= s) & (LAT15[:, None] <= n) & (LON15[None] >= w) & (LON15[None] <= e)
            wt = cosw_lat()[:, None] * mk
            x = r[1]
            ok = np.isfinite(x)
            out['nino34'] = ((np.where(ok, x, 0) * wt).sum((-2, -1)) / (ok * wt).sum((-2, -1))).astype(np.float32)
    cap = None
    for j, L in enumerate(CAP_LEVELS):
        r = ifs_daily(day_dir, f'z{L}') if f'z{L}' in IFS_FILES else None
        if r is None:
            continue
        x = r[1] / 1.0
        m60 = LAT15 >= 60
        wt = np.cos(np.deg2rad(LAT15[m60]))
        if cap is None:
            cap = np.full((x.shape[0], NLEAD, len(CAP_LEVELS)), np.nan, np.float32)
        cap[:, :, j] = (x[:, :, m60].mean(-1) * wt).sum(-1) / wt.sum()
    if cap is not None:
        out['cap_z'] = cap
    zl, z60, zeq = [], [], []
    for L in (10, 50):
        r = ifs_daily(day_dir, f'u{L}')
        if r is None:
            continue
        zm = r[1].mean(-1)                                                                 # [M, 46, 121]
        zl.append(L)
        z60.append(_zm_rows(LAT15, zm, 60.0))
        zeq.append(_zm_rows(LAT15, zm, 0.0, half=5.0))
    if zl:
        out['zm_levels'] = np.array(zl, float)
        out['zmu60'] = np.stack(z60, -1).astype(np.float32)
        out['zmueq'] = np.stack(zeq, -1).astype(np.float32)
    out['members'] = np.int32(M or 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.partial.npz')
    np.savez(tmp, **out)
    os.replace(tmp, out_path)
    if log:
        log(f'ifs {init} {M} members, keys {sorted(k for k in out if not k.startswith("em"))} {time.time() - t0:.0f}s')
    return M
