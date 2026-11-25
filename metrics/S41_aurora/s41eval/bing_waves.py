"""Convectively coupled equatorial waves (plan v2 W5b) for the S41 hindcasts: latitude-resolved 15S-15N extraction, Wheeler-Kiladis
space-time spectra, wave-band power ratios and WK-filtered wave skill. Exploratory (not pre-registered).

Conventions
  Grid      WB2 1.5 deg; the equatorial strip is the 21 rows |lat| <= 15 (lat -15..15 ascending), 240 lon (0..358.5 E).
            UC-S2S 1-deg fields -> 1.5 deg with ``bing_modes.fast15`` (same conservative weights as the modes extract). Only olr and
            u850 are read (the UC-S2S land/sea fill concerns sst/swvl1/stl1 only, so ``read_h25_daily``'s masks do not apply).
  OLR       positive up (= -net TOA LW of the files), W m-2.
  Sampling  as ``bing_modes.SAMPLING``: UC-S2S '00z' (u850 at 00 UTC of D, OLR = 24 h ending 00 UTC of D = ERA5 calendar day D-1),
            FuXi 'day' (calendar-day means; OLR of day D).
  Spectra   per start and member: days 1..45 (45 days), anomalies = minus the time mean of the segment and the linear trend, split-cosine
            taper (10 % each end) in time; symmetric / antisymmetric components about the equator (x(lat) +- x(-lat))/2 over 0..15N;
            2-D FFT (lon, time); power summed over latitudes 0..15N; averaged over members and starts. ERA5 = the same days, same
            processing. Wavenumber k > 0 = eastward. Frequencies 1/45 .. 22/45 cpd.
  Bands     (Wheeler & Kiladis 1999 / Kiladis et al. 2009 filter boxes, NCL kf_filter conventions):
            Kelvin  sym,  k 1..14,  period 2.5..20 d, equivalent depth 8..90 m
            ER      sym,  k -10..-1, period 9.7..48 d, h 8..90 m (n = 1)
            MRG     anti, k -10..-1, period 3..9.6 d, h 8..90 m  (n = 0)
            MJO     sym,  k 1..5,   period 20..96 d (with 45-day segments the resolvable frequencies are 1/45 and 1/22.5 cpd)
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from s41eval import bing_modes as BM
from s41eval import bing_scores as BS

EQ = np.abs(BS.LAT15) <= 15.0 + 1e-6
LATEQ = BS.LAT15[EQ]                          # 21 rows -15..15
NEQ = int(EQ.sum())
G = 9.80665
BETA = 2.28e-11                               # 1/(m s) at the equator
RE = 6.371e6
DAY = 86400.0
OUT = BS.R / 'runs' / 'bing' / 'eval' / 'v1.5' / 'helperC'
NKEEP = 10                                    # members stored lat-resolved (0..9); the 30-member ensemble mean is stored as well

BANDS = {
    'kelvin': dict(sym='sym', k=(1, 14), per=(2.5, 20.0), h=(8.0, 90.0), kind='kelvin'),
    'er': dict(sym='sym', k=(-10, -1), per=(9.7, 48.0), h=(8.0, 90.0), kind='er'),
    'mrg': dict(sym='anti', k=(-10, -1), per=(3.0, 9.6), h=(8.0, 90.0), kind='mrg'),
    'mjo': dict(sym='sym', k=(1, 5), per=(20.0, 96.0), h=None, kind=None),
}


# ====================================================================== extraction
def read_member_eq(path, init: date):
    """one h25 member file -> {'olr','u850': [46, 21, 240] float32} on the 1.5-deg equatorial strip (OLR positive up)."""
    import netCDF4
    with open(path, 'rb') as fh:
        buf = fh.read()
    out = {}
    with netCDF4.Dataset('mem', memory=buf) as ds:
        ds.set_auto_mask(False)
        ld = BM._lead_days_nc(ds, init)
        assert len(ld) == BM.NLEAD and (ld == np.arange(BM.NLEAD)).all(), f'lead days {ld[:3]}..{ld[-1]}'
        lat = np.asarray(ds['lat'][:], float)
        flip = lat[0] > lat[-1]
        for v in ('olr', 'u850'):
            x = np.asarray(ds[v][:], np.float32)
            fv = getattr(ds[v], '_FillValue', None)
            if fv is not None:
                x = np.where(x == fv, np.nan, x)
            if flip:
                x = x[..., ::-1, :]
            out[v] = BM.fast15(x)[:, EQ, :].astype(np.float32)
    out['olr'] = -out['olr']
    return out


def eq_path(system, init):
    return OUT / 'eq' / system / f'{init:%Y%m%d}.npz'


def extract_eq_start(system, init: date, out_path: Path, log=print):
    import time
    t0 = time.time()
    files = BM.member_files(system, init)
    if not files:
        raise FileNotFoundError(f'{system} {init}: no member files')
    M = len(files)
    keep = {v: np.full((min(M, NKEEP), BM.NLEAD, NEQ, 240), np.nan, np.float32) for v in ('olr', 'u850')}
    acc = {v: np.zeros((BM.NLEAD, NEQ, 240), np.float64) for v in ('olr', 'u850')}
    for j, f in enumerate(files):
        x = read_member_eq(f, init)
        for v in acc:
            acc[v] += x[v]
            if j < NKEEP:
                keep[v][j] = x[v]
    out = dict(members=np.int32(M), init=np.str_(init.isoformat()), system=np.str_(system),
               files=np.array([os.path.basename(f) for f in files]), lat=LATEQ)
    for v in acc:
        out[f'{v}_m'] = keep[v]
        out[f'{v}_em'] = (acc[v] / M).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.partial.npz')
    np.savez(tmp, **out)
    os.replace(tmp, out_path)
    if log:
        log(f'eq {system} {init} {M} members {time.time() - t0:.0f}s')
    return M


# ====================================================================== ERA5 equatorial strip, daily
ERA5_NEW = BS.DATA / 'era5_bing_1p5'
ERA5_STAGE = BS.DATA / 'clim_trend_bing' / 'stage'


def _year_arr(var, year):
    f = ERA5_NEW / var / f'{year}.npy'
    if f.exists():
        return np.load(f, mmap_mode='r'), 'new'                       # [days incl Feb 29, 4, 121, 240]
    f = ERA5_STAGE / f'{var}_{year}.npy'
    if f.exists():
        return np.load(f, mmap_mode='r'), 'stage'                     # [365, 4, 240, 121]
    return None, None


def era5_eq_year(var, year, kind):
    """-> (dates, [n, 21, 240]) for one year; kind 'flux' (olr slot 0, calendar day), '00' (slot 0), 'mean4' (slots 0-3)."""
    A, src = _year_arr(var, year)
    if A is None:
        return [], np.zeros((0, NEQ, 240), np.float32)
    slots = [0, 1, 2, 3] if kind == 'mean4' else [0]
    if src == 'new':
        X = np.asarray(A[:, slots][:, :, EQ, :], np.float32).mean(1)
        days = [date(year, 1, 1) + timedelta(days=i) for i in range(X.shape[0])]
    else:
        X = np.asarray(A[:, slots][:, :, :, EQ], np.float32).mean(1).transpose(0, 2, 1)
        days, d = [], date(year, 1, 1)
        while d.year == year:
            if not (d.month == 2 and d.day == 29):
                days.append(d)
            d += timedelta(days=1)
        if len(days) != X.shape[0]:
            days = days[:X.shape[0]]
    return days, X


def build_era5_eq(first_year=2017, last=date(2026, 11, 3), out=None):
    """ERA5 daily strips olr_flux / u850_00 / u850_mean4 [n, 21, 240] from first_year-01-01 to ``last`` (NaN where missing; Feb 29
    of stage years = mean of Feb 28 and Mar 1). Writes OUT/era5_eq_<name>.npy + .json."""
    import json
    out = Path(out or OUT)
    days = BM.daterange(date(first_year, 1, 1), last)
    ix = {d: i for i, d in enumerate(days)}
    meta = {}
    for name, (var, kind) in {'olr_flux': ('olr', 'flux'), 'u850_00': ('u850', '00'), 'u850_mean4': ('u850', 'mean4')}.items():
        A = np.full((len(days), NEQ, 240), np.nan, np.float32)
        for y in range(first_year, last.year + 1):
            dd, X = era5_eq_year(var, y, kind)
            for d, x in zip(dd, X):
                if d in ix:
                    A[ix[d]] = x
        bad = ~np.isfinite(A).all((1, 2)) | (np.abs(A).max((1, 2)) == 0)
        for i in np.flatnonzero(bad):                                   # fill isolated missing days (Feb 29 of stage years)
            if 0 < i < len(days) - 1 and not bad[i - 1] and not bad[i + 1]:
                A[i] = 0.5 * (A[i - 1] + A[i + 1])
                bad[i] = False
        A[bad] = np.nan
        np.save(out / f'era5_eq_{name}.npy', A)
        meta[name] = dict(var=var, kind=kind, first=days[0].isoformat(), last=days[-1].isoformat(), n=len(days),
                          n_missing=int(bad.sum()), missing=[days[i].isoformat() for i in np.flatnonzero(bad)][:50])
    (out / 'era5_eq.json').write_text(json.dumps(meta, indent=1))
    return meta


class Era5Eq:
    def __init__(self, out=None):
        import json
        out = Path(out or OUT)
        self.meta = json.loads((out / 'era5_eq.json').read_text())
        self.A = {k: np.load(out / f'era5_eq_{k}.npy') for k in self.meta}
        self.first = date.fromisoformat(self.meta['olr_flux']['first'])

    def get(self, name, days):
        ii = np.array([(d - self.first).days for d in days])
        A = self.A[name]
        o = np.full((len(days),) + A.shape[1:], np.nan, np.float32)
        ok = (ii >= 0) & (ii < A.shape[0])
        o[ok] = A[ii[ok]]
        return o

    def obs(self, var, samp, days):
        """ERA5 strip matching a system's sampling for valid label dates ``days``."""
        if var == 'olr':
            return self.get('olr_flux', [d - timedelta(days=1) for d in days] if samp == '00z' else days)
        return self.get('u850_00' if samp == '00z' else 'u850_mean4', days)


# ====================================================================== Wheeler-Kiladis spectra
def split_cosine_taper(n, frac=0.1):
    w = np.ones(n)
    m = max(1, int(round(frac * n)))
    t = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
    w[:m] = t
    w[-m:] = t[::-1]
    return w


def detrend_t(x):
    """remove mean and linear trend along axis 0 (NaN not allowed)."""
    n = x.shape[0]
    t = np.arange(n) - (n - 1) / 2.0
    xm = x.mean(0)
    b = (t.reshape((-1,) + (1,) * (x.ndim - 1)) * (x - xm)).sum(0) / (t ** 2).sum()
    return x - xm - b * t.reshape((-1,) + (1,) * (x.ndim - 1))


def sym_anti(x, lat=LATEQ):
    """x [..., lat, lon] with lat symmetric about 0 -> (sym, anti) [..., nlat_half(incl. 0), lon] over lat >= 0."""
    lat = np.asarray(lat)
    n0 = int(np.argmin(np.abs(lat)))
    assert abs(lat[n0]) < 1e-6 and np.allclose(lat[n0:], -lat[n0::-1]), 'latitudes must be symmetric about 0'
    north = x[..., n0:, :, ] if x.ndim >= 2 else x
    south = x[..., n0::-1, :]
    return 0.5 * (north + south), 0.5 * (north - south)


def wk_power(x, taper=0.1):
    """x [T, nlat, nlon] (one segment, finite) -> (sym, anti) power [nfreq (0..T//2 cpd/T), nk (sorted -L/2..L/2-1)], summed over
    lat >= 0 (one-sided in frequency: positive and negative frequencies folded). k > 0 = eastward (x ~ cos(k lon - w t))."""
    T, _, L = x.shape
    x = detrend_t(np.asarray(x, np.float64)) * split_cosine_taper(T, taper)[:, None, None]
    f = np.fft.fftfreq(T)
    kn = np.fft.fftfreq(L, 1.0 / L).astype(int)
    nf = T // 2 + 1
    ks = np.sort(kn)
    out = []
    for comp in sym_anti(x):
        P = (np.abs(np.fft.fft2(comp, axes=(0, 2)) / (T * L)) ** 2).sum(1)      # [T, L]
        # numpy: exp(-i(w t + k lon)) -> cos(k0 lon - w0 t) peaks at (f = +w0, kn = -k0) and (f = -w0, kn = +k0)
        Q = np.zeros((nf, L))
        for i in range(T):
            fi = f[i]
            j = int(round(abs(fi) * T))
            if j >= nf:
                continue
            kphys = -kn if fi >= 0 else kn
            Q[j, (kphys + L // 2) % L] += P[i]                          # ks[i] = i - L/2 (k = L/2 aliases to -L/2)
        out.append(Q)
    return out[0], out[1]


def wk_axes(T, L=240):
    freq = np.arange(T // 2 + 1) / T                                    # cpd
    ks = np.sort(np.fft.fftfreq(L, 1.0 / L).astype(int))
    return freq, ks


def disp_freq(k, h, kind, n=1):
    """dispersion-relation frequency (cpd) of shallow-water equatorial waves for zonal wavenumber k (global, > 0 east), depth h."""
    c = np.sqrt(G * h)
    kk = np.asarray(k, float) / RE                                      # rad/m for global wavenumber k
    with np.errstate(invalid='ignore', divide='ignore'):
        if kind == 'kelvin':
            w = c * kk
        elif kind == 'er':
            w = -BETA * kk / (kk ** 2 + (2 * n + 1) * BETA / c)
        elif kind == 'mrg':
            w = 0.5 * kk * c + np.sqrt((0.5 * kk * c) ** 2 + BETA * c)
        else:
            raise ValueError(kind)
    return np.asarray(w) * DAY / (2 * np.pi)


def band_mask(freq, ks, band):
    """boolean [nf, nk] for a BANDS entry on the (freq >= 0, k) plane."""
    b = BANDS[band]
    F, K = np.meshgrid(freq, ks, indexing='ij')
    with np.errstate(divide='ignore'):
        per = np.where(F > 0, 1.0 / np.where(F > 0, F, 1), np.inf)
    m = (K >= b['k'][0]) & (K <= b['k'][1]) & (per >= b['per'][0]) & (per <= b['per'][1]) & (F > 0)
    if b['h'] is not None:
        lo = disp_freq(K, b['h'][0], b['kind'])
        hi = disp_freq(K, b['h'][1], b['kind'])
        a, c = np.minimum(lo, hi), np.maximum(lo, hi)
        m &= (F >= a) & (F <= c)
    return m


def band_power(P, freq, ks, band):
    return float(P[band_mask(freq, ks, band)].sum())


# ====================================================================== WK filtering (wave-band reconstruction in time-lon)
def wk_filter(x, band, dt=1.0):
    """x [T, nlat, nlon] finite -> band-filtered field, same shape. Applied to the symmetric (Kelvin, ER, MJO) or antisymmetric
    (MRG) component (the filtered field is that component, on the full strip). No taper (the caller pads)."""
    b = BANDS[band]
    T, nl, L = x.shape
    sym, anti = sym_anti(x)
    comp = sym if b['sym'] == 'sym' else anti
    F = np.fft.fft2(comp, axes=(0, 2))
    f = np.fft.fftfreq(T, dt)                                           # cpd
    kn = np.fft.fftfreq(L, 1.0 / L)                                     # numpy wavenumber index (e^{-i k lon})
    Fm, Km = np.meshgrid(f, kn, indexing='ij')
    # physical eastward wavenumber for positive frequency: kphys = -kn when f > 0; for f < 0, kphys = +kn (conjugate pair)
    kphys = np.where(Fm >= 0, -Km, Km)
    fa = np.abs(Fm)
    with np.errstate(divide='ignore'):
        per = np.where(fa > 0, 1.0 / np.where(fa > 0, fa, 1), np.inf)
    m = (kphys >= b['k'][0]) & (kphys <= b['k'][1]) & (per >= b['per'][0]) & (per <= b['per'][1]) & (fa > 0)
    if b['h'] is not None:
        lo = disp_freq(kphys, b['h'][0], b['kind'])
        hi = disp_freq(kphys, b['h'][1], b['kind'])
        m &= (fa >= np.minimum(lo, hi)) & (fa <= np.maximum(lo, hi))
    y = np.real(np.fft.ifft2(F * m[:, None, :], axes=(0, 2)))
    n0 = int(np.argmin(np.abs(LATEQ))) if nl == NEQ else nl // 2
    full = np.empty_like(x, dtype=np.float64)
    full[:, n0:] = y
    full[:, n0::-1] = y if b['sym'] == 'sym' else -y
    return full
