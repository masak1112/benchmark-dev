"""S41 W3 (weaknesses of the raw output) and W4 (ensemble quality) diagnostics for Bing's UC-S2S hindcasts and FuXi-S2S.

Helper B of the S41 evaluation v1.5 (2026-11-12). Grid / regions / windows / drift correction are those of ``bing_eval`` (``core``).

Phases (``scripts/bing_weak.py``):
  era5   one task per valid day: ERA5 00 UTC pressure levels from the UC-S2S input h5 (``aurora_bundle_v5/h5data/<Y>_<4*doy>.h5``,
         1 deg, 18 levels 5..1000 hPa) -> g15 ``era5_pl/<YYYYMMDD>.npy`` [5 vars z t u v q, 18, 121, 240] and zonal spectra
         ``era5_spec/<YYYYMMDD>.npz`` (z500, KE250 on 1 deg 00 UTC; on g15 00 UTC; z500 on g15 daily mean from the 1.5 deg store).
  start  one task per start: per system, the ensemble-mean / SD files (``ensstats``) vs ERA5 -> daily region-mean bias, MSE and
         ensemble variance by variable x level x lead day, zonal means, window means (w1, w2, w34, w56) of z t u q for the
         drift-corrected profiles, surface drift curves (t2m, msl, sst, tp24, olr vs the package's ERA5 store and sampling),
         members 0-9: precipitation histograms (w1, w34, w56) and zonal spectra by lead day.
  profile one task per (system, var): LOYO drift-corrected (``core.drift_bias``) RMSE / bias by level, window, region, season.
  members one task per (system, field): fair CRPSS and ensemble-mean ACC (drift-corrected, w1 / w34 / w56) vs ensemble size.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from s41eval.bing_eval import core as K

R = Path('/lus/flare/projects/S2Storm/s2storm')
K.FIELDS.setdefault('v250', dict(units='m s-1'))          # member reader (read_h25_daily) needs a FIELDS entry; local only
H5 = R / 'aurora_bundle_v5' / 'h5data'
STORE = R / 'data' / 'era5_bing_1p5'
LEVELS = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
UA = ['z', 't', 'u', 'v', 'q']
UA_LONG = {'z': 'geopotential', 't': 'temperature', 'u': 'u_component_of_wind', 'v': 'v_component_of_wind', 'q': 'specific_humidity'}
WIN_UA = ['z', 't', 'u', 'q']                 # variables whose window-mean maps are kept for the drift-corrected profiles
SFC = ['t2m', 'msl', 'sst', 'tp24', 'olr']
REG4 = ['global', 'tropics', 'nh_extratropics', 'sh_extratropics']
WKEEP = ['w1', 'w2', 'w34', 'w56']
HWIN = ['w1', 'w34', 'w56']
HREG = ['global', 'tropics', 'nh_extratropics', 'sh_extratropics', 'land', 'ocean']
HEDGES = np.concatenate([[0.0], np.logspace(-2, 3, 101)])          # mm/day; first bin [0, 0.01)
BANDS = {'nh_mid': (30, 60), 'sh_mid': (-60, -30), 'tropics': (-20, 20)}
NMEM_SPEC = 10
SYSTEMS = {
    'ucs2s_sstanom': dict(ens=str(R / 'runs/bing/hc/sstanom/S2S_SSTANOM/S41u_sstanom_{d}/ensstats/ensstats_{d}00.nc'),
                          member=str(R / 'runs/bing/hc/sstanom/S2S_SSTANOM/S41u_sstanom_{d}/predictions/'
                                     'pangu_plasim_S41u_sstanom_{d}_24h_45step_{d}00_ens_{m}_h25.nc'), kind='ucs2s', time='00z'),
    'ucs2s_plain': dict(ens=str(R / 'runs/bing/hc/plain/S2S/S41u_plain_{d}/ensstats/ensstats_{d}00.nc'),
                        member=str(R / 'runs/bing/hc/plain/S2S/S41u_plain_{d}/predictions/'
                                   'pangu_plasim_S41u_plain_{d}_24h_45step_{d}00_ens_{m}_h25.nc'), kind='ucs2s', time='00z'),
    'fuxi': dict(ens=str(R / 'runs/bing/hc/fuxi/{d}/ensstats_{d}.nc'), member=str(R / 'runs/bing/hc/fuxi/{d}/m{m:02d}_h25.nc'),
                 kind='fuxi', time='day'),
}
UC_SFC = {'t2m': '2m_temperature', 'msl': 'mean_sea_level_pressure', 'sst': 'sea_surface_temperature',
          'tp24': 'total_precipitation_24hr', 'olr': 'mean_top_net_long_wave_radiation_flux'}


# ------------------------------------------------------------------ pure functions (tested in tests/test_bing_weak.py)
def h5_path(d: date) -> Path:
    return H5 / f'{d.year}_{4 * (d - date(d.year, 1, 1)).days:04d}.h5'


def zonal_spectrum(x, lat, band, kmax=None):
    """x [..., ny, nx] -> cos-lat weighted mean over the latitude band of the one-sided zonal power spectrum |c_k|^2 (c = rfft / nx,
    k >= 1 doubled so that sum_k P_k = zonal variance) -> [..., nk] (k = 0..nx/2; k = 0 set to 0)."""
    s, n = band
    sel = (lat >= s) & (lat <= n)
    xs = np.asarray(x, np.float64)[..., sel, :]
    nx = xs.shape[-1]
    c = np.fft.rfft(xs, axis=-1) / nx
    P = np.abs(c) ** 2
    P[..., 1:] *= 2.0
    if nx % 2 == 0:
        P[..., -1] /= 2.0
    P[..., 0] = 0.0
    w = np.cos(np.deg2rad(lat[sel]))
    out = np.einsum('...yk,y->...k', P, w) / w.sum()
    return out[..., :kmax + 1] if kmax else out


def wavg_hist(x, w, edges=HEDGES):
    """cos-lat weighted histogram of x [..., ny, nx] (NaN ignored) with weights w [ny, nx] broadcast -> [nbins]."""
    x = np.asarray(x)
    ww = np.broadcast_to(w, x.shape)
    ok = np.isfinite(x)
    return np.histogram(np.clip(x[ok], 0, None), bins=edges, weights=ww[ok])[0]


def hist_quantile(h, q, edges=HEDGES):
    """quantile q of a histogram (linear within a bin)."""
    c = np.cumsum(h) / max(h.sum(), 1e-300)
    i = int(np.searchsorted(c, q))
    i = min(i, len(h) - 1)
    c0 = c[i - 1] if i > 0 else 0.0
    f = (q - c0) / max(c[i] - c0, 1e-300)
    return float(edges[i] + f * (edges[i + 1] - edges[i]))


def hist_frac(h, lo, hi, edges=HEDGES):
    """fraction of the histogram mass with lo <= x < hi (lo, hi must be bin edges)."""
    i0, i1 = int(np.argmin(np.abs(edges - lo))), int(np.argmin(np.abs(edges - hi)))
    return float(h[i0:i1].sum() / max(h.sum(), 1e-300))


def rank_flatness(h):
    """rank histogram h [M+1] -> (normalised counts, chi2-like delta = sum((p - 1/(M+1))^2) * (M+1), U-shape index = mass in the two
    outer bins / expected)."""
    p = np.asarray(h, np.float64) / max(np.sum(h), 1e-300)
    n = len(p)
    return p, float(((p - 1 / n) ** 2).sum() * n), float((p[0] + p[-1]) / (2 / n))


def reliability_curve(counts):
    """counts [11 bins, 3] (sum w, sum w p, sum w o) -> (mean p per bin, observed frequency per bin, weight fraction per bin),
    reliability and resolution terms of the Brier score decomposition (area-weighted)."""
    c = np.asarray(counts, np.float64)
    W = c[:, 0].sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        pm = c[:, 1] / c[:, 0]
        of = c[:, 2] / c[:, 0]
    wf = c[:, 0] / max(W, 1e-300)
    obar = c[:, 2].sum() / max(W, 1e-300)
    ok = c[:, 0] > 0
    rel = float(np.sum(wf[ok] * (pm[ok] - of[ok]) ** 2))
    res = float(np.sum(wf[ok] * (of[ok] - obar) ** 2))
    return pm, of, wf, rel, res, float(obar)


def subset_scores(F, O, CC, C, starts, idx, mk, grid='g15', clip=False):
    """members-needed core: F [S, M, W, y, x] member window means, O [S, W, y, x], CC {ref: [S, W, y, x]} reference fair CRPS,
    C {ref: [S, W, y, x]} reference climatologies; idx = member indices of the subset. Drift correction = LOYO lead-dependent mean
    of the subset's ensemble mean (as the package). -> dict (metric, ref, region, w) -> value (pooled over starts)."""
    Fs = np.asarray(F[:, idx], np.float32)
    em = Fs.mean(1, dtype=np.float64).astype(np.float32)
    valid = np.isfinite(O).any((-2, -1)) & np.isfinite(em).any((-2, -1))
    b = K.drift_bias(em, O, starts, valid)
    use = valid & np.isfinite(b).any((-2, -1))
    X = Fs - np.nan_to_num(b)[:, None]
    if clip:
        X = np.clip(X, 0, None)
    crps = np.stack([K.fair_crps(X[s], O[s]) for s in range(len(starts))]).astype(np.float32)
    emc = X.mean(1)
    out = {}
    Wn = O.shape[1]
    for w in range(Wn):
        ix = np.flatnonzero(use[:, w])
        if ix.size < 3:
            continue
        for rn, m in mk.items():
            pc = K.amean(crps[ix, w], m, grid)
            for r in CC:
                pr = K.amean(np.asarray(CC[r][ix, w]), m, grid)
                ok = np.isfinite(pr) & np.isfinite(pc)
                out[('crpss', r, rn, w)] = float(1 - pc[ok].sum() / pr[ok].sum())
                Cr = np.asarray(C[r][ix, w])
                am = K.acc_map(emc[ix, w] - Cr, O[ix, w] - Cr)
                out[('acc', r, rn, w)] = float(K.amean(am, m, grid))
    return out


def member_offsets(F, m, grid='g15'):
    """F [S, M, y, x] member values of one window -> (r [M] pattern correlation of each member's mean deviation from the ensemble mean
    between odd and even starts, offset variance / ensemble variance (area means over mask m), per-member area-mean offset [M]).
    Exchangeable members: r ~ 0 and offset fraction ~ 1/S; persistent member-specific biases: r > 0, fraction >> 1/S."""
    S, M = F.shape[:2]
    dev = F - np.nanmean(F, 1, keepdims=True)
    d_all = np.nanmean(dev, 0)
    d_o, d_e = np.nanmean(dev[0::2], 0), np.nanmean(dev[1::2], 0)
    r = np.array([float(K.acc_pattern(d_o[i] - K.amean(d_o[i], m, grid), d_e[i] - K.amean(d_e[i], m, grid), m, grid)) for i in range(M)])
    off = float(K.amean(np.nanmean(d_all ** 2, 0) * M / (M - 1), m, grid))
    ens = float(K.amean(np.nanmean(np.nanvar(F, 1, ddof=1), 0), m, grid))
    return r, off / ens, np.array([float(K.amean(d_all[i], m, grid)) for i in range(M)])


def permember_scores(F, O, CC, starts, mk, grid='g15', clip=False):
    """fair CRPSS with (a) the package drift correction (LOYO mean of the ensemble mean, one shift for all members) and (b) a per-member
    LOYO drift correction (each member's own lead-dependent mean removed). F [S, M, W, y, x]. -> {(corr, ref, region, w): CRPSS}."""
    S, M = F.shape[:2]
    em = F.mean(1, dtype=np.float64).astype(np.float32)
    valid = np.isfinite(O).any((-2, -1)) & np.isfinite(em).any((-2, -1))
    b = K.drift_bias(em, O, starts, valid)
    bm = np.stack([K.drift_bias(np.ascontiguousarray(F[:, i]), O, starts, valid) for i in range(M)], 1)   # [S, M, W, y, x]
    use = valid & np.isfinite(b).any((-2, -1))
    out = {}
    for corr, X in (('dc_ens', F - np.nan_to_num(b)[:, None]), ('dc_member', F - np.nan_to_num(bm))):
        if clip:
            X = np.clip(X, 0, None)
        crps = np.stack([K.fair_crps(X[s], O[s]) for s in range(S)]).astype(np.float32)
        for w in range(O.shape[1]):
            ix = np.flatnonzero(use[:, w])
            for rn, m in mk.items():
                pc = K.amean(crps[ix, w], m, grid)
                for r in CC:
                    pr = K.amean(np.asarray(CC[r][ix, w]), m, grid)
                    ok = np.isfinite(pr) & np.isfinite(pc)
                    out[(corr, r, rn, w)] = float(1 - pc[ok].sum() / pr[ok].sum())
    return out


# ------------------------------------------------------------------ I/O helpers
def era5_day_pl(d: date):
    """[5, 18, 121, 240] g15 ERA5 00 UTC + 1 deg fields for the spectra, or (None, None) if the h5 is missing."""
    import h5py
    p = h5_path(d)
    if not p.exists():
        return None, None
    with h5py.File(p, 'r') as f:
        g = f['input']
        tv = g['time'][()]
        tv = tv.decode() if isinstance(tv, bytes) else str(tv)
        if not tv.startswith(f'{d:%Y-%m-%d}T00'):
            raise ValueError(f'{p}: time {tv} != {d}')
        x = np.stack([np.stack([np.asarray(g[f'{UA_LONG[v]}_{float(L)}'][:], np.float32) for L in LEVELS]) for v in UA])
    return K.to_g15(x), x


def era5_spec(d: date, x1):
    """spectra of one ERA5 day (1 deg 00 UTC fields x1 [5, 18, 180, 360] and g15)."""
    i5, i25 = LEVELS.index(500), LEVELS.index(250)
    lat1 = np.arange(-89.5, 90, 1.0)
    out = {}
    z1, u1, v1 = x1[0, i5], x1[2, i25], x1[3, i25]
    z15, u15, v15 = K.to_g15(np.stack([z1, u1, v1]))
    for b, rng in BANDS.items():
        out[f'z500_1deg_{b}'] = zonal_spectrum(z1, lat1, rng)
        out[f'ke250_1deg_{b}'] = 0.5 * (zonal_spectrum(u1, lat1, rng) + zonal_spectrum(v1, lat1, rng))
        out[f'z500_g15_{b}'] = zonal_spectrum(z15, K.LAT['g15'], rng)
        out[f'ke250_g15_{b}'] = 0.5 * (zonal_spectrum(u15, K.LAT['g15'], rng) + zonal_spectrum(v15, K.LAT['g15'], rng))
    # daily-mean z500 on g15 (FuXi's convention) from the 1.5 deg store
    f = STORE / 'z500' / f'{d.year}.npy'
    if f.exists():
        A = np.load(f, mmap_mode='r')
        zd = np.asarray(A[(d - date(d.year, 1, 1)).days], np.float64)
        if np.isfinite(zd).all():
            for b, rng in BANDS.items():
                out[f'z500_g15day_{b}'] = zonal_spectrum(zd.mean(0), K.LAT['g15'], rng)
    return out


class Store:
    """the package's 1.5 deg ERA5 store (``bing_eval.data.Era5``) with the package samplings."""

    def __init__(self):
        from s41eval.bing_eval import data as D
        self.E = D.Era5(STORE)

    def field(self, var, sys_time, d):
        flux = K.FIELDS[var].get('flux')
        samp = 'day' if sys_time == 'day' else ('prev' if flux else '00z')
        x = self.E.day(var, samp, d)
        return None if x is None else np.asarray(x, np.float32) * K.FIELDS[var].get('scale', 1.0)


def read_ensstats(sys, init: date):
    """-> (em_ua [46, 5, 18, 121, 240], var_ua, em_sfc [46, 5, 121, 240], var_sfc) on g15, output units (olr positive up,
    tp24 mm/day); NaN for levels a system does not have. UC-S2S sst masked to the ERA5 sea before the remap (v1.5 fix)."""
    import netCDF4
    cfg = SYSTEMS[sys]
    p = cfg['ens'].format(d=f'{init:%Y%m%d}')
    if not os.path.exists(p):
        return None
    T = K.NLEAD
    em_ua = np.full((T, len(UA), len(LEVELS), 121, 240), np.nan, np.float32)
    va_ua = np.full_like(em_ua, np.nan)
    em_s = np.full((T, len(SFC), 121, 240), np.nan, np.float32)
    va_s = np.full_like(em_s, np.nan)
    with netCDF4.Dataset(p) as ds:
        ds.set_auto_mask(False)
        if cfg['kind'] == 'ucs2s':
            lev = [int(x) for x in ds['level'][:]]
            li = [LEVELS.index(L) for L in lev]
            for j, v in enumerate(UA):
                em_ua[:, j][:, li] = K.to_g15(np.asarray(ds[f'{UA_LONG[v]}_mean'][:T], np.float32))
                va_ua[:, j][:, li] = K.to_g15(np.asarray(ds[f'{UA_LONG[v]}_sd'][:T], np.float32) ** 2)
            from s41eval.bing_eval import data as D
            for j, v in enumerate(SFC):
                m = np.asarray(ds[f'{UC_SFC[v]}_mean'][:T], np.float32)
                s = np.asarray(ds[f'{UC_SFC[v]}_sd'][:T], np.float32)
                if v == 'sst':
                    sea = D.native_mask('sea')
                    m, s = np.where(sea, m, np.nan), np.where(sea, s, np.nan)
                if v == 'tp24':
                    m, s = m * 1000.0, s * 1000.0
                if v == 'olr':
                    m = -m
                em_s[:, j] = K.to_g15(m)
                va_s[:, j] = K.to_g15(s ** 2)
        else:
            names = [str(x) for x in ds['variable'][:]]
            flip = ds['lat'][0] > ds['lat'][-1]
            M = np.asarray(ds['mean'][:T], np.float32)
            S = np.asarray(ds['sd'][:T], np.float32)
            if flip:
                M, S = M[..., ::-1, :], S[..., ::-1, :]
            for j, v in enumerate(UA):
                for L in LEVELS:
                    n = f'{v}{L}'
                    if n in names:
                        em_ua[:, j, LEVELS.index(L)] = M[:, names.index(n)]
                        va_ua[:, j, LEVELS.index(L)] = S[:, names.index(n)] ** 2
            fx = {'t2m': ('t2m', 1.0), 'msl': ('msl', 1.0), 'sst': ('sst', 1.0), 'tp24': ('tp', 24000.0), 'olr': ('ttr', -1 / 3600.0)}
            for j, v in enumerate(SFC):
                n, sc = fx[v]
                em_s[:, j] = M[:, names.index(n)] * sc
                va_s[:, j] = (S[:, names.index(n)] * abs(sc)) ** 2
    return em_ua, va_ua, em_s, va_s


def region_stats(diff, var, mk, grid='g15'):
    """diff, var [..., y, x] -> bias, mse, mean variance (where diff is finite) [..., n_regions]."""
    ok = np.isfinite(diff)
    v2 = np.where(ok, var, np.nan)
    b = np.stack([K.amean(diff, mk[r], grid) for r in REG4], -1)
    m = np.stack([K.amean(diff ** 2, mk[r], grid) for r in REG4], -1)
    v = np.stack([K.amean(v2, mk[r], grid) for r in REG4], -1)
    return b.astype(np.float32), m.astype(np.float32), v.astype(np.float32)


def zm(x):
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(x, -1).astype(np.float32)


def windows_of(x, keep=WKEEP):
    """x [T, ...rest, y, x] -> [len(keep), ...rest, y, x] window means (package windows)."""
    xt = np.moveaxis(x, 0, -3)
    w = K.window_means(xt)
    w = np.moveaxis(w, -3, 0)
    return w[[K.WK.index(k) for k in keep]]


def member_daily(sys, init, m, fields=('tp24', 'z500', 'u250', 'v250')):
    from s41eval.bing_eval import data as D
    p = SYSTEMS[sys]['member'].format(d=f'{init:%Y%m%d}', m=m)
    if not os.path.exists(p):
        return None, None
    out, ld, _ = D.read_h25_daily(p, init, list(fields), regrid=False)
    return out, ld
