"""Scores of the 2025-26 Monday-start hindcasts for Bing's comparison: UC-S2S (SST-anomaly and plain), FuXi-S2S and the
real-time ECMWF IFS extended-range ensemble, against ERA5, on the WeatherBench2 1.5-degree grid (121 x 240).

Grid and conventions
  * Everything is scored on the 1.5-degree WB2 grid. Internally lat runs -90..90 (as ``ucs2s_hindcast`` 1p5 output and
    ``data/clim_trend``); written maps are flipped to lat 90..-90 (WB2 order), lon 0..358.5. 1-degree inputs (UC-S2S native, ERA5
    ``era5_verif*``) are remapped with ``ucs2s_hindcast.to_1p5`` (first-order conservative, NaN-aware).
  * Lead day k = valid date init + k (00 UTC label). Windows: weeks w1..w6 = days 7(w-1)+1..7w, w34 = days 15-28, w56 = days 29-42.
    Window value = mean of the daily values (precipitation: mean daily total, mm/day); a window is NaN if any of its days is missing.
  * Verification sampling (decided per system and variable, see ``SAMPLING``):
      00z   instantaneous 00 UTC state of the valid date D (UC-S2S, IFS states) -> ``data/era5_verif00``;
      prev  24-h quantity ending 00 UTC of D = ERA5 daily value of date D-1 (UC-S2S/IFS tp24, IFS olr, IFS 2t daily means)
            -> ``data/era5_verif`` (tp24 = (D-1 00, D 00]; states = mean of 00/06/12/18 of D-1);
      day   daily mean / total of the calendar date D (FuXi-S2S, which outputs daily means labelled 00 UTC) -> ``data/era5_verif``
            (t2m, msl, z500, tp24) or the staged daily means in ``data/era5_bing_obs``. If no daily-mean ERA5 exists for a variable
            the 00 UTC state is used and the row is flagged ``sampling = day->00z``.
  * Climatology = trend-adjusted climatology (DECISIONS 2026-11-04 item 8): 1991-2025 linear trend per calendar day and grid point,
    evaluated at the year of the valid date (``data/clim_trend/<var>/clim_<Y>.npy``; extra variables in ``data/clim_trend_bing``),
    sampled like the verification (00z = hour 0; day = mean of the 4 hours; tp24/olr: daily value stored at hour 0).
  * Climatological reference ensemble (CRPSS): ERA5 of the same calendar dates in each year 1991-2020 (30 members, from the 1.5-degree
    stage arrays of ``s41eval.sci.clim_trend``), each shifted along the fitted trend to the verification year
    (x_y + b(d) (Y - y)), i.e. the S33 re-scoring convention applied at every grid point; precipitation clipped at 0.
  * Metrics per (variable, region, window, correction):
      crps   fair (ensemble-size corrected) CRPS, ``score_scan.fair_crps`` (all members, also all IFS members);
      crpss  1 - sum_s A(crps_fc) / sum_s A(crps_clim), A = cos-lat area mean, equal weight per start;
      acc    per grid point, Pearson correlation across starts of ensemble-mean and ERA5 anomalies (centred across starts), then
             area mean (S40 / score_v2 convention); acc_spatial = uncentred pattern ACC of the anomalies over the region per start,
             averaged over starts (gate_score global z500 convention);
      rmse   sqrt(A(mean_s (em - o)^2)); spread = sqrt(A(mean_s var_ddof1)); ssr = sqrt((M+1)/M) spread / rmse; bias = A(mean_s(em - o)).
    IFS: acc, acc_spatial, rmse, spread, ssr, bias from members 0-10 (11 members, ``--det_members``); crps/crpss from all members.
  * Corrections: raw; dc = lead-dependent mean-drift correction, leave-one-year-out: for start s the bias at each grid point and window
    is the mean over the other-year starts within +-30 calendar days (``biascorr.training_sets``) of (ensemble mean - ERA5); starts
    without training starts (Nov-Dec 2025) are dropped from dc; raw_dcsub = raw scores on exactly the dc starts. Applying the bias on
    window means equals applying the daily bias and averaging (no 7-day lead smoothing needed at window level).
  * MJO: Wheeler-Hendon RMM re-implemented with Bing's MJOcast conventions (Bing's code needs MJOcast, eofs/windspharm, chi200 obs
    to 2024 and per-system file layouts; impractical here): OLR (positive up), u850, u200, averaged over 15S-15N (unweighted, as MJOcast),
    anomalies vs a 1991-2020 mean + 3-harmonic annual cycle, minus the trailing 120-day mean (MJOcast: window includes the current day;
    for forecasts ERA5 days before the init + forecast days), each field divided by sqrt(zonal mean of its time variance), multivariate
    EOF over 1991-2020, PCs scaled to unit variance; sign: RMM1 > 0 = enhanced convection (negative OLR) over 100-150E, RMM2 sign so that
    RMM2 lags RMM1 by ~10 days (eastward propagation). Skill: bivariate correlation (uncentred) and RMSE of the ensemble-mean RMM
    (IFS: members 0-10) vs ERA5, per lead day. UC-S2S has no 200 hPa: u200 = log-p interpolation of u150/u250 when u150 exists
    (Bing's weights), else u250 (flagged in the output).

Inputs (adapters): ``read_h25`` (UC-S2S ``save_hindcast`` files and FuXi files in the same format; native 1 deg or 1.5 deg; extra
variables ignored), ``read_ifs`` (ECDS s2s-forecasts GRIB split by ``runs/bing/ifs/scripts/ifs_ext.py``: <dir>/<YYYYMMDD>/p1sfc.grib etc.).

    python scripts/bing_scores.py --systems ucs2s_sstanom ucs2s_plain fuxi ifs --tag v1
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

R = Path(os.environ.get('S2STORM_ROOT', '/lus/flare/projects/S2Storm/s2storm'))
DATA = R / 'data'
V00 = DATA / 'era5_verif00'
VDAY = DATA / 'era5_verif'
VEXTRA = DATA / 'era5_bing_obs'
CLIM_DIRS = [DATA / 'clim_trend', DATA / 'clim_trend_bing']
HC = R / 'runs' / 'bing' / 'hc'
IFS_DIR = R / 'runs' / 'bing' / 'ifs' / 'data'
OUT = R / 'runs' / 'bing' / 'scores'

LAT15 = np.linspace(-90.0, 90.0, 121)
LON15 = np.arange(240) * 1.5
WINDOWS = {'w1': (1, 7), 'w2': (8, 14), 'w3': (15, 21), 'w4': (22, 28), 'w5': (29, 35), 'w6': (36, 42), 'w34': (15, 28), 'w56': (29, 42)}
WK = list(WINDOWS)
NLEAD = 46
REGIONS = {'global': (-90, 90, 0, 360), 'tropics': (-20, 20, 0, 360), 'nh_extratropics': (20, 90, 0, 360),
           'sh_extratropics': (-90, -20, 0, 360), 'conus': (25, 50, 235, 295), 'midwest': (36, 46, 263, 274),
           'plains': (35, 43, 257, 265)}
# scale: output units (tp m -> mm); clip: clip at 0 after correction
VARS = {'t2m': dict(units='K'), 'tp24': dict(units='mm/day', scale=1000.0, clip=True), 'z500': dict(units='m2 s-2'),
        'msl': dict(units='Pa'), 'u850': dict(units='m s-1'), 'olr': dict(units='W m-2'), 't850': dict(units='K'),
        'u250': dict(units='m s-1'), 'v250': dict(units='m s-1'), 'sst': dict(units='K'), 'u200': dict(units='m s-1'),
        'v850': dict(units='m s-1'), 'swvl1': dict(units='m3 m-3')}
DEFAULT_VARS = ['t2m', 'tp24', 'z500', 'msl', 'u850', 'olr', 't850', 'u250', 'v250', 'sst']
SYSTEMS = {'ucs2s_sstanom': dict(kind='h25', sub='sstanom'), 'ucs2s_plain': dict(kind='h25', sub='plain'),
           'fuxi': dict(kind='h25', sub='fuxi'), 'ifs': dict(kind='ifs')}
CLIM_YEARS = list(range(1991, 2021))
DET_MEMBERS = {'ifs': 11}
MJO_BAND = 15.0
BAD_OSTS = {int(x) for x in os.environ.get('BING_BAD_OSTS', '9').split(',') if x.strip()}   # flare OST 9 hung since 11-11 00:30 UTC
STRIPE_OSTS = [27, 41, 63, 88]
W150 = math.log(250 / 200) / math.log(250 / 150)          # Bing's log-p weight of 150 hPa for 200 hPa (0.437)


def sampling(system: str, var: str) -> str:
    """verification sampling of ``var`` for ``system`` (see module docstring)."""
    if system == 'fuxi':
        return 'day'
    if var in ('tp24', 'olr'):
        return 'prev'
    if system == 'ifs' and var in ('t2m', 'sst'):
        return 'prev'                    # ECDS s2s 2t / sst are daily means of (D-1 00, D 00]
    return '00z'


# ------------------------------------------------------------------ Lustre guards (OST 9 hang, 2026-11-11)
@lru_cache(maxsize=None)
def readable(path) -> bool:
    """False if any stripe of ``path`` is on a hung OST (``lfs getstripe`` is metadata-only) or the file is missing."""
    import subprocess
    path = str(path)
    if not os.path.exists(path):
        return False
    if not path.startswith('/lus/') or not BAD_OSTS:
        return True
    try:
        r = subprocess.run(['lfs', 'getstripe', path], capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return False
    idx, on = set(), False
    for line in r.stdout.splitlines():
        t = line.split()
        if t[:1] == ['obdidx']:
            on = True
            continue
        if on and t and t[0].isdigit():
            idx.add(int(t[0]))
    for line in r.stdout.splitlines():                    # single-stripe short form
        if 'lmm_stripe_offset' in line:
            idx.add(int(line.split(':')[1]))
    return not (idx & BAD_OSTS)


_STRIPE_I = [0]


def mkdir_striped(d) -> Path:
    """mkdir -p and, on Lustre, ``lfs setstripe -c 1 -i k`` with k not a bad OST (new files avoid OST 9)."""
    import subprocess
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    if str(d).startswith('/lus/'):
        k = STRIPE_OSTS[_STRIPE_I[0] % len(STRIPE_OSTS)]
        _STRIPE_I[0] += 1
        subprocess.run(['lfs', 'setstripe', '-c', '1', '-i', str(k), str(d)], capture_output=True, timeout=60)
    return d


# ------------------------------------------------------------------ grid helpers
def to15(x):
    """[..., 180, 360] (1 deg, lat -89.5..89.5) -> [..., 121, 240]; 1.5-degree input returned as float32."""
    x = np.asarray(x)
    if x.shape[-2:] == (121, 240):
        return x.astype(np.float32)
    from s41eval.hindcast_grid import to_1p5
    return to_1p5(x)


@lru_cache(maxsize=1)
def cosw():
    return np.cos(np.deg2rad(LAT15))[:, None] * np.ones((1, 240))


@lru_cache(maxsize=1)
def masks():
    out = {}
    for k, (s, n, w, e) in REGIONS.items():
        out[k] = ((LAT15[:, None] >= s) & (LAT15[:, None] <= n) & (LON15[None] >= w) & (LON15[None] <= e)).astype(float)
    return out


def amean(x, m):
    """cos-lat area mean of x [..., 121, 240] over mask m (NaN-aware)."""
    w = cosw() * m
    ok = np.isfinite(x)
    num = (np.where(ok, x, 0.0) * w).sum((-2, -1))
    den = (ok * w).sum((-2, -1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def doy365(d: date) -> int:
    """0-based day of year in the 365-day calendar of clim_trend (Feb 29 -> Feb 28)."""
    if d.month == 2 and d.day == 29:
        d = d - timedelta(days=1)
    return (date(2001, d.month, d.day) - date(2001, 1, 1)).days


def window_means(x, lead_days=None):
    """x [..., T, y, x] daily with lead days ``lead_days`` (default 0..T-1) -> [..., len(WINDOWS), y, x]; NaN if a day is missing."""
    x = np.asarray(x, np.float32)
    T = x.shape[-3]
    ld = np.arange(T) if lead_days is None else np.asarray(lead_days)
    out = []
    for a, b in WINDOWS.values():
        sel = [int(np.flatnonzero(ld == k)[0]) for k in range(a, b + 1) if (ld == k).any()]
        if len(sel) < b - a + 1:
            out.append(np.full(x.shape[:-3] + x.shape[-2:], np.nan, np.float32))
            continue
        seg = x[..., sel, :, :]
        allnan = ~np.isfinite(seg).any((-2, -1))                      # a whole missing day -> NaN window
        m = seg.mean(-3)
        if allnan.any():
            m = np.where(allnan.any(-1)[..., None, None], np.nan, m)
        out.append(m)
    return np.stack(out, -3).astype(np.float32)


# ------------------------------------------------------------------ scores (pure array functions)
def fair_crps(ens, obs):
    """ens [m, ...], obs [...] -> fair CRPS [...] (Ferro 2014): E|X-y| - sum_ij |xi-xj| / (2 m (m-1)) (= score_scan.fair_crps)."""
    ens = np.asarray(ens, np.float64)
    m = ens.shape[0]
    a = np.abs(ens - np.asarray(obs, np.float64)[None]).mean(0)
    s = np.sort(ens, 0)
    k = (2 * np.arange(m) - m + 1).reshape((m,) + (1,) * (ens.ndim - 1))
    return a - (k * s).sum(0) / (m * (m - 1))


def drift_bias(em, O, starts, valid):
    """em, O [S, W, y, x]; valid [S, W] -> leave-one-year-out bias [S, W, y, x] (NaN where no training start)."""
    from s41eval.biascorr import training_sets
    tr = training_sets(list(starts))
    d = em - O
    b = np.full_like(d, np.nan)
    for s, idx in enumerate(tr):
        for w in range(d.shape[1]):
            j = [i for i in idx if valid[i, w]]
            if j:
                b[s, w] = np.nanmean(d[j, w], 0)
    return b


def _acc_map(fa, oa):
    """fa, oa [n, y, x] -> Pearson across n per point."""
    fc, oc = fa - np.nanmean(fa, 0), oa - np.nanmean(oa, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nansum(fc * oc, 0) / np.sqrt(np.nansum(fc ** 2, 0) * np.nansum(oc ** 2, 0))


def _acc_spatial(fa, oa, m):
    w = cosw() * m
    ok = np.isfinite(fa) & np.isfinite(oa)
    f, o = np.where(ok, fa, 0), np.where(ok, oa, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (w * f * o).sum((-2, -1)) / np.sqrt((w * f * f).sum((-2, -1)) * (w * o * o).sum((-2, -1)))


def score_var(E, O, C, crps_clim, starts, det_members=None, clip=False, corrections=('raw', 'dc', 'raw_dcsub')):
    """E [S, M, W, y, x] window means of the members; O, C, crps_clim [S, W, y, x] (ERA5, climatology, CRPS of the climatological
    ensemble). Returns (rows, maps): rows = list of dict(correction, region, window, metric, value, n_starts, members);
    maps = {correction: {'crpss','acc','crps','crps_clim','bias': [W, y, x]}}."""
    S, M = E.shape[:2]
    W = E.shape[2]
    md = min(det_members or M, M)
    valid = np.isfinite(O).any((-2, -1)) & np.isfinite(E[:, 0]).any((-2, -1))                 # [S, W] (crps_clim may be NaN)
    em = E[:, :md].mean(1)
    b = drift_bias(em, O, starts, valid) if ('dc' in corrections or 'raw_dcsub' in corrections) else None
    hasb = np.isfinite(b).any((-2, -1)) if b is not None else None
    rows, maps = [], {}
    MK = masks()
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='(Mean of empty slice|All-NaN|invalid value|divide by zero)')
    for corr in corrections:
        crps = np.full((S, W) + E.shape[-2:], np.nan, np.float32)
        emc = np.full_like(em, np.nan)
        var = np.full_like(em, np.nan)
        use = valid.copy() if corr == 'raw' else valid & hasb
        for s in range(S):
            if not use[s].any():
                continue
            X = E[s].astype(np.float64)
            if corr == 'dc':
                X = X - np.nan_to_num(b[s])[None]
                if clip:
                    X = np.clip(X, 0, None)
            crps[s] = fair_crps(X, O[s])
            emc[s] = X[:md].mean(0)
            var[s] = X[:md].var(0, ddof=1) if md > 1 else np.nan
        fa, oa = emc - C, O - C
        mp = {k: np.full((W,) + E.shape[-2:], np.nan, np.float32) for k in ('crpss', 'acc', 'crps', 'crps_clim', 'bias')}
        for w in range(W):
            ix = np.flatnonzero(use[:, w])
            if ix.size < 1:
                continue
            cf, cc = np.nanmean(crps[ix, w], 0), np.nanmean(crps_clim[ix, w], 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                mp['crpss'][w] = 1 - cf / cc
            mp['crps'][w], mp['crps_clim'][w] = cf, cc
            if ix.size >= 3:                                              # temporal ACC needs a few starts
                mp['acc'][w] = _acc_map(fa[ix, w], oa[ix, w])
            se = np.nanmean((emc[ix, w] - O[ix, w]) ** 2, 0)
            sp = np.nanmean(var[ix, w], 0)
            mp['bias'][w] = np.nanmean(emc[ix, w] - O[ix, w], 0)
            for rn, mk in MK.items():
                pc = amean(crps[ix, w], mk)
                pr = amean(crps_clim[ix, w], mk)
                mse, msp = float(amean(se, mk)), float(amean(sp, mk))
                rmse = math.sqrt(mse)
                vals = dict(crps=float(np.nanmean(pc)), crps_clim=float(np.nanmean(pr)),
                            crpss=float(1 - np.nansum(pc) / np.nansum(pr)) if np.isfinite(pr).any() else float('nan'),
                            acc=float(amean(mp['acc'][w], mk)),
                            acc_spatial=float(np.nanmean(_acc_spatial(fa[ix, w], oa[ix, w], mk))),
                            rmse=rmse, spread=math.sqrt(msp) if np.isfinite(msp) else float('nan'),
                            ssr=math.sqrt((md + 1) / md * msp) / rmse if rmse > 0 and md > 1 else float('nan'),
                            bias=float(amean(mp['bias'][w], mk)))
                for k, v in vals.items():
                    rows.append(dict(correction=corr, region=rn, window=WK[w] if W == len(WK) else w, metric=k, value=v,
                                     n_starts=int(ix.size), members=M if k.startswith('crps') else md))
        maps[corr] = mp
    return rows, maps


# ------------------------------------------------------------------ ERA5 verification and climatology
@lru_cache(maxsize=1600)
def _npz(path: str):
    if not readable(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files if k != 'meta'}


@lru_cache(maxsize=4000)
def _get(store: Path, d: date, key: str):
    z = _npz(str(store / f'{d:%Y%m%d}.npz'))
    if z is None or key not in z:
        return None
    return to15(z[key].astype(np.float64))


def obs_day(var: str, samp: str, d: date):
    """ERA5 field [121, 240] (lat ascending) for valid label date d with sampling samp; returns (field or None, used sampling)."""
    if samp == 'prev':
        x, s = obs_day(var, 'day', d - timedelta(days=1))
        return x, 'prev' if s == 'day' else 'prev->' + s
    if samp == 'day':
        for store, key in ((VDAY, var), (VEXTRA, f'{var}_day')):
            x = _get(store, d, key)
            if x is not None:
                return x, 'day'
        if var in ('tp24', 'olr'):
            return None, 'day'
        x, _ = obs_day(var, '00z', d)
        return x, 'day->00z'
    for store, key in ((V00, var), (VEXTRA, f'{var}_00')):
        x = _get(store, d, key)
        if x is not None:
            return x, '00z'
    if var == 'olr':
        x = _get(VEXTRA, d - timedelta(days=1), 'olr_day')
        return x, '00z->prev'
    return None, '00z'


def _clim_dir(var):
    for d in CLIM_DIRS:
        if (d / var / 'meta.json').exists():
            return d
    return None


@lru_cache(maxsize=16)
def _clim_year(var, year):
    d = _clim_dir(var)
    if d is None or not readable(d / var / f'clim_{year}.npy'):
        return None
    return np.load(d / var / f'clim_{year}.npy', mmap_mode='r')


def _hours(var, samp):
    return [0] if (samp == '00z' or var in ('tp24', 'olr')) else [0, 1, 2, 3]


def clim_day(var: str, samp: str, d: date):
    """trend-adjusted climatology [121, 240] at valid label date d (evaluated at d's year)."""
    if samp.startswith('prev'):
        return clim_day(var, 'day', d - timedelta(days=1))
    s = 'day' if samp.startswith('day') and samp != 'day->00z' else '00z'
    A = _clim_year(var, d.year)
    if A is None:
        return None
    return np.asarray(A[doy365(d), _hours(var, s)], np.float32).mean(0).T


@lru_cache(maxsize=40)
def _stage(var, year):
    for d in CLIM_DIRS:
        f = d / 'stage' / f'{var}_{year}.npy'
        if readable(f):
            return np.load(f, mmap_mode='r')
    return None


@lru_cache(maxsize=8)
def _slope(var):
    """trend slope b(d, h) [365, 4, 240, 121] per year (smoothed with the fit's harmonics)."""
    from s41eval.sci.clim_trend import harmonics
    f = _clim_dir(var) / var / 'coef_b.npy'
    assert readable(f), f
    cb = np.load(f)
    return np.einsum('dk,k...->d...', harmonics(), cb).astype(np.float32)


def clim_ensemble(var, samp, starts, years=CLIM_YEARS, log=None):
    """climatological ensemble window means [S, len(years), W, 121, 240]: ERA5 of the same calendar dates in each year y,
    shifted along the fitted trend to the start's year ((start.year - y) b(d)); NaN if the stage arrays are missing."""
    s0 = 'day' if samp.startswith('day') and samp != 'day->00z' else ('prev' if samp.startswith('prev') else '00z')
    hrs = _hours(var, 'day' if s0 == 'prev' else s0)
    shift_days = 1 if s0 == 'prev' else 0
    ld = range(0, NLEAD)
    out = np.full((len(starts), len(years), len(WINDOWS), 121, 240), np.nan, np.float32)
    if _clim_dir(var) is None:
        return out[:, :0]
    used = []
    sc = VARS[var].get('scale', 1.0)
    b = _slope(var)[:, hrs].mean(1)                                       # [365, 240, 121]
    for iy, y in enumerate(years):
        if any(_stage(var, y + o) is None for o in range(2)):          # member year and the next (windows into Jan)
            if log:
                log(f'clim ensemble {var}: stage {y} or {y + 1} missing/unreadable -> member dropped')
            continue
        used.append(iy)
        for i, st in enumerate(starts):
            daily = np.full((NLEAD, 121, 240), np.nan, np.float32)
            ds_ = [st + timedelta(days=k - shift_days) for k in range(1, NLEAD)]
            for off in sorted({d.year - st.year for d in ds_}):
                ks = [k for k, d in zip(range(1, NLEAD), ds_) if d.year - st.year == off]
                js = [doy365(st + timedelta(days=k - shift_days)) for k in ks]
                A = _stage(var, y + off)
                x = np.asarray(A[js][:, hrs], np.float32).mean(1) + b[js] * (st.year - y)      # [n, 240, 121]
                daily[ks] = np.swapaxes(x, 1, 2)
            if var in ('tp24',):
                daily = np.clip(daily, 0, None)
            out[i, iy] = window_means(daily) * sc
    if log:
        log(f'clim ensemble {var} {samp}: {len(used)} member years')
    return out[:, used]


def obs_windows(var, samp, starts):
    """ERA5 and climatology window means [S, W, 121, 240] + the sampling actually used."""
    O = np.full((len(starts), len(WINDOWS), 121, 240), np.nan, np.float32)
    C = np.full_like(O, np.nan)
    used = set()
    for i, st in enumerate(starts):
        od = np.full((NLEAD, 121, 240), np.nan, np.float32)
        cd = np.full_like(od, np.nan)
        for k in range(1, NLEAD):
            d = st + timedelta(days=k)
            x, u = obs_day(var, samp, d)
            if x is not None:
                od[k] = x
                used.add(u)
            c = clim_day(var, u if x is not None else samp, d)
            if c is not None:
                cd[k] = c
        n_int = fill_gaps(od)
        if n_int:
            used.add('interp')
        sc = VARS[var].get('scale', 1.0)
        O[i], C[i] = window_means(od) * sc, window_means(cd) * sc
    return O, C, sorted(used)


def fill_gaps(x, max_gap=2):
    """x [T, y, x] daily, in place: missing days (all-NaN fields) between valid days, gaps <= max_gap, linearly interpolated
    (ERA5 files on the hung OST 9 or otherwise missing). Leading/trailing gaps are left NaN. -> number of days filled."""
    ok = np.isfinite(x).any((-2, -1))
    idx = np.flatnonzero(ok)
    n = 0
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for k in range(a + 1, b):
                f = (k - a) / (b - a)
                x[k] = (1 - f) * x[a] + f * x[b]
                n += 1
    return n


# ------------------------------------------------------------------ forecast readers
def _lead_days(ds, init: date):
    if 'lead_day' in ds.coords or 'lead_day' in ds:
        return np.asarray(ds['lead_day'].values, int)
    t = ds['time'].values.astype('datetime64[D]')
    return ((t - np.datetime64(init, 'D')) / np.timedelta64(1, 'D')).astype(int)


def _u200(get, names):
    if 'u200' in names:
        return get('u200'), 'u200'
    if 'u150' in names and 'u250' in names:
        return W150 * get('u150') + (1 - W150) * get('u250'), 'u150/u250 log-p'
    if 'u250' in names:
        return get('u250'), 'u250 as u200'
    return None, 'missing'


def read_h25(path, init: date, vars_=DEFAULT_VARS, mjo=True):
    """one UC-S2S / FuXi member file -> dict(var -> window means [W, 121, 240] in output units), 'mjo' -> {olr, u850, u200:
    [46, 240] 15S-15N means by lead day 0..45 (NaN where absent)}, 'u200_src'."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        ld = _lead_days(ds, init)
        lat = np.asarray(ds['lat'].values)
        flip = lat[0] > lat[-1]
        names = set(ds.data_vars)

        def get(v):
            x = np.asarray(ds[v].values, np.float32)
            return x[..., ::-1, :] if flip else x
        out = {}
        for v in vars_:
            if v == 'u200':
                x, _ = _u200(get, names)
            elif v in names:
                x = get(v)
            else:
                continue
            if x is None:
                continue
            x = to15(x) * VARS[v].get('scale', 1.0)
            out[v] = window_means(x, ld)
        if mjo:
            band = np.abs(LAT15) <= MJO_BAND
            mj = {}
            u2, src = _u200(get, names)
            for v, x in (('olr', get('olr') if 'olr' in names else None), ('u850', get('u850') if 'u850' in names else None),
                         ('u200', u2)):
                arr = np.full((NLEAD, 240), np.nan, np.float32)
                if x is not None:
                    xb = to15(x)[..., band, :].mean(-2)
                    for i, k in enumerate(ld):
                        if 0 <= k < NLEAD:
                            arr[k] = xb[i]
                mj[v] = arr
            out['mjo'] = mj
            out['u200_src'] = src
    return out


def h25_members(root: Path, init: date):
    files = glob.glob(str(root / f'{init:%Y%m%d}' / 'm*_h25.nc'))
    key = lambda f: int(re.search(r'm(\d+)_h25\.nc$', f).group(1))  # noqa: E731
    return sorted([f for f in files if re.search(r'/m\d+_h25\.nc$', f) and readable(f)], key=key)


IFS_PART = {'t2m': 'p1sfc', 'msl': 'p1sfc', 'tp24': 'p1sfc', 'olr': 'p1sfc', 'z500': 'p1pl', 'u850': 'p1pl', 'u200': 'p1pl',
            'sst': 'p2sfc', 'v850': 'p2plb', 't850': 'p2plb', 'v200': 'p2plb', 't200': 'p2plb'}
IFS_KEY = {'t2m': (('2t', 'avg_2t', 'mean2t'), 0), 'msl': (('msl',), 0), 'tp24': (('tp',), 0), 'olr': (('ttr',), 0),
           'sst': (('sst', 'avg_sst'), 0)}                    # avg_*: recent ecCodes name daily-mean 2t/sst (paramId 228004/...)


def _ifs_key(v):
    if v in IFS_KEY:
        return IFS_KEY[v]
    m = re.match(r'([a-z])(\d+)$', v)
    return (({'z': 'gh'}.get(m.group(1), m.group(1)),), int(m.group(2)))


def _grib_var(path, sn, lev):
    """decode only the messages of (shortName, level) -> {number: {(kind, end step h): [121, 240] lat ascending}}, units."""
    import eccodes
    out, units = {}, ''
    with open(path, 'rb') as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                if eccodes.codes_get(h, 'shortName') not in sn or (lev and int(eccodes.codes_get(h, 'level')) != lev):
                    continue
                num = int(eccodes.codes_get(h, 'number'))
                sr = str(eccodes.codes_get(h, 'stepRange'))
                units = str(eccodes.codes_get(h, 'units'))
                st = str(eccodes.codes_get(h, 'stepType'))
                nj, ni = eccodes.codes_get(h, 'Nj'), eccodes.codes_get(h, 'Ni')
                lat0 = eccodes.codes_get(h, 'latitudeOfFirstGridPointInDegrees')
                x = eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
                x[x == eccodes.codes_get(h, 'missingValue')] = np.nan
                if lat0 > 0:
                    x = x[::-1]
                a, _, b = sr.partition('-')
                kind = 'mean' if (b and st == 'avg') else 'inst'
                out.setdefault(num, {})[(kind, int(b or a))] = x
            finally:
                eccodes.codes_release(h)
    return out, units


def ifs_var_daily(day_dir: Path, v):
    """one variable of one IFS start -> (members, [M, 46, 121, 240] by lead day, sampling kind) in our units, or None.
    Instantaneous fields at step 24k; tp/ttr accumulated from step 0 -> 24-h totals / means ending at step 24k (olr = -dttr/86400);
    2t/sst daily means (stepRange 24(k-1)-24k) at lead k."""
    part = IFS_PART.get(v)
    f = day_dir / f'{part}.grib' if part else None
    if f is None or not readable(f):
        return None
    sn, lev = _ifs_key(v)
    raw, units = _grib_var(f, sn, lev)
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
                    if v == 'tp24':
                        dl = np.clip(dl / 1000.0 if 'kg' in units else dl, 0, None)       # kg m-2 (= mm) -> m
                    else:
                        dl = -dl / 86400.0
                    arr[i, k] = dl
            elif e in d:
                arr[i, k] = d[e] * (9.80665 if 'gh' in sn else 1.0)
        if v not in ('tp24', 'olr') and kind == 'inst' and 0 in d:
            arr[i, 0] = d[0] * (9.80665 if 'gh' in sn else 1.0)
    return mem, arr, kind


def read_ifs(day_dir: Path, init: date, vars_=DEFAULT_VARS, mjo=True):
    """-> list of per-member dicts in the read_h25 format (window means + MJO band means); one variable decoded at a time."""
    band = np.abs(LAT15) <= MJO_BAND
    out, mem0 = None, None
    for v in sorted(set(vars_) | ({'olr', 'u850', 'u200'} if mjo else set())):
        r = ifs_var_daily(Path(day_dir), v)
        if r is None:
            continue
        mem, arr, kind = r
        if out is None:
            mem0 = mem
            out = [{'mjo': {}, 'u200_src': 'u200'} for _ in mem]
        if mem != mem0:
            ix = [mem.index(n) if n in mem else None for n in mem0]
        else:
            ix = list(range(len(mem)))
        for i, j in enumerate(ix):
            x = arr[j] if j is not None else np.full((NLEAD, 121, 240), np.nan, np.float32)
            if v in vars_:
                out[i][v] = window_means(x * VARS[v].get('scale', 1.0))
            if mjo and v in ('olr', 'u850', 'u200'):
                out[i]['mjo'][v] = x[:, band].mean(-2)
        del arr
    if out is None:
        return []
    for r in out:
        for v in ('olr', 'u850', 'u200'):
            r['mjo'].setdefault(v, np.full((NLEAD, 240), np.nan, np.float32))
    return out


# ------------------------------------------------------------------ MJO (RMM)
def _i365(d: date) -> int:
    return d.year * 365 + doy365(d)


def _harm3(doy):
    t = 2 * np.pi * np.asarray(doy, float) / 365.0
    return np.stack([np.ones_like(t)] + [f(k * t) for k in (1, 2, 3) for f in (np.cos, np.sin)], 1)


class RMM:
    """Wheeler-Hendon RMM from ERA5 band series (dict var -> {date: [240]}), conventions in the module docstring."""

    def __init__(self, series, eof_years=(1991, 2020), fields=('olr', 'u850', 'u200'), ref=None):
        """fields: ('olr', 'u850', 'u200') = WH04 RMM; ('u850', 'u200') = winds-only RMM (systems without usable OLR), its EOF
        order and signs aligned to ``ref`` (the OLR RMM) by maximum PC correlation."""
        self.series = series
        self.fields = tuple(fields)
        days = sorted(d for d in set.intersection(*[set(series[v]) for v in self.fields])
                      if not (d.month == 2 and d.day == 29))                    # 365-day calendar, as the stage arrays
        self.days = days
        self.ix = {d: i for i, d in enumerate(days)}
        self.ix365 = {_i365(d): i for i, d in enumerate(days)}
        X = {v: np.stack([series[v][d] for d in days]).astype(np.float64) for v in self.fields}
        doy = np.array([doy365(d) for d in days])
        tr = np.array([eof_years[0] <= d.year <= eof_years[1] for d in days])
        H = _harm3(doy)
        self.coef = {v: np.linalg.lstsq(H[tr], X[v][tr], rcond=None)[0] for v in X}
        A = {v: X[v] - H @ self.coef[v] for v in X}
        self.anom = A
        F = {v: self._filt(A[v], days) for v in A}
        ok = tr & np.all([np.isfinite(F[v]).all(1) for v in F], 0)
        self.norm = {v: float(np.sqrt(F[v][ok].var(0).mean())) for v in F}
        Z = np.concatenate([F[v][ok] / self.norm[v] for v in self.fields], 1)
        _, sv, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        eof = Vt[:2].copy()
        self.explained = (sv[:2] ** 2 / (sv ** 2).sum()).tolist()
        pcs = Z @ eof.T
        sd = pcs.std(0)
        eof /= sd[:, None]
        pcs /= sd
        if ref is None:
            lon = LON15
            mc = (lon >= 100) & (lon <= 150)
            if eof[0, :240][mc].mean() > 0:                  # RMM1 > 0 = negative OLR (convection) over the Maritime Continent
                eof[0] *= -1
                pcs[:, 0] *= -1
            lag = 10
            if np.corrcoef(pcs[:-lag, 0], pcs[lag:, 1])[0, 1] < 0:   # RMM2 lags RMM1 (eastward propagation)
                eof[1] *= -1
                pcs[:, 1] *= -1
        else:
            okd = [days[i] for i in np.flatnonzero(ok)]
            jj = [j for j, d in enumerate(okd) if d in ref.obs_rmm and np.isfinite(ref.obs_rmm[d]).all()]
            P = pcs[jj]
            Q = np.stack([ref.obs_rmm[okd[j]] for j in jj])
            Cm = np.corrcoef(P.T, Q.T)[:2, 2:]
            if abs(Cm[1, 0]) + abs(Cm[0, 1]) > abs(Cm[0, 0]) + abs(Cm[1, 1]):
                eof, pcs, Cm = eof[::-1].copy(), pcs[:, ::-1].copy(), Cm[::-1]
            for j in range(2):
                if Cm[j, j] < 0:
                    eof[j] *= -1
                    pcs[:, j] *= -1
            self.ref_corr = [float(abs(Cm[0, 0])), float(abs(Cm[1, 1]))]
        self.eof = eof
        self.obs_rmm = {}
        Fall = np.concatenate([F[v] / self.norm[v] for v in self.fields], 1)
        r = Fall @ eof.T
        for i, d in enumerate(days):
            self.obs_rmm[d] = r[i]

    @staticmethod
    def _filt(a, days):
        """minus the trailing 120-day mean including the current day (MJOcast), NaN where < 120 consecutive days."""
        out = np.full_like(a, np.nan)
        cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, 0)], 0)
        ordn = np.array([_i365(d) for d in days])
        for i in range(119, len(days)):
            if ordn[i] - ordn[i - 119] == 119:
                out[i] = a[i] - (cs[i + 1] - cs[i - 119]) / 120.0
        return out

    def clim(self, v, dates):
        return _harm3([doy365(d) for d in dates]) @ self.coef[v]

    def forecast(self, init: date, band, lead_days=range(NLEAD)):
        """band: {var: [46, 240]} ensemble-mean 15S-15N fields by lead day (lead 0 may be NaN) -> RMM [46, 2] (NaN where missing)."""
        out = np.full((NLEAD, 2), np.nan)
        prev = [self.ix365.get(_i365(init) - j) for j in range(119, 0, -1)]      # 119 days before the init (365-day calendar)
        if any(i is None for i in prev):
            return out
        Z = []
        for v in self.fields:
            pa = np.stack([self.anom[v][i] for i in prev])                       # [119, 240] ERA5 anomalies
            dates = [init + timedelta(days=k) for k in range(NLEAD)]
            fa = np.asarray(band[v], np.float64) - self.clim(v, dates)
            if not np.isfinite(fa[0]).all() and init in self.ix:
                fa[0] = self.anom[v][self.ix[init]]                                         # no lead 0 (FuXi): ERA5 init day
            seq = np.concatenate([pa, fa], 0)                                               # day index 119 = lead 0
            f = np.full((NLEAD, 240), np.nan)
            for k in range(NLEAD):
                win = seq[k:k + 120]                                                        # trailing 120 days incl. lead k
                if np.isfinite(win).all():
                    f[k] = fa[k] - win.mean(0)
            Z.append(f / self.norm[v])
        Z = np.concatenate(Z, 1)
        return Z @ self.eof.T


def rmm_skill(F, O):
    """F, O [n, L, 2] -> bivariate COR [L] and RMSE [L] over n (rows with any NaN dropped per lead)."""
    cor, rmse, n = [], [], []
    for k in range(F.shape[1]):
        ok = np.isfinite(F[:, k]).all(1) & np.isfinite(O[:, k]).all(1)
        f, o = F[ok, k], O[ok, k]
        n.append(int(ok.sum()))
        if ok.sum() < 2:
            cor.append(np.nan)
            rmse.append(np.nan)
            continue
        cor.append(float((f * o).sum() / np.sqrt((f ** 2).sum() * (o ** 2).sum())))
        rmse.append(float(np.sqrt(((f - o) ** 2).sum(1).mean())))
    return np.array(cor), np.array(rmse), np.array(n)


def era5_band_series(first: date, last: date, log=None):
    """ERA5 15S-15N band means by date for olr (daily mean of the date), u850, u200 (00 UTC): stage arrays (1991-2025, clim_trend[_bing]
    stage, hour 0) then ``era5_verif00`` / ``era5_bing_obs`` for dates not in the stage. -> {var: {date: [240]}}"""
    band = np.abs(LAT15) <= MJO_BAND
    out = {v: {} for v in ('olr', 'u850', 'u200')}
    for y in range(first.year, last.year + 1):
        for v in out:
            A = _stage(v, y)
            if A is None:
                continue
            x = np.asarray(A[:, 0][:, :, band], np.float32).mean(-1)        # [365, 240]
            d = date(y, 1, 1)
            while d.year == y:
                if first <= d <= last and not (d.month == 2 and d.day == 29):
                    out[v][d] = x[doy365(d)]
                d += timedelta(days=1)
        if log:
            log(f'band series {y}: {[len(out[v]) for v in out]}')
    d = first
    while d <= last:
        for v in out:
            if d in out[v]:
                continue
            x = _get(VEXTRA, d, 'olr_day') if v == 'olr' else _get(V00, d, v)
            if x is not None:
                out[v][d] = x[band].mean(0)
        d += timedelta(days=1)
    return out


# ------------------------------------------------------------------ driver
def list_starts(path=R / 'runs' / 'bing' / 'starts' / 'starts.csv', first='2025-01-01', last='2026-12-31'):
    with open(path) as f:
        return sorted({date.fromisoformat(r['start']) for r in csv.DictReader(f) if first <= r['start'] <= last})


def _read_member(args):
    kind, path, init, vars_ = args
    try:
        return path, read_h25(path, init, vars_)
    except Exception as e:  # noqa: BLE001
        return path, f'ERROR {type(e).__name__}: {e}'


def _read_ifs_start(args):
    d, init, vars_ = args
    try:
        return init, read_ifs(Path(d), init, vars_)
    except Exception as e:  # noqa: BLE001
        return init, f'ERROR {type(e).__name__}: {e}'


class Log:
    def __init__(self, path=None):
        self.path = path

    def __call__(self, *a):
        s = f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} ' + ' '.join(str(x) for x in a)
        print(s, flush=True)
        if self.path:
            with open(self.path, 'a') as f:
                f.write(s + '\n')


def load_system(system, starts, vars_, roots, cache: Path, procs=8, max_members=None, log=print):
    """read all members of all starts, streaming window means into memmaps.
    -> (starts kept, M, {var: memmap [S, M, W, y, x]}, {start: {olr, u850, u200: [46, 240]} ensemble mean of the first
    det_members members}, info)."""
    import multiprocessing as mp
    cfg = SYSTEMS[system]
    mkdir_striped(cache)
    detm = DET_MEMBERS.get(system)
    if cfg['kind'] == 'h25':
        root = Path(roots.get(system) or HC / cfg['sub'])
        files = {s: h25_members(root, s) for s in starts}
        if max_members:
            files = {s: f[:max_members] for s, f in files.items()}
        keep = [s for s in starts if len(files[s]) >= 2]
        log(f'{system}: {sum(len(files[s]) for s in keep)} member files for {len(keep)}/{len(starts)} starts under {root}')
        if not keep:
            return [], 0, {}, {}, {}
        M = min(len(files[s]) for s in keep)
        if any(len(files[s]) != M for s in keep):
            log(f'{system}: member counts differ {sorted({len(files[s]) for s in keep})}: using the first {M} of each start')
        jobs = [('h25', f, s, vars_) for s in keep for f in files[s][:M]]
        slot = {f: (i, m) for i, s in enumerate(keep) for m, f in enumerate(files[s][:M])}
        worker, items = _read_member, jobs
        nproc = procs
    else:
        root = Path(roots.get(system) or IFS_DIR)
        keep = [s for s in starts if readable(root / f'{s:%Y%m%d}' / 'p1sfc.grib')]
        log(f'{system}: {len(keep)}/{len(starts)} starts with GRIB under {root}')
        if not keep:
            return [], 0, {}, {}, {}
        M = None
        items = [(str(root / f'{s:%Y%m%d}'), s, vars_) for s in keep]
        worker, nproc = _read_ifs_start, min(procs, 3)                  # ~100 members per start: few processes
    E, mjs, mjn, bad, src = {}, {}, {}, set(), set()
    S = len(keep)
    sidx = {s: i for i, s in enumerate(keep)}

    def alloc(m):
        for v in vars_:
            E[v] = np.lib.format.open_memmap(cache / f'{system}_{v}.npy', mode='w+', dtype=np.float32,
                                             shape=(S, m, len(WINDOWS), 121, 240))
            E[v][:] = np.nan

    def put(i, m, rec):
        for v in vars_:
            if v in rec:
                E[v][i, m] = rec[v]
        if 'mjo' in rec and m < (detm or 10 ** 6):
            src.add(rec.get('u200_src', ''))
            for v, x in rec['mjo'].items():
                mjs.setdefault(i, {}).setdefault(v, np.zeros((NLEAD, 240)))
                mjs[i][v] += np.nan_to_num(x)
                mjn.setdefault(i, {}).setdefault(v, np.zeros((NLEAD, 1)))
                mjn[i][v] += np.isfinite(x).all(-1, keepdims=True)
    if M is not None:
        alloc(M)
    with mp.get_context('spawn').Pool(nproc) as pool:
        for c, (key, r) in enumerate(pool.imap_unordered(worker, items, chunksize=1)):
            if isinstance(r, str):
                log(f'{system}: {key}: {r}')
                bad.add(sidx[key] if key in sidx else slot[key][0])
                continue
            if cfg['kind'] == 'h25':
                i, m = slot[key]
                put(i, m, r)
            else:
                if max_members:
                    r = r[:max_members]
                if M is None:
                    M = len(r)                       # IFS: member count of the first start read (normally 101)
                    alloc(M)
                if len(r) < M:
                    log(f'{system}: {key}: {len(r)} < {M} members -> dropped')
                    bad.add(sidx[key])
                    continue
                for m in range(M):
                    put(sidx[key], m, r[m])
            if (c + 1) % 100 == 0:
                log(f'{system}: read {c + 1}/{len(items)}')
    for v in list(E):
        E[v].flush()
    ok = [i for i in range(S) if i not in bad]
    if len(ok) < S:
        log(f'{system}: {S - len(ok)} starts dropped (read errors)')
    for v in list(E):
        if not np.isfinite(np.asarray(E[v][ok[0] if ok else 0, 0])).any():
            log(f'{system}: {v} not in the files -> skipped')
            del E[v]
    mj = {}
    for i in ok:
        if i in mjs:
            mj[keep[i]] = {v: np.where(mjn[i][v] > 0, mjs[i][v] / np.maximum(mjn[i][v], 1), np.nan) for v in mjs[i]}
    info = {'u200_src': sorted(src), 'rows': ok}
    return keep, M, E, mj, info


def run(systems, vars_, starts, tag, roots=None, procs=8, max_members=None, common=False, mjo=True, figures=True,
        out_root=OUT, clim_years=CLIM_YEARS):
    out = mkdir_striped(Path(out_root) / tag)
    cache = out / 'cache'
    log = Log(out / 'log.txt')
    roots = roots or {}
    loaded = {}
    for sy in systems:
        loaded[sy] = load_system(sy, starts, vars_, roots, cache, procs, max_members, log)
        log(f'{sy}: {len(loaded[sy][0])} starts, {loaded[sy][1]} members, vars {sorted(loaded[sy][2])}')
    if common:
        cs = sorted(set.intersection(*[set(loaded[s][0]) for s in systems if loaded[s][0]]))
        log(f'common starts: {len(cs)}')
    rows, maps_all, meta = [], {}, dict(tag=tag, systems={}, vars=vars_, windows=WINDOWS, regions=REGIONS,
                                       clim_years=[clim_years[0], clim_years[-1]], det_members=DET_MEMBERS)
    sel_st = {}
    for sy in systems:
        keep, M, E, mj, info = loaded[sy]
        if not keep:
            log(f'{sy}: no data, skipped')
            continue
        sel = [i for i in info['rows'] if (not common or keep[i] in cs)]
        sel_st[sy] = (sel, [keep[i] for i in sel])
        meta['systems'][sy] = dict(starts=[str(keep[i]) for i in sel], members=M, u200_src=info['u200_src'])
    for v in vars_:
        clim_cache = {}                                   # ERA5 / climatology windows shared by systems with the same sampling+starts
        for sy, (sel, st) in sel_st.items():
            keep, M, E, mj, info = loaded[sy]
            if v not in E or not st:
                continue
            samp = sampling(sy, v)
            key = (samp, tuple(st))
            if key not in clim_cache:
                O, C, used = obs_windows(v, samp, st)
                CE = clim_ensemble(v, samp, st, clim_years, log)
                CC = (np.stack([fair_crps(CE[i], O[i]) for i in range(len(st))]).astype(np.float32) if CE.shape[1] >= 2
                      else np.full_like(O, np.nan))
                log(f'{v} {samp}: climatological ensemble {CE.shape[1]} members')
                del CE
                clim_cache[key] = (O, C, used, CC)
            O, C, used, CC = clim_cache[key]
            if not np.isfinite(O).any():
                log(f'{sy} {v}: no ERA5 verification ({samp}) -> skipped')
                continue
            Ev = np.asarray(E[v][sel]) if len(sel) < len(keep) else np.asarray(E[v])
            r, mp_ = score_var(Ev, O, C, CC, st, DET_MEMBERS.get(sy), VARS[v].get('clip', False))
            us = '+'.join(used)
            for x in r:
                x.update(system=sy, variable=v, sampling=us)
            rows += r
            maps_all[(sy, v)] = mp_
            g = {(x['correction'], x['metric']): x['value'] for x in r if x['region'] == 'global' and x['window'] == 'w34'}
            log(f"{sy} {v} ({us}): w34 global CRPSS raw {g.get(('raw', 'crpss'), np.nan):+.3f} dc {g.get(('dc', 'crpss'), np.nan):+.3f}"
                f" ACC raw {g.get(('raw', 'acc'), np.nan):.3f}")
            del Ev
    write_table(rows, out)
    write_maps(maps_all, out, vars_)
    if mjo:
        rows_m = run_mjo(systems, loaded, out, log, common_starts=cs if common else None)
        write_table(rows_m, out, name='mjo')
        meta['mjo'] = 'see mjo.csv / mjo.json'
    (out / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
    write_summary(rows, out, meta)
    if figures:
        plot_maps(out, vars_)
    log('done', out)
    return out


def write_table(rows, out: Path, name='scores'):
    cols = ['system', 'variable', 'region', 'window', 'correction', 'metric', 'value', 'n_starts', 'members', 'sampling']
    with open(out / f'{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / f'{name}.json').write_text(json.dumps(rows, default=float))


def write_maps(maps_all, out: Path, vars_):
    import xarray as xr
    systems = sorted({k[0] for k in maps_all})
    for sy in systems:
        data = {}
        for v in vars_:
            if (sy, v) not in maps_all:
                continue
            for corr, mp in maps_all[(sy, v)].items():
                for k in ('crpss', 'acc', 'bias'):
                    data[f'{v}_{k}_{corr}'] = (('window', 'lat', 'lon'), mp[k][:, ::-1].astype(np.float32))
        if not data:
            continue
        ds = xr.Dataset(data, coords=dict(window=WK, lat=LAT15[::-1].astype(np.float32), lon=LON15.astype(np.float32)))
        ds.attrs = dict(system=sy, description='per-grid-point CRPSS (vs trend-adjusted climatological ensemble), ACC (Pearson across '
                        'starts of anomalies vs the trend-adjusted climatology), mean bias; s41eval.bing_scores')
        ds.to_netcdf(out / f'maps_{sy}.nc')


def write_summary(rows, out: Path, meta):
    lines = [f"# Bing scores `{meta['tag']}`", '',
             'CRPSS vs the trend-adjusted climatological ensemble and ACC (per-point Pearson across starts, area mean) by window; '
             'raw | drift-corrected (leave-one-year-out). Members/starts per system in `meta.json`. Full table: `scores.csv`.', '']
    idx = {}
    for r in rows:
        idx[(r['system'], r['variable'], r['region'], r['window'], r['correction'], r['metric'])] = r
    systems = list(meta['systems'])
    for v in meta['vars']:
        for rn in ('global', 'tropics', 'nh_extratropics', 'conus'):
            have = [s for s in systems if (s, v, rn, 'w1', 'raw', 'crpss') in idx]
            if not have:
                continue
            lines += [f'## {v}, {rn}', '', '| system | metric | corr | n | ' + ' | '.join(WK) + ' |',
                      '|---|---|---|---|' + '---|' * len(WK)]
            for s in have:
                for m in ('crpss', 'acc'):
                    for c in ('raw', 'dc'):
                        vals = [idx.get((s, v, rn, w, c, m)) for w in WK]
                        n = next((x['n_starts'] for x in vals if x), '')
                        lines.append(f'| {s} | {m} | {c} | {n} | ' + ' | '.join(f"{x['value']:.2f}" if x else '' for x in vals) + ' |')
            lines.append('')
    (out / 'summary.md').write_text('\n'.join(lines))


def run_mjo(systems, loaded, out: Path, log, common_starts=None):
    """RMM skill per system, two variants: 'rmm' (OLR + u850 + u200, WH04) for systems whose OLR is usable, and 'rmm_wind'
    (u850 + u200 only, aligned to the OLR RMM) for every system (UC-S2S writes no usable OLR: the S40 writer sample is ~0)."""
    allst = sorted({s for sy in systems for s in loaded[sy][0]})
    if not allst:
        return []
    first = date(CLIM_YEARS[0], 1, 1)
    last = max(allst) + timedelta(days=NLEAD)
    ser = era5_band_series(first, last, log)
    if min(len(ser[v]) for v in ser) < 365 * 5:
        log(f'MJO: ERA5 band series too short {[len(ser[v]) for v in ser]} -> skipped')
        return []
    rm = RMM(ser)
    rw = RMM(ser, fields=('u850', 'u200'), ref=rm)
    log(f'MJO EOFs: OLR RMM explained {rm.explained}; winds-only explained {rw.explained}, PC corr with OLR RMM {rw.ref_corr}')
    rows, js = [], dict(rmm=dict(explained=rm.explained, norm=rm.norm), rmm_wind=dict(explained=rw.explained, norm=rw.norm,
                                                                                     corr_with_rmm=rw.ref_corr), systems={})
    for sy in systems:
        keep, M, E, mj, info = loaded[sy]
        st = [s for s in keep if mj.get(s) is not None and (common_starts is None or s in common_starts)]
        if not st:
            continue
        olr_sd = np.nanmean([np.nanstd(mj[s]['olr'][1:]) for s in st])
        variants = [('rmm_wind', rw)] + ([('rmm', rm)] if olr_sd > 5.0 else [])     # ERA5 band OLR SD ~ 15-25 W m-2
        if olr_sd <= 5.0:
            log(f'{sy}: model OLR band SD {olr_sd:.2f} W m-2 -> not usable, winds-only RMM only')
        md = min(DET_MEMBERS.get(sy) or M, M)
        for name, R_ in variants:
            F = np.stack([R_.forecast(s, mj[s]) for s in st])
            O = np.stack([np.stack([R_.obs_rmm.get(s + timedelta(days=k), np.full(2, np.nan)) for k in range(NLEAD)]) for s in st])
            cor, rmse, n = rmm_skill(F, O)
            js['systems'].setdefault(sy, {})[name] = dict(starts=[str(s) for s in st], cor=cor.tolist(), rmse=rmse.tolist(),
                                                          n=n.tolist(), members=md, u200=info.get('u200_src'),
                                                          model_olr_band_sd=float(olr_sd))
            for k in range(1, NLEAD):
                for m, val in (('bivariate_cor', cor[k]), ('rmse', rmse[k])):
                    rows.append(dict(system=sy, variable=name, region='15S-15N', window=f'd{k}', correction='raw', metric=m,
                                     value=float(val), n_starts=int(n[k]), members=md,
                                     sampling='u200 ' + '+'.join(info.get('u200_src', []))))
            log(f'{sy} {name} COR day 7/14/21/28: {cor[7]:.2f}/{cor[14]:.2f}/{cor[21]:.2f}/{cor[28]:.2f}')
    (out / 'mjo.json').write_text(json.dumps(js, indent=1, default=float))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        cols = ['#0072B2', '#D55E00', '#CC79A7', '#000000', '#E69F00', '#56B4E9']        # Okabe-Ito, no green
        for i, (sy, rr) in enumerate(js['systems'].items()):
            for name, ls in (('rmm', '-'), ('rmm_wind', '--')):
                if name in rr:
                    ax[0].plot(range(NLEAD), rr[name]['cor'], ls, color=cols[i % 6], label=f'{sy} {name}')
                    ax[1].plot(range(NLEAD), rr[name]['rmse'], ls, color=cols[i % 6])
        ax[0].axhline(0.5, color='k', lw=0.6)
        ax[1].axhline(math.sqrt(2), color='k', lw=0.6)
        ax[0].set(xlabel='lead day', ylabel='bivariate COR', title='RMM, ensemble mean')
        ax[1].set(xlabel='lead day', ylabel='RMSE')
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'mjo_skill.png', dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log('MJO figure failed', e)
    return rows


@lru_cache(maxsize=1)
def land_mask15():
    A = _clim_year('swvl1', 2025)
    if A is None:
        return None
    return np.isfinite(np.asarray(A[0, 0]).T)


def plot_maps(out: Path, vars_, window='w34'):
    """PNG panels (systems side by side) of the weeks 3-4 CRPSS and ACC maps, global and US; RdBu_r (no green); coastline = the
    ERA5 land-sea boundary from the swvl1 climatology (no cartopy data offline)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import xarray as xr
    files = sorted(out.glob('maps_*.nc'))
    if not files:
        return
    ds = {f.stem[5:]: xr.open_dataset(f) for f in files}
    land = land_mask15()
    figdir = mkdir_striped(out / 'figs')
    lat, lon = LAT15[::-1], LON15
    for v in vars_:
        for m in ('crpss', 'acc'):
            for corr in ('raw', 'dc'):
                key = f'{v}_{m}_{corr}'
                sy = [s for s in ds if key in ds[s]]
                if not sy:
                    continue
                for dom, ext in (('global', (0, 358.5, -90, 90)), ('us', (230, 300, 20, 55))):
                    fig, axs = plt.subplots(1, len(sy), figsize=(4.2 * len(sy) + 1, 3.0 if dom == 'us' else 2.6), squeeze=False)
                    for a, s in zip(axs[0], sy):
                        x = ds[s][key].sel(window=window).values
                        im = a.pcolormesh(lon, lat, x, cmap='RdBu_r', vmin=-0.5 if m == 'crpss' else -1, vmax=0.5 if m == 'crpss' else 1,
                                          shading='nearest')
                        if land is not None:
                            a.contour(LON15, LAT15, land.astype(float), levels=[0.5], colors='0.2', linewidths=0.6)
                        a.set_xlim(ext[0], ext[1])
                        a.set_ylim(ext[2], ext[3])
                        a.set_title(f'{s}', fontsize=9)
                        a.set_aspect('equal')
                    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label=f'{m} {window}')
                    fig.suptitle(f'{v} {m} {window} ({corr})', fontsize=10)
                    fig.savefig(figdir / f'{v}_{m}_{corr}_{window}_{dom}.png', dpi=110, bbox_inches='tight')
                    plt.close(fig)
