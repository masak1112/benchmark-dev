"""Grids, fields, windows and the pure score functions of the S41 evaluation package (no I/O).

Grid: the common conservative 1.5-degree WeatherBench2 grid ``g15`` (121 x 240; cells centred on lat -90..90 step 1.5 with half cells at
the poles, lon 0..358.5), internal latitude order **ascending** (-90..90); written maps are flipped to 90..-90 (WB2 order).
Sensitivity grid ``g30``: conservative 3 degrees (61 x 120, lat -90..90 step 3, lon 0..357), made from g15 by area overlap.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from functools import lru_cache

import numpy as np

LAT = {'g15': np.linspace(-90.0, 90.0, 121), 'g30': np.linspace(-90.0, 90.0, 61)}
LON = {'g15': np.arange(240) * 1.5, 'g30': np.arange(120) * 3.0}
GRIDS = tuple(LAT)
WINDOWS = {'w1': (1, 7), 'w2': (8, 14), 'w3': (15, 21), 'w4': (22, 28), 'w5': (29, 35), 'w6': (36, 42), 'w34': (15, 28), 'w56': (29, 42)}
# extra windows appended AFTER the 8 standard ones (indices of the standard windows unchanged), e.g. BING_EXTRA_WINDOWS=w12:1-14 for the
# W12 ablation tag (pre-registered weeks 1-2); never set for v1/v1.5 (their caches have 8 windows)
for _w in filter(None, __import__('os').environ.get('BING_EXTRA_WINDOWS', '').split(',')):
    _n, _r = _w.split(':')
    WINDOWS[_n] = tuple(int(v) for v in _r.split('-'))
WK = list(WINDOWS)
NLEAD = 46
REGIONS = {'global': (-90, 90, 0, 360), 'tropics': (-20, 20, 0, 360), 'nh_extratropics': (20, 90, 0, 360),
           'sh_extratropics': (-90, -20, 0, 360), 'conus': (25, 50, 235, 295), 'land': 'land', 'ocean': 'ocean'}
RK = list(REGIONS)
SEASONS = {'all': tuple(range(1, 13)), 'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}
PERIODS = {'2019-24': (2019, 2024, ()), '2025-26': (2025, 2026, ()), 'pooled': (2019, 2026, ()), 'pooled_no2019': (2019, 2026, (2019,))}
# canonical fields (output units). scale: file units -> output units; clip: clip at 0 after correction; flux: 24-h quantity
FIELDS = {'t2m': dict(units='K'), 'tp24': dict(units='mm/day', scale=1000.0, clip=True, flux=True), 'msl': dict(units='Pa'),
          'z500': dict(units='m2 s-2'), 'u850': dict(units='m s-1'), 'u200': dict(units='m s-1'), 'u250': dict(units='m s-1'),
          't850': dict(units='K'), 'sst': dict(units='K'), 'olr': dict(units='W m-2 (positive up)', flux=True),
          'swvl1': dict(units='m3 m-3'), 'z50': dict(units='m2 s-2'), 'u50': dict(units='m s-1')}
FK = list(FIELDS)
W150 = math.log(250 / 200) / math.log(250 / 150)          # Bing's log-p weight of 150 hPa for 200 hPa (0.437)
K_TERCILE = 3


# ------------------------------------------------------------------ calendar
def doy365(d: date) -> int:
    """0-based day of year in the 365-day calendar of the stage arrays (Feb 29 -> Feb 28)."""
    if d.month == 2 and d.day == 29:
        d = d - timedelta(days=1)
    return (date(2001, d.month, d.day) - date(2001, 1, 1)).days


def season_of(d: date) -> str:
    return next(k for k, v in SEASONS.items() if k != 'all' and d.month in v)


def in_period(d: date, period: str) -> bool:
    a, b, excl = PERIODS[period]
    return a <= d.year <= b and d.year not in excl


def subset_index(starts, period='pooled', season='all'):
    return np.array([i for i, s in enumerate(starts) if in_period(s, period) and s.month in SEASONS[season]], int)


# ------------------------------------------------------------------ grids
def _overlap(src_edges, dst_edges):
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.clip(hi - lo, 0, None)


def _cell_edges_lat(c, step):
    return np.sin(np.deg2rad(np.clip(np.concatenate([c - step / 2, [c[-1] + step / 2]]), -90, 90)))


@lru_cache(maxsize=4)
def cons_weights(src='g15', dst='g30'):
    """separable conservative weights (Wlat [ny_dst, ny_src], Wlon [nx_dst, nx_src]) between two regular global grids."""
    sl, dl = LAT[src], LAT[dst]
    Wlat = _overlap(_cell_edges_lat(sl, sl[1] - sl[0]), _cell_edges_lat(dl, dl[1] - dl[0]))
    ss, ds_ = LON[src][1] - LON[src][0], LON[dst][1] - LON[dst][0]
    n = len(LON[src])
    s = np.concatenate([LON[src] - 360 - ss / 2, LON[src] - ss / 2, LON[src] + 360 - ss / 2, [LON[src][-1] + 360 + ss / 2]])
    d = np.concatenate([LON[dst] - ds_ / 2, [LON[dst][-1] + ds_ / 2]])
    W3 = _overlap(s, d)
    Wlon = W3[:, :n] + W3[:, n:2 * n] + W3[:, 2 * n:]
    return Wlat, Wlon


def regrid(x, src='g15', dst='g30', min_frac=0.5):
    """conservative, NaN-aware: x [..., ny_src, nx_src] -> [..., ny_dst, nx_dst] (NaN where < min_frac of the cell is valid)."""
    if src == dst:
        return np.asarray(x, np.float32)
    Wlat, Wlon = cons_weights(src, dst)
    x = np.asarray(x, np.float64)
    m = np.isfinite(x)
    num = np.einsum('ia,...ab,jb->...ij', Wlat, np.where(m, x, 0.0), Wlon, optimize=True)
    den = np.einsum('ia,...ab,jb->...ij', Wlat, m.astype(np.float64), Wlon, optimize=True)
    full = Wlat.sum(1)[:, None] * Wlon.sum(1)[None, :]
    out = num / np.where(den > 0, den, 1.0)
    out[den < min_frac * full] = np.nan
    return out.astype(np.float32)


def to_g15(x):
    """[..., 180, 360] 1-degree (UC-S2S, lat ascending -89.5..89.5) -> g15 (``ucs2s_hindcast.to_1p5``, conservative); g15 passes."""
    x = np.asarray(x)
    if x.shape[-2:] == (121, 240):
        return x.astype(np.float32)
    if x.shape[-2:] != (180, 360):
        raise ValueError(f'unexpected grid {x.shape[-2:]}')
    from s41eval.hindcast_grid import to_1p5
    return to_1p5(x)


@lru_cache(maxsize=4)
def cosw(grid='g15'):
    return np.cos(np.deg2rad(LAT[grid]))[:, None] * np.ones((1, len(LON[grid])))


_LSM = {}


def set_lsm(lsm15):
    """land-sea fraction on g15 [121, 240] lat ascending (ERA5 lsm); land = lsm >= 0.5."""
    _LSM['g15'] = np.asarray(lsm15, np.float64)
    _LSM['g30'] = regrid(lsm15, 'g15', 'g30').astype(np.float64)
    masks.cache_clear()


@lru_cache(maxsize=4)
def masks(grid='g15'):
    lat, lon = LAT[grid], LON[grid]
    out = {}
    for k, r in REGIONS.items():
        if isinstance(r, str):
            if grid not in _LSM:
                continue
            land = _LSM[grid] >= 0.5
            out[k] = (land if r == 'land' else ~land).astype(float)
            continue
        s, n, w, e = r
        out[k] = ((lat[:, None] >= s) & (lat[:, None] <= n) & (lon[None] >= w) & (lon[None] <= e)).astype(float)
    return out


def amean(x, m, grid=None):
    """cos-lat area mean of x [..., ny, nx] over mask m (NaN-aware)."""
    grid = grid or ('g15' if x.shape[-2] == 121 else 'g30')
    w = cosw(grid) * m
    ok = np.isfinite(x)
    num = (np.where(ok, x, 0.0) * w).sum((-2, -1))
    den = (ok * w).sum((-2, -1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def region_means(x, grid):
    """x [..., ny, nx] -> [..., n_regions] (order RK; NaN for a region without mask)."""
    mk = masks(grid)
    return np.stack([amean(x, mk[r], grid) if r in mk else np.full(x.shape[:-2], np.nan) for r in RK], -1)


# ------------------------------------------------------------------ windows
def window_means(x, lead_days=None):
    """x [..., T, y, x] daily with lead days ``lead_days`` (default 0..T-1) -> [..., len(WINDOWS), y, x]; NaN if a day is missing
    (a whole-field-NaN day or an absent lead). Precipitation: mean daily total."""
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
        allnan = ~np.isfinite(seg).any((-2, -1))
        m = seg.mean(-3)
        if allnan.any():
            m = np.where(allnan.any(-1)[..., None, None], np.nan, m)
        out.append(m)
    return np.stack(out, -3).astype(np.float32)


def fill_gaps(x, max_gap=2):
    """x [T, y, x] daily, in place: interior missing days (whole field NaN), gaps <= max_gap, linearly interpolated -> days filled."""
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


# ------------------------------------------------------------------ scores
def _sorted_last(ens):
    """members axis 0 -> sorted along a contiguous last axis (fast sort), float32."""
    e = np.ascontiguousarray(np.moveaxis(np.asarray(ens, np.float32), 0, -1))
    e.sort(-1)
    return e


def fair_crps(ens, obs, sorted_last=None):
    """ens [m, ...], obs [...] -> fair CRPS [...] (Ferro 2014): E|X-y| - sum_ij |xi-xj| / (2 m (m-1)); NaN where obs is NaN.
    (sorted-member form: sum_ij |xi-xj| / 2 = sum_k (2k - m + 1) x_(k), k = 0..m-1, accumulated in float64)"""
    s = _sorted_last(ens) if sorted_last is None else sorted_last
    m = s.shape[-1]
    o = np.asarray(obs, np.float64)[..., None]
    a = np.abs(s.astype(np.float64) - o).mean(-1)
    k = (2 * np.arange(m) - m + 1).astype(np.float64)
    return a - np.einsum('...m,m->...', s.astype(np.float64), k) / (m * (m - 1))


def tercile_edges(ref_ens, sorted_last=None):
    """ref_ens [n, ...] -> [2, ...] lower / upper tercile boundaries (numpy 'linear' quantiles 1/3, 2/3)."""
    s = _sorted_last(ref_ens) if sorted_last is None else sorted_last
    n = s.shape[-1]
    out = []
    for q in (1 / 3, 2 / 3):
        h = (n - 1) * q
        j = int(np.floor(h))
        f = h - j
        out.append(s[..., j] if j + 1 >= n else s[..., j] + f * (s[..., j + 1] - s[..., j]))
    return np.stack(out).astype(np.float32)


def loyo_weights(starts, valid=None, window=30):
    """leave-one-year-out drift training weights: W [S, S] (rows sum to 1 or 0); row s = equal weights on the starts of OTHER years
    whose day of year is within +-window days (circular) of s (``biascorr.training_sets``), restricted to ``valid`` [S] starts."""
    from s41eval.biascorr import training_sets
    tr = training_sets(list(starts), window)
    S = len(starts)
    W = np.zeros((S, S))
    for s, idx in enumerate(tr):
        j = [i for i in idx if valid is None or valid[i]]
        assert all(starts[i].year != starts[s].year for i in j)
        if j:
            W[s, j] = 1.0 / len(j)
    return W


def drift_bias(em, O, starts, valid):
    """em, O [S, W, y, x]; valid [S, W] -> LOYO mean bias [S, W, y, x] (NaN where no training start). NaN-aware per point."""
    d = (em - O).astype(np.float64)
    S, Wn = d.shape[:2]
    b = np.full(d.shape, np.nan, np.float32)
    for w in range(Wn):
        Wt = loyo_weights(starts, valid[:, w]) > 0
        X = d[:, w].reshape(S, -1)
        ok = np.isfinite(X)
        num = Wt.astype(np.float64) @ np.where(ok, X, 0.0)
        den = Wt.astype(np.float64) @ ok
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        b[:, w] = r.reshape(d[:, w].shape)
    return b


def acc_map(fa, oa):
    """fa, oa [n, y, x] -> Pearson correlation across n per point (centred), NaN-aware (pairs with both finite)."""
    ok = np.isfinite(fa) & np.isfinite(oa)
    n = ok.sum(0)
    f, o = np.where(ok, fa, 0.0), np.where(ok, oa, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        fm, om = f.sum(0) / n, o.sum(0) / n
        fc, oc = np.where(ok, f - fm, 0.0), np.where(ok, o - om, 0.0)
        r = (fc * oc).sum(0) / np.sqrt((fc ** 2).sum(0) * (oc ** 2).sum(0))
    return np.where(n >= 3, r, np.nan)


def acc_pattern(fa, oa, m, grid):
    """uncentred cos-lat pattern ACC over mask m, per leading index: fa, oa [..., y, x] -> [...]"""
    w = cosw(grid) * m
    ok = np.isfinite(fa) & np.isfinite(oa)
    f, o = np.where(ok, fa, 0), np.where(ok, oa, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (w * f * o).sum((-2, -1)) / np.sqrt((w * f * f).sum((-2, -1)) * (w * o * o).sum((-2, -1)))


def tercile_probs(ens, edges):
    """ens [m, ...], edges [2, ...] -> (p_below [...], p_above [...]) = fraction of members < lower / > upper boundary."""
    e = np.asarray(ens)
    return (e < edges[0][None]).mean(0), (e > edges[1][None]).mean(0)


def rps3(p_lo, p_hi, obs, edges):
    """ranked probability score (3 categories, cumulative, not normalised) of probabilities vs the observed category."""
    o_lo = (obs < edges[0]).astype(np.float64)
    o_mid_or_lo = (obs <= edges[1]).astype(np.float64)
    c1, c2 = p_lo, 1.0 - p_hi
    r = (c1 - o_lo) ** 2 + (c2 - o_mid_or_lo) ** 2
    return np.where(np.isfinite(obs) & np.isfinite(edges[0]), r, np.nan)


def rps_clim3(obs, edges):
    """RPS of the climatological 1/3-1/3-1/3 forecast."""
    return rps3(np.full(np.shape(obs), 1 / 3), np.full(np.shape(obs), 1 / 3), obs, edges)


def rpss_debias_D(M, K=K_TERCILE):
    """Weigel et al. (2007) ensemble-size correction D = (K^2 - 1) / (6 K M) for equiprobable categories (K=3: 4/(9M))."""
    return (K * K - 1) / (6.0 * K * M)


def obs_rank(ens, obs, rng):
    """rank of obs among the m members, 0..m, ties broken at random -> int array [...] (-1 where obs NaN)."""
    e = np.asarray(ens)
    lt = (e < obs[None]).sum(0)
    eq = (e == obs[None]).sum(0)
    r = lt + np.floor(rng.random(obs.shape) * (eq + 1)).astype(int)
    return np.where(np.isfinite(obs), r, -1)


REL_BINS = np.linspace(-0.05, 1.05, 12)              # 11 bins centred 0, 0.1, ..., 1


def reliability_counts(p, o, w):
    """p (forecast probability), o (0/1 event), w (area weight) [...] -> [11 bins, 3] (sum w, sum w p, sum w o)."""
    ok = np.isfinite(p) & np.isfinite(o) & (w > 0)
    b = np.clip(np.digitize(p[ok], REL_BINS) - 1, 0, 10)
    ww = w[ok]
    out = np.zeros((11, 3))
    np.add.at(out[:, 0], b, ww)
    np.add.at(out[:, 1], b, ww * p[ok])
    np.add.at(out[:, 2], b, ww * o[ok])
    return out


def ssr(M, mean_var, mse):
    """spread/error ratio with the finite-ensemble factor: sqrt((M+1)/M * mean variance) / RMSE (Fortin et al. 2014)."""
    if not (mse > 0) or M < 2 or not np.isfinite(mean_var):
        return float('nan')
    return math.sqrt((M + 1) / M * mean_var) / math.sqrt(mse)


def draws11(M, n=10, k=11, seed=11):
    """n fixed random k-member subsets of 0..M-1 (same for every start of a system)."""
    rng = np.random.default_rng(seed)
    if M <= k:
        return [np.arange(M)]
    return [np.sort(rng.choice(M, k, replace=False)) for _ in range(n)]
