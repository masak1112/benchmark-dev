"""Shared utilities for the UC-S2S monsoon benchmark."""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath

import config as C


# ============================================================================
# Grid helpers
# ============================================================================
def _edges_overlap(src_c, src_w, tgt_lo, tgt_hi, use_sin):
    """Weights (n_tgt, n_src) = overlap of each target interval with each source cell.

    For latitude (use_sin=True) the overlap is measured in sin(lat), i.e. area on
    the sphere, so the result is exact first-order conservative remapping for a
    regular lat-lon grid (same as CDO remapcon for this case).
    """
    s_lo = src_c - src_w / 2.0
    s_hi = src_c + src_w / 2.0
    lo = np.maximum(tgt_lo[:, None], s_lo[None, :])
    hi = np.minimum(tgt_hi[:, None], s_hi[None, :])
    valid = hi > lo
    if use_sin:
        w = np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo))
    else:
        w = hi - lo
    return np.where(valid, w, 0.0)


class LatLonAverager:
    """Conservative averaging of a regular lat-lon field onto target cells.

    Separable: out = Wlat @ field @ Wlon.T with each row normalised, so it handles
    any number of leading dims.  Works for (a) regridding to the 2-deg India grid
    and (b) a single box average (one target cell = the box).
    """

    def __init__(self, src_lat, src_lon, tgt_lat_lo, tgt_lat_hi, tgt_lon_lo, tgt_lon_hi):
        src_lat = np.asarray(src_lat, float)
        src_lon = np.asarray(src_lon, float)
        dlat = float(np.abs(np.diff(src_lat)).mean())
        dlon = float(np.abs(np.diff(src_lon)).mean())
        wlat = _edges_overlap(src_lat, dlat, np.asarray(tgt_lat_lo, float), np.asarray(tgt_lat_hi, float), True)
        wlon = _edges_overlap(src_lon, dlon, np.asarray(tgt_lon_lo, float), np.asarray(tgt_lon_hi, float), False)
        if (wlat.sum(1) == 0).any() or (wlon.sum(1) == 0).any():
            raise ValueError("Target cell outside source grid")
        # Trim to the source rows/cols actually used (speed)
        self.ilat = np.where(wlat.sum(0) > 0)[0]
        self.ilon = np.where(wlon.sum(0) > 0)[0]
        self.ilat_sl = slice(self.ilat.min(), self.ilat.max() + 1)
        self.ilon_sl = slice(self.ilon.min(), self.ilon.max() + 1)
        wlat = wlat[:, self.ilat_sl]
        wlon = wlon[:, self.ilon_sl]
        self.wlat = wlat / wlat.sum(1, keepdims=True)
        self.wlon = wlon / wlon.sum(1, keepdims=True)

    def __call__(self, arr):
        """arr[..., lat, lon] on the full source grid -> [..., n_tgt_lat, n_tgt_lon]."""
        sub = arr[..., self.ilat_sl, self.ilon_sl]
        return np.einsum("il,...lm,jm->...ij", self.wlat, sub, self.wlon, optimize=True)


def india_regridder(src_lat, src_lon):
    lat = np.array(C.INDIA_LATS)
    lon = np.array(C.INDIA_LONS)
    h = C.INDIA_RES / 2
    return LatLonAverager(src_lat, src_lon, lat - h, lat + h, lon - h, lon + h)


def box_averager(src_lat, src_lon, box):
    s, n, w, e = box
    return LatLonAverager(src_lat, src_lon, [s], [n], [w], [e])


def cmz_mask(lats=None, lons=None):
    """Boolean (lat, lon) mask of 2-deg cells inside the CMZ polygon (as in the core)."""
    lats = np.asarray(C.INDIA_LATS if lats is None else lats, float)
    lons = np.asarray(C.INDIA_LONS if lons is None else lons, float)
    path = MplPath(list(zip(C.CMZ_POLY_LON, C.CMZ_POLY_LAT)))
    lo, la = np.meshgrid(lons, lats)
    inside = path.contains_points(np.c_[lo.ravel(), la.ravel()]).reshape(lo.shape)
    return xr.DataArray(inside, coords=dict(lat=lats, lon=lons), dims=("lat", "lon"))


# ============================================================================
# Observations
# ============================================================================
def load_imd(year):
    """IMD 2x2 daily rainfall as (time, lat, lon) with float coords.

    Dimension names differ between years (TIME/latitude/longitude up to 2022,
    TIME/lat/lon for 2023-24), so they are normalised here.
    """
    fn = C.IMD_DIR / f"data_{year}.nc"
    da = xr.open_dataset(fn)["RAINFALL"]
    ren = {k: v for k, v in {"TIME": "time", "latitude": "lat", "longitude": "lon",
                             "LATITUDE": "lat", "LONGITUDE": "lon"}.items() if k in da.dims}
    da = da.rename(ren).transpose("time", "lat", "lon")
    da = da.assign_coords(lat=da.lat.astype(float), lon=da.lon.astype(float),
                          time=pd.to_datetime(da.time.values).normalize())
    return da.astype("float32").load()


@lru_cache(maxsize=None)
def imd_series(start_year, end_year):
    """Concatenated IMD daily (time, lat, lon) for a range of years."""
    return xr.concat([load_imd(y) for y in range(start_year, end_year + 1)], dim="time")


def imd_valid_mask(years):
    """Cells with valid IMD data in every year listed (2023-24 files have a larger mask)."""
    m = None
    for y in years:
        v = load_imd(y).notnull().any("time")
        m = v if m is None else (m & v)
    return m


def load_threshold():
    th = xr.open_dataset(C.THRESH_FILE)["MWmean"]
    return th.assign_coords(lat=th.lat.astype(float), lon=th.lon.astype(float)).load()


def load_obs_indices():
    ds = xr.open_dataset(C.OBS_INDEX_FILE).load()
    ds["time"] = pd.to_datetime(ds.time.values).normalize()
    return ds


# ============================================================================
# Model forecasts
# ============================================================================
def _open_model_tp(model_key, year):
    m = C.MODELS[model_key]
    fn = m["tp_dir"] / f"{year}.nc"
    if not fn.exists():
        raise FileNotFoundError(fn)
    ds = xr.open_dataset(fn)
    d = m["dims"]
    da = ds[m["tp_var"]].rename({d["init"]: "init_time", d["member"]: "member", d["lead"]: "day"})
    da = da.assign_coords(lat=da.lat.astype(float), lon=da.lon.astype(float))
    return da


def init_shift(model_key):
    return pd.Timedelta(days=int(C.MODELS[model_key].get("init_shift_days", 0)))


def model_init_dates(model_key, year, months=None):
    """Hindcast START dates in the model file for `year` (before any init_shift), optionally by month.

    Start dates are what models are matched on ("common" mode); the loaders return the
    effective init (start + init_shift_days) that the scores use."""
    da = _open_model_tp(model_key, year)
    t = pd.DatetimeIndex(pd.to_datetime(da.init_time.values))
    if months is not None:
        t = t[t.month.isin(list(months))]
    return t.sort_values()


def common_init_dates(model_keys, year, months=C.ONSET_INIT_MONTHS):
    out = None
    for k in model_keys:
        t = model_init_dates(k, year, months)
        out = t if out is None else out.intersection(t)
    return out.sort_values()


def load_model_tp(model_key, year, mem_num=C.MEM_NUM, init_dates=None, months=None, max_step=None):
    """Standardised forecast precip: DataArray tp(init_time, member, step, lat, lon) in mm/day.

    step k = 24 h accumulation ending at init + k days (see config docstring).
    Only steps >= 1 are kept.  The first `mem_num` members are used, as in the
    legacy loader.
    """
    m = C.MODELS[model_key]
    da = _open_model_tp(model_key, year)
    if init_dates is not None:
        want = pd.DatetimeIndex(init_dates)
        have = pd.DatetimeIndex(pd.to_datetime(da.init_time.values))
        sel = have[have.isin(want)]
        if len(sel) == 0:
            raise ValueError(f"No matching initialization times found for {model_key} {year}")
        da = da.sel(init_time=sel)
    elif months is not None:
        t = pd.DatetimeIndex(pd.to_datetime(da.init_time.values))
        da = da.isel(init_time=np.where(t.month.isin(list(months)))[0])
    da = da.isel(member=slice(0, mem_num))
    if m.get("drop_last_day"):
        da = da.isel(day=slice(0, -1))
    step = da["day"].values.astype(int) + int(m["day_offset"])
    da = da.assign_coords(day=step).rename(day="step")
    keep = step >= 1
    if max_step is not None:
        keep &= step <= max_step
    da = da.isel(step=np.where(keep)[0])
    units = str(da.attrs.get("units", "mm/day"))
    da = da.transpose("init_time", "member", "step", "lat", "lon").load()
    if init_shift(model_key):
        da = da.assign_coords(init_time=pd.DatetimeIndex(da.init_time.values) + init_shift(model_key))
    if units.strip() == "m":
        da = da * 1000.0
    da.name = "tp"
    da.attrs["units"] = "mm/day"
    return da


def load_model_indices(model_key, year, mem_num=C.MEM_NUM, init_dates=None, months=None):
    """Per-member indices (init_time, member, lead) where lead = days after the effective init.

    UC-S2S: from the postprocessed indices file (lead 0 = initial state).
    NGCM  : only the Webster-Yang index exists (pre-computed, 6-hourly); the
            00 UTC values at step = 24*k hours are used, sign flipped to U850-U200.
    """
    m = C.MODELS[model_key]
    if "indices_dir" in m:
        ds = xr.open_dataset(m["indices_dir"] / f"{year}.nc").rename(day="lead")
    elif "wy_dir" in m:
        w = xr.open_dataset(m["wy_dir"] / f"{year}.nc")
        w = w.rename({"time": "init_time", "number": "member"})
        steps = w.step.values
        keep = steps[(steps % 24) == 0]
        wy = w["webster_yang_index"].sel(step=keep) * m["wy_sign"]
        wy = wy.assign_coords(step=(keep // 24).astype(int)).rename(step="lead")
        ds = xr.Dataset({"wy": wy})
        ds["member"] = np.arange(ds.sizes["member"])
    else:
        raise KeyError(f"No indices for {model_key}")
    t = pd.DatetimeIndex(pd.to_datetime(ds.init_time.values))
    if init_dates is not None:
        ds = ds.sel(init_time=t[t.isin(pd.DatetimeIndex(init_dates))])
    elif months is not None:
        ds = ds.isel(init_time=np.where(t.month.isin(list(months)))[0])
    ds = ds.isel(member=slice(0, mem_num))
    ds = ds.transpose("init_time", "member", "lead").load()
    if init_shift(model_key):
        ds = ds.assign_coords(init_time=pd.DatetimeIndex(ds.init_time.values) + init_shift(model_key))
    return ds


def index_obs_spec(model_key, name):
    """(ERA5 variable, day shift) verifying index `name` of `model_key`:
    verifying date = effective init + lead + shift."""
    m = C.MODELS[model_key]
    if "wy_dir" in m and "indices_dir" not in m:
        return "wy_true200", 0                    # NGCM pre-computed WY uses the real 200 hPa
    spec = m.get("index_obs", {})
    return name + spec.get("suffix", ""), int(spec.get("shift", 0))


# ============================================================================
# Statistics
# ============================================================================
def noleap_doy(dates):
    """Day of year 1..365 ignoring 29 Feb (mapped onto 28 Feb)."""
    d = pd.DatetimeIndex(dates)
    doy = d.dayofyear.values.copy()
    leap = d.is_leap_year
    doy[leap & (doy >= 60)] -= 1
    return doy


def harmonic_smooth_clim(values, n_harm=3):
    """Smooth a (365, ...) daily climatology with the mean + first n harmonics."""
    v = np.asarray(values, float)
    n = v.shape[0]
    t = 2 * np.pi * np.arange(n) / n
    basis = [np.ones(n)]
    for k in range(1, n_harm + 1):
        basis += [np.cos(k * t), np.sin(k * t)]
    X = np.stack(basis, 1)                                  # (365, 2n+1)
    flat = v.reshape(n, -1)
    good = np.isfinite(flat).all(0)
    out = np.full_like(flat, np.nan)
    if good.any():
        coef, *_ = np.linalg.lstsq(X, flat[:, good], rcond=None)
        out[:, good] = X @ coef
    return out.reshape(v.shape)


def daily_climatology(series_da, years, n_harm=3):
    """(365, ...) smoothed day-of-year climatology from a (time, ...) DataArray."""
    t = pd.DatetimeIndex(series_da.time.values)
    sel = t.year.isin(years)
    da = series_da.isel(time=np.where(sel)[0])
    doy = noleap_doy(da.time.values)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        raw = np.stack([np.nanmean(da.values[doy == d], axis=0) for d in range(1, 366)])
    return harmonic_smooth_clim(raw, n_harm)


def fair_crps(ens, obs, axis=-1):
    """Fair (ensemble-size unbiased) CRPS, Ferro (2014).  ens has members on `axis`."""
    ens = np.moveaxis(np.asarray(ens, float), axis, -1)
    obs = np.asarray(obs, float)
    m = ens.shape[-1]
    t1 = np.nanmean(np.abs(ens - obs[..., None]), axis=-1)
    s = np.sort(ens, axis=-1)
    i = np.arange(1, m + 1)
    # sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - m - 1) x_(i)
    pair = 2.0 * np.sum((2 * i - m - 1) * s, axis=-1)
    t2 = pair / (2.0 * m * (m - 1))
    out = t1 - t2
    out[~np.isfinite(obs)] = np.nan
    return out


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def year_block_bootstrap(stat_fn, years, n_boot=1000, seed=0, ci=(5, 95)):
    """Percentile CI of stat_fn(idx) when resampling whole years with replacement."""
    rng = np.random.default_rng(seed)
    years = np.asarray(years)
    uy = np.unique(years)
    groups = [np.where(years == y)[0] for y in uy]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uy), len(uy))
        idx = np.concatenate([groups[p] for p in pick])
        vals.append(stat_fn(idx))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return tuple(np.percentile(vals, ci))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
