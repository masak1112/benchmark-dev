#!/usr/bin/env python
"""
Drift / bias correction of the postprocessed forecasts -> "<model>_bc" datasets.

Method: the S41 "dc" (drift-corrected) level of
metrics/S41_aurora/s41eval/biascorr.py (mean level) and bing_eval/core.py::drift_bias,
applied to DAILY lead days:

    b[s, L, cell] = mean over training starts t of ( ensmean_t[L, cell] - obs_t[L, cell] )
    training starts of s = starts of the OTHER years whose day of year is within
                           +-BC_WINDOW_DAYS of s (circular)   -> leave-one-year-out
    b is smoothed over leads with a centred BC_LEAD_SMOOTH-day running mean (truncated
    at the ends), as biascorr.py does for its statistics;
    corrected x = x - b for every member (precipitation clipped at 0 afterwards, as the
    S41 pipeline does for tp24).

The mean is NaN-aware and paired (a start/lead/cell without an observation contributes to
neither the model nor the observed mean).  Where no bias can be estimated (cells IMD never
observes) the raw value is kept, as the S41 pipeline does (`nan_to_num(b)`).

References: IMD 2-deg for rainfall (the dataset everything is verified against here) and
ERA5 for the indices, sampled like each model (00 UTC states for UC-S2S, calendar-day
means for FuXi-S2S; see config.MODELS[..]['index_obs']).  Only starts with verifying
observations (config.BC_TRAIN_YEARS) are corrected and used for training.

Output files have exactly the layout of the input files (same start dates, day index and
day_offset), so every downstream script reads "<model>_bc" like any other model.

Usage:  python bias_correct.py --models ucs2s_plain ucs2s_sstanom fuxi
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xarray as xr

import config as C
import common as U


def training_matrix(starts, window, train_years):
    """W [S, S] bool: W[s, t] = start t trains start s (other year, |doy diff| <= window, circular)."""
    doy = U.noleap_doy(starts).astype(int)
    yrs = np.asarray(pd.DatetimeIndex(starts).year)
    dd = np.abs(doy[:, None] - doy[None, :])
    dd = np.minimum(dd, 365 - dd)
    return (yrs[:, None] != yrs[None, :]) & (dd <= window) & np.isin(yrs, train_years)[None, :]


def smooth_leads(a, n):
    """centred running mean over axis 1 (leads), truncated at the ends, NaN-aware."""
    if not n or n <= 1:
        return a
    h = n // 2
    out = np.empty_like(a)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        for l in range(a.shape[1]):
            out[:, l] = np.nanmean(a[:, max(0, l - h):l + h + 1], axis=1)
    return out


def drift_bias(em, obs, W, lead_smooth):
    """em, obs [S, L, ...] -> LOYO paired mean bias b [S, L, ...] (NaN where not estimable)."""
    d = (np.asarray(em, np.float64) - np.asarray(obs, np.float64))
    S = d.shape[0]
    X = d.reshape(S, -1)
    ok = np.isfinite(X)
    Wf = W.astype(np.float64)
    num = Wf @ np.where(ok, X, 0.0)
    den = Wf @ ok
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan).reshape(d.shape)
    return smooth_leads(b, lead_smooth)


# ----------------------------------------------------------------------------
def correct_tp(base, out_rows):
    t0 = time.time()
    m = C.MODELS[base]
    years = C.BC_TRAIN_YEARS
    parts, inits = [], []
    for y in years:
        da = U.load_model_tp(base, y, mem_num=C.MEM_NUM)
        parts.append(da.values.astype(np.float32))
        inits += list(pd.to_datetime(da.init_time.values))
        steps, lat, lon = da.step.values, da.lat.values, da.lon.values
    F = np.concatenate(parts)                                          # [S, M, L, y, x]
    inits = pd.DatetimeIndex(inits)
    S, M, L = F.shape[:3]
    last = max(y for y in range(2019, 2031) if (C.IMD_DIR / f"data_{y}.nc").exists())
    obs = U.imd_series(min(years), min(max(years) + 1, last))
    vd = inits.values[:, None] + steps.astype("timedelta64[D]")[None, :]
    O = obs.reindex(time=vd.ravel()).values.reshape(S, L, lat.size, lon.size)
    W = training_matrix(inits, C.BC_WINDOW_DAYS, years)
    b = drift_bias(F.mean(1), O, W, C.BC_LEAD_SMOOTH)                  # [S, L, y, x]
    Fbc = np.clip(F - np.nan_to_num(b)[:, None], 0, None).astype(np.float32)

    # diagnostics: CMZ-mean bias by lead week for May-Jul starts
    cmz = (U.cmz_mask() & U.imd_valid_mask(years)).values
    mjj = inits.month.isin(C.ONSET_INIT_MONTHS)
    for wk, (k1, k2) in C.WEEKS.items():
        sl = (steps >= k1) & (steps <= k2)
        out_rows.append(dict(model=base, target="rain_cmz", week=wk,
                             bias_removed=float(np.nanmean(b[mjj][:, sl][..., cmz])),
                             raw_bias=float(np.nanmean((F.mean(1) - O)[mjj][:, sl][..., cmz])),
                             corrected_bias=float(np.nanmean((Fbc.mean(1) - O)[mjj][:, sl][..., cmz]))))

    # write in the base file layout: start date = init - shift, day = step - day_offset (day 0 = 0)
    file_starts = inits - U.init_shift(base)
    days = steps - int(m["day_offset"])
    nday = int(days.max()) + 1
    out_dir = U.ensure_dir(C.MODELS[f"{base}_bc"]["tp_dir"])
    for y in years:
        sel = np.where(file_starts.year == y)[0]
        arr = np.zeros((len(sel), M, nday, lat.size, lon.size), np.float32)
        arr[:, :, days] = Fbc[sel]
        ds = xr.Dataset({"tp": (("init_time", "member", "day", "lat", "lon"), arr)},
                        coords=dict(init_time=file_starts[sel].values, member=np.arange(M),
                                    day=np.arange(nday), lat=lat, lon=lon))
        ds["tp"].attrs = dict(units="mm/day", long_name="Total precipitation (24 h), drift/bias corrected",
                              convention=f"same day/init convention as {base}; day 0 = 0")
        ds.attrs = dict(model=C.MODELS[f"{base}_bc"]["label"], bias_correction=(
            f"LOYO mean drift vs IMD 2-deg, +-{C.BC_WINDOW_DAYS} d day-of-year window, "
            f"{C.BC_LEAD_SMOOTH}-d lead smoothing, clipped at 0 (S41 'dc' level)"),
            train_years=json.dumps(years), source=str(m["tp_dir"]))
        ds.to_netcdf(out_dir / f"{y}.nc", encoding={"tp": dict(zlib=True, complevel=1)})
    print(f"  {base} tp: S={S} starts x M={M} x L={L}; written to {out_dir} in {time.time() - t0:.0f}s", flush=True)


def correct_indices(base, obs_ds, out_rows):
    t0 = time.time()
    years = C.BC_TRAIN_YEARS
    ds = xr.concat([U.load_model_indices(base, y) for y in years], dim="init_time")
    inits = pd.DatetimeIndex(pd.to_datetime(ds.init_time.values))
    leads = ds.lead.values
    k = leads >= 1
    W = training_matrix(inits, C.BC_WINDOW_DAYS, years)
    obs_t = pd.DatetimeIndex(obs_ds.time.values)
    mjj = inits.month.isin(C.ONSET_INIT_MONTHS)
    out = ds.copy(deep=True)
    for name in ds.data_vars:
        ovar, shift = U.index_obs_spec(base, name)
        series = pd.Series(obs_ds[ovar].values, index=obs_t)
        vd = inits.values[:, None] + (leads[k] + shift).astype("timedelta64[D]")[None, :]
        O = series.reindex(vd.ravel()).values.reshape(len(inits), k.sum())
        F = ds[name].values[:, :, k]
        b = drift_bias(F.mean(1), O, W, C.BC_LEAD_SMOOTH)               # [S, L]
        Fc = F - np.nan_to_num(b)[:, None]
        out[name].values[:, :, k] = Fc
        for wk, (k1, k2) in C.WEEKS.items():
            sl = (leads[k] >= k1) & (leads[k] <= k2)
            out_rows.append(dict(model=base, target=name, week=wk,
                                 bias_removed=float(np.nanmean(b[mjj][:, sl])),
                                 raw_bias=float(np.nanmean((F.mean(1) - O)[mjj][:, sl])),
                                 corrected_bias=float(np.nanmean((Fc.mean(1) - O)[mjj][:, sl]))))
    out = out.assign_coords(init_time=(inits - U.init_shift(base)).values).rename(lead="day")
    out.attrs.update(bias_correction=f"LOYO mean drift vs ERA5 ({U.index_obs_spec(base, 'wy')[0]} sampling), "
                                     f"+-{C.BC_WINDOW_DAYS} d, {C.BC_LEAD_SMOOTH}-d lead smoothing; lead 0 unchanged")
    d = U.ensure_dir(C.MODELS[f"{base}_bc"]["indices_dir"])
    t = pd.DatetimeIndex(out.init_time.values)
    for y in years:
        out.isel(init_time=np.where(t.year == y)[0]).to_netcdf(d / f"{y}.nc")
    print(f"  {base} indices corrected in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=C.BC_BASES, choices=C.BC_BASES)
    ap.add_argument("--skip", nargs="*", default=[], choices=["tp", "indices"])
    a = ap.parse_args()
    obs_ds = U.load_obs_indices() if "indices" not in a.skip else None
    rows = []
    for base in a.models:
        print(f"bias-correcting {base}", flush=True)
        if "tp" not in a.skip:
            correct_tp(base, rows)
        if "indices" not in a.skip:
            correct_indices(base, obs_ds, rows)
    df = pd.DataFrame(rows)
    out = U.ensure_dir(C.OUT_ROOT / "bias_correction")
    df.to_csv(out / f"bc_diagnostics_{'_'.join(a.models)}.csv", index=False)
    print(df.pivot_table(index=["target", "week"], columns="model",
                         values=["raw_bias", "corrected_bias"]).round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
