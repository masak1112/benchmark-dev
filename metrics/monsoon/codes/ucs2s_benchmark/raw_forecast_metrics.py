#!/usr/bin/env python
"""
Monsoon-relevant verification of the RAW forecasts (beyond onset dates).

For one model and one init selection it writes, to
  <OUT_ROOT>/raw_metrics/<init_mode>_<months>/<model>/ :

  scores.csv            tidy table: target x aggregation (daily lead / week) x metric
                        targets  rain_cmz (IMD, 2-deg CMZ mean)
                                 wy, imi, llj, mhi, olr_cmz (ERA5; NGCM: wy only)
                        metrics  mean_f, mean_o, bias, ratio, acc (+90% CI),
                                 rmse, spread_ratio, crps, crps_clim, crpss (+90% CI)
  rain_maps_weekly.nc   per-cell weekly-mean rainfall bias (%) and ACC, weeks 1-6
  rain_daily_stats.csv  CMZ wet-day (>1 mm) frequency, onset-criterion exceedance
                        frequency (day1 > 1 mm & 5-day sum > MWmean threshold,
                        i.e. the exact onset test) and daily-rain quantiles vs lead
  lead_alignment.csv    corr(CMZ forecast step k, IMD[init+k+off]), off=-1,0,1
                        (step k must peak at off=0 if the lead handling is right)
  mok_onset.csv         (UC-S2S, native) circulation onset over Kerala from the LLJ
                        index per init; mok_obs.csv = ERA5 LLJ onset vs IMD MOK
  samples_*.nc          the CMZ / index samples behind the scores (for re-use)

Conventions
  * ACC = Pearson correlation across inits of ensemble-mean anomalies,
    anomalies w.r.t. the observed 1991-2018 day-of-year climatology (mean +
    3 harmonics), separately for every lead / week.
  * CRPS is the fair CRPS (Ferro 2014).  The reference is a climatological
    ensemble: the observed value for the same calendar window in each year
    1991-2018 (28 members, all before the verification years).
  * spread_ratio = sqrt((M+1)/M) * mean ensemble SD / RMSE(ens mean); 1 = calibrated.
  * 90% CIs from a year-block bootstrap (whole years resampled), which is the
    honest unit given only six verification years.

Usage:
  python raw_forecast_metrics.py --model ucs2s_plain --init-mode native --months 5 6 7
  python raw_forecast_metrics.py --model ngcm --init-mode common
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xarray as xr

import config as C
import common as U

N_BOOT = 1000
QUANT_LEADS = [1, 3, 5, 10, 15, 20, 30, 40]


# ============================================================================
# Helpers
# ============================================================================
def select_dates(model_key, mode, years, months):
    out = {}
    for y in years:
        d = (U.model_init_dates(model_key, y, months) if mode == "native"
             else U.common_init_dates(C.COMPARISON_SET, y, months))
        if len(d):
            out[y] = d
    return out


def valid_dates(inits, leads):
    return inits.values[:, None] + np.asarray(leads).astype("timedelta64[D]")[None, :]


def shift_to_year(dates, year):
    d = pd.DatetimeIndex(dates)
    day = np.where((d.month == 2) & (d.day == 29), 28, d.day)
    return pd.to_datetime(dict(year=np.full(len(d), year), month=d.month, day=day))


def clim_ensemble(series, start_dates, nd, clim_years):
    """Observed mean over [start, start+nd-1] shifted to each climatology year -> (N, n_years)."""
    fwd = series.rolling(nd).mean().shift(-(nd - 1))
    out = np.empty((len(start_dates), len(clim_years)))
    for j, y in enumerate(clim_years):
        out[:, j] = fwd.reindex(shift_to_year(start_dates, y)).values
    return out


def clim_value(clim365, dates):
    """Smoothed climatology (365,) evaluated at dates (any shape)."""
    d = pd.DatetimeIndex(np.asarray(dates).ravel())
    return clim365[U.noleap_doy(d) - 1].reshape(np.shape(dates))


def score(f_ens, o, c_mean, c_ens, years):
    """All scalar scores for one (target, lead) block.  f_ens (N, M); o, c_mean (N,); c_ens (N, Y)."""
    ok = np.isfinite(o) & np.isfinite(f_ens).all(1) & np.isfinite(c_mean)
    if c_ens is not None:
        ok &= np.isfinite(c_ens).all(1)
    f_ens, o, c_mean, years = f_ens[ok], o[ok], c_mean[ok], np.asarray(years)[ok]
    if c_ens is not None:
        c_ens = c_ens[ok]
    n, m = f_ens.shape
    out = dict(n=int(n), n_years=int(len(np.unique(years))), members=int(m))
    if n < 3:
        return out
    fm = f_ens.mean(1)
    fa, oa = fm - c_mean, o - c_mean
    rmse = np.sqrt(np.mean((fm - o) ** 2))
    spread = np.sqrt(np.mean(f_ens.var(1, ddof=1)))
    out.update(mean_f=fm.mean(), mean_o=o.mean(), bias=(fm - o).mean(),
               ratio=fm.mean() / o.mean() if o.mean() != 0 else np.nan,
               acc=U.pearson(fa, oa), rmse=rmse, spread=spread,
               spread_ratio=np.sqrt((m + 1) / m) * spread / rmse if rmse > 0 else np.nan)
    out["acc_lo"], out["acc_hi"] = U.year_block_bootstrap(
        lambda i: U.pearson(fa[i], oa[i]), years, N_BOOT)
    if c_ens is not None:
        cf = U.fair_crps(f_ens, o, axis=1)
        cc = U.fair_crps(c_ens, o, axis=1)
        out.update(crps=cf.mean(), crps_clim=cc.mean(), crpss=1 - cf.mean() / cc.mean())
        out["crpss_lo"], out["crpss_hi"] = U.year_block_bootstrap(
            lambda i: 1 - cf[i].mean() / cc[i].mean(), years, N_BOOT)
    return out


def score_target(name, F, O, inits, leads, series, clim365, years_of_init, rows, weekly=True):
    """F (N, M, L), O (N, L) at leads `leads`; series = full observed daily pd.Series."""
    lead_pos = {int(l): i for i, l in enumerate(leads)}
    for l in leads:
        if l < 1:
            continue
        i = lead_pos[int(l)]
        vd = valid_dates(inits, [l])[:, 0]
        c_ens = clim_ensemble(series, vd, 1, C.CLIM_YEARS)
        r = score(F[:, :, i], O[:, i], clim_value(clim365, vd), c_ens, years_of_init)
        rows.append(dict(target=name, window="daily", lead=int(l), lead_label=f"d{int(l)}", **r))
    if not weekly:
        return
    for wk, (k1, k2) in C.WEEKS.items():
        if k1 not in lead_pos or k2 not in lead_pos:
            continue
        sl = slice(lead_pos[k1], lead_pos[k2] + 1)
        vd = valid_dates(inits, np.arange(k1, k2 + 1))
        c_mean = clim_value(clim365, vd).mean(1)
        c_ens = clim_ensemble(series, vd[:, 0], k2 - k1 + 1, C.CLIM_YEARS)
        r = score(F[:, :, sl].mean(2), O[:, sl].mean(1), c_mean, c_ens, years_of_init)
        rows.append(dict(target=name, window="weekly", lead=int(wk[2:]), lead_label=f"{k1}-{k2}", **r))


def corr_along0(a, b):
    a = a - np.nanmean(a, 0)
    b = b - np.nanmean(b, 0)
    return np.nansum(a * b, 0) / np.sqrt(np.nansum(a * a, 0) * np.nansum(b * b, 0))


def first_persistent(x, thr, persist, allowed, mean_window=0):
    """First i with allowed[i], x[i:i+persist] > thr (all) and, if mean_window,
    mean(x[i:i+mean_window]) > thr (window must fit in the series); else -1."""
    above = x > thr
    n = len(x)
    for i in range(n - max(persist, mean_window) + 1):
        if allowed[i] and above[i:i + persist].all() and \
                (mean_window == 0 or np.mean(x[i:i + mean_window]) > thr):
            return i
    return -1


# ============================================================================
# Rainfall vs IMD
# ============================================================================
def rain_metrics(model_key, dates, out_dir, rows):
    t0 = time.time()
    years = sorted(dates)
    parts, inits = [], []
    for y in years:
        da = U.load_model_tp(model_key, y, init_dates=dates[y], max_step=45)
        parts.append(da.values)
        inits += list(pd.to_datetime(da.init_time.values))
        steps = da.step.values
    F = np.concatenate(parts)                                   # (N, M, L, lat, lon)
    inits = pd.DatetimeIndex(inits)
    yrs = inits.year.values
    N, M, L = F.shape[:3]

    last_imd = max(y for y in range(2019, 2031) if (C.IMD_DIR / f"data_{y}.nc").exists())
    obs = U.imd_series(min(years), min(max(years) + 1, last_imd))
    vd = valid_dates(inits, steps)
    O = obs.reindex(time=vd.ravel()).values.reshape(N, L, obs.sizes["lat"], obs.sizes["lon"])

    valid = U.imd_valid_mask(sorted(set(C.EVAL_YEARS) | set(C.CLIM_YEARS))).values
    cmz = U.cmz_mask().values & valid
    th = U.load_threshold().values
    print(f"  rain: N={N} inits, M={M}, L={L}; valid cells {valid.sum()}, CMZ cells {cmz.sum()}", flush=True)

    # --- CMZ mean series & climatology -------------------------------------
    clim_obs = U.imd_series(min(C.CLIM_YEARS), max(C.CLIM_YEARS))
    s_clim = pd.Series(clim_obs.values[:, cmz].mean(-1), index=pd.DatetimeIndex(clim_obs.time.values))
    s_eval = pd.Series(obs.values[:, cmz].mean(-1), index=pd.DatetimeIndex(obs.time.values))
    series = pd.concat([s_clim, s_eval])
    series = series[~series.index.duplicated()].sort_index()
    clim365 = U.harmonic_smooth_clim(np.array([s_clim[U.noleap_doy(s_clim.index) == d].mean()
                                               for d in range(1, 366)]))
    F_cmz = F[..., cmz].mean(-1)                                 # (N, M, L)
    O_cmz = O[..., cmz].mean(-1)                                 # (N, L)
    score_target("rain_cmz", F_cmz, O_cmz, inits, steps, series, clim365, yrs, rows)

    # --- lead alignment check ------------------------------------------------
    al = []
    for k in (1, 2, 3):
        for off in (-1, 0, 1):
            o_sh = s_eval.reindex(valid_dates(inits, [k + off])[:, 0]).values
            al.append(dict(step=k, offset=off, corr=U.pearson(F_cmz[:, :, k - 1].mean(1), o_sh)))
    pd.DataFrame(al).to_csv(out_dir / "lead_alignment.csv", index=False)

    # --- weekly maps ---------------------------------------------------------
    cell_clim = U.daily_climatology(clim_obs, C.CLIM_YEARS)      # (365, lat, lon)
    bias = np.full((len(C.WEEKS),) + valid.shape, np.nan)
    acc = np.full_like(bias, np.nan)
    for w, (k1, k2) in enumerate(C.WEEKS.values()):
        sl = slice(k1 - 1, k2)
        fw = F[:, :, sl].mean((1, 2))
        ow = O[:, sl].mean(1)
        doy = U.noleap_doy(pd.DatetimeIndex(vd[:, sl].ravel())).reshape(N, -1)
        cw = cell_clim[doy - 1].mean(1)                           # (N, lat, lon)
        with np.errstate(all="ignore"):
            mo = np.nanmean(ow, 0)
            ok = valid & (mo > 0.5)          # skip cells with < 0.5 mm/day observed (desert, early May)
            bias[w] = np.where(ok, (np.nanmean(fw, 0) - mo) / mo * 100, np.nan)
            acc[w] = np.where(valid, corr_along0(fw - cw, ow - cw), np.nan)
    xr.Dataset(
        dict(bias_pct=(("week", "lat", "lon"), bias), acc=(("week", "lat", "lon"), acc)),
        coords=dict(week=np.arange(1, len(C.WEEKS) + 1), lat=obs.lat.values, lon=obs.lon.values),
        attrs=dict(model=C.MODELS[model_key]["label"], n_inits=N, members=M,
                   weeks=json.dumps(C.WEEKS))).to_netcdf(out_dir / "rain_maps_weekly.nc")

    # --- daily-rain statistics in CMZ cells -----------------------------------
    Fc, Oc, thc = F[..., cmz], O[..., cmz], th[cmz]               # (N,M,L,c), (N,L,c), (c,)
    stats = []
    for k in range(L):
        r = dict(lead=int(steps[k]))
        r["wet_freq_f"] = np.mean(Fc[:, :, k] > 1)
        ok = np.isfinite(Oc[:, k])
        r["wet_freq_o"] = np.mean(Oc[:, k][ok] > 1)
        if k + 5 <= L:
            s5f = Fc[:, :, k:k + 5].sum(2)
            s5o = Oc[:, k:k + 5].sum(1)
            r["onset_crit_freq_f"] = np.mean((Fc[:, :, k] > 1) & (s5f > thc))
            ok5 = np.isfinite(s5o)
            r["onset_crit_freq_o"] = np.mean(((Oc[:, k] > 1) & (s5o > thc))[ok5])
        if int(steps[k]) in QUANT_LEADS:
            fv, ov = Fc[:, :, k].ravel(), Oc[:, k][ok].ravel()
            for q in (50, 90, 99):
                r[f"q{q}_f"], r[f"q{q}_o"] = np.percentile(fv, q), np.percentile(ov, q)
        stats.append(r)
    pd.DataFrame(stats).to_csv(out_dir / "rain_daily_stats.csv", index=False)

    xr.Dataset(dict(f_cmz=(("init_time", "member", "step"), F_cmz.astype("float32")),
                    o_cmz=(("init_time", "step"), O_cmz.astype("float32"))),
               coords=dict(init_time=inits.values, member=np.arange(M), step=steps)
               ).to_netcdf(out_dir / "samples_rain_cmz.nc")
    print(f"  rain done in {time.time() - t0:.0f}s", flush=True)


# ============================================================================
# Circulation / OLR indices vs ERA5
# ============================================================================
def index_metrics(model_key, dates, out_dir, rows, obs_ds):
    t0 = time.time()
    m = C.MODELS[model_key]
    parts = [U.load_model_indices(model_key, y, init_dates=d) for y, d in sorted(dates.items())]
    ds = xr.concat(parts, dim="init_time")
    inits = pd.DatetimeIndex(pd.to_datetime(ds.init_time.values))
    yrs = inits.year.values
    leads = ds.lead.values
    names = [k for k in ("wy", "imi", "llj", "mhi", "olr_cmz") if k in ds.data_vars]
    obs_t = pd.DatetimeIndex(obs_ds.time.values)
    clim_sel = obs_t.year.isin(C.CLIM_YEARS)
    lead0 = {}
    for name in names:
        ovar, shift = U.index_obs_spec(model_key, name)         # ERA5 sampling matching the model
        series = pd.Series(obs_ds[ovar].values, index=obs_t)
        clim365 = U.harmonic_smooth_clim(np.array(
            [series[clim_sel][U.noleap_doy(obs_t[clim_sel]) == d].mean() for d in range(1, 366)]))
        F = ds[name].values                                       # (N, M, L)
        O = series.reindex(valid_dates(inits, leads + shift).ravel()).values.reshape(len(inits), len(leads))
        if shift:   # score_target evaluates climatologies at init + lead -> pass a shifted series instead
            series = series.shift(-shift, freq="D")
            clim365 = np.roll(clim365, -shift)       # value at label d = clim(d + shift)
        keep = leads >= 1
        score_target(name, F[:, :, keep], O[:, keep], inits, leads[keep], series, clim365, yrs, rows)
        if 0 in leads and not shift and np.isfinite(F[:, :, leads == 0]).any():
            i0 = int(np.where(leads == 0)[0][0])
            lead0[name] = float(np.sqrt(np.nanmean((F[:, :, i0].mean(1) - O[:, i0]) ** 2)))
    if lead0:
        with open(out_dir / "index_lead0_rmse.json", "w") as fh:
            json.dump(lead0, fh, indent=1)
    ds.to_netcdf(out_dir / "samples_indices.nc")
    print(f"  indices done in {time.time() - t0:.0f}s (lead-0 RMSE vs ERA5: {lead0})", flush=True)


# ============================================================================
# Circulation-based monsoon onset over Kerala (LLJ index)
# ============================================================================
def mok_metrics(model_key, out_dir, obs_ds, years):
    for variant, par in C.LLJ_ONSET_VARIANTS.items():
        mok_metrics_variant(model_key, out_dir, obs_ds, years, variant, **par)


def mok_metrics_variant(model_key, out_dir, obs_ds, years, variant, thr, persist, mean_window):
    per, mw = persist, mean_window
    obs_t = pd.DatetimeIndex(obs_ds.time.values)
    ovar, shift = U.index_obs_spec(model_key, "llj")             # FuXi: calendar-day means, shift -1
    llj = pd.Series(obs_ds[ovar].values, index=obs_t)
    obs_rows, obs_onset = [], {}
    for y in sorted(set(obs_t.year)):
        s = llj[(llj.index >= f"{y}-04-01") & (llj.index <= f"{y}-09-30")]
        i = first_persistent(s.values, thr, per, s.index >= pd.Timestamp(f"{y}-05-01"), mw)
        d = s.index[i] if i >= 0 else pd.NaT
        obs_onset[y] = d
        if y in C.IMD_OFFICIAL_MOK:
            imd = pd.Timestamp(C.IMD_OFFICIAL_MOK[y])
            obs_rows.append(dict(year=y, era5_llj_onset=d.date() if i >= 0 else None,
                                 imd_mok=imd.date(), diff_days=(d - imd).days if i >= 0 else None))
    pd.DataFrame(obs_rows).to_csv(out_dir / f"mok_obs_{variant}.csv", index=False)
    # climatological reference forecast: median ERA5 onset (noleap day of year), 1991-2018
    clim_on = pd.DatetimeIndex([d for y, d in obs_onset.items() if y in C.CLIM_YEARS and not pd.isna(d)])
    clim_md = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(np.median(U.noleap_doy(clim_on))) - 1)

    rows = []
    for y in years:
        od = obs_onset.get(y, pd.NaT)
        if pd.isna(od):
            continue
        dts = U.model_init_dates(model_key, y, months=(4, 5, 6))
        if not len(dts):
            continue
        ds = U.load_model_indices(model_key, y, init_dates=dts)
        eff = pd.DatetimeIndex(ds.init_time.values)
        ds = ds.isel(init_time=np.where(eff < od)[0])              # effective init before onset
        for ti, init in enumerate(pd.DatetimeIndex(ds.init_time.values)):
            x = ds["llj"].isel(init_time=ti).values                # (M, L) lead 0..45
            lead = ds.lead.values + shift                            # days from init to the verifying date
            k = ds.lead.values >= 1
            vdates = init + pd.to_timedelta(lead[k], "D")
            allowed = vdates >= pd.Timestamp(f"{y}-05-01")
            mem_days = []
            for mm in range(x.shape[0]):
                i = first_persistent(x[mm, k], thr, per, allowed, mw)
                mem_days.append(int(lead[k][i]) if i >= 0 else np.nan)
            mem_days = np.array(mem_days, float)
            frac = np.isfinite(mem_days).mean()
            ens = int(round(np.nanmean(mem_days))) if frac >= 0.5 else np.nan
            obs_day = (od - init).days
            ens_date = init + pd.Timedelta(days=ens) if np.isfinite(ens) else pd.NaT
            clim_date = shift_to_year([clim_md], y)[0]   # climatological (median) onset
            rows.append(dict(
                year=y, init=init.date(), obs_onset=od.date(), lead_to_onset=obs_day,
                frac_members_onset=frac, ens_onset=ens_date.date() if np.isfinite(ens) else None,
                error_days=(ens_date - od).days if np.isfinite(ens) else np.nan,
                member_mae=(np.nanmean(np.abs(mem_days - obs_day)) if np.isfinite(mem_days).any() else np.nan),
                frac_within3=np.mean(np.abs(mem_days - obs_day) <= 3),
                clim_error_days=(clim_date - od).days))
    df = pd.DataFrame(rows)
    df["max_detectable_lead"] = int(45 - max(per, mw) + 1)
    df.to_csv(out_dir / f"mok_onset_{variant}.csv", index=False)
    print(f"  MOK [{variant}] (LLJ > {thr} m/s for {per} d, {mw}-d mean): {len(df)} inits; ERA5 vs IMD MOK:\n"
          f"{pd.DataFrame(obs_rows).to_string(index=False)}", flush=True)


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--init-mode", default="native", choices=["native", "common"])
    ap.add_argument("--months", nargs="+", type=int, default=list(C.ONSET_INIT_MONTHS))
    ap.add_argument("--years", nargs="+", type=int, default=C.EVAL_YEARS)
    ap.add_argument("--skip", nargs="*", default=[], choices=["rain", "indices", "mok"])
    a = ap.parse_args()

    tag = f"{a.init_mode}_m{''.join(str(x) for x in a.months)}"
    out_dir = U.ensure_dir(C.OUT_ROOT / "raw_metrics" / tag / a.model)
    dates = select_dates(a.model, a.init_mode, a.years, a.months)
    print(f"{a.model} [{tag}] inits/yr {{{', '.join(f'{y}: {len(d)}' for y, d in dates.items())}}}", flush=True)
    with open(out_dir / "run_info.json", "w") as fh:
        json.dump(dict(model=a.model, label=C.MODELS[a.model]["label"], init_mode=a.init_mode,
                       months=a.months, mem_num=C.MEM_NUM, clim_years=[min(C.CLIM_YEARS), max(C.CLIM_YEARS)],
                       init_dates={str(y): [x.strftime("%Y-%m-%d") for x in d] for y, d in dates.items()}),
                  fh, indent=1)

    rows = []
    if "rain" not in a.skip:
        rain_metrics(a.model, dates, out_dir, rows)
    m = C.MODELS[a.model]
    has_idx = "indices_dir" in m or "wy_dir" in m
    obs_ds = U.load_obs_indices() if (has_idx and ("indices" not in a.skip or "mok" not in a.skip)) else None
    if has_idx and "indices" not in a.skip:
        index_metrics(a.model, dates, out_dir, rows, obs_ds)
    if "indices_dir" in m and "mok" not in a.skip and a.init_mode == "native":
        mok_metrics(a.model, out_dir, obs_ds, a.years)
    df = pd.DataFrame(rows)
    df.insert(0, "model", a.model)
    df.to_csv(out_dir / "scores.csv", index=False)
    wk = df[df["window"] == "weekly"].pivot(index="lead", columns="target", values="acc").round(2)
    print("weekly ACC:\n", wk.to_string(), flush=True)


if __name__ == "__main__":
    main()
