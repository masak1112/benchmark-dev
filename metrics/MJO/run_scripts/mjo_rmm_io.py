"""
Shared loader for step4 RMM output (MJO_*_Forecast_Init_YYYYMMDD.nc), used by the
multi-model MJO figure scripts in run_scripts/mjo_figs/.

Fixes applied consistently for every dataset:
  * Lead alignment. Lead = valid date - init date (init from the filename). UC-S2S and
    SI-S2S files start at lead 0, FuXi / NeuralGCM at lead 1, so array position is not
    lead. Arrays returned here are on leads 1..max_lead (index 0 = lead day 1).
  * One RMM sign convention. Runs projected after the 2026-09-26 obs rewrite (NeuralGCM
    reruns, FuXi 200 hPa extrap., SI-S2S 20-member rerun) used a sign-flipped EOF set.
    For chi200 files, a file whose stored ERA5 RMM anti-correlates with the reference
    ERA5 RMM is multiplied by -1 (forecast and obs together).
  * Init-day ERA5 state from the reference table (ours_si_v3 + fuxi_s2s, original EOF
    convention), not from each file's first time step.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

R = "/glade/campaign/univ/uchi0018/bing/mjo"
MAX_LEAD = 45
OBS_RMM = "/glade/work/bgong/MJO_results/obs/chi200_1p0deg/MJO_obs_created.nc"
REF_SOURCES = ("ours_si_v3", "fuxi_s2s")   # original EOF convention; never add reruns

_REF = None
_AMP_FULL = None


def date_of(f):
    return re.search(r"\d{8}", os.path.basename(f)).group()


def get_dates(pattern, months=None):
    out = {date_of(f) for f in glob.glob(pattern)}
    return {d for d in out if months is None or int(d[4:6]) in months}


def era5_ref():
    """ERA5 chi200 RMM (r1, r2) by date in the original convention."""
    global _REF
    if _REF is None:
        recs = []
        for name in REF_SOURCES:
            for fp in glob.glob(f"{R}/{name}/chi200_1p0deg/rmm_output/MJO_chi200_Forecast_Init_????????.nc"):
                with xr.open_dataset(fp) as ds:
                    if ds.time.dtype.kind != "M":
                        continue
                    recs.append(pd.DataFrame({"t": pd.to_datetime(ds.time.values),
                                              "r1": ds.RMM1_obs.values, "r2": ds.RMM2_obs.values}))
        df = pd.concat(recs).dropna()
        spread = df.groupby("t")[["r1", "r2"]].agg(lambda x: x.max() - x.min()).max().max()
        if spread > 1e-3:
            raise RuntimeError(f"reference ERA5 RMM sources disagree ({spread:.3g})")
        _REF = df.groupby("t")[["r1", "r2"]].mean()
    return _REF


def era5_amp_full():
    """Amplitude for dates outside the reference table (sign-independent)."""
    global _AMP_FULL
    if _AMP_FULL is None:
        ref = era5_ref()
        amp = np.hypot(ref.r1, ref.r2)
        with xr.open_dataset(OBS_RMM) as o:
            full = pd.Series(np.hypot(o.RMM1_obs.values, o.RMM2_obs.values),
                             index=pd.to_datetime(o.time.values))
        _AMP_FULL = amp.combine_first(full)
    return _AMP_FULL


def load(pattern, date_filter=None, month_filter=None, year_filter=None,
         max_lead=MAX_LEAD, align=None):
    """Return dict with inits (list), r1e, r2e, r1o, r2o (n_init, max_lead) on leads
    1..max_lead, init_r1, init_r2, init_amp (ERA5 on the init day), n, max_lead.
    align: sign-align to the reference (default: True for chi200 files)."""
    files = sorted(glob.glob(pattern))
    if date_filter is not None:
        files = [f for f in files if date_of(f) in date_filter]
    if month_filter is not None:
        files = [f for f in files if int(date_of(f)[4:6]) in {int(m) for m in month_filter}]
    if year_filter is not None:
        files = [f for f in files if int(date_of(f)[:4]) in {int(y) for y in year_filter}]
    if align is None:
        align = "chi200" in pattern
    ref = era5_ref() if align else None

    inits, arrs, n_flip = [], {k: [] for k in ("r1e", "r2e", "r1o", "r2o")}, 0
    for fp in files:
        d = date_of(fp)
        init = pd.Timestamp(d)
        with xr.open_dataset(fp) as ds:
            if ds.time.dtype.kind == "M":
                t = pd.to_datetime(ds.time.values)
                lead = ((t - init) / pd.Timedelta("1D")).astype(int).values
            else:  # raw lead index, no units
                t = init + pd.to_timedelta(np.arange(ds.sizes["time"]), "D")
                lead = np.arange(ds.sizes["time"])
            src = dict(r1e=ds.RMM1_emean.values, r2e=ds.RMM2_emean.values,
                       r1o=ds.RMM1_obs.values, r2o=ds.RMM2_obs.values)
        if ref is not None:
            rr = ref.reindex(t)
            if np.nansum(src["r1o"] * rr.r1.values + src["r2o"] * rr.r2.values) < 0:
                src = {k: -v for k, v in src.items()}
                n_flip += 1
        keep = (lead >= 1) & (lead <= max_lead)
        for k, v in src.items():
            row = np.full(max_lead, np.nan)
            row[lead[keep] - 1] = v[keep]
            arrs[k].append(row)
        inits.append(d)

    out = {k: (np.stack(v) if v else np.empty((0, max_lead))) for k, v in arrs.items()}
    idx = pd.to_datetime(inits)
    r0 = era5_ref().reindex(idx)
    out.update(inits=inits, n=len(inits), max_lead=max_lead, n_flip=n_flip,
               init_r1=r0.r1.values, init_r2=r0.r2.values,
               init_amp=era5_amp_full().reindex(idx).values)
    return out
