import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ---- helpers ----
def _detect_member_dim(da, prefer=None):
    if prefer and prefer in da.dims:
        return prefer
    for cand in ("number", "member", "ens", "realization", "ensemble"):
        if cand in da.dims:
            return cand
    raise ValueError("Could not detect ensemble member dim; pass member_dim.")

def _crps_skill_vec(targets, preds, member_dim):
    # E|X - y| over ensemble members
    return np.abs(preds - targets).mean(dim=member_dim, skipna=False)

def _sort_along_dim(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Sort values along `dim` using the data (not the coord labels)."""
    return xr.apply_ufunc(
        np.sort,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        vectorize=False,
        dask_gufunc_kwargs={"allow_rechunk": True}, # safe on larger tiles
    )
def _mean_pair_abs_diff_fast(preds: xr.DataArray, member_dim: str) -> xr.DataArray:
    """
    E|X - X'| over ordered pairs via order stats:
    (2 / [n(n-1)]) * sum_k (2k - n - 1) X_(k), with X_(k) sorted along `member_dim`.
    """
    n = preds.sizes[member_dim]
    if n < 2:
        return xr.zeros_like(preds.isel({member_dim: 0}, drop=True))

    # IMPORTANT: member axis must be one dask chunk for performance/correctness.
    preds_sorted = _sort_along_dim(preds, member_dim)

    w = xr.DataArray(2 * np.arange(1, n + 1) - n - 1, dims=(member_dim,))
    S = (preds_sorted * w).sum(dim=member_dim, skipna=False)
    return (2.0 / (n * (n - 1))) * S

def _crps_ensemble_vec(targets, preds, member_dim):
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    return _crps_skill_vec(targets, preds, member_dim) - 0.5 * _mean_pair_abs_diff_fast(preds, member_dim)

# ---- main: fully vectorized across all leads ----
def evaluate_crps_fast(
    forecast: xr.Dataset,
    analysis: xr.Dataset,
    var: str = "2m_temperature",
    *,
    time_dim: str = "time",
    lead_dim: str = "prediction_timedelta",
    member_dim: str | None = None,
    compute: bool = False,
):
    """
    Returns:
      Dataset with:
        - crps : DataArray, same dims as the input forecast minus member_dim
        - ssr  : DataArray, aggregated by lead only (variance-form spread–skill ratio)
    Notes:
      * 'ssr' is computed as sqrt(mean(spread^2) / mean(error^2)) where
        spread = ensemble std (ddof=1) and error = (ensemble_mean - target).
      * The mean() above is over all dims except 'lead_dim' (and of course excluding member_dim).
    """
    if var not in forecast or var not in analysis:
        raise KeyError(f"'{var}' must exist in both forecast and analysis.")

    f = forecast[var]
    mdim = _detect_member_dim(f, member_dim)

    # 1) Build valid_time and attach as a coord (fixed)
    valid_time = forecast[time_dim] + forecast[lead_dim]   # DataArray (time, lead)
    f = f.assign_coords(valid_time=valid_time)

    # 2) Vectorized selection of targets at those valid times
    t = analysis[var].sel({time_dim: f["valid_time"]})

    # 3) CRPS with O(n log n) spread term
    crps = _crps_ensemble_vec(t, f, mdim)
    crps.name = "crps"

    # 4) Spread–skill ratio (variance-form), aggregated by lead
    nmem = f.sizes[mdim]
    if nmem > 1:
        spread = f.std(dim=mdim, ddof=1, skipna=False)
    else:
        # with a single member, define spread=0 to avoid NaNs
        spread = xr.zeros_like(f.isel({mdim: 0}, drop=True))

    ens_mean = f.mean(dim=mdim, skipna=False)
    err2 = (ens_mean - t) ** 2
    # reduce over all non-lead, non-member dims (time, lat, lon, etc.)
    reduce_dims = [d for d in f.dims if d not in (lead_dim, mdim)]
    spread2_mean = (spread ** 2).mean(dim=reduce_dims, skipna=False) if reduce_dims else (spread ** 2)
    err2_mean = err2.mean(dim=reduce_dims, skipna=False) if reduce_dims else err2

    ssr = np.sqrt(spread2_mean / err2_mean)
    ssr.name = "ssr"
    out = xr.Dataset({"crps": crps, "ssr": ssr, "ens_mean": ens_mean})
    return out.compute() if compute else out

# ---- example usage ----
era5 = xr.open_zarr("/glade/derecho/scratch/yqsun/PANGU-S2S/ERA5/era5-benchX",chunks={}).rename({"lat": "latitude", "lon": "longitude", "plev":"level"})
# era5 = xr.open_dataset("/project/pedramh/bing/PanguWeather/v2.0/data/2018_180x360.nc", chunks={}).expand_dims(plev=1).rename({"lat": "latitude", "lon": "longitude", "plev":"level"})
era5['latitude'] = era5['latitude'].astype("float32")
era5['longitude'] = era5['longitude'].astype("float32")
era5['level'] = era5['level'].astype("int32")


def preprocess(ds):
    ens = np.int32(ds.encoding["source"].split("/")[-1].split(".")[0].split("_")[-1])
    ds = ds.expand_dims(dim={"number": [ens]})
    ds = ds.rename({"time":"prediction_timedelta"})
    ds = ds.expand_dims(dim={"time": [ds["prediction_timedelta"][0].values]})
    ds["prediction_timedelta"] = np.arange(0, 24*len(ds["prediction_timedelta"]), 24).astype("timedelta64[h]").astype("timedelta64[ns]")
    return ds

import glob
#files = [f"/glade/campaign/univ/uchi0014/bing/predictions/pangu_plasim_2_24h_45step_201903{day:02d}00_ens_{ens}.nc" for day in range(1, 10) for ens in range(0, 10)]
files = glob.glob("/glade/campaign/univ/uchi0014/bing/predictions/pangu_plasim_2_24h_45step_201905*.nc")
#files = [f"/glade/campaign/univ/uchi0014/bing/predictions/pangu_plasim_2_24h_45step_201903*.nc"]
s2s = xr.open_mfdataset(files, chunks={}, engine='netcdf4', preprocess=preprocess)
t2m = s2s[['2m_temperature']].chunk({'time':1, 'number':-1, 'prediction_timedelta':1, 'latitude':-1, 'longitude':-1}) 
ttp =  s2s[["total_precipitation_24hr"]].chunk({'time':1, 'number':-1, 'prediction_timedelta':1, 'latitude':-1, 'longitude':-1}) 
gpt = s2s[['geopotential']].chunk({'time':1, 'number':-1, 'prediction_timedelta':1, 'latitude':-1, 'longitude':-1})
sst = s2s[["sea_surface_temperature"]].chunk({'time':1, 'number':-1, 'prediction_timedelta':1, 'latitude':-1, 'longitude':-1})
print("gpt",gpt)
res = evaluate_crps_fast(gpt, era5, var="geopotential", compute=True)
print("res",res)
res2 = res.sel(level=500)
print("res2",res2)

# Aggregate CRPS by lead (keep your existing pattern)
by_lead_crps = res2.crps.mean(dim=[d for d in res2.crps.dims if d not in ("prediction_timedelta",)])
# SSR is already aggregated by lead; just rename for symmetry
by_lead_ssr = res.ssr

crps = []
ssr = []
ens_mean = []
hrss = []
for lead_td in by_lead_crps.prediction_timedelta.values:
    hrs = int(np.round(lead_td / np.timedelta64(1, "h")))
    hrss.append(hrs)
    crps_val = float(by_lead_crps.sel(prediction_timedelta=lead_td).values)
    ssr_val  = float(by_lead_ssr.sel(prediction_timedelta=lead_td).values)
    crps.append(crps_val)
    ssr.append(ssr_val)
    ens_mean.append(res2.ens_mean.sel(prediction_timedelta=lead_td).values)
    print(f"Step {hrs:>4} hours: CRPS={crps_val:.4f} | SSR={ssr_val:.3f}")
    
np.save("geopotential_500_crps.npy", np.array(crps))
np.save("geopotential_500_ssr.npy", np.array(ssr))
np.save("geopotential_500_ens_mean.npy", np.array(ens_mean))



