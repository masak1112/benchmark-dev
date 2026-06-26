import xarray as xr
import numpy as np
import datetime
import os
import glob
from concurrent.futures import ProcessPoolExecutor

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
        dask_gufunc_kwargs={"allow_rechunk": True},

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
        spread = f.std(dim=mdim, ddof=1, skipna=True)
    else:
        # with a single member, define spread=0 to avoid NaNs
        spread = xr.zeros_like(f.isel({mdim: 0}, drop=True))

    ens_mean = f.mean(dim=mdim, skipna=True)
    err2 = (ens_mean - t) ** 2
    # reduce over all non-lead, non-member dims (time, lat, lon, etc.)
    reduce_dims = [d for d in f.dims if d not in (lead_dim, mdim)]
    spread2_mean = (spread ** 2).mean(dim=reduce_dims, skipna=True) if reduce_dims else (spread ** 2)
    err2_mean = err2.mean(dim=reduce_dims, skipna=True) if reduce_dims else err2

    ssr = np.sqrt(spread2_mean / err2_mean)
    ssr.name = "ssr"

    out = xr.Dataset({"crps": crps, "ssr": ssr, "spread": np.sqrt(spread2_mean), "error": np.sqrt(err2_mean)})
    return out.compute() if compute else out


def preprocess(ds):
    ens = np.int32(ds.encoding["source"].split("/")[-1].split(".")[0].split("_")[-1])
    ds = ds.expand_dims(dim={"number": [ens]})
    ds = ds.rename({"time":"prediction_timedelta"})
    ds = ds.expand_dims(dim={"time": [ds["prediction_timedelta"][0].values]})
    ds["prediction_timedelta"] = np.arange(0, 24*len(ds["prediction_timedelta"]), 24).astype("timedelta64[h]").astype("timedelta64[ns]")
    ds["longitude"] = ds["longitude"].astype("float32")
    return ds


def get_ERA5(variable, level=None):
    era5 = xr.open_mfdataset(f"/scratch/10000/amarchakitus/ERA5/{variable}/*_180x360.nc", chunks={}, engine='h5netcdf').rename({"lat": "latitude", "lon": "longitude"})
    era5['latitude'] = era5['latitude'].astype("float32")
    era5['longitude'] = era5['longitude'].astype("float32")
    if level:
        era5 = era5.sel(level=level).drop_vars("level")
    return era5

def get_dates(target_year):
    base_year = 2024
    hour = 0
    base_dates = []
    for month in range(5, 8):
        if month in [5, 7]:
            max_day = 31
        else:
            max_day = 30
        for day in range(1, max_day + 1):
            date = datetime.datetime(base_year, month, day, hour, 0)
            if date.weekday() not in (0, 3):
                continue
            date_f = date.strftime("%Y%m%d%H")
            base_dates.append(date_f)
    dates_f_year = [str(target_year) + date[4:] for date in base_dates]
    dates_year = [datetime.datetime.strptime(date, "%Y%m%d%H") for date in dates_f_year]
    return dates_year

def get_file_list(in_dir, years, ens):
    file_list = []
    for year in years:
        dates = get_dates(year)
        for date in dates:
            date_str = date.strftime("%Y%m%d00")
            for e in ens:
                file_path = f"{in_dir}/pangu_plasim_2_24h_45step_{date_str}_ens_{e}.nc"
                file_list.append(file_path)
    return file_list

def open_file(file, variable, level=None):
    try:
        if level:
            ds = xr.open_dataset(file, chunks={}, engine='h5netcdf')[variable].sel(level=level).drop_vars("level")
        else:
            ds = xr.open_dataset(file, chunks={}, engine='h5netcdf')[variable]
        return preprocess(ds)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def evaluate_s2s_forecast(in_dir, out_dir, variable, level=None):
    if level:
        print(f"Evaluating variable: {variable} at level: {level}")
    else:
        print(f"Evaluating surface variable: {variable}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    era5 = get_ERA5(variable, level)

    years = range(2019, 2025)
    ens = range(0, 30)

    file_list = get_file_list(in_dir, years, ens)
    if len(file_list) == 0:
        print(f"No files found for variable: {variable} at level: {level}. Skipping evaluation.")
        return

    datasets = []
    with ProcessPoolExecutor() as executor:
        for ds in executor.map(open_file, file_list, [variable]*len(file_list), [level]*len(file_list)):
            if ds is not None:
                datasets.append(ds)
            else:
                print("A dataset was skipped due to an error.")

    s2s = xr.combine_by_coords(datasets)

    s2s_eval = s2s.chunk({'time':1, 'number':-1, 'prediction_timedelta':-1, 'latitude':-1, 'longitude':-1})

    res = evaluate_crps_fast(s2s_eval, era5, var=variable, compute=True)

    # Aggregate CRPS by lead (keep your existing pattern)
    by_lead_crps = res.crps.mean(dim=[d for d in res.crps.dims if d not in ("prediction_timedelta",)])

    # SSR is already aggregated by lead; just rename for symmetry
    by_lead_ssr = res.ssr

    for lead_td in by_lead_crps.prediction_timedelta.values:
        hrs = int(np.round(lead_td / np.timedelta64(1, "h")))
        crps_val = float(by_lead_crps.sel(prediction_timedelta=lead_td).values)
        ssr_val  = float(by_lead_ssr.sel(prediction_timedelta=lead_td).values)
        print(f"Step {hrs:>4} hours: CRPS={crps_val:.4f} | SSR={ssr_val:.3f}")


    by_lead_crps_ds = by_lead_crps.to_dataset(name="crps")
    by_lead_ssr_ds = by_lead_ssr.to_dataset(name="ssr")
    merged = xr.merge([by_lead_crps_ds, by_lead_ssr_ds])

    if level:
        merged.to_netcdf(f"{out_dir}/S2S_ens_eval_{variable}_{level}.nc")
    else:
        merged.to_netcdf(f"{out_dir}/S2S_ens_eval_{variable}.nc")

def main():

    eval_dict = {
        "2m_temperature": [None],
        "sea_surface_temperature": [None],
        "total_precipitation_24hr": [None],
        "geopotential": [500],
        }

    model_dict = {"v2_std01":
                    {
                      "in_path": "/scratch/10786/bgong1/PanguWeather/v2.0/HPC_scripts/results/S2S/2/predictions_v2_std01",
                      "out_path": "/scratch/10000/amarchakitus/s2s/v2_std01_results",
                      "eval_dict": eval_dict
                    },
                "v2_std1":
                    {
                       "in_path": "/scratch/10786/bgong1/PanguWeather/v2.0/HPC_scripts/results/S2S/2/predictions_v2_std1",
                       "out_path": "/scratch/10000/amarchakitus/s2s/v2_std1_results",
                       "eval_dict": eval_dict
                    }
                }
    
    for model_name, meta in model_dict.items():
        print(f"Evaluating model: {model_name}")
        in_dir = meta["in_path"]
        out_dir = meta["out_path"]
        model_eval_dict = meta["eval_dict"]

        for variable, levels in model_eval_dict.items():
            for level in levels:
                print(f"Processing variable: {variable}, level: {level}") if level else print(f"Processing variable: {variable}")
                evaluate_s2s_forecast(in_dir, out_dir, variable, level)

if __name__ == "__main__":
    main()