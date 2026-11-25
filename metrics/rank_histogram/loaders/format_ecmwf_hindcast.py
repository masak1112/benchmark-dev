"""
ECMWF S2S *reforecast/hindcast* loader (GRIB).

Differs from format_ecmwf_grib (the real-time set) in three important ways:

1. Each file holds THREE init dates, not one: `time` carries the hindcast years
   (2019, 2020, 2021) for a given 2025 model cycle. Filenames look like
   `ecmwf_rf_{group}_pf_{YYYYMMDD}_h2019-2021.grib`, where YYYYMMDD is the 2025
   cycle date; the actual init dates come from the `time` coordinate.
2. Only TEN perturbed members (vs 50 real-time), so the histogram has 11 bins.
   RI is normalised by bin count so it stays comparable, but note it in captions.
3. Variable groups differ: t2m, pl500_z (geopotential only -- no u), tp,
   uv200_850. There is NO pl850_z_t_q group.

Unit handling:
  * `gh` is geopotential HEIGHT in gpm -> config sets model_scale 9.80665.
  * `tp` is ACCUMULATED precipitation from the init (kg m-2, monotonic in step),
    while ERA5 total_precipitation_24hr is a daily total -- so it is differenced
    along step here. Verified: the mean daily increment is 2.39 kg m-2 against
    ERA5's 2.54 (0.00254 m/day).

Returned dims: (time=init, number, prediction_timedelta, lat, lon) on the ERA5
1.0 deg grid.
"""

from pathlib import Path
import warnings

import numpy as np
import xarray as xr

_ERA5_LAT = np.arange(-89.5, 90.0, 1.0)
_ERA5_LON = np.arange(0.0, 360.0, 1.0)


def _regrid_to_era5(da: xr.DataArray) -> xr.DataArray:
    if float(da.lon[0]) == 0.0 and float(da.lon[-1]) < 360.0:
        wrap = da.isel(lon=0).assign_coords(lon=360.0)
        da = xr.concat([da, wrap], dim="lon")
    return da.interp(lat=_ERA5_LAT, lon=_ERA5_LON,
                     method="linear").astype("float32")


def load_forecast(
    input_dir: str,
    years: list,
    var_name: str,
    var_cfg: dict,
    max_members: int | None,
    on_date=None,
) -> xr.Dataset | None:
    group     = var_cfg["grib_group"]
    short     = var_cfg["model_var"]
    era5_var  = var_cfg["era5_var"]
    scale     = var_cfg.get("model_scale", 1.0)
    deaccum   = var_cfg.get("deaccumulate", False)

    files = sorted(Path(input_dir).glob(f"ecmwf_rf_{group}_pf_*.grib"))
    if not files:
        raise RuntimeError(
            f"No ecmwf_rf_{group}_pf_*.grib in {input_dir}.")
    print(f"  Found {len(files)} ECMWF hindcast files for group '{group}'.")

    acc, n_dates = [], 0
    for i, f in enumerate(files):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_dataset(f, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
            if short not in ds:
                print(f"  SKIP {f.name}: '{short}' absent")
                continue
            da = ds[short]
            for c in ("isobaricInhPa", "surface", "heightAboveGround"):
                if c in da.coords and da.coords[c].size == 1:
                    da = da.drop_vars(c, errors="ignore")
            da = da.rename({"step": "prediction_timedelta",
                            "latitude": "lat", "longitude": "lon"})
            da = da.drop_vars("valid_time", errors="ignore")

            if deaccum:
                # tp is accumulated from the init; convert to per-day totals.
                # diff() drops the first step, which is the step-0 zero field.
                da = da.diff("prediction_timedelta")

            if float(da.lat.values[0]) > float(da.lat.values[-1]):
                da = da.isel(lat=slice(None, None, -1))
            da = da.assign_coords(number=np.arange(da.sizes["number"]))
            if max_members is not None:
                da = da.isel(number=slice(max_members))
                da = da.assign_coords(number=np.arange(da.sizes["number"]))

            # one file = several hindcast years; emit each as its own init date
            init_times = np.atleast_1d(da["time"].values)
            for t in init_times:
                if int(str(t)[:4]) not in years:
                    continue
                one = (da.sel(time=t) if da["time"].size > 1 else da)
                one = one.drop_vars("time", errors="ignore").expand_dims(time=[t])
                one = one.transpose("time", "number",
                                    "prediction_timedelta", "lat", "lon")
                one = _regrid_to_era5(one).load()
                if scale != 1.0:
                    one = one * scale
                n_dates += 1
                if on_date is not None:
                    on_date(one.to_dataset(name=era5_var))
                    del one
                else:
                    acc.append(one)
            ds.close()
        except Exception as exc:
            print(f"  ERROR {f.name}: {type(exc).__name__}: {str(exc)[:90]}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(files)} files, {n_dates} init dates",
                  flush=True)

    if on_date is not None:
        return None
    if not acc:
        raise RuntimeError(f"No data loaded for '{var_name}' from {input_dir}.")
    return xr.concat(acc, dim="time").sortby("time").to_dataset(name=era5_var)
