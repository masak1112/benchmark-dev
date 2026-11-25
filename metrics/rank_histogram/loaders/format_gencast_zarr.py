"""
GenCast zarr loader.

Directory layout:
    {input_dir}/gencast_{YYYY}.zarr        (one store per year)

Store structure (verified 2026-09-21):
    dims  : sample=52 (members), time=26 (init dates), step=90 (lead),
            lat=721, lon=1440, level=10
    vars  : 2m_temperature, temperature, u/v_component_of_wind,
            specific_humidity, sea_surface_temperature,
            total_precipitation_12hr
            -- NOTE: there is NO geopotential in these stores, so z500/z850
               cannot be evaluated from this source.
    level : [200 250 300 400 500 600 700 850 925 1000] in hPa
    step  : 12, 24, ... 1080 hours (12-hourly). Sub-sampled to 24-hourly here
            so the lead axis matches the daily models (0..1080h).
    time  : days since {YYYY}-05-02, i.e. init dates (May-Jul window).

Returned dataset dimensions:
    (time=init_dates, number=members, prediction_timedelta, lat, lon)
regridded to the ERA5 1.0 deg grid and renamed to var_cfg['era5_var'].
"""

from pathlib import Path

import numpy as np
import xarray as xr


# ERA5 target grid (matches {variable}/{year}_180x360.nc)
_ERA5_LAT = np.arange(-89.5, 90.0, 1.0)
_ERA5_LON = np.arange(0.0, 360.0, 1.0)


def _regrid_to_era5(da: xr.DataArray) -> xr.DataArray:
    """Bilinear interpolation onto the 180x360 ERA5 grid.

    GenCast is 0.25 deg (721x1440), finer than ERA5's 1.0 deg, so this is a
    downsampling interpolation. `lon` is periodic but both grids start at 0.0
    and the ERA5 grid stops at 359.0, so no wrap-around handling is needed.
    """
    # .interp() upcasts float32 -> float64, doubling peak memory on the
    # final concat (this caused repeated OOM kills). Cast back.
    return da.interp(lat=_ERA5_LAT, lon=_ERA5_LON,
                     method="linear").astype("float32")


def load_forecast(
    input_dir: str,
    years: list,
    var_name: str,
    var_cfg: dict,
    max_members: int | None,
) -> xr.Dataset:
    model_var = var_cfg["model_var"]
    era5_var  = var_cfg["era5_var"]
    level     = var_cfg.get("level")
    scale     = var_cfg.get("model_scale", 1.0)

    stores = [Path(input_dir) / f"gencast_{y}.zarr" for y in years]
    stores = [s for s in stores if s.exists()]
    if not stores:
        raise RuntimeError(f"No gencast_*.zarr stores for years {years} in {input_dir}.")
    print(f"  Found {len(stores)} GenCast zarr stores.")

    per_year = []
    for store in stores:
        # decode_times=False: the `time` units string is per-store
        # ("days since {year}-05-02"), decoded manually below.
        ds = xr.open_zarr(store, decode_times=False, consolidated=False)

        if model_var not in ds:
            raise KeyError(
                f"'{model_var}' not in {store.name}. Available: {list(ds.data_vars)}")

        da = ds[model_var]

        if level is not None:
            if "level" not in da.dims:
                raise KeyError(f"'{model_var}' in {store.name} has no level dim.")
            da = da.sel(level=float(level)).drop_vars("level", errors="ignore")

        # ── init-date axis ───────────────────────────────────────────────────
        units = ds["time"].attrs.get("units", "")
        if "since" not in units:
            raise ValueError(f"Unexpected time units in {store.name}: {units!r}")
        origin = np.datetime64(units.split("since")[-1].strip().replace(" ", "T"))
        offs   = ds["time"].values.astype("int64")
        # Guard against the malformed entries seen in some stores: a valid
        # offset is a small day count within the ~May-Jul init window.
        ok = (offs >= 0) & (offs < 400)
        if not ok.all():
            print(f"  {store.name}: dropping {int((~ok).sum())} init(s) with "
                  f"out-of-range time offsets (e.g. {offs[~ok][:1]}).")
        init_times = origin + offs[ok].astype("timedelta64[D]")

        da = da.isel(time=np.flatnonzero(ok))

        # ── lead axis: keep 24-hourly steps only ─────────────────────────────
        step_h = ds["step"].values.astype("int64")
        keep_step = np.flatnonzero(step_h % 24 == 0)
        da = da.isel(step=keep_step)
        lead = (step_h[keep_step]).astype("timedelta64[h]").astype("timedelta64[ns]")

        da = (
            da
            .rename({"sample": "number", "step": "prediction_timedelta"})
            .assign_coords(
                time=init_times,
                prediction_timedelta=lead,
                number=np.arange(da.sizes["sample"]),
            )
        )

        if max_members is not None:
            da = da.isel(number=slice(max_members))
            da = da.assign_coords(number=np.arange(da.sizes["number"]))

        da = _regrid_to_era5(da)
        per_year.append(da)

    fc = xr.concat(per_year, dim="time").sortby("time")
    if scale != 1.0:
        fc = fc * scale

    fc = fc.transpose("time", "number", "prediction_timedelta", "lat", "lon")
    return fc.to_dataset(name=era5_var)
