"""
NeuralGCM (NGCM) zarr loader -- campaign "climatology" layout.

Directory layout:
    {input_dir}/{YYYY}/{YYYY-MM-DD}T00.zarr     (one store per init date)

Store structure (verified 2026-09-21):
    dims  : time=1 (the init), number=25 (members),
            prediction_timedelta_daily=46 (lead 0..45 days),
            longitude=128, latitude=64            <- lon BEFORE lat
            (also prediction_timedelta=180 for the 6-hourly variables)
    vars  : pressure level is BAKED INTO THE NAME, there is no level coord --
            geopotential_{200,500,850}_avg, temperature_{200,500,850}_avg,
            temperature_{925,1000}, specific_humidity_*, u/v_component_*,
            total_precipitation_24hr
            -- NOTE: there is NO 2m_temperature in these stores.
    grid  : 2.8125 deg Gaussian, latitude non-uniform (-87.86..87.86)
    years : 2019, 2020, 2021 only (67 inits/yr, ~6-day cadence)

Because the level is part of the variable name, config entries for this format
should give the full name in `model_var` (e.g. `geopotential_500_avg`) and omit
`level`, or give a plain `model_var`/`level` pair which is combined into
`{model_var}_{level:g}_avg` as a fallback.

The init date is parsed from the STORE DIRECTORY NAME, not the `time`
coordinate, because `time` is malformed in some stores.

Returned dataset dimensions:
    (time=init_dates, number=members, prediction_timedelta, lat, lon)
regridded to the ERA5 1.0 deg grid and renamed to var_cfg['era5_var'].
"""

from pathlib import Path

import numpy as np
import xarray as xr


_ERA5_LAT = np.arange(-89.5, 90.0, 1.0)
_ERA5_LON = np.arange(0.0, 360.0, 1.0)


def _regrid_to_era5(da: xr.DataArray) -> xr.DataArray:
    """Interpolate the 2.8125 deg Gaussian grid onto ERA5 1.0 deg.

    This is an upsampling interpolation (64x128 -> 180x360). Two fixes are
    needed to avoid spurious NaNs, which would otherwise shrink the verified
    domain relative to the other models:

    1. Longitude is periodic: the source stops at 357.1875 but the ERA5 grid
       needs values up to 359.0, so a wrapped 360.0 column is appended.
    2. The Gaussian latitude grid stops at +-87.86, inside ERA5's +-89.5, so
       linear interpolation cannot reach the 4 polar rows (2.2% of points).
       `fill_value=None` with bounded extrapolation makes those rows take the
       nearest source latitude instead of NaN.
    """
    if float(da.lon[0]) == 0.0:
        wrap = da.isel(lon=0).assign_coords(lon=360.0)
        da = xr.concat([da, wrap], dim="lon")
    # .interp() upcasts float32 -> float64, doubling peak memory. Cast back.
    out = da.interp(lat=_ERA5_LAT, lon=_ERA5_LON, method="linear",
                    kwargs={"fill_value": None}).astype("float32")
    # `fill_value=None` extrapolates linearly, which can overshoot at the pole;
    # clamp the polar rows to the nearest in-range source value instead.
    lat_min, lat_max = float(da.lat.min()), float(da.lat.max())
    edge_lo = out.sel(lat=_ERA5_LAT[_ERA5_LAT >= lat_min][0])
    edge_hi = out.sel(lat=_ERA5_LAT[_ERA5_LAT <= lat_max][-1])
    out = out.where(out.lat >= lat_min, edge_lo)
    out = out.where(out.lat <= lat_max, edge_hi)
    return out


def _resolve_var(ds: xr.Dataset, model_var: str, level) -> str:
    """Pick the level-baked variable name present in the store."""
    if model_var in ds:
        return model_var
    if level is not None:
        for cand in (f"{model_var}_{level:g}_avg", f"{model_var}_{level:g}"):
            if cand in ds:
                return cand
    raise KeyError(
        f"Cannot resolve '{model_var}' (level={level}) in store. "
        f"Available: {sorted(ds.data_vars)}")


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

    stores = []
    for y in years:
        ydir = Path(input_dir) / str(y)
        if ydir.is_dir():
            stores.extend(sorted(ydir.glob("*.zarr")))
    if not stores:
        raise RuntimeError(f"No NGCM zarr stores for years {years} in {input_dir}.")
    print(f"  Found {len(stores)} NGCM init-date stores.")

    per_init = []
    for store in stores:
        # Init date from the directory name (`time` is unreliable/corrupt).
        init = np.datetime64(store.name.split("T")[0])

        ds = xr.open_zarr(store, decode_times=False, consolidated=False)
        vname = _resolve_var(ds, model_var, level)
        da = ds[vname]

        if "prediction_timedelta_daily" not in da.dims:
            raise KeyError(
                f"{store.name}: '{vname}' lacks prediction_timedelta_daily "
                f"(dims {da.dims}); 6-hourly variables are not supported here.")

        lead_days = ds["prediction_timedelta_daily"].values.astype("int64")
        lead = (lead_days * 24).astype("timedelta64[h]").astype("timedelta64[ns]")

        da = (
            da
            .rename({"prediction_timedelta_daily": "prediction_timedelta",
                     "latitude": "lat", "longitude": "lon"})
            .assign_coords(prediction_timedelta=lead)
        )

        # Store carries time=1; replace it with the filename-derived init.
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        da = da.expand_dims(time=[init])

        da = da.assign_coords(number=np.arange(da.sizes["number"]))
        if max_members is not None:
            da = da.isel(number=slice(max_members))
            da = da.assign_coords(number=np.arange(da.sizes["number"]))

        # Some fields carry an extra degenerate dim (e.g. total_precipitation_24hr
        # has surface=1), which breaks the explicit transpose below.
        for d in list(da.dims):
            if d not in ("time", "number", "prediction_timedelta", "lat", "lon") \
                    and da.sizes[d] == 1:
                da = da.isel({d: 0}, drop=True)

        # Source is (..., lon, lat); put lat first for the interp + downstream.
        da = da.transpose("time", "number", "prediction_timedelta", "lat", "lon")
        da = _regrid_to_era5(da)
        per_init.append(da)

    fc = xr.concat(per_init, dim="time").sortby("time")
    if scale != 1.0:
        fc = fc * scale

    fc = fc.transpose("time", "number", "prediction_timedelta", "lat", "lon")
    return fc.to_dataset(name=era5_var)
