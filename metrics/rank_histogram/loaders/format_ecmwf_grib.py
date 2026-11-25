"""
ECMWF S2S reforecast loader (GRIB).

Layout (verified 2026-10-12):
    {input_dir}/ecmwf_{group}_pf_{YYYYMMDD}.grib   50 perturbed members
    {input_dir}/ecmwf_{group}_cf_{YYYYMMDD}.grib    1 control member

Groups and their GRIB short names:
    t2m          -> t2m
    pl500_z_u    -> gh (geopotential HEIGHT, gpm), u      @ 500 hPa
    pl850_z_t_q  -> gh, t, q                              @ 850 hPa
    tp           -> tp
    olr, uv200_850 (not used here)

Per-file structure: (number=50, step=46..47, latitude=121, longitude=240),
1.5 deg grid with latitude running NORTH->SOUTH, `step` as a timedelta from the
init date. Regridded here to the ERA5 1.0 deg (180x360) grid.

IMPORTANT unit conversion: ECMWF supplies geopotential HEIGHT in gpm, while
ERA5 stores GEOPOTENTIAL in m2 s-2. The config must therefore set
`model_scale: 9.80665` for any gh-derived variable (verified: the ERA5/ECMWF
ratio of the global 500 hPa mean is 9.788 ~ g).

Only the perturbed (pf) files are read; the control member is excluded so the
ensemble is homogeneous. With max_members=20 the first 20 perturbed members are
used, matching the other models in this comparison.

Returned dims: (time=init, number, prediction_timedelta, lat, lon)
"""

from pathlib import Path
import re
import warnings

import numpy as np
import xarray as xr

_ERA5_LAT = np.arange(-89.5, 90.0, 1.0)
_ERA5_LON = np.arange(0.0, 360.0, 1.0)

_DATE_RE = re.compile(r"_(\d{8})\.grib$")


def _regrid_to_era5(da: xr.DataArray) -> xr.DataArray:
    """Bilinear interpolation from the 1.5 deg grid onto ERA5 1.0 deg.

    Longitude is periodic (source stops at 358.5) so a wrapped 360.0 column is
    appended; .interp() upcasts float32->float64 so the result is cast back.
    """
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
    group    = var_cfg["grib_group"]      # e.g. "pl500_z_u"
    short    = var_cfg["model_var"]       # GRIB shortName, e.g. "gh"
    era5_var = var_cfg["era5_var"]
    scale    = var_cfg.get("model_scale", 1.0)

    files = sorted(
        p for p in Path(input_dir).glob(f"ecmwf_{group}_pf_*.grib")
        if (m := _DATE_RE.search(p.name)) and int(m.group(1)[:4]) in years
    )
    if not files:
        raise RuntimeError(
            f"No ecmwf_{group}_pf_*.grib for years {years} in {input_dir}.")
    print(f"  Found {len(files)} ECMWF perturbed files for group '{group}'.")

    acc = []
    for i, f in enumerate(files):
        init = np.datetime64(
            f"{f.name[-13:-9]}-{f.name[-9:-7]}-{f.name[-7:-5]}T00:00:00", "ns")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_dataset(f, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
            if short not in ds:
                print(f"  SKIP {f.name}: '{short}' absent "
                      f"(has {list(ds.data_vars)})")
                continue
            da = ds[short]
            # drop the degenerate pressure-level coord if present
            for c in ("isobaricInhPa", "surface", "heightAboveGround"):
                if c in da.coords and da.coords[c].size == 1:
                    da = da.drop_vars(c, errors="ignore")
            da = (da.rename({"step": "prediction_timedelta",
                             "latitude": "lat", "longitude": "lon"})
                    .drop_vars("time", errors="ignore")
                    .drop_vars("valid_time", errors="ignore")
                    .expand_dims(time=[init]))
            da = da.assign_coords(number=np.arange(da.sizes["number"]))
            if max_members is not None:
                da = da.isel(number=slice(max_members))
                da = da.assign_coords(number=np.arange(da.sizes["number"]))
            if float(da.lat.values[0]) > float(da.lat.values[-1]):
                da = da.isel(lat=slice(None, None, -1))
            da = da.transpose("time", "number",
                              "prediction_timedelta", "lat", "lon")
            da = _regrid_to_era5(da).load()
            if scale != 1.0:
                da = da * scale
            ds.close()
        except Exception as exc:
            print(f"  ERROR {f.name}: {type(exc).__name__}: {str(exc)[:90]}")
            continue

        if on_date is not None:
            # Streaming: hand this init date over and release it, so the full
            # cube is never held (see format_fuxi for the same pattern).
            on_date(da.to_dataset(name=era5_var))
            del da
        else:
            acc.append(da)

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(files)} init dates read", flush=True)

    if on_date is not None:
        return None
    if not acc:
        raise RuntimeError(f"No data loaded for '{var_name}' from {input_dir}.")
    return xr.concat(acc, dim="time").sortby("time").to_dataset(name=era5_var)
