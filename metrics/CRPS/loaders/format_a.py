"""
Format A loader: amip_s2s_ensembles / amip_s2s_train_scratch style.

Directory layout:
    {input_dir}/{YYYYMMDD}/ensemble_{YYYYMMDD}_seed{N}.nc

Each seed file has dims (member, time, lat, lon[, plev]) where `time`
contains the 46 valid-time steps starting from the init date.

Returned dataset dimensions:
    (time=init_dates, number=ensemble_members, prediction_timedelta, lat, lon)
Variable is renamed to var_cfg['era5_var'] for downstream compatibility.
"""

from pathlib import Path
import numpy as np
import xarray as xr


def _preprocess_seed_file(
    file_path: str,
    model_var: str,
    level,
    level_dim: str,
) -> xr.DataArray:
    """Load one seed file and reshape to (time=[init], prediction_timedelta, number, lat, lon)."""
    ds = xr.open_dataset(file_path, chunks={"member": -1, "time": 1})
    seed = int(Path(file_path).stem.split("seed")[-1])

    init_time = ds["time"].values[0]

    da = ds[model_var]
    if level is not None:
        da = da.sel({level_dim: float(level)}).drop_vars(level_dim, errors="ignore")

    valid_times = da["time"].values
    lead_ns = (valid_times - init_time).astype("timedelta64[ns]")

    n_mem = da.sizes["member"]
    new_ids = seed * n_mem + np.arange(n_mem)

    da = (
        da
        .assign_coords(member=new_ids, time=lead_ns)
        .rename({"time": "prediction_timedelta", "member": "number"})
        .expand_dims(time=[init_time])
    )
    return da


def load_forecast(
    input_dir: str,
    years: list,
    var_name: str,
    var_cfg: dict,
    max_members: int | None,
) -> xr.Dataset:
    """
    Load Format A forecast data.

    Returns xr.Dataset with variable named var_cfg['era5_var'] and dims
    (time, number, prediction_timedelta, lat, lon).
    """
    model_var  = var_cfg["model_var"]
    era5_var   = var_cfg["era5_var"]
    level      = var_cfg.get("level")
    level_dim  = var_cfg.get("level_dim", "plev")
    scale      = var_cfg.get("model_scale", 1.0)

    date_dirs = sorted(
        d for d in Path(input_dir).iterdir()
        if d.is_dir() and d.name.isdigit() and int(d.name[:4]) in years
    )
    print(f"  Found {len(date_dirs)} init-date directories.")

    all_date_das = []
    for date_dir in date_dirs:
        date_str   = date_dir.name
        seed_files = sorted(date_dir.glob(f"ensemble_{date_str}_seed*.nc"))
        if not seed_files:
            print(f"  WARNING: no seed files in {date_dir}, skipping.")
            continue

        seed_das = []
        for sf in seed_files:
            try:
                da = _preprocess_seed_file(str(sf), model_var, level, level_dim)
                seed_das.append(da)
            except Exception as exc:
                print(f"  ERROR loading {sf.name}: {exc}")

        if seed_das:
            date_da = xr.concat(seed_das, dim="number")
            if max_members is not None:
                date_da = date_da.isel(number=slice(max_members))
            # Reassign a plain per-date positional index. The seed-derived IDs
            # (seed * n_mem + i) are only unique within a date; dates with a
            # different number/order of seed files (e.g. a missing seed file)
            # end up with different ID subsets. Concatenating across `time`
            # with mismatched `number` coordinates triggers an outer join that
            # fills most cells with NaN, which poisons CRPS (computed with
            # skipna=False on the member dim).
            date_da = date_da.assign_coords(number=np.arange(date_da.sizes["number"]))
            all_date_das.append(date_da)

    if not all_date_das:
        raise RuntimeError(f"No data loaded for '{var_name}' from {input_dir}.")

    fc_da = xr.concat(all_date_das, dim="time")
    if scale != 1.0:
        fc_da = fc_da * scale

    return fc_da.to_dataset(name=era5_var)
