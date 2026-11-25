"""
loaders/fuxi_s2s_ext200.py — FuXi S2S with winds extrapolated from 250 to 200 hPa.

FuXi only outputs winds at 850/500/250 hPa, so the plain fuxi_s2s loader uses u250/v250
in place of 200 hPa. This loader extrapolates linearly in log-pressure from 500 and
250 hPa before computing chi and the U200 slot:

    x200 = x250 + W * (x250 - x500),   W = ln(250/200) / ln(500/250) ≈ 0.322

Validated on ERA5 2020 (15S-15N mean, vs true 200 hPa): chi mean-offset RMS 0.41 -> 0.15
(fraction of anomaly std), amplitude ratio 0.73 -> 1.02; u: offset 0.25 -> 0.07,
amplitude 0.83 -> 1.07; anomaly correlation essentially unchanged.

Everything else (archive extraction, chi computation, lon regridding, output layout)
is the fuxi_s2s loader; only the per-member reader is swapped.
"""

import glob
import os

import numpy as np
import xarray as xr

from loaders import fuxi_s2s as _base

W = np.log(250.0 / 200.0) / np.log(500.0 / 250.0)

list_dates = _base.list_dates


def _load_member_ext200(member_dir: str, init_dt: np.datetime64):
    """Same return signature as fuxi_s2s._load_member, but the first two arrays are
    u/v extrapolated to 200 hPa instead of u250/v250."""
    day_files = sorted(glob.glob(os.path.join(member_dir, "*.nc")))
    if not day_files:
        raise FileNotFoundError(f"No NC files in {member_dir}")

    u_days, v_days, u850_days, olr_days, t_list = [], [], [], [], []
    for df in day_files:
        ds = xr.open_dataset(df)
        data = ds["__xarray_dataarray_variable__"].squeeze(["time", "lead_time"])
        lead = int(ds.lead_time.values[0])
        t_list.append(init_dt + np.timedelta64(lead, "D"))

        u250, u500 = data.sel(channel="u250").values, data.sel(channel="u500").values
        v250, v500 = data.sel(channel="v250").values, data.sel(channel="v500").values
        u_days.append(u250 + W * (u250 - u500))
        v_days.append(v250 + W * (v250 - v500))
        u850_days.append(data.sel(channel="u850").values)
        olr_days.append(-data.sel(channel="ttr").values)   # FuXi stores negative OLR
        ds.close()

    times = np.array(t_list, dtype="datetime64[ns]")
    return (np.stack(u_days).astype(np.float32),
            np.stack(v_days).astype(np.float32),
            np.stack(u850_days).astype(np.float32),
            np.stack(olr_days).astype(np.float32),
            times)


def process_date(date_str, src_dir, out_dir, variable, resolution,
                 max_members=None, scratch_dir="/tmp"):
    orig = _base._load_member
    _base._load_member = _load_member_ext200
    try:
        out_file = _base.process_date(date_str, src_dir, out_dir, variable, resolution,
                                      max_members=max_members, scratch_dir=scratch_dir)
    finally:
        _base._load_member = orig
    # relabel the U200 slot / provenance (base writes "U250 (proxy U200)")
    with xr.open_dataset(out_file) as ds:
        ds = ds.load()
    if "extrap" not in ds.attrs.get("source", ""):
        ds["ua_200"].attrs["long_name"] = "U200 extrapolated (log-p) from U250/U500"
        ds.attrs["source"] += "  winds: 200 hPa log-p extrapolated from 250/500 hPa"
        ds.to_netcdf(out_file + ".tmp")
        os.replace(out_file + ".tmp", out_file)
    return out_file
