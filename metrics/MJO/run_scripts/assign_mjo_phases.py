"""
Assign Wheeler-Hendon MJO phase (1-8) and amplitude to every forecast
(init date x member x lead day) from step4 RMM output.

Phase convention (angle = atan2(RMM2, RMM1), degrees):
  1: [-180,-135)  2: [-135,-90)  3: [-90,-45)  4: [-45,0)
  5: [0,45)       6: [45,90)     7: [90,135)   8: [135,180)
Amplitude < 1 is weak/inactive. Active/weak for stratifying forecasts is based on
ERA5 (ground truth) amplitude at the init date; forecast-amplitude flag is kept separately.

Output per dataset:
  <root>/<dataset>/chi200_1p0deg/mjo_phases_<dataset>.nc   (init, lead, number)
  <root>/<dataset>/chi200_1p0deg/mjo_phases_<dataset>.csv  (long table)

Usage:
  python run_scripts/assign_mjo_phases.py ours_si_v3 gencast amip_s2s ...
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

ROOT = "/glade/campaign/univ/uchi0018/bing/mjo"


def rmm_phase(rmm1, rmm2):
    deg = np.degrees(np.arctan2(rmm2, rmm1))
    phase = np.floor((deg + 180.0) / 45.0).astype("float") + 1
    phase = np.where(phase > 8, 8, phase)  # deg == 180 edge case
    return xr.where(np.isfinite(rmm1) & np.isfinite(rmm2), phase, np.nan)


def rmm_amp(rmm1, rmm2):
    return np.sqrt(rmm1**2 + rmm2**2)


def process_file(path):
    ds = xr.open_dataset(path)
    # init date always from the filename; the first time step is not necessarily lead 0
    # (FuXi files start at lead day 1)
    init = pd.Timestamp(os.path.basename(path)[-11:-3])
    if ds.time.dtype.kind != "M":
        # some step4 files store time as raw lead index with no units
        # (e.g. ours_si_v2 Init_20190509); rebuild from the init date in the filename
        print(f"  warning: {os.path.basename(path)} has non-datetime time; using init from filename")
        ds = ds.assign_coords(time=init + pd.to_timedelta(np.arange(ds.sizes["time"]), "D"))
    lead = ((pd.to_datetime(ds.time.values) - init) / pd.Timedelta("1D")).astype(int).values

    out = xr.Dataset()
    for suffix, r1, r2 in [
        ("", ds.RMM1, ds.RMM2),                         # per member (time, number)
        ("_emean", ds.RMM1_emean, ds.RMM2_emean),       # ensemble mean
        ("_obs", ds.RMM1_obs, ds.RMM2_obs),             # verifying ERA5
    ]:
        out[f"RMM1{suffix}"] = r1
        out[f"RMM2{suffix}"] = r2
        out[f"amp{suffix}"] = rmm_amp(r1, r2)
        out[f"phase{suffix}"] = rmm_phase(r1, r2)

    valid = ds.time.values
    out = out.rename(time="lead").assign_coords(lead=lead)
    # ensemble size can differ between inits (e.g. NGCM 25 vs 51); index members so
    # concat pads missing ones with NaN
    out = out.assign_coords(number=np.arange(ds.sizes["number"]))
    out = out.expand_dims(init=[init])
    out = out.assign_coords(valid_time=(("init", "lead"), valid[None, :]))
    ds.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--root", default=ROOT)
    args = p.parse_args()

    for name in args.datasets:
        base = os.path.join(args.root, name, "chi200_1p0deg")
        files = sorted(glob.glob(os.path.join(base, "rmm_output", "MJO_chi200_Forecast_Init_*.nc")))
        if not files:
            print(f"[{name}] no RMM files, skipping")
            continue
        print(f"[{name}] {len(files)} init dates")

        parts = [process_file(f) for f in files]
        ds = xr.concat(parts, dim="init", join="outer")
        ds = ds.assign_coords(valid_time=ds.init + ds.lead * np.timedelta64(1, "D"))
        # ERA5 RMM on the init day, looked up by date from the obs stored in all files
        # (works when lead 0 is missing, e.g. FuXi; the first init may stay NaN)
        tab = pd.DataFrame({
            "t": ds.valid_time.values.ravel(),
            "r1": ds.RMM1_obs.values.ravel(), "r2": ds.RMM2_obs.values.ravel(),
        }).dropna().groupby("t").mean()
        r1_0 = tab.r1.reindex(ds.init.values).values
        r2_0 = tab.r2.reindex(ds.init.values).values
        ds["init_RMM1_obs"] = ("init", r1_0)
        ds["init_RMM2_obs"] = ("init", r2_0)
        ds["init_amp_obs"] = ("init", rmm_amp(r1_0, r2_0))
        ds["init_phase_obs"] = ("init", np.asarray(rmm_phase(r1_0, r2_0)))
        n_miss = int(np.isnan(r1_0).sum())
        if n_miss:
            print(f"  warning: no ERA5 RMM on init day for {n_miss} init(s): "
                  f"{list(pd.to_datetime(ds.init.values[np.isnan(r1_0)]).strftime('%Y%m%d'))}")
        if 0 not in ds.lead.values:
            print(f"  note: no lead-0 step; leads run {int(ds.lead.min())}..{int(ds.lead.max())}")
        ds.attrs["description"] = "Wheeler-Hendon MJO phase/amplitude per forecast; phase 1-8, amp<1 = weak"

        nc_out = os.path.join(base, f"mjo_phases_{name}.nc")
        ds.to_netcdf(nc_out)

        # long table: one row per (init, lead, member)
        df = ds[["RMM1", "RMM2", "amp", "phase"]].to_dataframe().reset_index()
        obs = ds[["amp_obs", "phase_obs", "amp_emean", "phase_emean",
                  "init_phase_obs", "init_amp_obs"]].to_dataframe().reset_index()
        obs = obs.drop(columns="valid_time", errors="ignore")
        df = df.merge(obs, on=["init", "lead"])
        df["is_active_fcst"] = df["amp"] >= 1.0            # member's own forecast amplitude
        df["is_active_obs"] = df["amp_obs"] >= 1.0         # ERA5 on the valid date
        df["init_active_obs"] = df["init_amp_obs"] >= 1.0  # ERA5 on the init date (use for stratifying)
        csv_out = os.path.join(base, f"mjo_phases_{name}.csv")
        df.to_csv(csv_out, index=False)

        # per-init summary: which (observed) MJO phase each forecast starts in
        init_df = pd.DataFrame({
            "init": pd.to_datetime(ds.init.values).strftime("%Y%m%d"),
            "RMM1_obs": ds.init_RMM1_obs.values,
            "RMM2_obs": ds.init_RMM2_obs.values,
            "amp_obs": ds.init_amp_obs.values,
            "phase_obs": ds.init_phase_obs.values,
            f"phase_emean_lead{int(ds.lead.min())}": ds.phase_emean.isel(lead=0).values,
        }).dropna(subset=["amp_obs"])
        init_df["phase_obs"] = init_df["phase_obs"].astype(int)
        init_df["is_active"] = init_df["amp_obs"] >= 1.0  # ground truth (ERA5) at init
        init_out = os.path.join(base, f"mjo_init_phases_{name}.csv")
        init_df.to_csv(init_out, index=False)
        print(f"  -> {nc_out}\n  -> {csv_out}\n  -> {init_out}")
        print(init_df.groupby(["phase_obs", "is_active"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
