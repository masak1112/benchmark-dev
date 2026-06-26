import os
import shutil
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# USER CONFIG
# ============================================================

BASE_DIR = "/glade/derecho/scratch/bgong/amip_s2s_ensembles"

GRID_FILE = "/glade/work/rajatm/benchmark-dev/metrics/monsoon/grid_2deg_india.txt"

VAR = "PRATEsfc_24h"

N_SEEDS = 10

TEMP_DIR = "./temp_regrid"

OUT_DIR = "./yearly_outputs"


# ============================================================
# CDO WRAPPER
# ============================================================

def run_cdo(infile, outfile):
    """
    Extract + regrid in one safe CDO call.
    """
    cmd = [
        "cdo",
        f"remapcon,{GRID_FILE}",
        f"-selname,{VAR}",
        infile,
        outfile,
    ]

    subprocess.run(cmd, check=True)


# ============================================================
# DATE GROUPING
# ============================================================

def get_year_map():
    years = defaultdict(list)

    for d in os.listdir(BASE_DIR):
        if len(d) == 8 and d.isdigit():
            years[d[:4]].append(d)

    for y in years:
        years[y] = sorted(years[y])

    return dict(years)


# ============================================================
# PROCESS ONE INIT DATE
# ============================================================

def process_init_date(init_date):

    all_members = []

    init_path = os.path.join(BASE_DIR, init_date)

    print(f"Processing {init_date}")

    for seed in range(N_SEEDS):

        infile = os.path.join(
            init_path,
            f"ensemble_{init_date}_seed{seed}.nc"
        )

        if not os.path.exists(infile):
            print(f"  Missing seed {seed}")
            continue

        tmp_extract = os.path.join(
            TEMP_DIR,
            f"{init_date}_seed{seed}_rain.nc"
        )

        tmp_regrid = os.path.join(
            TEMP_DIR,
            f"{init_date}_seed{seed}_rg.nc"
        )

        # ----------------------------------------------------
        # STEP 1: extract + FIX dimension order
        # ----------------------------------------------------

        ds = xr.open_dataset(infile)

        rain = ds[VAR].transpose(
            "time",
            "member",
            "lat",
            "lon"
        )

        rain.to_dataset(name=VAR).to_netcdf(tmp_extract)

        ds.close()

        # ----------------------------------------------------
        # STEP 2: regrid using CDO
        # ----------------------------------------------------

        run_cdo(tmp_extract, tmp_regrid)

        # ----------------------------------------------------
        # STEP 3: read regridded output
        # ----------------------------------------------------

        rg = xr.open_dataset(tmp_regrid)[VAR]

        all_members.append(rg)

    if len(all_members) == 0:
        return None

    # ========================================================
    # CONCAT seeds -> total member dimension (10 × 4 = 40)
    # ========================================================

    da = xr.concat(all_members, dim="member")

    da = da.assign_coords(
        member=np.arange(da.sizes["member"])
    )

    # ========================================================
    # convert time → day index
    # ========================================================

    da = da.rename({"time": "day"})

    da = da.assign_coords(
        day=np.arange(da.sizes["day"])
    )

    # ========================================================
    # add init_time
    # ========================================================

    da = da.expand_dims(
        init_time=[
            pd.to_datetime(init_date, format="%Y%m%d")
        ]
    )

    # final ordering
    da = da.transpose(
        "init_time",
        "member",
        "day",
        "lat",
        "lon"
    )

    return da


# ============================================================
# PROCESS ONE YEAR
# ============================================================

def process_year(year, dates):

    print("\n" + "=" * 60)
    print(f"YEAR {year}")
    print("=" * 60)

    yearly = []

    for d in dates:
        da = process_init_date(d)
        if da is not None:
            yearly.append(da)

    if len(yearly) == 0:
        print("No data for year")
        return

    # --------------------------------------------------------
    # combine init_time dimension
    # --------------------------------------------------------

    out = xr.concat(yearly, dim="init_time")

    out_ds = out.to_dataset(name=VAR)

    outfile = os.path.join(OUT_DIR, f"{year}.nc")

    print(f"Writing {outfile}")

    out_ds.to_netcdf(outfile)

    out_ds.close()


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    year_map = get_year_map()

    print("Available years:")
    print(sorted(year_map.keys()))

    choice = input("Enter years (comma separated) or ALL: ").strip()

    if choice.upper() == "ALL":
        years = sorted(year_map.keys())
    else:
        years = [y.strip() for y in choice.split(",") if y.strip() in year_map]

    for y in years:
        process_year(y, year_map[y])

    print("\nCleaning temporary files...")
    shutil.rmtree(TEMP_DIR)
    print("Done.")


if __name__ == "__main__":
    main()