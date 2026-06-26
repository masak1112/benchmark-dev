import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import sys

# ============================================================
# USER CONFIG
# ============================================================

BASE_DIR = "/glade/derecho/scratch/bgong/amip_s2s_train_scratch"

GRID_FILE = "/glade/work/rajatm/benchmark-dev/metrics/monsoon/grid_2deg_india.txt"

VAR = "PRATEsfc_24h"

N_SEEDS = 13

TEMP_DIR = "./temp_regrid_parallel"

OUT_DIR = "/glade/work/rajatm/benchmark-dev/metrics/monsoon/data_files/regrid_out/amip_s2s_train_scratch"


# ============================================================
# CDO FUNCTION
# ============================================================

def run_cdo(infile, outfile):

    cmd = [
        "cdo",
        f"remapcon,{GRID_FILE}",
        f"-selname,{VAR}",
        infile,
        outfile,
    ]

    subprocess.run(cmd, check=True)


# ============================================================
# SEED WORKER (PARALLEL UNIT)
# ============================================================

def process_seed(init_date, seed):

    infile = os.path.join(
        BASE_DIR,
        init_date,
        f"ensemble_{init_date}_seed{seed}.nc"
    )

    if not os.path.exists(infile):
        return None

    tmp_extract = os.path.join(
        TEMP_DIR,
        f"{init_date}_seed{seed}_rain.nc"
    )

    tmp_regrid = os.path.join(
        TEMP_DIR,
        f"{init_date}_seed{seed}_rg.nc"
    )

    ds = xr.open_dataset(infile)

    rain = ds[VAR].transpose("time", "member", "lat", "lon")

    rain.to_dataset(name=VAR).to_netcdf(tmp_extract)

    ds.close()

    run_cdo(tmp_extract, tmp_regrid)

    ds2 = xr.open_dataset(tmp_regrid)
    da = ds2[VAR]
    ds2.close()

    # attach seed explicitly (IMPORTANT)
    # da = da.assign_coords(seed=seed)

    return da


# ============================================================
# PROCESS ONE INIT DATE (PARALLEL SEEDS)
# ============================================================

def process_init_date(init_date):

    print(f"\nProcessing {init_date}")

    with ProcessPoolExecutor(max_workers=N_SEEDS) as executor:

        futures = {
            seed: executor.submit(process_seed, init_date, seed)
            for seed in range(N_SEEDS)
        }

        # deterministic collection (NOT as_completed)
        results = [None] * N_SEEDS

        for seed in range(N_SEEDS):
            results[seed] = futures[seed].result()

    # remove missing seeds safely
    results = [r for r in results if r is not None]

    if len(results) == 0:
        return None

    da = xr.concat(results, dim="member")
    da = da.assign_coords(member=np.arange(da.sizes["member"]))

    # --------------------------------------------------------
    # time → day
    # --------------------------------------------------------

    da = da.rename({"time": "day"})
    da = da.assign_coords(day=np.arange(da.sizes["day"]))

    # --------------------------------------------------------
    # init_time
    # --------------------------------------------------------

    da = da.expand_dims(
        init_time=[pd.to_datetime(init_date, format="%Y%m%d")]
    )

    return da


# ============================================================
# YEAR PROCESSING
# ============================================================

def get_year_map():

    years = defaultdict(list)

    for d in os.listdir(BASE_DIR):

        if len(d) == 8 and d.isdigit():
            years[d[:4]].append(d)

    for y in years:
        years[y] = sorted(years[y])

    return dict(years)


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
        print("No data found")
        return

    out = xr.concat(yearly, dim="init_time")

    out = out * 86400  # convert from kg/m2/s to mm/day
    #ds_out = out.rename({VAR: "tp"})
    ds_out = out.to_dataset(name="tp")
    
    # update attributes
    ds_out["tp"].attrs["units"] = "mm/day"
    ds_out["tp"].attrs["long_name"] = "Total precipitation"

    outfile = os.path.join(OUT_DIR, f"{year}.nc")

    print(f"Writing {outfile}")

    ds_out.to_netcdf(outfile)

    ds_out.close()


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    years_map = get_year_map()

    available_years = sorted(years_map.keys())

    print("Available years:")
    print(available_years)

    # ========================================================
    # READ YEARS FROM COMMAND LINE
    # ========================================================

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python regrid_pipeline.py 2019")
        print("  python regrid_pipeline.py 2019 2020")
        print("  python regrid_pipeline.py ALL")
        sys.exit(1)

    args = sys.argv[1:]

    # ========================================================
    # SELECT YEARS
    # ========================================================

    if len(args) == 1 and args[0].upper() == "ALL":
        years = available_years

    else:
        years = []

        for y in args:
            if y in years_map:
                years.append(y)
            else:
                print(f"Warning: year {y} not found, skipping")

    if len(years) == 0:
        print("No valid years selected.")
        sys.exit(1)

    print("\nProcessing years:", years)

    # ========================================================
    # RUN PIPELINE
    # ========================================================

    for y in years:
        process_year(y, years_map[y])

    print("\nCleaning temp files...")
    shutil.rmtree(TEMP_DIR)
    print("Done.")


if __name__ == "__main__":
    main()