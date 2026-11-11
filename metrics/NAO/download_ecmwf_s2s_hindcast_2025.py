"""
Download ECMWF S2S REFORECASTS (hindcasts) of the 2025 model version for selected hindcast years
(default 2019-2021; --hyears 2022-2024 for the later block).

Model-version 2025 runs every other day (e.g. Jan 1, 3, 5, ...); each model date re-forecasts the same
calendar day of the 20 previous years (2005-2024). Only the selected hindcast years are taken
(default 2019, 2020, 2021), so every file holds those hindcast inits. 11 members: 1 control + 10 perturbed. 00 UTC, global.

Variable groups (one GRIB per model date, control / perturbed):
  t2m       2 m temperature, daily means, leadtime windows "0_24" ... "1080_1104"
  tp        total precipitation, accumulated from init (kg m-2), leadtime 0, 24, ..., 1104 h
  pl500_z   geopotential height at 500 hPa, instantaneous, 0, 24, ..., 1104 h
  uv200_850 u and v at 200 and 850 hPa, instantaneous, 0, 24, ..., 1104 h
  sm20      soil moisture top 20 cm (kg m-3; /1000 -> m3 m-3), daily means, "0_24" ... "1080_1104"
            (S2S has no ERA5-style swvl1 0-7 cm layer; sm20 is the closest)

The valid model-version days of every month are read from the data store constraints endpoint.
Output: /glade/derecho/scratch/bgong/ec_s2s_model_2025_hindcast/
        ecmwf_rf_{group}_{cf|pf}_{model date YYYYMMDD}_h{first}-{last hindcast year}.grib
Credentials: ~/.ecdsapirc. Re-running skips files that already exist.

Usage:  nohup /glade/work/bgong/.venv/bin/python -u download_ecmwf_s2s_hindcast_2025.py [--hyears Y0-Y1] [GROUP ...] > LOG 2>&1 &
        (no GROUP = all groups; e.g. `... download_ecmwf_s2s_hindcast_2025.py sm20` for soil moisture only,
         `... download_ecmwf_s2s_hindcast_2025.py --hyears 2022-2024` for hindcast years 2022-2024)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cdsapi
import requests

OUT = "/glade/derecho/scratch/bgong/ec_s2s_model_2025_hindcast"
DATASET = "s2s-reforecasts"
MODEL_YEAR = 2025
HINDCAST_YEARS = [2019, 2020, 2021]
N_PARALLEL = 3
LEADS = [str(h) for h in range(0, 1105, 24)]                  # 0..1104 h = days 0..46
LEADS_DAILY = [f"{h}_{h + 24}" for h in range(0, 1081, 24)]   # daily-mean windows for t2m

GROUPS = {
    "t2m":       {"level_type": "single_level", "variable": ["2_m_temperature"], "leadtime_hour": LEADS_DAILY},
    "tp":        {"level_type": "single_level", "variable": ["total_precipitation"], "leadtime_hour": LEADS},
    "pl500_z":   {"level_type": "pressure", "variable": ["geopotential_height"], "level_value": ["500_hpa"],
                  "leadtime_hour": LEADS},
    "uv200_850": {"level_type": "pressure", "variable": ["u_component_of_wind", "v_component_of_wind"],
                  "level_value": ["200_hpa", "850_hpa"], "leadtime_hour": LEADS},
    "sm20":      {"level_type": "single_level", "variable": ["soil_moisture_top_20_cm"], "leadtime_hour": LEADS_DAILY},
}

rc = dict(line.split(":", 1) for line in open(os.path.expanduser("~/.ecdsapirc")) if ":" in line)
URL, KEY = rc["url"].strip(), rc["key"].strip()


def model_days(month):
    """Valid model-version days of (MODEL_YEAR, month) from the data store constraints."""
    for attempt in range(5):                                  # the endpoint sometimes returns 502
        r = requests.post(f"{URL}/retrieve/v1/processes/{DATASET}/constraints", timeout=120,
                          json={"inputs": {"origin": "ecmwf", "level_type": "single_level",
                                           "variable": "2_m_temperature",
                                           "year": str(MODEL_YEAR), "month": f"{month:02d}"}})
        if r.status_code < 500:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    return sorted(r.json().get("day", []))


def fetch(job):
    month, day, group, ftype, tag = job
    target = (f"{OUT}/ecmwf_rf_{group}_{tag}_{MODEL_YEAR}{month:02d}{day}"
              f"_h{HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]}.grib")
    if os.path.exists(target):
        return target, 'exists'
    client = cdsapi.Client(url=URL, key=KEY, quiet=True)
    for attempt in range(3):
        try:
            client.retrieve(DATASET, {
                "origin": "ecmwf", **GROUPS[group],
                "year": str(MODEL_YEAR), "month": f"{month:02d}", "day": day,
                "hyear": [str(h) for h in HINDCAST_YEARS], "hmonth": [f"{month:02d}"], "hday": [day],
                "time": "00:00", "forecast_type": ftype, "data_format": "grib",
            }).download(target + ".part")
            os.replace(target + ".part", target)
            return target, 'ok'
        except Exception as e:
            err = str(e)[:200]
            time.sleep(60 * (attempt + 1))
    return target, f'FAILED: {err}'


def main():
    os.makedirs(OUT, exist_ok=True)
    global HINDCAST_YEARS
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", help=f"variable groups (default: all of {list(GROUPS)})")
    ap.add_argument("--hyears", default=f"{HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]}",
                    help="hindcast-year range Y0-Y1 (default %(default)s)")
    args = ap.parse_args()
    y0, y1 = map(int, args.hyears.split("-"))
    HINDCAST_YEARS = list(range(y0, y1 + 1))
    groups = args.groups or list(GROUPS)
    unknown = set(groups) - set(GROUPS)
    if unknown:
        sys.exit(f"Unknown group(s) {sorted(unknown)}; choose from {list(GROUPS)}")
    days = [(m, d) for m in range(1, 13) for d in model_days(m)]
    jobs = [(m, d, g, ftype, tag) for m, d in days for g in groups
            for ftype, tag in (("control_forecast", "cf"), ("perturbed_forecast", "pf"))]
    print(f"{len(days)} model-version dates in {MODEL_YEAR}, hindcast years {HINDCAST_YEARS}, "
          f"{len(jobs)} requests", flush=True)
    with ThreadPoolExecutor(N_PARALLEL) as ex:
        for k, (target, msg) in enumerate(ex.map(fetch, jobs), 1):
            print(f"[{k}/{len(jobs)}] {os.path.basename(target)}: {msg}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
