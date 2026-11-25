"""
Download ECMWF S2S reforecasts (hindcasts) of Z500 (geopotential height): currently model year 2018
(hindcast years 1998-2017), Nov and Dec init dates (PLAN / MONTHS below), control + perturbed members,
00 UTC, days 0-46, North Atlantic box.

ECMWF reforecasts are organised by model-version date: each Monday/Thursday of model year Y re-runs
the same calendar day for the 20 previous years (Y-20 .. Y-1). Each hindcast year is taken once:
  model year 2017 -> hindcast year 1997
  model year 2018 -> hindcast years 1998-2017
  model year 2019 -> hindcast year 2018
The valid model-version days of every month are queried from the data store (constraints endpoint),
so no date is guessed.

Credentials: ~/.ecdsapirc ("url: https://ecds.ecmwf.int/api" / "key: ...").
Output: /glade/derecho/scratch/bgong/ec_s2s_model_2018/
  ecmwf_rf_gh500_{cf,pf}_{model date YYYYMMDD}_h{first}-{last hindcast year}.grib

Usage:  /glade/work/bgong/.venv/bin/python download_ecmwf_s2s_reforecast.py
"""

import os
import time

import cdsapi
import requests

OUT = "/glade/derecho/scratch/bgong/ec_s2s_model_2018"
DATASET = "s2s-reforecasts"
# model-version year -> hindcast years taken from it
# (full plan for 1997-2018 was {2017: [1997], 2018: 1998-2017, 2019: [2018]}; model year 2017 and
#  2018 Jan-May are already downloaded)
PLAN = {2018: list(range(1998, 2018))}
MONTHS = [11, 12]                                     # model-version (= hindcast) init months
LEADS = [str(h) for h in range(0, 1105, 24)]          # 0..1104 h = days 0..46
AREA = [85, -100, 15, 50]                             # N, W, S, E: NAO box 20-80N, 90W-40E plus margin
BASE = {"origin": "ecmwf", "level_type": "pressure", "variable": "geopotential_height",
        "level_value": "500_hpa"}

rc = dict(line.split(":", 1) for line in open(os.path.expanduser("~/.ecdsapirc")) if ":" in line)
URL, KEY = rc["url"].strip(), rc["key"].strip()
client = cdsapi.Client(url=URL, key=KEY)


def model_days(year, month):
    """Valid model-version days of (year, month) from the data store constraints."""
    for attempt in range(5):                          # the endpoint sometimes returns 502
        r = requests.post(f"{URL}/retrieve/v1/processes/{DATASET}/constraints", timeout=120,
                          json={"inputs": {**BASE, "year": str(year), "month": f"{month:02d}"}})
        if r.status_code < 500:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    return sorted(r.json().get("day", []))


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(y, m, d) for y in PLAN for m in MONTHS for d in model_days(y, m)]
    hys = sorted(h for v in PLAN.values() for h in v)
    print(f"{len(jobs)} model-version dates (months {MONTHS}); hindcast years {hys[0]}-{hys[-1]}", flush=True)
    for k, (y, m, d) in enumerate(jobs, 1):
        hy = PLAN[y]
        for ftype, tag in (("control_forecast", "cf"), ("perturbed_forecast", "pf")):
            target = f"{OUT}/ecmwf_rf_gh500_{tag}_{y}{m:02d}{d}_h{hy[0]}-{hy[-1]}.grib"
            if os.path.exists(target):
                continue
            print(f"[{k}/{len(jobs)}] {y}-{m:02d}-{d} {tag} hindcasts {hy[0]}-{hy[-1]}", flush=True)
            try:
                client.retrieve(DATASET, {
                    **BASE,
                    "variable": [BASE["variable"]], "level_value": [BASE["level_value"]],
                    "year": str(y), "month": f"{m:02d}", "day": d,
                    "hyear": [str(h) for h in hy], "hmonth": [f"{m:02d}"], "hday": [d],
                    "time": "00:00",
                    "forecast_type": ftype,
                    "leadtime_hour": LEADS,
                    "area": AREA,
                    "data_format": "grib",
                }).download(target)
            except Exception as e:
                print(f"  WARN: {y}-{m:02d}-{d} {tag} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
