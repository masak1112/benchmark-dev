"""
Download ECMWF S2S real-time ensemble Z500 (geopotential height) for the 68 UC-S2S V2 DJF init dates
(Mondays/Thursdays, 2019-01-03 .. 2021-02-25): control + 50 perturbed members, 00 UTC, days 0-46,
North Atlantic box (NAO 20-80N, 90W-40E plus margin).

Check the dataset name and request keys against "Show API request" on the ECMWF data store form.
Output: /glade/derecho/scratch/bgong/ec_s2s_model_2018/ecmwf_gh500_{cf,pf}_{YYYYMMDD}.grib
"""

import os

import cdsapi

DATES = "/glade/work/bgong/benchmark-dev/metrics/NAO/dates_ucs2s_v2_djf.txt"
OUT = "/glade/derecho/scratch/bgong/ec_s2s_model_2018"
os.makedirs(OUT, exist_ok=True)

leads = [str(h) for h in range(0, 1105, 24)]          # 0..1104 h = days 0..46
# The S2S datasets are on the ECMWF Data Store (ecds.ecmwf.int), not the Copernicus CDS that
# ~/.cdsapirc points to; its url/key are read from ~/.ecdsapirc (same "url: / key:" format).
rc = dict(line.split(":", 1) for line in open(os.path.expanduser("~/.ecdsapirc")) if ":" in line)
client = cdsapi.Client(url=rc["url"].strip(), key=rc["key"].strip())

for d in open(DATES).read().split():                  # YYYYMMDD
    for ftype, tag in (("control_forecast", "cf"), ("perturbed_forecast", "pf")):
        target = f"{OUT}/ecmwf_gh500_{tag}_{d}.grib"
        if os.path.exists(target):
            continue
        client.retrieve("s2s-forecasts", {             # real-time dataset (confirm exact name)
            "origin": "ecmwf",
            "year": d[:4], "month": d[4:6], "day": d[6:],
            "time": "00:00",
            "level_type": "pressure",
            "level_value": ["500_hpa"],
            "variable": ["geopotential_height"],
            "forecast_type": ftype,
            "leadtime_hour": leads,
            "area": [85, -100, 15, 50],               # N, W, S, E
            "data_format": "grib",
        }).download(target)
