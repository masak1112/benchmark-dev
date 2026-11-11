"""
Download ECMWF S2S real-time ensemble for MJO evaluation (2019-2025).

Variables downloaded for every available ECMWF init date (Mon + Thu, 00 UTC):
  - U + V at 200 hPa  → velocity potential (chi200) for VP200-based RMM
  - U + V at 850 hPa  → wind components for OLR-based RMM (Wheeler & Hendon 2004)
  - OLR (ttr)         → outgoing longwave radiation for OLR-based RMM
    Note: ECMWF stores OLR as 'ttr' (top net thermal radiation, accumulated J/m², negative).
    Convert to W/m2: OLR = -ttr / 86400

Dates: all available ECMWF S2S init dates 2019-2025 (queried from API).
Grid: global (no area restriction).
Lead times: days 0-46 (0, 24, ..., 1104 h).
Output: /glade/derecho/scratch/bgong/ec_s2s_model_2025/

API: ECMWF Data Store (ecds.ecmwf.int), credentials from ~/.ecdsapirc
"""

import os
import json
import pandas as pd
import cdsapi

# ── Configuration ──────────────────────────────────────────────────────────────

OUT = "/glade/derecho/scratch/bgong/ec_s2s_model_2025"
os.makedirs(OUT, exist_ok=True)

START_YEAR = 2019
END_YEAR   = 2024

# Lead times: days 0-46
LEADS = [str(h) for h in range(0, 1105, 24)]

# ── ECMWF Data Store client ────────────────────────────────────────────────────
rc = {k.strip(): v.strip()
      for line in open(os.path.expanduser("~/.ecdsapirc"))
      if ":" in line
      for k, v in [line.split(":", 1)]}
client = cdsapi.Client(url=rc["url"], key=rc["key"])

# ── Build date list: Mon + Thu each year 2019-2025 ────────────────────────────
# ECMWF S2S real-time runs every Monday and Thursday at 00 UTC.
all_dates = []
for year in range(START_YEAR, END_YEAR + 1):
    for d in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="B"):
        if d.weekday() in (0, 3):   # Monday=0, Thursday=3
            all_dates.append(d.strftime("%Y%m%d"))

print(f"Total init dates to download: {len(all_dates)}  ({all_dates[0]} – {all_dates[-1]})")

# ── Download loop ──────────────────────────────────────────────────────────────
for d in all_dates:
    year, month, day = d[:4], d[4:6], d[6:]

    for ftype, tag in (("control_forecast", "cf"), ("perturbed_forecast", "pf")):

        # 1. U + V at 200 & 850 hPa
        target_uv = f"{OUT}/ecmwf_uv200_850_{tag}_{d}.grib"
        if not os.path.exists(target_uv):
            print(f"  UV 200/850  {tag}  {d}")
            try:
                client.retrieve("s2s-forecasts", {
                    "origin":        "ecmwf",
                    "year":          year,
                    "month":         month,
                    "day":           day,
                    "time":          "00:00",
                    "level_type":    "pressure",
                    "level_value":   ["200_hpa", "850_hpa"],
                    "variable":      ["u_component_of_wind", "v_component_of_wind"],
                    "forecast_type": ftype,
                    "leadtime_hour": LEADS,
                    "data_format":   "grib",
                }).download(target_uv)
            except Exception as e:
                print(f"  WARN: {d} {tag} UV failed — {e}")
                continue
        else:
            print(f"  Skip (exists): {target_uv}")

        # 2. OLR (ttr)
        target_olr = f"{OUT}/ecmwf_olr_{tag}_{d}.grib"
        if not os.path.exists(target_olr):
            print(f"  OLR         {tag}  {d}")
            try:
                client.retrieve("s2s-forecasts", {
                    "origin":        "ecmwf",
                    "year":          year,
                    "month":         month,
                    "day":           day,
                    "time":          "00:00",
                    "level_type":    "surface",
                    "variable":      ["top_net_thermal_radiation"],
                    "forecast_type": ftype,
                    "leadtime_hour": LEADS,
                    "data_format":   "grib",
                }).download(target_olr)
            except Exception as e:
                print(f"  WARN: {d} {tag} OLR failed — {e}")
        else:
            print(f"  Skip (exists): {target_olr}")

print("Done.")
