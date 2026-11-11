"""
Download ECMWF S2S real-time ensemble (2019-2024) for additional variables.

Variables downloaded for every ECMWF init date (Mon + Thu, 00 UTC):
  - 2 m temperature (t2m)    → daily mean; ECDS leads are windows "0_24", ..., "1080_1104"
  - Total precipitation (tp) → accumulated from init (kg/m² = mm); downloaded at 0, 24, ..., 1104 h
    Note: 24 h precip for day k = tp(24k) - tp(24(k-1))
  - Z500 (gh @ 500 hPa)      → geopotential height (S2S only archives gh, not z)
  - U500 (u @ 500 hPa)
  - T850, Z850 (gh), Q850 (specific humidity) @ 850 hPa

Dates: Mon + Thu 2019-2024 (same as download_ecmwf_s2s_mjo.py).
Grid: global (no area restriction).
Output: /glade/derecho/scratch/bgong/ec_s2s_model_2025/

API: ECMWF Data Store (ecds.ecmwf.int), credentials from ~/.ecdsapirc
"""

import os
import pandas as pd
import cdsapi

# ── Configuration ──────────────────────────────────────────────────────────────

OUT = "/glade/derecho/scratch/bgong/ec_s2s_model_2025"
os.makedirs(OUT, exist_ok=True)

START_YEAR = 2019
END_YEAR   = 2024

# Instantaneous / accumulated fields: days 0-46 (0, 24, ..., 1104 h)
LEADS = [str(h) for h in range(0, 1105, 24)]
# Daily-mean fields (t2m): 24 h windows "0_24", ..., "1080_1104"
LEADS_DAILY = [f"{h}_{h + 24}" for h in range(0, 1081, 24)]

# tag → request fields (origin/date/forecast_type/format are added in the loop)
REQUESTS = {
    "t2m": {
        "level_type":    "single_level",
        "variable":      ["2_m_temperature"],
        "leadtime_hour": LEADS_DAILY,
    },
    "tp": {
        "level_type":    "single_level",
        "variable":      ["total_precipitation"],
        "leadtime_hour": LEADS,
    },
    "pl500_z_u": {
        "level_type":    "pressure",
        "level_value":   ["500_hpa"],
        "variable":      ["geopotential_height", "u_component_of_wind"],
        "leadtime_hour": LEADS,
    },
    "pl850_z_t_q": {
        "level_type":    "pressure",
        "level_value":   ["850_hpa"],
        "variable":      ["geopotential_height", "temperature", "specific_humidity"],
        "leadtime_hour": LEADS,
    },
}

# ── ECMWF Data Store client ────────────────────────────────────────────────────
rc = {k.strip(): v.strip()
      for line in open(os.path.expanduser("~/.ecdsapirc"))
      if ":" in line
      for k, v in [line.split(":", 1)]}
client = cdsapi.Client(url=rc["url"], key=rc["key"])

# ── Build date list: Mon + Thu each year ──────────────────────────────────────
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
        for vtag, req in REQUESTS.items():
            target = f"{OUT}/ecmwf_{vtag}_{tag}_{d}.grib"
            if os.path.exists(target):
                print(f"  Skip (exists): {target}")
                continue
            print(f"  {vtag:<12} {tag}  {d}")
            try:
                client.retrieve("s2s-forecasts", {
                    "origin":        "ecmwf",
                    "year":          year,
                    "month":         month,
                    "day":           day,
                    "time":          "00:00",
                    "forecast_type": ftype,
                    "data_format":   "grib",
                    **req,
                }).download(target)
            except Exception as e:
                print(f"  WARN: {d} {tag} {vtag} failed — {e}")

print("Done.")
