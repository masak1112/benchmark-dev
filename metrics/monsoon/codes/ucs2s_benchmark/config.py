"""
Central configuration for the UC-S2S monsoon benchmark.

Everything path- or model-specific lives here so the other scripts stay generic.
Add a new model by adding an entry to MODELS.

Lead-day convention used throughout this package
------------------------------------------------
"step k" (k = 1, 2, ...) is the 24 h rainfall accumulation ENDING at
init + k days 00 UTC.  The onset core labels step k with the date init + k,
and IMD_2by2 day D (accumulation ending 03 UTC on D) is compared with it.
`day_offset` below converts each file's own day index to this step:

    step = day_index + day_offset          (rows with step < 1 are dropped)

  * UC-S2S files: lead_day k is the 24 h total ending at the valid time and
    lead 0 is identically zero  -> day_offset = 0 (lead 0 dropped).
  * NGCM files:   day 0 is already the first 24 h (verified by lagged
    correlation with IMD: day d matches IMD[init+d+1], r=0.95 vs 0.83 for
    IMD[init+d]) -> day_offset = 1.
    The legacy pipeline dropped NGCM day 0 and so scored NGCM one day out of
    phase; `ngcm_legacy` reproduces that behaviour for comparison only.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
import os as _os

PKG = Path(__file__).resolve().parent                       # .../monsoon/codes/ucs2s_benchmark
CODES = PKG.parent                                           # .../monsoon/codes (legacy core scripts)
REPO = Path(_os.environ.get("MONSOON_REPO", str(CODES.parent)))     # .../metrics/monsoon of this checkout
DATA = Path(_os.environ.get("MONSOON_DATA", str(REPO / "data_files")))   # data + grid files (Derecho)

IMD_DIR = DATA / "IMD_2by2"
THRESH_FILE = DATA / "mwset2x2.nc4"
SHAPEFILE = DATA / "ind_map_shpfile" / "india_shapefile.shp"

# Existing onset "core" scripts (imported unchanged, only the loader is swapped).
# Looked up in <PKG>/legacy_core/ first (self-contained copy), else in the monsoon repo's codes/.
def _core(sub, name):
    local = PKG / "legacy_core" / name
    return local if local.exists() else CODES / sub / name


CORE_MAE = _core("mae_far_mr_probablistic_models", "mae_far_mr_probabilistic_models.py")
CORE_RELIABILITY = _core("realibility_diagram", "onset_reliability_diagram_cmz.py")
CORE_SKILL = _core("probabilistic_skill_scores", "binned_skill_score_cmz.py")

# Raw UC-S2S hindcasts (new format)
RAW_ROOT = Path("/glade/derecho/scratch/jhayron/hindcasts_bing")

# ERA5 at 1 deg (same 180x360 grid as UC-S2S), 6-hourly, one file per year
ERA5_DIR = Path("/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata")
OBS_INDEX_FILE = DATA / "obs_indices" / "era5_monsoon_indices_daily.nc"

OUT_ROOT = Path(_os.environ.get("MONSOON_OUT", str(REPO / "codes" / "output" / "ucs2s_eval")))

# ----------------------------------------------------------------------------
# Evaluation set-up (kept identical to the legacy onset benchmark)
# ----------------------------------------------------------------------------
EVAL_YEARS = list(range(2019, 2025))      # IMD_2by2 and ERA5 both end in 2024
ONSET_INIT_MONTHS = (5, 6, 7)             # inits 1 May - 31 Jul (as before)
MEM_NUM = 30                              # UC-S2S has 30 members; NGCM subsampled
CLIM_YEARS = list(range(1991, 2019))      # anomaly + CRPSS reference climatology
                                          # (ends before the first verification year)

# Models compared in "common" init mode (dates shared by all of them)
COMPARISON_SET = ["ucs2s_plain", "ucs2s_sstanom", "fuxi", "ngcm"]   # matched on hindcast start date

# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
# tp_dir       : folder with {year}.nc holding 2-deg India precip (mm/day)
# tp_var       : variable name in those files
# dims         : how the file names init/member/lead dims
# day_offset   : see module docstring
# indices_dir  : per-member circulation/OLR indices written by postprocess
# wy_dir       : (NGCM) pre-computed Webster-Yang files, 6-hourly steps in h
# init_shift_days : added to the file start date to get the effective init (FuXi: its
#                lead 0 is the ERA5 mean of the start day, i.e. it has seen the
#                atmosphere up to D+1 00 UTC -> effective init D+1, lead L = step L)
# index_obs    : ERA5 sampling of the indices: suffix ("" = 00 UTC states, "_day" =
#                calendar-day means) and shift (verifying day = init + lead + shift)
MODELS = {
    "ucs2s_plain": dict(
        label="UC-S2S plain",
        raw_dir=RAW_ROOT / "plain" / "S2S",
        run_prefix="S41u_plain",
        tp_dir=DATA / "regrid_out" / "ucs2s_plain",
        tp_var="tp",
        dims=dict(init="init_time", member="member", lead="day"),
        day_offset=0,
        n_members=30,
        indices_dir=DATA / "regrid_out" / "ucs2s_plain" / "indices",
        color="#1f77b4",
    ),
    "ucs2s_sstanom": dict(
        label="UC-S2S SST-anomaly",
        raw_dir=RAW_ROOT / "sstanom" / "S2S_SSTANOM",
        run_prefix="S41u_sstanom",
        tp_dir=DATA / "regrid_out" / "ucs2s_sstanom",
        tp_var="tp",
        dims=dict(init="init_time", member="member", lead="day"),
        day_offset=0,
        n_members=30,
        indices_dir=DATA / "regrid_out" / "ucs2s_sstanom" / "indices",
        color="#d62728",
    ),
    "fuxi": dict(
        label="FuXi-S2S",
        raw_dir=RAW_ROOT / "fuxi",
        raw_layout="fuxi",                 # <raw_dir>/<YYYYMMDD>/mNN_h25.nc, 1.5 deg, lat N->S
        tp_dir=DATA / "regrid_out" / "fuxi_s2s_bing",
        tp_var="tp",
        dims=dict(init="init_time", member="member", lead="day"),
        day_offset=0,
        init_shift_days=1,                 # lead_day 0 = ERA5 daily mean of start day (dropped)
        n_members=30,
        indices_dir=DATA / "regrid_out" / "fuxi_s2s_bing" / "indices",
        index_obs=dict(suffix="_day", shift=-1),   # FuXi fields are daily means of UTC day D+L
        color="#eda100",
    ),
    "ngcm": dict(
        label="NGCM",
        tp_dir=DATA / "model_forecast_data" / "ngcm51" / "twice_weekly_0z" / "tp_2p0",
        tp_var="tp",
        dims=dict(init="time", member="number", lead="day"),
        day_offset=1,
        n_members=51,
        drop_last_day=True,    # day 44 holds ~75% of a daily total (partial accumulation)
        wy_dir=DATA / "model_forecast_data" / "ngcm51" / "twice_weekly_0z" / "wy",
        wy_sign=-1.0,          # files store u200 - u850; we use U850 - U200
        color="#2ca02c",
    ),
    # Legacy (misaligned) NGCM lead handling -- only to quantify the effect
    "ngcm_legacy": dict(
        label="NGCM (legacy lead alignment)",
        tp_dir=DATA / "model_forecast_data" / "ngcm51" / "twice_weekly_0z" / "tp_2p0",
        tp_var="tp",
        dims=dict(init="time", member="number", lead="day"),
        day_offset=0,
        n_members=51,
        drop_last_day=True,
        color="#98df8a",
    ),
}

# Drift / bias-corrected copies ("<key>_bc"), written by bias_correct.py.
# Method = S41 "dc" level (metrics/S41_aurora/s41eval/biascorr.py + bing_eval.core.drift_bias):
# b = mean(ens-mean - obs) per grid point and lead day over the starts of the OTHER years
# whose day of year is within +-BC_WINDOW_DAYS (leave-one-year-out), smoothed over leads
# with a centred BC_LEAD_SMOOTH-day running mean; corrected = x - b (precip clipped at 0).
# Reference: IMD 2-deg for rainfall (the verifying dataset here), ERA5 for the indices.
BC_BASES = ["ucs2s_plain", "ucs2s_sstanom", "fuxi"]
BC_WINDOW_DAYS = 30
BC_LEAD_SMOOTH = 7
BC_TRAIN_YEARS = list(range(2019, 2025))      # starts with verifying observations
for _b in BC_BASES:
    _m = {k: v for k, v in MODELS[_b].items() if k not in ("raw_dir", "raw_layout", "run_prefix")}
    _m.update(label=MODELS[_b]["label"] + " (bias-corr.)", bc_of=_b,
              tp_dir=DATA / "regrid_out" / f"{_b}_bc", indices_dir=DATA / "regrid_out" / f"{_b}_bc" / "indices")
    MODELS[f"{_b}_bc"] = _m

# Order used in tables and figures
MODEL_ORDER = ["ucs2s_plain", "ucs2s_plain_bc", "ucs2s_sstanom", "ucs2s_sstanom_bc",
               "fuxi", "fuxi_bc", "ngcm"]

# Weekly onset bins for the probabilistic skill scores (step 7 of run_onset_core.py)
ONSET_WEEK_BINS = [(1, 7), (8, 14), (15, 21), (22, 28)]

# ----------------------------------------------------------------------------
# Grids / regions
# ----------------------------------------------------------------------------
# 2-deg India grid (identical to grid_2deg_india.txt / IMD_2by2 / mwset2x2)
INDIA_LATS = [8.0 + 2 * i for i in range(16)]     # 8 .. 38
INDIA_LONS = [68.0 + 2 * i for i in range(17)]    # 68 .. 100
INDIA_RES = 2.0

# Core monsoon zone polygon used by the legacy onset scripts (2-deg version)
CMZ_POLY_LON = [83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83]
CMZ_POLY_LAT = [17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17]

# Circulation indices: boxes are (lat_s, lat_n, lon_w, lon_e), area-weighted.
#  wy  : Webster & Yang (1992)   U850 - U200, 0-20N 40-110E
#  imi : Wang et al. (2001) IMI  U850(5-15N,40-80E) - U850(20-30N,70-90E)
#  llj : U850 5-15N 40-80E (southern Arabian Sea westerlies; Wang et al. 2009
#        onset index)
#  mhi : Goswami et al. (1999)   V850 - V200, 10-30N 70-110E
# UC-S2S has no 200 hPa level, so U200/V200 are interpolated linearly in ln(p)
# between 150 and 250 hPa.  The SAME operator is applied to ERA5; ERA5 values
# using the true 200 hPa level are stored as *_true200 (used for NGCM, whose
# pre-computed WY uses the real 200 hPa).
BOX_WY = (0.0, 20.0, 40.0, 110.0)
BOX_IMI_S = (5.0, 15.0, 40.0, 80.0)
BOX_IMI_N = (20.0, 30.0, 70.0, 90.0)
BOX_MHI = (10.0, 30.0, 70.0, 110.0)
P200_WEIGHT_150 = 0.4368   # ln(250/200)/ln(250/150); u200 = u250 + w*(u150-u250)

INDEX_INFO = {
    "wy": dict(label="Webster-Yang (U850-U200)", units="m s-1"),
    "imi": dict(label="Indian monsoon index (Wang 2001)", units="m s-1"),
    "llj": dict(label="Low-level jet U850 5-15N 40-80E", units="m s-1"),
    "mhi": dict(label="Monsoon Hadley index (V850-V200)", units="m s-1"),
    "olr_cmz": dict(label="OLR over CMZ", units="W m-2"),
}

# Circulation-based onset over Kerala (LLJ index U850 5-15N 40-80E).
# onset = first day >= 1 May with LLJ > thr for `persist` days and (if mean_window)
# the mean over the next `mean_window` days > thr.  Same rule for forecasts and ERA5.
#  * "wang2009": commonly quoted form of Wang et al. (2009) (6.2 m/s, 6 days).  On
#    ERA5 1-deg it fires 14-21 days before IMD MOK in 2020-22 (cyclone-driven May
#    westerlies; Amphan/Tauktae/Asani).
#  * "era5cal" (default): 8 m/s, 6 days, 20-day mean > 8 m/s.  ERA5 - IMD MOK =
#    0,0,+4,+1,-1,+9 days for 2019-24 (MAE 2.5 d); 1991-2018 median 3 Jun (IMD normal
#    1 Jun).  NB the threshold was chosen looking at 2019-24, so treat it as a choice.
LLJ_ONSET_VARIANTS = {
    "era5cal": dict(thr=8.0, persist=6, mean_window=20),
    "wang2009": dict(thr=6.2, persist=6, mean_window=0),
}
LLJ_ONSET_DEFAULT = "era5cal"
IMD_OFFICIAL_MOK = {2019: "2019-06-08", 2020: "2020-06-01", 2021: "2021-06-03",
                    2022: "2022-05-29", 2023: "2023-06-08", 2024: "2024-05-30"}

# Weekly windows (lead days, inclusive) used by the raw-forecast metrics
WEEKS = {f"wk{w}": (7 * (w - 1) + 1, 7 * w) for w in range(1, 7)}
