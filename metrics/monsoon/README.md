# Indian summer monsoon evaluation

This folder evaluates subseasonal-to-seasonal (S2S) hindcasts of the Indian summer monsoon against
IMD gridded rainfall and ERA5. The evaluation has three steps:

1. **Regridding.** Forecast precipitation is put on the IMD 2° grid over India. Monsoon circulation
   indices are computed from the forecast winds and OLR.
2. **Onset benchmarking.** Local monsoon onset dates are detected in every forecast member and
   scored against IMD over the Core Monsoon Zone (CMZ). The scores are MAE, false-alarm rate, miss
   rate, reliability, and fair Brier, RPS and AUC skill scores.
3. **General rainfall evaluation.** The daily and weekly rainfall forecasts and the circulation
   forecasts are verified directly: bias, ACC, CRPSS, spread/error and rain statistics.

`codes/ucs2s_benchmark/` runs all three steps as one PBS chain on Derecho, for these models:

- UC-S2S (plain and SST-anomaly)
- FuXi-S2S
- drift/bias-corrected versions of those three
- NGCM

Its README has the details of each script. The standalone scripts in `codes/` run steps 1 and 2
for any other model.

## Folder layout

| Path | What |
|---|---|
| `codes/ucs2s_benchmark/` | the full pipeline (regridding, bias correction, onset benchmark, raw-forecast metrics, figures) for the S41 hindcasts (`metrics/S41_aurora`) and NGCM |
| `codes/postprocess_raw_forecast_s2s/` | older regridding scripts (CDO) for the PanguWeather/PlaSim S2S output format |
| `codes/onset_threshold_imd/` | computes the onset rainfall threshold `mwset2x2.nc4` from IMD |
| `codes/mae_far_mr_probablistic_models/`, `codes/mae_far_mr_deterministic_models/`, `codes/mae_far_mr_climatology/` | onset MAE / FAR / miss-rate maps and CMZ averages |
| `codes/realibility_diagram/` | onset reliability diagram over the CMZ |
| `codes/probabilistic_skill_scores/` | onset fair Brier, RPS and AUC skill scores in lead-time bins, against an IMD climatological ensemble |
| `codes/cdo_grid_config/`, `grid_2deg_india.txt` | CDO grid descriptions of the 1°, 2° and 4° India grids |
| `codes/complete_onset_benchmarking_pipeline_for_{ncar,stampede}.sh` | runs the onset scripts in sequence for one model |
| `data_files/` | IMD 2° rainfall `IMD_2by2/data_<year>.nc` (1901–2024), `mwset2x2.nc4`, `grid_2deg_india.txt`, India shapefile. Generated data (regridded forecasts, ERA5 indices) is also written here and is git-ignored |
| `jupyter_notebooks/`, `regrid_forecast_output.ipynb` | exploratory notebooks. `jupyter_notebooks/regrid.py` and `UCAR.sh` are an older CDO regridding helper from the S41 repository |

## Input data

| Data | Used for | Location |
|---|---|---|
| IMD 0.25° rainfall, aggregated to 2° (`IMD_2by2`) | observed onset, rainfall verification | `data_files/IMD_2by2/data_<year>.nc`, 1901–2024 |
| Onset threshold `MWmean` | onset definition | `data_files/mwset2x2.nc4` (made by `codes/onset_threshold_imd/`) |
| CMZ polygon | spatial aggregation | hard-coded in the onset scripts, and in `ucs2s_benchmark/config.py` |
| ERA5 1° (u, v at 850/250/200/150 hPa; TOA net longwave) | circulation and OLR verification | `/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata` |
| Hindcasts | forecasts | see `codes/ucs2s_benchmark/config.py` (`MODELS`) |

IMD 2023–24 files have a larger land mask (106 valid cells against 62) and dimensions named
`lat/lon`. The onset scripts handle this, and the rainfall metrics use only cells that are valid in
every year (30 CMZ cells).

Keep only `data_<year>.nc` files in `IMD_2by2/`, because the onset and threshold scripts read every
`.nc` file in that folder. An older 2023 file whose missing values are not flagged as missing
(fill values give a June–September mean of −65 mm/day) is kept in `IMD_2by2_unused/` for reference
only.

## 1. Regridding

The target is the 2° IMD grid: 17 × 16 cell centres, 68–100°E and 8–38°N
(`grid_2deg_india.txt`). Precipitation is remapped **first-order conservatively**: every target cell
is the area-weighted mean of the source cells that overlap it, weighted by overlap area. This is
`cdo remapcon`, and it conserves the rainfall total.

There are two implementations.

- **`codes/ucs2s_benchmark/postprocess_ucs2s.py`** reads the S41 per-member files (`*_h25.nc`):
  UC-S2S on 1° and FuXi-S2S on 1.5° (latitude north to south, flipped on read). It computes the
  overlap weights exactly in NumPy (sin-latitude for the latitude weights). `--check-cdo` compares
  one member with `cdo remapcon` (agreement about 1e-6 mm/day). In the same pass it computes daily
  monsoon indices on the native grid:
  - Webster–Yang shear (`wy`)
  - Wang et al. (2001) index (`imi`)
  - Somali low-level jet U850 (`llj`)
  - Goswami monsoon Hadley index (`mhi`)
  - CMZ-mean OLR (`olr_cmz`)

  `era5_indices.py` computes the same indices from ERA5 with identical operators. The S41 files have
  no 200 hPa level, so U200/V200 are interpolated in ln p between 150 and 250 hPa for both forecasts
  and ERA5.
- **`codes/postprocess_raw_forecast_s2s/`** (`postprocess_precip_forecast[_parallel].py`,
  `run_postprocess*.sh`, `amip_preprocessing_parallel_ncar.py`) handles the older
  PanguWeather/PlaSim output (`pangu_plasim_<exp>_24h_<N>step_<date>_ens_<m>.nc`). It extracts
  precipitation, runs `cdo setgrid` and `cdo remapcon,codes/cdo_grid_config/2p0_grid_ind.txt`, and
  merges the members of one year. The 1° and 4° grid files support the 1° and 4° versions of the
  onset scripts.

Both write one file per year, `{year}.nc`, containing `tp(init_time, member, day, lat, lon)` in
mm/day. That is the layout every onset script reads.

**Lead convention.** "Step k" is the 24 h total ending at init + k days, 00 UTC. It is compared with
IMD date init + k, because IMD labels date D with the 24 h ending 08:30 IST (03 UTC) on D. Each
model's file day index is mapped to step k by `config.MODELS[..]['day_offset']`:

| Model | File layout | Mapping |
|---|---|---|
| UC-S2S | day k = step k; day 0 is all zeros | offset 0; day 0 dropped |
| FuXi-S2S | `lead_day` L is the UTC calendar day start + L; L = 0 is ERA5's start-day mean, not a forecast | effective init = start + 1 d (`init_shift_days = 1`), so L = step L; L = 0 dropped |
| NGCM | day 0 is already the first 24 h | offset 1 |

The pipeline writes `lead_alignment.csv`, which checks this for every model: step k should
correlate best with IMD at offset 0. Earlier NGCM onset scores in this repository dropped NGCM
day 0, so they were one day out of phase. The `ngcm_legacy` entry reproduces them.

## 2. Onset benchmarking

**Onset definition** (Moron and Robertson 2014, doi:10.1002/joc.3745). At each 2° cell, onset is
the first day after 2 June (the climatological monsoon onset over Kerala, the "MOK filter") on which:

- rainfall is above 1 mm, and
- the 5-day total starting that day is above `MWmean`.

`MWmean` is the cell's mean 5-day wet-spell rainfall for April–October in IMD
(`codes/onset_threshold_imd/`). The MOK filter removes early false onsets that are followed by a dry
spell. Observed onset comes from IMD in every year.

**Forecasts.** The inits are every hindcast start from 1 May to 31 Jul. An init counts at a cell
only if it starts before that cell's observed onset. Onset is detected in every member separately,
with the same rule, within the first 15 or 30 forecast days.

**Deterministic scores** (`mae_far_mr_*`):

- The ensemble predicts onset when at least 50% of members do. The forecast onset is then the mean
  of the members' onset dates.
- Forecast onsets are verified in two windows: days 1–15 with a tolerance of ±3 days, and days
  16–30 with ±5 days. A forecast onset in the window within the tolerance of the observed onset is a
  hit; one outside it is a false positive.
- MAE is the mean absolute onset error, in days, over every forecast onset in the window: hits and
  false positives.
- FAR is false positives / (false positives + true negatives).
- Miss rate is missed onsets / observed onsets.

Maps are written per cell. CMZ values are the mean over CMZ cells; for MAE, the CMZ mean per year is
averaged over years, with ±1 standard error.

**Probabilistic scores** (`realibility_diagram/`, `probabilistic_skill_scores/`):

- The probability of onset in a lead bin is the fraction of members with onset in that bin.
- The bins are days 1–5, 6–10, 11–15 (and so on to 30), or the weekly bins days 1–7, 8–14, 15–21
  and 22–28. Each set adds a final "no onset by the last day" bin.
- The scores are fair Brier score per bin, fair RPS across bins, and AUC.
- The reference forecast is an IMD climatological ensemble: each IMD year is one member, its
  observed onset day of year applied to the same init date. The climatology has extra bins for onset
  before the init.
- Skill = 1 − score / score of climatology.
- All forecast–observation pairs in the CMZ are pooled.
- The reliability diagram uses 10 probability bins.

**Init sets.** `native` uses every init a model has. `common` uses only the start dates shared by
all compared models: 48 inits for 2019 and 2021–24 (2020 has none), with 30 members for every model.
This is the like-for-like comparison.

## 3. General rainfall and circulation evaluation

`codes/ucs2s_benchmark/raw_forecast_metrics.py` verifies the forecasts themselves. The inits are
1 May to 31 Jul by default, or May–Sep (`--months 5 6 7 8 9`).

| Target | Truth | Scores |
|---|---|---|
| CMZ-mean rainfall | IMD 2° | bias, ACC, RMSE, spread/RMSE, fair CRPS and CRPSS, for daily leads 1–45 and weeks 1–6 |
| Rainfall maps | IMD 2° | weekly-mean bias (%) and ACC per cell, weeks 1–6 |
| CMZ daily-rain statistics | IMD 2° | wet-day (> 1 mm) frequency, frequency of meeting the onset criterion, 50th/90th/99th percentiles, all against lead |
| `wy`, `imi`, `llj`, `mhi`, `olr_cmz` | ERA5 | as for rainfall. Lead-0 RMSE checks the initial state. ERA5 is sampled like each model (00 UTC for UC-S2S, daily means for FuXi-S2S) |
| Monsoon onset over Kerala from the LLJ | ERA5 | onset = first day after 1 May with LLJ > 8 m/s for 6 days and a 20-day mean > 8 m/s (`era5cal`, default), or > 6.2 m/s for 6 days (Wang et al. 2009). Scores: ensemble onset error by lead, member hit rate (±3 d), compared with a climatological reference |

Conventions:

- **ACC** is the centred correlation, across inits, of ensemble-mean anomalies from the observed
  1991–2018 day-of-year climatology (mean plus 3 harmonics).
- **CRPSS** uses the fair CRPS. The reference ensemble is the observed value in the same calendar
  window in each of 1991–2018.
- **Uncertainty**: 90% intervals come from a year-block bootstrap.

## Drift/bias-corrected versions

`codes/ucs2s_benchmark/bias_correct.py` writes a bias-corrected copy of each S41 model (`<model>_bc`).
It uses the mean drift correction of `metrics/S41_aurora/s41eval/biascorr.py`:

- For each grid point and lead day, the bias is the mean of (ensemble mean − observation) over the
  starts of the *other* years within ±30 days of day of year (leave-one-year-out).
- The bias is smoothed with a 7-day running mean over leads and subtracted from every member.
- Rainfall is clipped at 0.
- Rainfall is corrected against IMD 2° and the indices against ERA5.

All three evaluation steps then treat `<model>_bc` as one more model.

## Running

```bash
cd codes/ucs2s_benchmark
./submit_all.sh                                   # regrid + ERA5 indices -> bias correction -> onset + rainfall metrics -> figures
SKIP_PP=1 SKIP_ERA5=1 SKIP_BC=1 ./submit_all.sh   # evaluation only, re-using regridded data
```

The jobs use the Derecho `develop` queue. Set the project code and Python environment in
`pbs/env.sh` and the `#PBS -A` lines.

Paths default to this folder. Override them with these environment variables:

- `MONSOON_REPO`: this folder
- `MONSOON_DATA`: where `IMD_2by2/`, the threshold file and the regridded data live
- `MONSOON_OUT`: outputs

The hindcast and ERA5 paths are in `config.py`. To run the onset scripts alone for another model,
see `codes/complete_onset_benchmarking_pipeline_for_ncar.sh` and the README in each `codes/` folder.

Outputs go to `codes/output/ucs2s_eval/`:

| Folder | Contents |
|---|---|
| `onset/<native\|common>/<model>/` | onset scores (NetCDF maps and CSVs) |
| `raw_metrics/<mode>_m<months>/<model>/` | rainfall and index scores |
| `bias_correction/` | bias before and after correction |
| `comparison/` | multi-model tables and figures |

The `comparison/` figures include:

- `fig_scorecard_onset_weekly_{bss,auc}_common`: onset BSS and AUC per week, in the style of
  Masiwal et al. 2026, arXiv:2602.03767, Fig. 3a,b
- `fig_onset_<mode>`
- `fig_rain_cmz_*`, `fig_rain_maps_*`, `fig_indices_*`, `fig_mok_*`
- `fig_scorecard_<target>_*`
