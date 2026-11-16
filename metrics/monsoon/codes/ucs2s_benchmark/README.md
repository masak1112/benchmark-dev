# UC-S2S monsoon benchmark

Evaluates the UC-S2S and FuXi-S2S hindcasts in `/glade/derecho/scratch/jhayron/hindcasts_bing`
for the Indian summer monsoon, raw and drift/bias-corrected, and compares them with the older
NGCM hindcast.

| key | label | source |
|---|---|---|
| `ucs2s_plain` | UC-S2S plain | `hindcasts_bing/plain/S2S/S41u_plain_<YYYYMMDD>` |
| `ucs2s_sstanom` | UC-S2S SST-anomaly | `hindcasts_bing/sstanom/S2S_SSTANOM/S41u_sstanom_<YYYYMMDD>` |
| `fuxi` | FuXi-S2S | `hindcasts_bing/fuxi/<YYYYMMDD>/mNN_h25.nc` (1.5°) |
| `<key>_bc` | … (bias-corr.) | the three above after `bias_correct.py` |
| `ngcm` | NGCM | `data_files/model_forecast_data/ngcm51/twice_weekly_0z/tp_2p0` (+ `wy/`) |
| `ngcm_legacy` | NGCM, old lead handling | same files; only used to measure the effect of the lead fix |

Two parts:

1. **Onset core**: the existing benchmark, run *unchanged*. That is MAE/FAR/MR, the reliability
   diagram, and fair BSS/RPSS/AUC on the 2° grid, aggregated over CMZ cells, with inits from 1 May to
   31 Jul, the MOK (2 June) filter and the `mwset2x2` thresholds. `run_onset_core.py` imports the three
   core scripts and swaps out only their forecast loader (see "Init dates" below). Step 7 adds the
   probabilistic scores in **weekly** onset bins (days 1–7, 8–14, 15–21, 22–28), using the core's own
   functions.
2. **Raw-forecast metrics** (new): verification of the rainfall and circulation forecasts
   themselves (see "Raw-forecast metrics").

## Run it

```bash
cd codes/ucs2s_benchmark
./submit_all.sh       # postprocess + ERA5 -> bias correction -> onset core + raw metrics -> plots
SKIP_PP=1 SKIP_ERA5=1 SKIP_BC=1 ./submit_all.sh   # re-run the evaluation only
```

| stage | script | PBS | output |
|---|---|---|---|
| postprocess | `postprocess_ucs2s.py` | `pbs/postprocess.pbs` | `data_files/regrid_out/{ucs2s_plain,ucs2s_sstanom,fuxi_s2s_bing}/{year}.nc`, `.../indices/{year}.nc` |
| ERA5 indices | `era5_indices.py` | `pbs/era5_indices.pbs` | `data_files/obs_indices/era5_monsoon_indices_daily.nc` |
| bias correction | `bias_correct.py` | `pbs/bias_correct.pbs` | `data_files/regrid_out/<key>_bc/`, diagnostics in `codes/output/ucs2s_eval/bias_correction/` |
| onset core | `run_onset_core.py` | `pbs/onset_core.pbs` | `codes/output/ucs2s_eval/onset/<mode>/<model>/` |
| raw metrics | `raw_forecast_metrics.py` | `pbs/raw_metrics.pbs` | `codes/output/ucs2s_eval/raw_metrics/<mode>_m<months>/<model>/` |
| comparison | `compare_models.py` | `pbs/plots.pbs` | `codes/output/ucs2s_eval/comparison/` |

All paths, years, members, boxes and thresholds are set in `config.py`. Use the `ngcm_env`
environment. Logs are written to `logs/`.

Single runs:

```bash
python postprocess_ucs2s.py --model ucs2s_plain --years 2019 2020 --workers 64
python postprocess_ucs2s.py --model ucs2s_plain --check-cdo     # regrid vs `cdo remapcon`
python bias_correct.py --models fuxi
python run_onset_core.py --model ngcm --init-mode common --step all      # steps 1-7
python run_onset_core.py --model fuxi_bc --init-mode common --step 7     # weekly bins only
python raw_forecast_metrics.py --model ucs2s_sstanom --init-mode native --months 5 6 7 8 9
python compare_models.py --init-mode common native
```

## Postprocessing the new format

- Each init has 30 members `..._ens_<m>_h25.nc` on the native 1° grid. `tp24` is in **m** and is
  the 24 h total ending at the valid time (00 UTC). `lead_day` runs 0–45, and lead 0 is identically 0.
  The 1.1 GB `..._ens_0.nc` "full" file exists for member 0 only and is not used.
- Precipitation is regridded **conservatively** (exact area-overlap weights) to the 2° India grid
  of `grid_2deg_india.txt`, converted to mm/day, and written as `tp(init_time, member, day, lat, lon)`
  with day 0–45. This is the same layout as the old `pp_si_infer_*` files, so the legacy scripts can
  read it too. `--check-cdo` agrees with `cdo remapcon` to about 1e-6 mm/day.
- Per-member daily indices are computed from `u850, v850, u250, v250, u150, v150, olr`:
  - `wy`: Webster–Yang, U850 − U200 over 0–20°N, 40–110°E.
  - `imi`: Wang et al. 2001, U850(5–15°N, 40–80°E) − U850(20–30°N, 70–90°E).
  - `llj`: U850 over 5–15°N, 40–80°E.
  - `mhi`: Goswami 1999, V850 − V200 over 10–30°N, 70–110°E.
  - `olr_cmz`: OLR, 24 h mean, averaged over the 2° CMZ cells.

  **UC-S2S has no 200 hPa level.** U200/V200 are interpolated linearly in ln p between 150 and
  250 hPa, and the same operator is applied to ERA5. ERA5 values with the real 200 hPa level are kept
  as `*_true200`; these are the reference for NGCM's pre-computed WY.
- All inits of every year (2019-01 to 2026-09) are processed. Only 2019–2024 can be verified,
  because IMD_2by2 and ERA5 end in 2024.

## FuXi-S2S

FuXi-S2S is a daily-mean model. Its files are on 1.5° (latitude north to south), and `lead_day` L holds
the mean (or total, for `tp24`) of the UTC calendar day init + L. Lead 0 is **ERA5's mean of the start
day**, not a forecast: against IMD, lead 0 correlates at r = 0.94 with the observed start day. FuXi has
therefore seen the atmosphere up to D+1 00 UTC, and the loader uses an **effective init of D+1**
(`init_shift_days = 1`). Lead L is then step L, aligned with IMD exactly as for UC-S2S, and lead 0 is
dropped. Models are still matched on the hindcast start date D in `common` mode. FuXi indices are
verified against ERA5 calendar-day means (`*_day` variables), the same sampling as the S41
evaluation (`metrics/S41_aurora`, `time: day`).

## Drift / bias correction (`*_bc`)

`bias_correct.py` applies the S41 "dc" level (`metrics/S41_aurora/s41eval/biascorr.py`, mean level, and
`bing_eval/core.py::drift_bias`) to daily leads, separately for every grid point and lead day:

- b = mean over training starts of (ensemble mean − observation).
- The training starts are those of the **other years** within **±30 days of day of year**
  (leave-one-year-out). Pairs are NaN-aware.
- b is smoothed over leads with a centred 7-day running mean.
- Every member gets x − b; rainfall is then clipped at 0.

The reference is **IMD 2°** for rainfall (the verifying dataset) and ERA5 for the indices, sampled
like each model. Only 2019–2024 starts, which have observations, are corrected and used for training.
The output files have the input layout, so every downstream script treats `<key>_bc` as one more
model. `bc_diagnostics_*.csv` lists the CMZ bias by week before and after correction.

## Init dates and the "native" / "common" modes

The new hindcasts have a year-specific, roughly weekly schedule (e.g. 2019: 9, 13, 23, 27 May, ...;
2025–26: every Monday). The legacy loader kept only the Mon/Thu dates of the 2024 calendar
(`date_filter_year`), which would silently drop most 2020, 2021 and 2023 inits. The new loader:

- `native`: every init the model has between 1 May and 31 Jul. That is about 13 per year for
  UC-S2S and 26 for NGCM, whose inits are the 2024 Mon/Thu set in every year.
- `common`: only start dates present in **all** of UC-S2S plain, UC-S2S SST-anomaly, FuXi-S2S and NGCM
  (`config.COMPARISON_SET`). That is 48 inits: all of 2019, 2022 and 2024, 4 in 2021, 6 in 2023 and
  **none in 2020**, so 2020 drops out. This is the apples-to-apples comparison.

Both modes use the first 30 members (NGCM has 51), as was done for the earlier 20-member comparisons.

## Lead alignment (important for NGCM)

Throughout, "step k" means the 24 h total ending at init + k days, and it is compared with IMD day
init + k. `config.MODELS[...]['day_offset']` maps each file's day index onto that step.

- UC-S2S: day k is already step k (offset 0; lead 0 is dropped).
- **NGCM: day 0 is already the first 24 h** (offset 1). Against IMD, NGCM day d correlates best with
  IMD[init+d+1] (CMZ, r = 0.95, against 0.83 for IMD[init+d]). The legacy loader dropped NGCM day 0,
  so earlier NGCM onset scores were computed one day out of phase, with the first forecast day
  discarded. `ngcm_legacy` reproduces that behaviour so its effect can be measured.
  `lead_alignment.csv` (raw metrics) repeats this check for every model; step k should peak at
  offset 0.

## Raw-forecast metrics

These are computed per model and per init selection. Default inits are 1 May to 31 Jul; the
May–Sep run (`native_m56789`) excludes NGCM, whose inits stop in July.

| target | truth | metrics |
|---|---|---|
| CMZ-mean rainfall (30 cells valid in all IMD years) | IMD 2° | bias, ACC, RMSE, spread/RMSE, fair CRPS and CRPSS (daily leads 1–45 and weeks 1–6) |
| rainfall maps | IMD 2° | weekly-mean bias (%) and ACC per cell, weeks 1–6 |
| CMZ daily-rain statistics | IMD 2° | wet-day (>1 mm) frequency, onset-criterion exceedance (day 1 > 1 mm and 5-day sum > MWmean, the exact onset test), q50/q90/q99 vs lead |
| `wy, imi, llj, mhi, olr_cmz` (NGCM: `wy` only) | ERA5 1° | same scores as rainfall; lead-0 RMSE checks the initial state |
| circulation onset over Kerala (UC-S2S, FuXi-S2S) | ERA5 | LLJ-based onset, two variants (`config.LLJ_ONSET_VARIANTS`): **era5cal** (default) = first day ≥ 1 May with LLJ > 8 m/s for 6 days and a 20-day mean > 8 m/s; **wang2009** = > 6.2 m/s for 6 days. Per-init ensemble onset error, member hit rate (±3 d) and a climatological (1991–2018 median) reference; `mok_obs_<variant>.csv` compares ERA5 onset with IMD official MOK |

Conventions:

- ACC is the Pearson correlation across inits of ensemble-mean anomalies, relative to the observed
  1991–2018 day-of-year climatology (mean plus 3 harmonics).
- CRPSS is computed from the fair CRPS. The reference is the observed value for the same calendar
  window in each of 1991–2018 (28 members, all before the verification years).
- 90% CIs come from a year-block bootstrap, which is wide with six years. Read overlapping CIs as
  "not distinguishable".

## Comparison figures (`compare_models.py`)

- `fig_scorecard_onset_weekly_<mode>`: scorecards in the style of Masiwal et al. (2026,
  arXiv:2602.03767) Fig. 3a,b. Panel (a) is the onset fair BSS (%) and panel (b) the AUC, per weekly
  bin; rows are climatology plus every model. `_5day_30` is the same with the paper's 5-day bins.
- `fig_scorecard_<target>_<tag>`: weekly CRPSS (%) and ACC scorecards of the raw-forecast metrics,
  for `rain_cmz`, `wy`, `llj` and `olr_cmz`.
- `fig_onset_<mode>`: CMZ MAE/FAR/MR bars, weekly BSS/AUC lines and the reliability diagram.
- `fig_rain_cmz_*`, `fig_rain_maps_*`, `fig_rain_stats_*`, `fig_indices_*`, `fig_mok_*`: as described
  in the script header.

In every figure colour identifies the model. Solid lines, filled markers and plain bars mean raw;
dashed lines, open markers and hatched bars mean bias-corrected.

## Known data caveats

- IMD_2by2 2023–24 files have a larger land mask (106 valid cells against 62 in other years) and
  dims named `lat/lon`. The core onset code still runs, and those years add one CMZ cell (31 against
  30). The raw metrics use only cells valid in all years.
- The `mwset2x2` threshold and the climatological onset reference in the core are unchanged. That
  includes the skill-score climatology using all IMD years, the target year among them.
- LLJ onset: with Wang et al. (2009, J. Climate 22, 3303)'s 6.2 m/s, ERA5 1° onset comes 14–21 days
  before IMD MOK in 2020–22, triggered by cyclone-driven May westerlies. The 8 m/s + 20-day-mean
  variant matches IMD MOK within 0–9 days for 2019–24 (MAE 2.5 d; 1991–2018 median 3 Jun), but it was
  chosen by looking at those same years. The same rule is applied to forecasts and ERA5, so the choice
  changes which event is scored, not which model is favoured. With the 20-day condition a forecast can
  only place onset on lead ≤ 26 d.
- NGCM's last day (day 44) is a partial (~75%) accumulation and is dropped.
- UC-S2S training years are not documented in the files. If 2019–2024 were seen in training, all
  scores here are in-sample.

## Adding a model

Add an entry to `config.MODELS`: `tp_dir` with `{year}.nc`, dim names, `day_offset`, optionally
`init_shift_days`, and `n_members`. Optionally add `indices_dir` (same layout as the UC-S2S index
files, with `index_obs` giving the ERA5 sampling) or `wy_dir`. Then run `run_onset_core.py` /
`raw_forecast_metrics.py` with `--model <key>`. Add it to `COMPARISON_SET` to include it in
`common` mode, and to `BC_BASES` to get a bias-corrected copy. Always check `lead_alignment.csv`.
