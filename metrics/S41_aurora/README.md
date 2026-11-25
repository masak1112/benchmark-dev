# S41_aurora — evaluation of the UC-S2S hindcasts (SST-anomaly and plain) and FuXi-S2S, as run on Aurora (Nov 2026)

A self-contained copy of the evaluation used for the 2026-11 UC-S2S assessment (Jhayron's s2storm repository, see `PROVENANCE.md`).
Nothing outside `metrics/S41_aurora/` and `loaders/format_s41.py` was added or changed in this repository.

## What is here
| Path | What |
|---|---|
| `s41eval/bing_eval/` | the config-driven, resumable evaluation package: readers (`data.py`), metrics and grids (`core.py`), pipeline (`pipeline.py`), paired statistics (`stats.py`), scorecards over weeks and system pairs (`scorecard.py`), extremes / RPSS (`extremes.py`), FDR difference maps (`diffmaps.py`), summary writer (`report.py`) |
| `s41eval/bing_modes.py`, `bing_waves.py`, `bing_helperc.py` | modes of variability (MJO RMM, NAO / PNA / AO / SAM, ENSO, stratosphere, land), Wheeler–Kiladis wave spectra and filtered waves, windows of opportunity |
| `s41eval/bing_weak.py` | weaknesses and ensemble quality: drift, bias, vertical profiles, QBO, precipitation distribution, spectra, rank histograms, reliability, members needed, member-offset diagnostic |
| `s41eval/bing_scores.py`, `biascorr.py`, `sci/clim_trend.py`, `hindcast_grid.py` | shared helpers: v0 scorer and reference ensembles, LOYO training sets, trend-adjusted climatology fit, the 1° → 1.5° conservative remap |
| `scripts/` | command-line entry points (`bing_eval.py` = pipeline; `bing_headline_mjo.py`, `bing_response_test.py`, `bing_w7_extras.py`, `bing_ablation_*`, `bing_reseed_compare.py`, `bing_helperA*.py`, `bing_weak*.py`, `bing_modes*.py`, `bing_waves.py`, `helperc_c0*.py`, figure scripts) |
| `configs/bing_eval/aurora.yaml`, `derecho.yaml` | site configs (every path the package reads or writes; Derecho is a template — edit the `EDIT` lines) |
| `tests/` | CPU tests (synthetic metric tests + data checks; see "Tests") |
| `../../loaders/format_s41.py` | reader for the S41 per-member file layout, returning xarray (OLR sign and land/sea masks handled) |

## Metrics (what each does)
- **Fair CRPS / CRPSS** (Ferro 2014 fair estimator): CRPSS = 1 − ΣCRPS(forecast) / ΣCRPS(reference), area-weighted sums over the region
  and starts. The reference is a 30-member **climatological ensemble of ERA5 on the same calendar date** (see References), so forecast
  and reference are scored the same way. Also with the spread inflated to spread/error = 1 (`crpss_inflated`) to separate calibration
  from resolution, and with random 11-member draws (comparability with 11-member IFS reforecasts).
- **ACC**: **centred** anomaly correlation per grid point across starts (the default; anomalies are drift-corrected forecast minus the
  reference climatology and ERA5 minus the same climatology, the mean over starts removed), then area-averaged. `acc_pattern` is the
  **uncentred** spatial pattern correlation per start (anomalies not centred over the area), averaged over starts. They answer different
  questions (temporal skill at a point vs pattern skill on a map); do not mix them.
- **RMSE, bias, spread, spread/error** (spread/error = sqrt(ensemble variance · (M+1)/M) / RMSE of the ensemble mean).
- **Tercile RPSS** (debiased, Weigel et al. 2007) and **BSS for weekly 10th/90th-percentile events** (thresholds from the reference
  ensembles; `extremes.py`).
- **Rank histograms** (cos-lat weighted), **reliability** (terciles, weeks 3–4), **members-needed** curves.
- **MJO**: RMM from 15°S–15°N OLR, u850, u200 (EOFs from ERA5 1979–2018, Wheeler & Hendon 2004 normalisation, one fixed 37.5°
  rotation to the BoM convention — provisional, identical for all systems); bivariate COR, RMSE, amplitude and phase error, skill day
  (COR = 0.5). u200 for UC-S2S / FuXi = log-p interpolation of u150 and u250.
- **Teleconnections**: NAO / PNA / AO / SAM weekly indices (rotated EOFs of ERA5 1979–2018 monthly z500), COR and skill days vs ERA5
  persistence. The NAO reference test against CPC passes only in DJF ("test failed" outside DJF).
- **Waves**: Wheeler–Kiladis symmetric/antisymmetric space–time spectra of 15°S–15°N OLR and u850, forecast/ERA5 power ratios in the MJO,
  Kelvin, ER and MRG bands, band variance by lead window, filtered-wave skill.
- **Paired comparisons** (`stats.py`, `scorecard.py`): system A − system B per start on common starts, **moving-block bootstrap over starts
  in date order (block 4, 1,000 draws), 95 % percentile CI**.

## Conventions
- **Grid:** forecasts and ERA5 on a common **conservative 1.5°** grid (WeatherBench2 121 × 240, lat −90…90 ascending); 3° sensitivity.
  UC-S2S native 1° fields are masked outside their ERA5 domain before the remap (UC-S2S fills land SST and sea stl1 with 270 K and sea swvl1
  with 0; without the mask the fill leaks into coastal cells).
- **Windows:** week 1 = days 1–7 … week 6 = days 36–42; w34 = days 15–28, w56 = days 29–42. Verification sampling follows the model:
  UC-S2S 00 UTC states (daily means for fluxes), FuXi daily means.
- **Drift correction:** the lead-dependent mean error (per grid point, window and system) is removed **leave-one-year-out**, estimated from
  the starts of the other years within **±30 days of calendar date**. Scores are "raw" or "dc" (drift-corrected); conclusions use dc.
- **References:** (a) **fixed 1991–2020** climatology; (b) **trend-adjusted** climatology (harmonic seasonal cycle + linear trend fitted
  1991–2025, evaluated at the target year). Both are 30-member same-calendar-date ERA5 ensembles. Report both: the trend explains much of the
  skill against (a).
- **Grades:** **robust** = 95 % CI of the pooled mean difference excludes 0 **and** the mean difference has the same sign in both halves
  (2019–24 and 2025–26; for 2025–26-only experiments: 2025 and 2026); **suggestive** = CI excludes 0 only; **noise** = otherwise.
  Only comparisons fixed before computing are "pre-registered"; everything else is exploratory.
- **OLR** is positive up. The h25 files store ERA5's net TOA long-wave flux (negative); readers flip the sign. UC-S2S's saved OLR is the
  00 UTC hourly-mean quantity, so it has ≈ 2.5× more power than a daily mean at periods < 3 days.

## Caveats of the Aurora hindcasts
- **Seeds:** the Aurora runs used one start per run, so member m had the same random seed at every start (upstream seeds
  `100000 * batch_idx + m`, `batch_idx` = the date's index within one run). Members therefore carry start-independent offset patterns.
  A rerun with per-start seeds (89 starts × 10 members) left spread and skill unchanged except absolute SST skill (≈ 0.02 CRPSS lower with
  independent seeds). Paired comparisons with shared seeds are unaffected.
- 2026-09-21 was run with a zero 2026-11-04 boundary file (lead days ≥ 44 suspect; scores use ≤ 42); 2026-09-28 not run yet.

## How to run
Every command reads one site config; outputs go to `<out_root>/<tag>/`. Tasks are independent and resumable (a task is skipped when its
`done` marker exists). On Aurora they were run with a simple MPI task farm (`s2storm.aurora.taskfarm`, not included); any farm, a job
array, or `bing_eval.py local` (sequential) works.

```bash
cd metrics/S41_aurora
export PYTHONPATH=$PWD                                  # makes `s41eval` importable
CFG=configs/bing_eval/derecho.yaml                      # or aurora.yaml

# 0. ERA5 staging (login / data node, needs internet): WB2 + ARCO -> the 1.5-degree verification store and references.
#    The staging and modes scripts take their root from S2STORM_ROOT (default: the Aurora path), not from the config.
export S2STORM_ROOT=/glade/derecho/scratch/$USER/s2storm      # same as `root` in the config
python scripts/bing_eval_stage.py verif --workers 16 [--first 2019-01-01 --last 2026-11-04]
python scripts/bing_eval_stage.py clim
python scripts/bing_eval_stage.py lsm
# 1. task lists, then run them (phase 1 = member reduction + verification, phase 2 = scores + checks)
python scripts/bing_eval.py tasks --config $CFG --tag v1 --phase 1
python scripts/bing_eval.py tasks --config $CFG --tag v1 --phase 2
python scripts/bing_eval.py local --config $CFG --tag v1 --phase 1      # sequential; or feed tasks_phase1.jsonl to a farm / PBS array
python scripts/bing_eval.py local --config $CFG --tag v1 --phase 2
# 2. collect: scores.csv / scores.nc, fail_summary.csv, provenance.json, summary.md
python scripts/bing_eval.py collect --config $CFG --tag v1
# 3. paired headline (A - B, weeks 3-4, both references, CIs, grades) and its MJO part
python scripts/bing_eval.py headline --config $CFG --tag v1 --a ucs2s_sstanom --b ucs2s_plain [--grid g30]
python scripts/bing_headline_mjo.py --modes <out_root>/modes --a ucs2s_sstanom --b ucs2s_plain --out headline_mjo.csv
```
- **Derecho (PBS, NCAR):** copy `configs/bing_eval/derecho.yaml`, set `root`, `python`, `out_root` and the system `path` templates;
  run phases with a CPU job, e.g. `qsub -A <proj> -q main -l select=1:ncpus=128:mem=200GB -l walltime=02:00:00` running
  `bing_eval.py local ...` (or a job array over the lines of `tasks_phase*.jsonl`). Memory: score tasks hold all starts × members of one
  field (≈ 10–30 GB for 400 starts × 30 members at 1.5°).
- **Aurora (PBS, ALCF):** `module load frameworks/2026.1.0`; CPU task-farm jobs on `debug` / `debug-scaling` (12–24 ranks per node);
  `HDF5_USE_FILE_LOCKING=FALSE` on Lustre.
- Modes, waves and weaknesses: `scripts/bing_modes.py` / `bing_modes_score.py` / `bing_modes_figs.py`, `scripts/bing_waves.py`,
  `scripts/bing_weak.py` (see each script's `--help`; they take the root from `S2STORM_ROOT`, except `s41eval/bing_weak.py` and `s41eval/bing_helperc.py`,
  which hard-code the Aurora root in their `R = ...` line: edit it on another site).
- Reading member files from other code: `loaders/format_s41.py` (`open_start(system, start, fields, regrid='1p5')`).

## Tests
`python -m pytest -q tests` (CPU, ≈ 4 min). On Aurora (2026-11-12): **54 passed, 2 failed**. The two failures are the documented
reference tests in `test_bing_modes.py`, which fail as specified: MJO vs BoM RMM (r 0.938 / 0.968 and 79 % phase match vs the required 0.95
/ 90 %; accepted provisionally with the fixed rotation) and NAO vs CPC (all-month r 0.745, DJF 0.882 vs 0.9). Tests that read data skip
or fail without the Aurora files; the synthetic metric tests (perfect forecast, fair-CRPS identities, bootstrap, spectra with prescribed
Kelvin / ER / MRG waves, BSS/RPSS) need no data.
