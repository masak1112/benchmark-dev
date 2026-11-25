# Weekly TCC: method and how to run

Weekly temporal correlation coefficient (TCC) of the ensemble-mean forecast against ERA5,
for UC-S2S V2, FuXi, SI-S2S and NeuralGCM. Code lives in this directory.

## Method

For each model, variable and lead week:

1. **Ensemble mean.** Average all members for each init date and lead day L = 1..42.
   FuXi (1.5 deg) and NeuralGCM (2.8125 deg Gaussian) are bilinearly interpolated to the
   ERA5 1 deg grid (180 x 360). NeuralGCM rows poleward of +-87.86 take the outermost row.

2. **Weekly means.** Week k averages lead days 7k-6 .. 7k (week 1 = days 1-7,
   week 6 = days 36-42). Day 0 is never used.

   F_ik = (1/7) * sum_{L = 7k-6 .. 7k} F_i(L)

   Weeks 3-4 and 5-6 are the averages of the two weekly means.

3. **Truth (ERA5).** Matched to what each model outputs:
   - UC-S2S V2, SI-S2S (instantaneous output): ERA5 at 00 UTC of each valid day.
   - FuXi, NeuralGCM (daily-mean output): ERA5 daily mean (avg of 00/06/12/18 UTC).

   Averaged over the same 7 valid days to give O_ik.

4. **Anomalies.** Subtract the same ERA5 1979-2018 day-of-year climatology from both,
   averaged over the same 7 valid days (C_ik): the 00Z climatology (`ACC/clim_00z/`) for 00Z
   models, the daily-mean climatology (`ACC/clim_dailymean/`) for daily-mean models.

   f'_ik = F_ik - C_ik,   o'_ik = O_ik - C_ik

5. **TCC at each grid point.** Pearson correlation across the N init dates:

   TCC_k(x) = sum_i (f'_ik - <f'>)(o'_ik - <o'>)
              / sqrt( sum_i (f'_ik - <f'>)^2 * sum_i (o'_ik - <o'>)^2 )

   where <.> is the mean over inits. Any bias that is the same for every init cancels.

6. **Regional mean.** cos(latitude)-weighted average of the TCC map over
   global, tropics (30S-30N), NH (30N-90N) and SH (30S-90S).

7. **Uncertainty (block bootstrap).** Steps 5-6 are repeated on 1000 resampled init sets.
   Whole calendar months of inits are drawn with replacement, because inits a few days apart
   verify over overlapping weeks. The 95% CI is the 2.5-97.5 percentile range. All models use
   the same resamples, so (model A - model B) is a paired comparison; a difference is called
   significant when its CI excludes 0.

## Optional: month mean removed (`--monthly-mean`, tags `*_mm`)

After step 4, for each variable, lead week and grid point, subtract the sample's own mean anomaly over all
inits in the same calendar month of the init date (years pooled), separately for forecast and ERA5. This
removes seasonal residuals of the fixed 1979-2018 climatology (e.g. warming since then) and each model's
season-dependent drift. It is in-sample and it is not a linear-in-time detrend. On 2019-2021 (20 members) it
changes global TCC by at most ~0.035 and does not change the model ranking.

## Model-specific details

| Model | Members | Truth / climatology | Variables not available | Unit conversion |
|---|---|---|---|---|
| UC-S2S V2 | 20 | ERA5 00Z / `clim_00z` | OLR (written as all zeros) | none |
| SI-S2S | 56 (14 seeds x 4) | ERA5 00Z / `clim_00z` | OLR | tp: kg m-2 s-1 x 86.4 -> m/day |
| FuXi | 51 | ERA5 daily mean / `clim_dailymean` | none | tp: mm/h x 0.024 -> m/day |
| NeuralGCM | 25 | ERA5 daily mean / `clim_dailymean` | t2m, u10, v10, OLR | none (tp already m/day) |

Variables: t2m, z500, tp, u10, v10, u850, v850, OLR. Variables a model does not provide are
set to NaN and never scored.

## Date sets

Models are compared on the same init dates. Each model's "own record" (hatched bars in the
paper-style figure) shows typical skill on its own dates, not a head-to-head.

| File | Inits | Models | Tag |
|---|---|---|---|
| `common_dates_ucs2sv2_fuxi.txt` | 86 | UC-S2S V2, FuXi | `common86` |
| `common_dates_ucs2sv2_fuxi_ngcm.txt` | 72 | UC-S2S V2, FuXi, NeuralGCM | `common72` |
| `common_dates_ucs2sv2_fuxi_sis2s.txt` | 39 (May-Jul only) | UC-S2S V2, FuXi, SI-S2S | `common39` |
| `common_dates_ucs2sv2_sis2s.txt` | 104 (May-Jul only) | UC-S2S V2, SI-S2S | `si104` |
| `dates_fuxi_2017_2021.txt` | 520 | FuXi (FuXi-S2S paper test period) | `y2017_2021` |
| `dates_ngcm_2019_2021.txt` | 454 | NeuralGCM | `y2019_2021` |
| `dates_ucs2sv2_cached151.txt` | 151 (104 in May-Jul) | UC-S2S V2 | `cached151` |

### 2019-2021 own-record sets (20 members; `plot_TCC_bars_own_record.py --period 2019-2021 --mem-suffix _m20`)

| Model | Inits | Tag |
|---|---|---|
| UC-S2S V2 | 302 (`dates_ucs2sv2_2019_2021.txt`) | `ucall1921_m20` |
| FuXi | 312 (`dates_fuxi_2019_2021.txt`) | `fx1921_m20` |
| SI-S2S | 78, May-Jul only (`common_dates_ucs2sv2_sis2s_2019_2021.txt`) | `si1921_m20` |
| NeuralGCM | 454 (`dates_ngcm_2019_2021.txt`) | `y2019_2021_m20` |

All four models on the same dates: 39 inits, May-Jul 2019-2021
(`common_dates_ucs2sv2_fuxi_sis2s_ngcm.txt`, tag `common39x4_m20`).

## Files

- `tcc_weekly.py`: readers for each model, ERA5 truth, stage 1 (`extract`: weekly means per
  init, cached to `/glade/derecho/scratch/bgong/tcc_cache/{model}/`) and stage 2
  (`aggregate`: TCC maps and regional means -> `results/tcc_weekly_{model}_{tag}.nc/.pkl`).
  The core calculation is `load_anomalies()` and `pearson()`.
- `tcc_bootstrap.py`: month-block bootstrap -> `results/tcc_bootstrap_{tag}_{block}.pkl`.
- `plot_TCC.py`: all figures -> `/glade/work/bgong/plots/tcc_weekly_*` (non-default tags get
  a `_{tag}` suffix). Model colors come from `/glade/work/bgong/model_colors.py`.
- `plot_TCC_bars_own_record.py`: paper-style bars with every model on ALL of its own inits
  (UC-S2S V2 151, FuXi 520, NeuralGCM 454, optionally SI-S2S 104)
  -> `tcc_weekly_paper_style_bars_own_record.png`.
- `compute_clim_dailymean.py`: builds the daily-mean (and OLR 00Z) ERA5 climatologies.
- `job_scripts/submit_tcc.sh`: main PBS job (`qsub -v MODEL=,STAGE=,DATES=,TAG= submit_tcc.sh`).

## How to run

```bash
cd /glade/work/bgong/benchmark-dev/metrics/TCC

# 1. weekly means per init (slow part; run as a job)
qsub -v MODEL=ngcm,STAGE=extract,DATES=dates_ngcm_2019_2021.txt,TAG=y2019_2021 job_scripts/submit_tcc.sh

# 2. TCC maps + regional means
python tcc_weekly.py --model ngcm --stage aggregate --dates-file dates_ngcm_2019_2021.txt --tag y2019_2021

# 3. bootstrap CIs (paired across the listed models)
python tcc_bootstrap.py --models ucs2s_v2 fuxi ngcm \
    --dates-file common_dates_ucs2sv2_fuxi_ngcm.txt --tag common72 --block month

# 4. figures
python plot_TCC.py --models ucs2s_v2 fuxi ngcm --tag common72
```

`--models` keys: `ucs2s_v2`, `fuxi`, `si_s2s`, `ngcm`. The first model listed is the reference
in the difference plots (first minus each other).

## Caveats

- **Truth sampling.** All models use the same ERA5 files and 1979-2018 climatology period, but
  00Z models are scored against ERA5 00 UTC and daily-mean models against the ERA5 daily mean.
  The two weekly-anomaly truths correlate at 0.99 (z500), 0.98 (tp), 0.94-0.97 (t2m, winds),
  0.89 (OLR). Scoring UC-S2S V2 against FuXi's daily-mean truth instead changes its global TCC by
  -0.005 to -0.025 in week 1 and within +-0.01 at weeks 3-6 (86 common dates), much smaller than
  the UC-S2S vs FuXi differences at weeks 4-6 (0.03-0.08).
- **Ensemble size.** Ensemble-mean TCC rises with the number of members. Native sizes are 20
  (UC-S2S V2), 51 (FuXi), 56 (SI-S2S), 25 (NeuralGCM). `--n-members 20` in `tcc_weekly.py` /
  `tcc_bootstrap.py` keeps the first 20 of each (separate caches `{model}_m20`, result tags `*_m20`,
  plot option `--mem-suffix _m20`).

- **ERA5 time reference.** ERA5 t2m/u10/v10 files for 1979-2018 use
  `hours since 1959-01-01`; all other files use 1900. `era5_dates()` decodes each file from its
  own units. Scripts that assume 1900 misdate those files.
- **Significance line.** The dotted r=0.21 (n=86) line on the line plots is the usual test for a
  single correlation and assumes independent inits; it is optimistic. Use the bootstrap CIs.
- **Small date sets.** The 39-init set has only 9 month blocks, so its CIs are wide.
- **NeuralGCM initial state** differs from ERA5 already on day 1 (u850 about +2 m/s, z500 about
  +220 m2/s2). A fixed bias does not affect TCC, but it would affect ACC/RMSE.
- **FuXi vs the FuXi-S2S paper (Fig. 1).** On 2017-2021 our weeks 3-4 values are within ~0.04
  of the paper; ours decay a little more slowly (week 3 lower, weeks 5-6 higher for TP/Z500).

## Results summary (global TCC)

On the 72 inits shared by UC-S2S V2 / FuXi / NeuralGCM:

| Variable | Weeks 3-4 | Weeks 5-6 |
|---|---|---|
| z500 | 0.37 / 0.39 / 0.33 | 0.23 / 0.30 / 0.12 |
| tp | 0.22 / 0.27 / 0.16 | 0.15 / 0.22 / 0.10 |
| u850 | 0.28 / 0.30 / 0.24 | 0.16 / 0.23 / 0.13 |

Weeks 1-2 are similar across models (NeuralGCM lower for tp). From week 3 on the order is
FuXi > UC-S2S V2 > NeuralGCM; the FuXi advantage comes mostly from the tropics.
