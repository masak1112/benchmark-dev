# Benchmarking Development Environment

## Naming & Data Conventions

### Naming

#### Probalistic: pangu_plasim_{EXPERIMENT}_24h_{N_STEPS}step_{DATE}_ens_{MEMBER}.nc

EXPERIMENT = {EXP} (1, 2, 3, ...)
N_STEPS = {N} (15, 30, 45)
DATE = {YYYYMMDDHH} (2021060806)
MEMBER = {M} (1, 2, 3, ... N)

### Data

#### Index Coords: [time: datetime64, prediction_timedelta: timedelta64, level: int32, latitude: float32, longitude: float32]

#### Coords: [time: datetime64, number: int32, prediction_timedelta: timedelta64, level: int32, latitude: float32, longitude: float32]

*Number is only a coord–not an index*

#### Chunks: {time: 1, number: -1, prediction_timedelta: 1, level: 1, latitude: -1, longitude: -1}

The S41 hindcasts (UC-S2S, FuXi-S2S) use a different per-member layout; see
`loaders/format_s41.py` and `metrics/S41_aurora/README.md`.

## HPC Clusters

### Derecho

#### Environment:

#### Data:
##### Verification Data:

*Data_name*: PATH

##### Inference Data: `/glade/campaign/univ/uchi0014/bing/predictions/`

##### Ancillary Data: `/glade/campaign/univ/uchi0014/bing/gt_data`

Land-sea mask: PATH

### Stampede3

#### Environment:

#### Data:
##### Verification Data:

- ERA5 ground truth data: `/scratch/10786/bgong1/gt_data/2018_180x360.nc`

##### Inference Data: `/scratch/10786/bgong1/predictions`

##### Ancillary Data:

- Land-sea mask: `/scratch/10786/bgong1/gt_data/land_sea_mask_180x360_c.nc`

### Midway3

#### Environment:

#### Data:
##### Verification Data:

- ERA5 ground truth data: `/project/pedramh/bing/PanguWeather/v2.0/data/2018_180x360.nc`

##### Inference Data: `/project/pedramh/bing/PanguWeather/v2.0/HPC_scripts/results/S2S/1/predictions`

##### Ancillary Data:

- Land-sea mask: `/project/pedramh/bing/PanguWeather/v2.0/data/land_sea_mask_180x360_c.nc`

## Metrics

### Monsoon

`metrics/monsoon/` evaluates S2S hindcasts of the Indian summer monsoon against IMD 2° rainfall
and ERA5. Full description: [`metrics/monsoon/README.md`](metrics/monsoon/README.md).

1. **Regridding.** Forecast precipitation is remapped conservatively (as `cdo remapcon`) to the IMD
   2° grid over India (68–100°E, 8–38°N). Monsoon circulation indices are computed from the native
   winds and OLR: Webster–Yang, Wang IMI, Somali jet, Goswami MHI, CMZ OLR.
2. **Onset benchmarking.** Local onset at each 2° cell is the first day after 2 June with more
   than 1 mm of rain and a 5-day total above the IMD threshold (Moron and Robertson 2014). Inits
   run from 1 May to 31 Jul. Scores over the Core Monsoon Zone:
   - MAE, false-alarm rate and miss rate for days 1–15 and 16–30
   - reliability
   - fair Brier, RPS and AUC skill scores in 5-day and weekly lead bins, against an IMD
     climatological ensemble
3. **General rainfall evaluation.** Daily and weekly CMZ and gridded rainfall verified against IMD:
   bias, ACC, fair CRPSS, spread/error, wet-day and extreme statistics. Circulation indices and the
   jet-based onset over Kerala are verified against ERA5.

`metrics/monsoon/codes/ucs2s_benchmark/` runs everything as one PBS chain for:

- UC-S2S (plain and SST-anomaly)
- FuXi-S2S
- drift/bias-corrected versions of those three (the S41 `biascorr.py` method)
- NGCM

### S41 / Aurora evaluation (UC-S2S, FuXi-S2S)

`metrics/S41_aurora/` holds the evaluation and inference code for the S41 UC-S2S hindcasts: fair
CRPSS, ACC, modes of variability, waves, drift and bias correction, and scorecards. See its README.

### MJO

### Power Spectrum

### CRPS

### Spread Skill Score

### Briar Skill Score

### Ranked Histogram

### QBO

### Rainfall Distribution
