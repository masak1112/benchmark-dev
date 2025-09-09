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

## HPC Clusters

### Stampede3
#### Inference Data:
The inference example can be found: `/scratch/10786/bgong1/predictions`
#### Ancillary Data:
- Land-sea mask: `/scratch/10786/bgong1/gt_data/land_sea_mask_180x360_c.nc`
- ERA5 ground truth data:  `/scratch/10786/bgong1/gt_data/2018_180x360.nc`
### Midway

#### Inference Data:`/project/pedramh/bing/PanguWeather/v2.0/HPC_scripts/results/S2S/1/predictions`
#### Ancillary Data:
- Land-sea mask: `/project/pedramh/bing/PanguWeather/v2.0/data/land_sea_mask_180x360_c.nc`
- ERA5 ground truth data: `/project/pedramh/bing/PanguWeather/v2.0/data/2018_180x360.nc`

### Derecho: 
#### Inference Data: `/glade/campaign/univ/uchi0014/bing/predictions/`
#### Ancillary Data: `/glade/campaign/univ/uchi0014/bing/gt_data`

## Metrics

### Monsoon

[PUT DOCUMENTATION HERE]

### MJO

### Power Spectrum

### CRPS

### Spread Skill Score

### Briar Skill Score

### Ranked Histogram

### QBO

### Rainfall Distribution