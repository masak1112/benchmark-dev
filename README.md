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

### Derecho

#### Environment:

#### Data:
##### Verification Data:

*Data_name*: PATH

##### Inference Data:

##### Ancillary Data:

Land-sea mask: PATH

### Stampede3

#### Environment:

#### Data:
##### Verification Data:

##### Inference Data:

##### Ancillary Data:

### Midway3

#### Environment:

#### Data:
##### Verification Data:

##### Inference Data:

##### Ancillary Data:

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