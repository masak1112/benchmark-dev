# Monsoon Onset Probabilistic Skill Score Analysis

## Overview

This Python script (`binned_skill_score_cmz.py`) performs comprehensive probabilistic skill score analysis for monsoon onset forecasts, aggregated across the core monsoon zone grids, compared against IMD (India Meteorological Department) observations and climatological baselines.

## Features

- **Probabilistic Skill Assessment**: Evaluates forecast model performance using multiple metrics including Brier Score, Ranked Probability Score (RPS), and area under the Receiver Operating Characteristic curve (AUC)
- **Climatological Comparison**: Computes skill scores relative to climatological forecasts treating ground turth of individual years avaialble as ensembles
- **Core Monsoon Zone Analysis**: Focuses analysis on the Indian Core Monsoon Zone using predefined polygon coordinates
- **Multi-Year Processing**: Handles multiple years of data for robust statistical analysis
- **Binned Forecast Verification**: Organizes forecasts into 5-day bins bins

### Required Data Files
1. **Model Data**: NetCDF files with precipitation forecasts
   - Both the forecast and ground truth (IMD data and threshold) needs to be on the same grid. This generally requires regridding the forecast output to the IMD grid. This can be done using **CDO** : `cdo rempacon,<gridfile.txt> <input_file.nc> <output_file.nc>`  
   - Format: `{year}.nc` or specify the yearly file name format with the input `file_format`
   - Variables: `tp` (total precipitation) daily precip in mm
   - Dimensions: `init_time/time`:intialization date in datetime64[ns], `day`/`step`: forecast step (0-35 in days) in int64, `lat`, `lon`, `number/member`:ensemble member number (1,2,3...n, where n is the total number of ensemble members) in int64 

2. **IMD Rainfall Data**: NetCDF files with observed rainfall
   - Format: `data_{year}.nc` or `{year}.nc`
   - Variables: `RAINFALL`
   - Dimensions: `lat`/`latitude`, `lon`/`longitude`, `time`/`TIME`

3. **Threshold File**: NetCDF file with onset thresholds
   - Mean wet spell threshold based on Moron and Robertson (2023). This file can be generated using `imd_onset_threshold.py` file
   - Variables: `MWmean` 
   - Dimensions: `lat`, `lon`

4. **India Shapefile**: Shapefile for country boundaries
   - Format: `.shp` with associated files (`.shx`, `.dbf`, etc.)

## Monsoon Onset Detection Methodology

### Onset Criteria
The script uses a **5-day wet spell criterion**:
1. **First day condition**: Daily rainfall > 1 mm
2. **Cumulative condition**: 5-day rolling sum > local threshold
3. **Date filter**: Optional MOK (Monsoon Onset over Kerala) filter starting June 2nd

### Spatial Focus
Analysis is restricted to the **Core Monsoon Zone (CMZ)** with resolution-dependent polygon coordinates:
- **1-degree resolution**: 22 vertices defining detailed CMZ boundary
- **2-degree resolution**: 17 vertices for coarser grid
- **4-degree resolution**: 9 vertices for very coarse grid

## Forecast Verification Framework

### Time Bins
Forecasts are organized into day bins for verification:
- **15-day forecasts**: [(1,5), (6,10), (11,15)]
- **30-day forecasts**: [(1,5), (6,10), (11,15), (16,20), (21,25), (26,30)]


### Probabilistic Approach
- Ensemble members provide probability estimates for each time bin
- Binary verification (onset/no-onset) for each forecast-observation pair

## Skill Score Metrics

### 1. Brier Score (BS)
- **Formula**: BS = (1/n) × Σ(forecast_prob - observed)²
- **Fair Brier Score**: Adjusts for finite ensemble size
- **Range**: 0 (perfect) to 1 (worst)

### 2. Ranked Probability Score (RPS)
- **Purpose**: Evaluates multi-category probabilistic forecasts
- **Fair RPS**: Ensemble-size adjusted version
- **Accounts for**: Cumulative probability differences across all bins

### 3. Area Under Curve (AUC)
- **Method**: Mann-Whitney U statistic approach
- **Range**: 0.5 (no skill) to 1.0 (perfect discrimination)
- **Interpretation**: Probability of correctly ranking onset/no-onset cases

### 4. Skill Scores
- **Brier Skill Score**: BSS = 1 - (BS_forecast / BS_climatology)
- **RPS Skill Score**: RPSS = 1 - (RPS_forecast / RPS_climatology)
- **Interpretation**: >0 indicates forecast better than climatology

## Climatological Baseline

### Construction Method
1. **Historical Dataset**: Computes onset dates for all available years in IMD folder
2. **Ensemble Approach**: Each historical year serves as an ensemble member
3. **Day-of-Year Matching**: Uses Julian day comparisons for seasonal alignment
4. **Probability Calculation**: Based on historical frequency of onset in each time bin


## Usage

### Command Line Interface
```bash
python binned_skill_score_cmz.py \
    --years 2019 2020 2021 2022 2023 2024 \
    --model_forecast_dir ../../model_forecast_data/ngcm51/climatology/tp_2p0 \
    --imd_folder ../../imd_rainfall_data/2p0 \
    --thres_file ../../imd_onset_threshold/mwset2x2.nc4 \
    --mem_num 51 \
    --file_pattern "{}.nc" \
    --max_forecast_day 15 \
    --model_name abc \
    --mok
```

### Key Parameters
- `--years`: Years to process for analysis
- `--model_forecast_dir`: Directory containing S2S model NetCDF files
- `--imd_folder`: Directory with IMD rainfall observations
- `--thres_file`: Path to rainfall threshold NetCDF file
- `--mem_num`: Number of ensemble members to use 
- `file_pattern`: Naming convention of individual forecast file
- `--max_forecast_day`: Maximum forecast horizon (15 or 30 days)
- `--model_name`: Model identifier for output file naming
- `--mok/--no_mok`: Enable/disable MOK date filter

## Output Files

### 1. Overall Skill Scores CSV
**Filename**: `overall_skill_scores_{model_name}_{max_forecast_day}day.csv`

Contains:
- AUC 
- Fair Brier Score 
- Fair Brier Skill Score (improvement over climatology)
- Fair RPS
- Fair RPS Skill Score (improvement over climatology)

### 2. Binned Skill Scores CSV
**Filename**: `binned_skill_scores_{model_name}_{max_forecast_day}day.csv`

Contains bin-wise metrics:
- Fair Brier Skill Score for each time bin
- AUC for each time bin
- Fair Brier Scores (forecast and climatology)

### 3. Skill Score Heatmap
**Filename**: `skill_scores_heatmap_{model_name}_{max_forecast_day}day.png`

Visual representation:
- **Upper panel**: Brier Skill Score heatmap (%)
- **Lower panel**: AUC values with climatological comparison


## Dependencies

### Required Python Packages
```python
numpy
xarray  
pandas
matplotlib
scipy
seaborn
argparse
glob
pathlib
datetime
```
