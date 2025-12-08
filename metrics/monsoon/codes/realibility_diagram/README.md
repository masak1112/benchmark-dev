# Monsoon Onset Reliability Analysis

A Python script for evaluating the reliability of probabilsitic monsoon onset forecast compared against IMD (India Meteorological Department) observations over the Core Monsoon Zone (CMZ).

## Overview

This tool performs reliability analysis by comparing ensemble forecast probabilities with observed monsoon onset events, generating reliability diagrams that show how well forecast probabilities correspond to actual occurrence frequencies.

## Features

- **Multi-year Analysis**: Process multiple years of data in a single run
- **Forecast Horizons**: 5-day forecast bins for 1-15-day forecast and 16-30-day forecasts windows
- **MOK Filter Control**: Option to enable/disable Monsoon Onset over Kerala (MOK) date filtering
- **Output**: Generates both visual reliability diagrams and detailed CSV results

## Required Data Files
1. **Model Data**: NetCDF files with precipitation forecasts
   - Both the forecast and ground truth (IMD data and threshold) needs to be on the same grid. This generally requires regridding the forecast output to the IMD grid. This can be done using **CDO** : `cdo rempacon,<gridfile.txt> <input_file.nc> <output_file.nc>`  
   - Format: `{year}.nc`
   - Variables: `tp` (total precipitation) daily precip in mm
   - Dimensions: `init_time/time`:initialization date in datetime64[ns], `day`/`step`: forecast step (0-35 in days) in int64, `lat`, `lon`, `number/member`:ensemble member number (1,2,3...n, where n is the total number of ensemble members) in int64 

2. **IMD Rainfall Data**: NetCDF files with observed rainfall
   - Format: `data_{year}.nc` or `{year}.nc`
   - Variables: `RAINFALL`
   - Dimensions: `lat`/`latitude`, `lon`/`longitude`, `time`/`TIME`
3. **Threshold Data**: NetCDF file 
   - Variables: `MWmean`: Monsoon onset threshold values
   - Dimensions: `lat`, `lon`
   - Resolution: Must match IMD data resolution

## Usage

### Command Line Interface

```bash
# With MOK filter (default)
python onset_reliability_diagram_cmz.py \
    --model_forecast_dir "../../model_forecast_data/ngcm51/climatology/tp_2p0" \
    --imd_folder "../../imd_rainfall_data/2p0" \
    --thres_file "../../imd_onset_threshold/mwset2x2.nc4" \
    --mem_num 51\
    --max_forecast_day 15 \
    --save_path "./output" \
    --years 2019 2020 2021 2022 2023 2024 \
    --file_pattern "{}.nc" \
    --mok

# Without MOK filter
python onset_reliability_diagram_cmz.py \
    --model_forecast_dir "../../model_forecast_data/ngcm51/climatology/tp_2p0" \
    --imd_folder "../../imd_rainfall_data/2p0" \
    --thres_file "../../imd_onset_threshold/mwset2x2.nc4" \
    --mem_num 51 \
    --max_forecast_day 15 \
    --save_path "./output" \
    --years 2019 2020 2021 2022 2023 2024 \
    --file_pattern "{}.nc" \
    --no-mok
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--model_forecast_dir` | str | Yes | - | Directory containing model forecast data files  |
| `--imd_folder` | str | Yes | - | Directory containing IMD rainfall NetCDF files |
| `--thres_file` | str | Yes | - | Path to monsoon onset threshold NetCDF file |
| `--mem_num` | int | Yes | - |  Number of ensemble members to be used (should be less than or equal to the total number of memebers avaialbe in the forecast) |
| `--save_path` | str | Yes | - | Output directory for results and figures |
| `--years` | list | Yes | - | Years to process (space-separated integers) |
| `--max_forecast_day` | int | No | 15 | Maximum forecast day (15 for 1-15 day forecast realiability or 30 for 16-30 day forecast reliability) |
| `--file_pattern` | str | No | `{}.nc` | File naming pattern for forecast files (single file for each year) |
| `--mok` | flag | No | True | Enable MOK filter (onset after June 2nd only) |
| `--no-mok` | flag | No | False | Disable MOK filter (allow onset from May 1st) |

### Day Bins

The script automatically sets day bins based on `max_forecast_day`:
- **15-day forecasts**: Days 1-5, 6-10, 11-15
- **30-day forecasts**: Days 16-20, 21-25, 26-30

### MOK Filter Options

The script provides two modes for onset detection:

1. **MOK Filter Enabled** (`--mok`, default):
   - Only considers onset events occurring after June 2nd (traditional MOK date)
   - Uses "MOK (June 2nd filter)" methodology
   - More conservative, focuses on post-MOK onset events

2. **MOK Filter Disabled** (`--no-mok`):
   - Considers onset events from May 1st onwards
   - Uses "no date filter" methodology  
   - Captures earlier monsoon onset events but susceptible  to false onset

## Supported Resolutions

The script automatically detects data resolution and applies appropriate CMZ boundaries:

- **1-degree**: High-resolution polygon with 22 vertices
- **2-degree**: Medium-resolution polygon with 17 vertices  
- **4-degree**: Coarse-resolution polygon with 9 vertices

## Output Files

### 1. Reliability Diagram (PNG)
- **Filename**: `reliability_{max_forecast_day}day.png`
- **Content**: Reliability diagram with error bars and frequency histogram
- **Resolution**: 600 DPI

### 2. Reliability Results (CSV)
- **Filename**: `reliability_results_{max_forecast_day}day_mok.csv` (with MOK filter) or `reliability_results_{max_forecast_day}day_no_mok.csv` (without MOK filter)
- **Columns**:
  - `Bin_Range`: Probability bin range (e.g., "0.0-0.1")
  - `N_Forecasts`: Number of forecasts in bin
  - `Mean_Forecast_Prob`: Average forecast probability in bin
  - `Observed_Frequency`: Actual occurrence rate
  - `Frequency`: Relative frequency of forecasts in bin
  - `Error_Bar`: Confidence interval

## Methodology

### Onset Detection Algorithm
1. **Observed Onset**: Uses IMD rainfall with 5-day rolling sum criterion
2. **Forecast Onset**: Applies same criteria to ensemble members
3. **MOK Filter Options**:
   - **Enabled**: Only considers onset after June 2nd (Monsoon Onset over Kerala date)
   - **Disabled**: Considers onset from May 1st onwards

### Reliability Calculation
1. **Probability Bins**: 10 equal bins from 0.0 to 1.0
2. **Ensemble Probability**: Fraction of members predicting onset in each day bin
3. **Conditional Frequency**: Observed occurrence rate for each probability bin


## Example Output

```
Processing years: [2019, 2020, 2021]
MOK filter enabled: True

Reliability Analysis:
Bin Range		N_Forecasts	Mean_Forecast_Prob	Reliability	Frequency	Error_Bar
------------------------------------------------------------------------------------------
0.0-0.1			1256		0.000			0.045		0.423		0.006
0.1-0.2			234		0.143			0.167		0.079		0.024
0.2-0.3			187		0.251			0.273		0.063		0.033
...
```

The output filenames will include the MOK filter status for easy identification of results from different analysis modes.