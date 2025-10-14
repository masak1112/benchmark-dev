# Monsoon Onset MAE, FAR, MR Analysis for Probablistic Models

This Python script computes Mean Absolute Error (MAE), False Alarm Rate (FAR), and Miss Rate (MR) for monsoon onset predictions from probablisitc forecast (models with multiple ensemble members). It processes multiple years of model forecast data (merged and regridded as individual files for each year) and IMD (India Meteorological Department) observational data to evaluate model performance.

## Features
- **Ensemble handling**: This script looks for onset forecast of each ensemble member for a given initialization and computes the mean onset date if more than 50\% of the members have an onset. If less than 50% of the members detect onset, the forecast is considrered to have no onset for the initiliaztion. 
- **Multi-year Analysis**: Process multiple years of data simultaneously
- **Spatial Metrics**: Generate spatial maps of MAE, FAR, and Miss Rate
- **Flexible Windows**: Configure verification and forecast windows (1-15 day and 16-30 day)
- **MOK Filtering**: Optional filtering based on climatological Monsoon Onset Kerala (MOK) date (June 2nd)
- **NetCDF Output**: Save results in standard NetCDF format and plots the spatial maps of MAE,FAR, and MR

## Requirements

### Python Dependencies
```bash
pip install numpy xarray pandas matplotlib geopandas argparse pathlib
```

### Required Data Files
1. **Model Data**: NetCDF files with precipitation forecasts
   - Both the forecast and ground truth (IMD data and threshold) needs to be on the same grid. This generally requires regridding the forecast output to the IMD grid. This can be done using **CDO** : `cdo rempacon,<gridfile.txt> <input_file.nc> <output_file.nc>`  
   - Format: `{year}.nc`
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


## Command Line Arguments

### Required Arguments
| Argument | Description |
|----------|-------------|
| `--years` | Years to process (e.g., `2019 2020 2021`) |
| `--model_forecast_dir` | Directory containing model forecast data  files in the right format (should have the same resolution as IMD data)|
| `--imd_folder` | Directory containing IMD rainfall data files|
| `--thres_file` | Path to threshold NetCDF file (should have the same resolution as IMD data)|
| `--shpfile_path` | Path to India shapefile |

### Optional Arguments
| Argument | Default | Description | Typical Value | 
|----------|---------|-------------|---------------|
| `--tolerance_days` | 3 | Tolerance in days for onset prediction | 3 for 1-15 day; 5 for 16-30 day forecast |
| `--verification_window` | 1 | Days after initialization to start validation window | 1 for 1-15 day; 16 for 16-30 day forecast |
| `--forecast_days` | 15 | Length of forecast window in days | 15 for 1-15 day; 30 for 16-30 day forecast |
| `--max_forecast_day` | 15 | Maximum forecast day to consider for onset | 15 for 1-15 day; 30 for 16-30 day forecast |
| `--mok` | False | Use MOK date filter (June 2nd) - flag, no value needed | - |
| `--output_file` | `spatial_metrics.nc` | Output NetCDF file name | -|
| `--plot_dir` | None | Directory to save plot PNG file (optional) | - |
| `--figsize` | `18 6` | Figure size in inches [width height] | - |

## Methodology

### Onset Detection Criteria 
- Partially based on Moron and Robertson, 2013 (https://doi.org/10.1002/joc.3745)
- **First Day Condition**: Rainfall > 1 mm on the first day
- **5-Day Sum Condition**: Total rainfall over 5 consecutive days > threshold
- **MOK Filter** (optional, this is a modification to avoid false early onset that are followed by dry spell): Only count onsets occurring on or after June 2nd
- **Ensemble handling**: TOnset forecast of each ensemble member for a given initialization is examined for onset. If onset is detected for 50% or more members, the mean onset date of all the members is considrered as the onset for that initialization. If less than 50% of the members detect onset, the forecast is considrered to have no onset.
### Verification Windows
- **1-15 Day Window**: Extended-range forecasing window
  - `--verification_window 1 --forecast_days 15 --tolerance_days 3`
- **16-30 Day Window**: Subseasonal forecasting window  
  - `--verification_window 16 --forecast_days 30 --tolerance_days 5`

### Contingency Matrix and Metrics
- **True Positive (TP)**: Model predicts onset within tolerance of observed onset
- **False Positive (FP)**: Model predicts onset in the forecast window but outside the tolerance range
- **False Negative (FN)**: Observed onset in forecast window but model doesn't predict
- **True Negative (TN)**: No onset observed or predicted

**Calculated Metrics:**
- **MAE**: Mean Absolute Error between predicted and observed onset dates (days)
- **FAR**: False Alarm Rate = FP / (FP + TN) × 100%
- **Miss Rate**: FN / (Total observed onsets) × 100%

## Output Files

### NetCDF File Structure
```
Dimensions:
    lat: N
    lon: M

Variables:
    false_alarm_rate(lat, lon): False Alarm Rate (0-1)
    miss_rate(lat, lon): Miss Rate (0-1) 
    mean_mae(lat, lon): Mean MAE across all years (days)
    mae_YYYY(lat, lon): Individual MAE for year YYYY (days)

Global Attributes:
    title: "Monsoon Onset MAE, FAR, MR Analysis"
    years: "[2019, 2020, 2021, ...]"
    tolerance_days: 5
    verification_window: 16
    forecast_days: 30
    mok_filter: 1 (0=False, 1=True)
```

### Plot File (Optional)
If `--plot_dir` is specified, generates a PNG file with:
- **Three-panel spatial plot**: MAE, False Alarm Rate, Miss Rate
- **India outline**: Country boundaries from shapefile
- **Core Monsoon Zone (CMZ) polygon**: Automatically detected based on grid resolution
- **Grid values**: Numerical values displayed on each grid cell
- **CMZ statistics**: Regional averages displayed in bottom-right corner
- **Filename format**: `spatial_metrics_{years}_{window}_{MOK_status}.png`

## Core Monsoon Zone (CMZ) Definition

The script automatically detects grid resolution and defines CMZ polygon:

- **2-degree resolution**: 16-point polygon
- **4-degree resolution**: 8-point polygon
- **1-degree resolution**: 20-point polygon from Rajeevan et al., 2010 (https://doi.org/10.1007/s12040-010-0019-4) 
- **Other resolutions**: No CMZ polygon (plots without regional statistics)

## Examples

### Standard 1-15 Day Evaluation
```bash
python mae_far_mr_probabilistic_models.py \
    --years 2019 2020 2021 \
    --model_forecast_dir ../../model_forecast_data/ngcm51/climatology/tp_2p0 \
    --imd_folder ../../imd_rainfall_data/2p0 \
    --thres_file ../../imd_onset_threshold/mwset2x2.nc4 \
    --shpfile_path ../../ind_map_shpfile/india_shapefile.shp \
    --tolerance_days 3 \
    --verification_window 1 \
    --forecast_days 15 \
    --mok \
    --output_file ./output/results_1-15day_MOK.nc \
    --plot_dir ./output/plots
```

### Extended 16-30 Day Evaluation
```bash
python mae_far_mr_probabilistic_models.py \
    --years 2019 2020 2021 2022 2023 2024 \
    --model_forecast_dir ../../model_forecast_data/ngcm51/climatology/tp_2p0 \
    --imd_folder ../../imd_rainfall_data/2p0 \
    --thres_file ../../imd_onset_threshold/mwset2x2.nc4 \
    --shpfile_path ../../ind_map_shpfile/india_shapefile.shp \
    --tolerance_days 5 \
    --verification_window 16 \
    --forecast_days 30 \
    --mok \
    --output_file ./output/results_16-30day_MOK.nc \
    --plot_dir /output/plots
```

### No MOK Filter
```bash
python mae_far_mr_probabilistic_models.py \
    --years 2019 2020 2021 \
    --model_forecast_dir ../../model_forecast_data/ngcm51/climatology/tp_2p0 \
    --imd_folder ../../imd_rainfall_data/2p0 \
    --thres_file ../../imd_onset_threshold/mwset2x2.nc4 \
    --shpfile_path ../../ind_map_shpfile/india_shapefile.shp \
    --tolerance_days 3 \
    --verification_window 1 \
    --forecast_days 15 \
    --output_file ./output/results_1-15day_noMOK.nc
```

## Console Output

The script provides comprehensive progress information:

```
Processing years: [2019, 2020, 2021]
Model forecast data directory: /path/to/data
...
==================================================
Processing year 2019
==================================================
Loading IMD rainfall from: /path/to/imd/data_2019.nc
Using MOK date (June 2nd) as start date for onset detection
Processing 26 init times x 9 lats x 11 lons...

Processing Summary:
Total potential initializations: 2574
Skipped (no observed onset): 234
Skipped (initialized after observed onset): 567
Valid initializations processed: 1773
Onsets found: 432
Onset rate: 0.244

=== CORE MONSOON ZONE (CMZ) AVERAGES ===
CMZ Mean MAE (avg across years): 4.2 ± 0.8 days
CMZ False Alarm Rate: 23.5 %
CMZ Miss Rate: 31.2 %

=== SUMMARY STATISTICS ===
Mean MAE: 5.1 days
Mean FAR: 28.3%
Mean Miss Rate: 34.7%
```

## Data Processing Notes

- **Twice-weekly Filtering**: Only Mondays and Thursdays from May-July are processed
- **Temporal Filtering**: Only forecasts initialized before observed onset are evaluated


## File Naming Conventions

**Output NetCDF**: User-specified via `--output_file`

**Plot PNG**: Auto-generated as `spatial_metrics_{min_year}-{max_year}_{verification_window}-{forecast_days}day_{MOK|noMOK}.png`

Examples:
- `spatial_metrics_2019-2024_1-15day_MOK.png`
- `spatial_metrics_2019-2024_16-30day_noMOK.png`


## MOK Flag Usage

The `--mok` argument is a boolean flag:

 **Correct usage:**
```bash
# To enable MOK filtering (True)
--mok

# To disable MOK filtering (False) 
# (simply omit the --mok flag)
```


## Script Files

- `mae_far_mr_deterministic_models.py`: Main analysis script
