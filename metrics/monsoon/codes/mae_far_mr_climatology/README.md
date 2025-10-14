# Monsoon Onset MAE, FAR, MR Analysis for Climatology Baseline

This Python script computes Mean Absolute Error (MAE), False Alarm Rate (FAR), and Miss Rate (MR) for monsoon onset predictions using climatology as a baseline forecast. It processes multiple years of IMD (India Meteorological Department) observational data to compute climatological onset dates and evaluate them against observed onset dates for each year. The climatology is treated operationally similar to model forecast where the onset is looked for in the forecasting windows (1-15 or 16-30 day) for each initialization.

## Features

- **Climatological Baseline**: Uses long-term climatological onset dates as "forecasts". These files are needed in the IMD folder
- **Spatial Metrics**: Generate spatial maps of MAE, FAR, and Miss Rate
- **Flexible Windows**: Configure verification and forecast windows (1-15 day and 16-30 day)
- **MOK Filtering**: Optional filtering based on climatological Monsoon Onset Kerala (MOK) date (June 2nd)
- **Flexible File Naming**: Handles both `data_{year}.nc` and `{year}.nc` naming conventions for IMD files
- **NetCDF Output**: Save results in standard NetCDF format and plots the spatial maps of MAE, FAR, and MR

## Requirements

### Python Dependencies
```bash
pip install numpy xarray pandas matplotlib geopandas argparse pathlib
```

### Required Data Files
1. **IMD Rainfall Data**: NetCDF files with observed rainfall
   - Format: `data_{year}.nc` or `{year}.nc`
   - Variables: `RAINFALL`
   - Dimensions: `lat`/`latitude/LATITUDE`, `lon`/`longitude/LONGITUDE`, `time`/`TIME`

2. **Threshold File**: NetCDF file with onset thresholds
   - Mean wet spell threshold based on Moron and Robertson (2023). This file can be generated using `imd_onset_threshold.py` file
   - Variables: `MWmean` 
   - Dimensions: `lat`, `lon`

3. **India Shapefile**: Shapefile for country boundaries
   - Format: `.shp` with associated files (`.shx`, `.dbf`, etc.)

## Command Line Arguments

### Required Arguments
| Argument | Description |
|----------|-------------|
| `--years` | Years to process (e.g., `2019 2020 2021`) |
| `--imd_folder` | Directory containing IMD rainfall data files |
| `--thres_file` | Path to threshold NetCDF file |
| `--shpfile_path` | Path to India shapefile |

### Optional Arguments
| Argument | Default | Description | Typical Value | 
|----------|---------|-------------|---------------|
| `--tolerance_days` | 5 | Tolerance in days for onset prediction | 3 for 1-15 day; 5 for 16-30 day forecast |
| `--verification_window` | 16 | Days after initialization to start validation window | 1 for 1-15 day; 16 for 16-30 day forecast |
| `--forecast_days` | 30 | Length of forecast window in days | 15 for 1-15 day; 30 for 16-30 day forecast |
| `--max_forecast_day` | 30 | Maximum forecast day to consider for onset | 15 for 1-15 day; 30 for 16-30 day forecast |
| `--mok` | False | Use MOK date filter (June 2nd) - flag, no value needed | - |
| `--output_file` | `climatology_spatial_metrics.nc` | Output NetCDF file name | -|
| `--plot_dir` | None | Directory to save plot PNG file (optional) | - |
| `--figsize` | `18 6` | Figure size in inches [width height] | - |

## Methodology

### Climatological Baseline Approach
1. **Climatology Computation**: 
   - Computes onset dates for all available years in the IMD folder
   - Calculates mean day-of-year for each grid point across all years
   - Uses this climatological onset date as a "forecast" for each evaluation year

2. **Forecast Generation**:
   - For each initialization date (Mondays and Thursdays from May-July)
   - Checks if climatological onset date falls within the forecast window
   - Only processes forecasts initialized before the observed onset date

### Onset Detection Criteria 
- Partially based on Moron and Robertson, 2013 (https://doi.org/10.1002/joc.3745)
- **First Day Condition**: Rainfall > 1 mm on the first day
- **5-Day Sum Condition**: Total rainfall over 5 consecutive days > threshold
- **MOK Filter** (optional): Only count onsets occurring on or after June 2nd

### Verification Windows
- **1-15 Day Window**: Extended-range forecasting window
  - `--verification_window 1 --forecast_days 15 --tolerance_days 3`
- **16-30 Day Window**: Subseasonal forecasting window  
  - `--verification_window 16 --forecast_days 30 --tolerance_days 5`

### Contingency Matrix and Metrics
- **True Positive (TP)**: Climatology predicts onset within tolerance of observed onset
- **False Positive (FP)**: Climatology predicts onset in the forecast window but outside the tolerance range
- **False Negative (FN)**: Observed onset in forecast window but climatology doesn't predict
- **True Negative (TN)**: No onset observed or predicted by climatology

**Calculated Metrics:**
- **MAE**: Mean Absolute Error between climatological and observed onset dates (days)
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
    title: "Monsoon Onset MAE, FAR, MR Analysis - Climatology Baseline"
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
- **Filename format**: `climatology_spatial_metrics_{years}_{window}_{MOK_status}.png`

## Core Monsoon Zone (CMZ) Definition

The script automatically detects grid resolution and defines CMZ polygon:

- **2-degree resolution**: 16-point polygon
- **4-degree resolution**: 8-point polygon
- **1-degree resolution**: 20-point polygon from Rajeevan et al., 2010 (https://doi.org/10.1007/s12040-010-0019-4) 
- **Other resolutions**: No CMZ polygon (plots without regional statistics)

## Examples

### Standard 1-15 Day Evaluation with Multiple Years
```bash
python mae_far_mr_climatology.py \
    --years 2019 2020 2021 \
    --imd_folder ../../imd_rainfall_data/2p0 \
    --thres_file ../../imd_onset_threshold/mwset2x2.nc4 \
    --shpfile_path ../../ind_map_shpfile/india_shapefile.shp \
    --tolerance_days 3 \
    --verification_window 1 \
    --forecast_days 15 \
    --max_forecast_day 15 \
    --mok \
    --output_file ./output/climatology_1-15day_MOK.nc \
    --plot_dir ./output/plots
```

### Extended 16-30 Day Evaluation
```bash
python mae_far_mr_climatology.py \
    --years 2019 2020 2021 2022 2023 2024 \
    --imd_folder ../../imd_rainfall_data/2p0 \
    --thres_file ../../imd_onset_threshold/mwset2x2.nc4 \
    --shpfile_path ../../ind_map_shpfile/india_shapefile.shp \
    --tolerance_days 5 \
    --verification_window 16 \
    --forecast_days 30 \
    --max_forecast_day 30 \
    --mok \
    --output_file ./output/climatology_16-30day_MOK.nc \
    --plot_dir ./output/plots
```


### No MOK Filter
```bash
python mae_far_mr_climatology.py \
    --years 2019 2020 2021 \
    --imd_folder ../../imd_rainfall_data/2p0 \
    --thres_file ../../imd_onset_threshold/mwset2x2.nc4 \
    --shpfile_path ../../ind_map_shpfile/india_shapefile.shp \
    --tolerance_days 3 \
    --verification_window 1 \
    --forecast_days 15 \
    --max_forecast_day 15 \
    --output_file ./output/climatology_1-15day_noMOK.nc
```

## Console Output

The script provides comprehensive progress information:

```
Processing years: [2019, 2020, 2021]
IMD folder: /data/imd
Computing climatological onset reference...
Computing climatological onset from 40 years: 1985-2024
Climatological onset computed from 40 valid years

==================================================
Evaluating climatology baseline for year 2019
==================================================
Loading IMD rainfall from: /data/imd/data_2019.nc
Using MOK date (June 2nd) as start date for onset detection
Processing climatology as forecast for 26 init times x 9 lats x 11 lons...
Only processing forecasts initialized before observed onset dates

Climatology Forecast Summary:
Total potential initializations: 2574
Skipped (no observed onset): 234
Skipped (initialized after observed onset): 567
Valid initializations processed: 1773
Onsets forecasted: 432
Forecast rate: 0.244

Year 2019 completed. Grid points processed: 89
Summary stats: TP=156, FP=87, FN=123, TN=1407

=== CORE MONSOON ZONE (CMZ) AVERAGES ===
CMZ Mean MAE (avg across years): 6.8 ± 1.2 days
CMZ False Alarm Rate: 35.4 %
CMZ Miss Rate: 44.1 %

=== SUMMARY STATISTICS ===
Mean MAE: 7.3 days
Mean FAR: 38.7%
Mean Miss Rate: 46.2%
```

## Data Processing Notes

- **Twice-weekly Filtering**: Only Mondays and Thursdays from May-July are processed (matching model forecast schedule)
- **Temporal Filtering**: Only forecasts initialized before observed 
- **Climatology Years**: Uses all available years in the IMD folder to compute climatology (typically 1901-2024)

## Passing Multiple Years

You can specify multiple years in several ways:

### Method 1: List individual years
```bash
--years 2019 2020 2021 2022 2023 2024
```

### Method 2: Use shell expansion (bash/zsh)
```bash
--years {1985..2024}
```

### Method 3: Use seq command
```bash
--years $(seq 1985 2024)
```

## File Naming Conventions

**Output NetCDF**: User-specified via `--output_file`

**Plot PNG**: Auto-generated as `climatology_spatial_metrics_{min_year}-{max_year}_{verification_window}-{forecast_days}day_{MOK|noMOK}.png`

Examples:
- `climatology_spatial_metrics_2019-2024_1-15day_MOK.png`
- `climatology_spatial_metrics_1985-2024_16-30day_noMOK.png`

## Climatology vs. Model Forecasts
1. **No Model Data Required**: Uses only IMD observational data
2. **Climatological Reference**: Computes long-term mean onset dates as "forecasts"


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

- `mae_far_mr_climatology.py`: Main climatology baseline analysis script
