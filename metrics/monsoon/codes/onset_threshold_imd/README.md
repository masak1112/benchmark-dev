# IMD Onset Threshold Calculation Script

This script calculates 5-day wet spell thresholds from Indian Meteorological Department (IMD) rainfall data needed for first wet spell detection. It processes NetCDF files containing rainfall data and computes mean wet thresholds from April to October (Moron and Robertson (MR), 2013: https://doi.org/10.1002/joc.3745).

## Overview

The script implements the MR difination to calculate onset thresholds based on:
- 5-day wet spell detection
- April to October filtering 

## Features

- **Flexible Input Handling**: Automatically detects various coordinate and variable naming conventions (useful for different resolution input IMD files)
- **Multi-file Support**: Processes multiple NetCDF files in a directory

## Requirements

```bash
python >= 3.6
numpy
xarray
pathlib
```

Install dependencies:
```bash
pip install numpy xarray
```

## Usage

### Basic Usage
```bash
python imd_onset_threshold.py <input_directory> <output_directory>
```

### With Verbose Output
```bash
python imd_onset_threshold.py <input_directory> <output_directory> --verbose
```

### Examples
```bash
# Process IMD data from current directory, save to results folder
python imd_onset_threshold.py ./imd_data ./results

# Process with detailed output
python imd_onset_threshold.py /path/to/imd/data /path/to/output --verbose
```

## Input Requirements

### Directory Structure
- Input directory containing NetCDF files (*.nc)
- Files should contain daily rainfall data
- Multiple files are automatically combined

### Data Format
The script automatically handles various naming conventions:

**Coordinate Names:**
- Latitude: `lat`, `latitude`, `Latitude`, `LAT`, `LATITUDE`
- Longitude: `lon`, `longitude`, `Longitude`, `LON`, `LONGITUDE`
- Time: `time`, `Time`, `TIME`, `date`, `Date`, `DATE`

**Rainfall Variable Names:**
- `RAINFALL`, `rainfall`, `Rainfall`
- `precip`, `precipitation`, `Precipitation`, `PRECIPITATION`
- `tp`, `TP`, `total_precipitation`
- `rain`, `Rain`, `RAIN`
- `pr`, `PR`

## Output

### Output File
The script generates a NetCDF4 file with dynamic naming based on spatial resolution:
- **1° resolution** → `mwset1x1.nc4`
- **0.25° resolution** → `mwset0p25x0p25.nc4`
- **4° resolution** → `mwset4x4.nc4`

### Output Content
The output file contains:
- **Variable**: `MWmean` (Mean wet threshold)
- **Units**: mm
- **Description**: Mean wet threshold for first wet spell detection
- **Dimensions**: [lat, lon]

### Metadata Attributes
- `long_name`: Mean wet threshold
- `units`: mm
- `spatial_resolution_lat/lon`: Resolution in degrees
- `algorithm_parameters`: Algorithm configuration used
- `source_variable`: Original rainfall variable name



## Command Line Options

| Option | Description |
|--------|-------------|
| `input_dir` | Directory containing input NetCDF files (required) |
| `output_dir` | Directory to save output NetCDF file (required) |
| `--verbose`, `-v` | Enable detailed output during processing |



## Example Output

```
Loading data from /path/to/input
Found rainfall variable: RAINFALL
Filtering data for monsoon season (April-October)...
Detected resolution:
  - Latitude: 0.25°
  - Longitude: 0.25°
  - Resolution string: 0p25x0p25
Calculating onset thresholds...
Saved MWmean threshold to: /path/to/output/mwset0p25x0p25.nc4
Processing completed successfully!
Output file: mwset0p25x0p25.nc4
```

## Technical Details

### Spatial Resolution Detection
The script automatically detects spatial resolution by analyzing coordinate spacing and rounds to common resolutions (0.25°, 0.5°, 1.0°, etc.).

### Memory Efficiency
The script uses xarray's lazy loading capabilities to handle large datasets efficiently.

