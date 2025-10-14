#!/usr/bin/env python3
"""
Script to calculate onset thresholds from rainfall data.
Usage: python onset_threshold_calculator.py <input_dir> <output_dir>
Written by: Rajat Masiwal and Tyler Yang
"""

import sys
import os
import argparse
import numpy as np
import xarray as xr
from pathlib import Path


def standardize_coordinate_names(ds):
    """
    Standardize coordinate names to 'lat', 'lon', 'time'.
    
    Args:
        ds (xarray.Dataset): Input dataset
        
    Returns:
        xarray.Dataset: Dataset with standardized coordinate names
    """
    # Common variations of coordinate names
    lat_variations = ['lat', 'latitude', 'Latitude', 'LAT', 'LATITUDE']
    lon_variations = ['lon', 'longitude', 'Longitude', 'LON', 'LONGITUDE']
    time_variations = ['time', 'Time', 'TIME', 'date', 'Date', 'DATE']
    
    rename_dict = {}
    
    # Find and standardize latitude
    for coord in ds.coords:
        if coord in lat_variations and coord != 'lat':
            rename_dict[coord] = 'lat'
            break
    
    # Find and standardize longitude
    for coord in ds.coords:
        if coord in lon_variations and coord != 'lon':
            rename_dict[coord] = 'lon'
            break
    
    # Find and standardize time
    for coord in ds.coords:
        if coord in time_variations and coord != 'TIME':
            rename_dict[coord] = 'TIME'
            break
    
    # Apply renaming if needed
    if rename_dict:
        print(f"Renaming coordinates: {rename_dict}")
        ds = ds.rename(rename_dict)
    
    return ds


def detect_rainfall_variable(ds):
    """
    Detect the rainfall variable name in the dataset.
    
    Args:
        ds (xarray.Dataset): Input dataset
        
    Returns:
        str: Name of the rainfall variable
    """
    # Common variations of rainfall variable names
    rainfall_variations = [
        'RAINFALL', 'rainfall', 'Rainfall',
        'precip', 'precipitation', 'Precipitation', 'PRECIPITATION',
        'tp', 'TP', 'total_precipitation',
        'rain', 'Rain', 'RAIN',
        'pr', 'PR'
    ]
    
    for var_name in rainfall_variations:
        if var_name in ds.data_vars:
            print(f"Found rainfall variable: {var_name}")
            return var_name
    
    # If no standard name found, list available variables
    available_vars = list(ds.data_vars.keys())
    raise ValueError(
        f"Could not find rainfall variable. Available variables: {available_vars}. "
        f"Expected one of: {rainfall_variations}"
    )


def onset_agro_bis(X, lseason, defdry, sw, wet, sd, dry, window):
    N, C = np.shape(X)
    nyear = N // lseason
    W = np.zeros(np.shape(X))
    W[X > defdry] = 1
    
    swet = None
    if sw > 1:
        swet = sequence_overlap(np.transpose([np.arange(lseason)]), lseason, sw)
        swet = np.transpose(swet[sw - 1:lseason,:])
        swet = (swet.reshape((-1, 1), order='F') @ np.ones((1, C))) + np.ones(((lseason - (sw - 1)) * sw, 1)) @ (np.arange(0, lseason * C, lseason).reshape(1, -1))
        swet = swet.reshape((sw,C*(lseason-(sw-1))), order='F')
    
    sdry = None
    if sd > 1:
        sdry = sequence_overlap(np.transpose([np.arange(lseason)]), lseason, sd)
        sdry = np.transpose(sdry[sd - 1:lseason,:])
        sdry = (sdry.reshape((-1, 1), order='F') @ np.ones((1, C))) + np.ones(((lseason - (sd - 1)) * sd, 1)) @ (np.arange(0, lseason * C, lseason).reshape(1, -1))
        sdry = sdry.reshape((sd,C*(lseason-(sd-1))), order='F')
    
    O1 = np.full((nyear, C), np.nan)
    O2 = np.full((nyear, C), np.nan)
    
    S = window - (sd - 1)
    S2 = sequence_overlap(np.transpose([np.arange(lseason)]), lseason, S)
    S2 = np.transpose(S2[S - 1:lseason])
    
    Lw = lseason - (sw - 1)
    SWmean = np.zeros((nyear * Lw, C))
    
    for i in range(nyear):
        sample = X[(i * lseason): ((i + 1) * lseason), :]
        sample_flat = sample.ravel(order="F")
        if sw > 1:
            SWmean[(i * Lw):(Lw * (i + 1)),:] = np.reshape(np.sum(sample_flat[swet.astype(int)], axis=0), (lseason - (sw - 1), C), order="F")
        else:
            SWmean[(i * Lw):(Lw * (i + 1)),:] = sample
    
    MWmean = np.zeros(C)
    for i in range(C):
        MWmean[i] = np.mean(SWmean[SWmean[:,i] > defdry, i])
    
    if wet == 0:
        wet = MWmean
    else:
        wet = wet * np.ones((1, C))
    
    for i in range(nyear):
        sample = X[(i * lseason): ((i + 1) * lseason), :]
        wsample = W[(i * lseason): ((i + 1) * lseason), :]
        sample_flat = sample.ravel(order="F")
        SW = sample
        SD = sample
        if sw > 1:
            SW = np.reshape(np.sum(sample_flat[swet.astype(int)], axis=0), (lseason - (sw - 1), C), order="F")
        if sd > 1:
            SD = np.reshape(np.sum(sample_flat[sdry.astype(int)], axis=0), (lseason - (sd - 1), C), order="F")
        
        for j in range(C):
            SW_extension = np.concatenate([SW[:,j], np.ones(sw - 1) * SW[lseason - sw, j]])
            SD_extension = np.concatenate([SD[:,j], np.zeros(sd - 1)])
            tab = np.column_stack([sample[:, j], wsample[:,j], SW_extension, SD_extension])
            o1 = np.where((tab[:, 2] >= wet[j]) & (tab[:, 1] == 1))[0]
            D = tab[:, 3]
            D = np.transpose(D[S2.astype(int)])
            D = np.vstack([D, np.zeros((window - sd, S))])
            if o1.size > 0:
                O1[i, j] = o1[0]
                tab2 = D[o1, :]
                o2 = o1[np.min(tab2, axis=1) > dry]
                if o2.size > 0:
                    O2[i, j] = o2[0]
    
    return O1, O2, MWmean


def sequence_overlap(X, lseason, nday):
    nr, nv = np.shape(X)
    nyear = nr // lseason
    indice = []
    for i in range(nday):
        row = np.arange(i, lseason + i)
        indice.append(row)
    indice = np.array(indice)
    nseq, lseq = np.shape(indice)
    Y = np.zeros((lseq * nyear, nv * nday))
    
    for i in range(nyear):
        sample = X[i * lseason: (i + 1) * lseason]
        sample = np.vstack([np.tile(sample[0], (nday - 1, 1)), sample])
        sample1 = np.zeros((lseq, nday * nv))
        for j in range(nday):
            sample1[:lseq, (j * nv):(j + 1) * nv] = sample[indice[j], :nv]
        Y[(i * lseq) : (i + 1) *lseq,:nv*nday] = sample1
    
    return Y


def determine_resolution(lats, lons):
    """
    Determine the spatial resolution of the data in degrees.
    
    Args:
        lats (array): Latitude values
        lons (array): Longitude values
        
    Returns:
        tuple: (lat_res, lon_res) resolution in degrees
    """
    # Calculate resolution as the difference between consecutive points
    if len(lats) > 1:
        lat_res = abs(lats[1] - lats[0])
    else:
        lat_res = 1.0  # Default fallback
        
    if len(lons) > 1:
        lon_res = abs(lons[1] - lons[0])
    else:
        lon_res = 1.0  # Default fallback
    
    # Round to common resolutions to handle floating point precision
    common_resolutions = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0, 5.0]
    
    def find_closest_resolution(res):
        return min(common_resolutions, key=lambda x: abs(x - res))
    
    lat_res_rounded = find_closest_resolution(lat_res)
    lon_res_rounded = find_closest_resolution(lon_res)
    
    return lat_res_rounded, lon_res_rounded


def format_resolution_string(lat_res, lon_res):
    """
    Format resolution values into a string for filename.
    
    Args:
        lat_res (float): Latitude resolution
        lon_res (float): Longitude resolution
        
    Returns:
        str: Formatted resolution string (e.g., '1x1', '0p25x0p25')
    """
    def format_number(num):
        if num == int(num):
            return str(int(num))
        else:
            # Replace decimal point with 'p' for filename compatibility
            return str(num).replace('.', 'p')
    
    lat_str = format_number(lat_res)
    lon_str = format_number(lon_res)
    
    return f"{lat_str}x{lon_str}"

def process_rainfall_data(input_dir, output_dir):
    """
    Process rainfall data to calculate onset thresholds.
    
    Args:
        input_dir (str): Directory containing NetCDF files
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_dir}")
    filen = "*.nc"
    
    try:
        dat = xr.open_mfdataset(
            os.path.join(input_dir, filen),
            preprocess=standardize_coordinate_names
        )
    except Exception as e:
        print(f"Error loading NetCDF files: {e}")
        raise
    
    print(f"Original dataset coordinates: {list(dat.coords.keys())}")
    print(f"Original dataset variables: {list(dat.data_vars.keys())}")
        
    # Detect rainfall variable
    rainfall_var = detect_rainfall_variable(dat)
    
    # Verify we have the expected coordinates
    required_coords = ['lat', 'lon', 'TIME']
    missing_coords = [coord for coord in required_coords if coord not in dat.coords]
    if missing_coords:
        available_coords = list(dat.coords.keys())
        raise ValueError(
            f"Missing required coordinates: {missing_coords}. "
            f"Available coordinates after standardization: {available_coords}"
        )
    
    # Filter for monsoon season only (April to October)
    print("Filtering data for monsoon season (April-October)...")
    dat_filtered = dat.sel(TIME=(dat.TIME.dt.month >= 4) & (dat.TIME.dt.month <= 10))
    
    print(f"Filtered data shape: {dat_filtered.dims}")
    print(f"Coordinate ranges:")
    print(f"  - lat: {dat_filtered.lat.min().values:.2f} to {dat_filtered.lat.max().values:.2f}")
    print(f"  - lon: {dat_filtered.lon.min().values:.2f} to {dat_filtered.lon.max().values:.2f}")
    print(f"  - TIME: {dat_filtered.TIME.min().values} to {dat_filtered.TIME.max().values}")
    
    # Determine resolution
    lat_res, lon_res = determine_resolution(dat_filtered.lat.values, dat_filtered.lon.values)
    resolution_str = format_resolution_string(lat_res, lon_res)
    
    print(f"Detected resolution:")
    print(f"  - Latitude: {lat_res}°")
    print(f"  - Longitude: {lon_res}°")
    print(f"  - Resolution string: {resolution_str}")
    
    # Stack spatial dimensions and extract rainfall values
    rainfall = dat_filtered[rainfall_var].stack(grid=["lat", "lon"])
    
    print(f"Rainfall data shape after stacking: {rainfall.shape}")
    print(f"Number of NaN values: {np.isnan(rainfall.values).sum()}")
    
    # Calculate onset thresholds
    print("Calculating onset thresholds...")
    O1, O2, MWmean = onset_agro_bis(rainfall.values, 214, 1, 5, 0, 10, 5, 30)
    
    # Reshape MWmean back to spatial grid
    MWmean_regrid = MWmean.reshape(len(dat_filtered["lat"]), len(dat_filtered["lon"]))
    
    # Convert to xarray DataArray
    MWmean_da = xr.DataArray(
        MWmean_regrid,
        coords={
            'lat': dat_filtered['lat'].values,
            'lon': dat_filtered['lon'].values
        },
        dims=['lat', 'lon'],
        name='MWmean',
        attrs={
            'long_name': 'Mean wet threshold',
            'units': 'mm',
            'description': 'Mean wet threshold for first wet spell detection',
            'source_variable': rainfall_var,
            'spatial_resolution_lat': f'{lat_res} degrees',
            'spatial_resolution_lon': f'{lon_res} degrees',
            'algorithm_parameters': 'lseason=214, defdry=1, sw=5, wet=0, sd=10, dry=5, window=30',
        }
    )
    
    # Create dynamic filename based on resolution
    output_filename = f'mwset{resolution_str}.nc4'
    output_file = os.path.join(output_dir, output_filename)
    
    # Save to output directory
    MWmean_da.to_netcdf(output_file)
    
    print(f"Saved MWmean threshold to: {output_file}")
    print(f"DataArray shape: {MWmean_da.shape}")
    print(f"DataArray statistics:")
    print(f"  - Min: {MWmean_da.min().values:.2f}")
    print(f"  - Max: {MWmean_da.max().values:.2f}")
    print(f"  - Mean: {MWmean_da.mean().values:.2f}")
    print(f"  - NaN count: {np.isnan(MWmean_da.values).sum()}")
    
    return MWmean_da, output_file


def main():
    parser = argparse.ArgumentParser(
        description='Calculate onset thresholds from rainfall data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python onset_threshold_calculator.py /path/to/input/data /path/to/output

The script automatically handles different coordinate naming conventions:
    - Latitude: lat, latitude, Latitude, LAT, LATITUDE
    - Longitude: lon, longitude, Longitude, LON, LONGITUDE  
    - Time: time, Time, TIME, date, Date, DATE

And different rainfall variable names:
    - RAINFALL, rainfall, Rainfall
    - precip, precipitation, Precipitation, PRECIPITATION
    - tp, TP, total_precipitation
    - rain, Rain, RAIN
    - pr, PR

Output filename is automatically determined based on resolution:
    - 1° resolution → mwset1x1.nc4
    - 0.25° resolution → mwset0p25x0p25.nc4
    - 4° resolution → mwset4x4.nc4
        """
    )
    
    parser.add_argument('input_dir', 
                       help='Directory containing input NetCDF files')
    parser.add_argument('output_dir', 
                       help='Directory to save output NetCDF file')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Check if directory contains NetCDF files
    nc_files = list(Path(args.input_dir).glob("*.nc"))
    if not nc_files:
        print(f"Error: No NetCDF files (*.nc) found in '{args.input_dir}'")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(nc_files)} NetCDF files:")
        for f in nc_files:
            print(f"  - {f.name}")
    
    # Process the data
    try:
        result, output_path = process_rainfall_data(args.input_dir, args.output_dir)
        print(f"Processing completed successfully!")
        print(f"Output file: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()