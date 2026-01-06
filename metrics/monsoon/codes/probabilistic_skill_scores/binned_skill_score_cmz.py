#!/usr/bin/env python3
"""
Monsoon Onset Reliability Analysis Script

This script performs reliability analysis for monsoon onset predictions
using model forecast data and IMD observations.
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
from matplotlib.path import Path
import argparse
from scipy import stats
import seaborn as sns
import glob



def get_forecast_probabilistic_twice_weekly(yr, model_forecast_dir, mem_num, date_filter_year = 2024, file_pattern='tp_4p0_{}.nc'):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    """
    fname = file_pattern.format(yr)
    file_path = os.path.join(model_forecast_dir, fname)
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Filter for twice weekly data from daily for the specified year
    start_date = datetime(date_filter_year, 5, 1)
    end_date = datetime(date_filter_year, 7, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
        
    # Find Mondays and Thursdays
    is_monday = date_range.weekday == 0
    is_thursday = date_range.weekday == 3
    filtered_dates = date_range[is_monday | is_thursday]
    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{yr}-%m-%d'))
        
    # Load data using xarray
    ds = xr.open_dataset(file_path)
    if 'time' in ds.dims:
        ds = ds.rename({'time': 'init_time'})
    if 'number' in ds.dims:
        ds = ds.rename({'number': 'member'})
    if 'sample' in ds.dims:
        ds = ds.rename({'sample': 'member'})
    
    # Find common dates between desired dates and available dates
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
        
    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {yr}")
        
    # Select only the matching initialization times
    ds = ds.sel(init_time=matching_times)
    if "total_precipitation_24hr" in ds.data_vars:
        ds = ds.rename({"total_precipitation_24hr": "tp"}) # For the quantile-mapped variable change the var name from total_precipitation_24hr to tp
        ds = ds[['tp']]*1000  # Convert from m to mm
    if 'day' in ds.dims:
        if ds['day'][0].values == 0:
            ds = ds.sel(day=slice(1, None))
    
    if 'step' in ds.dims:
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None)) 
    
    if 'day' in ds.dims:        
        ds = ds.rename({'day': 'step'})

    ds = ds.isel(member =slice(0, mem_num))  # limit to first mem_num members (0-mem_num)
    p_model = ds['tp']  # in mm
    init_times = p_model.init_time.values
    ds.close()
    return p_model, init_times

def load_imd_rainfall(year, imd_folder):
    """Load IMD daily rainfall NetCDF for a given year."""
    file_patterns = [f"data_{year}.nc", f"{year}.nc"]
    
    imd_file = None
    for pattern in file_patterns:
        test_path = f"{imd_folder}/{pattern}"
        if os.path.exists(test_path):
            imd_file = test_path
            break
    
    if imd_file is None:
        available_files = [f for f in os.listdir(imd_folder) if f.endswith('.nc')]
        raise FileNotFoundError(
            f"No IMD file found for year {year} in {imd_folder}. "
            f"Tried patterns: {file_patterns}. "
            f"Available files: {available_files}"
        )
    
    print(f"Loading IMD rainfall from: {imd_file}")
    
    ds = xr.open_dataset(imd_file)
    rainfall = ds['RAINFALL']
    
    # Standardize dimension names
    dim_mapping = {}
    if 'latitude' in rainfall.dims:
        dim_mapping['latitude'] = 'lat'
    if 'LATITUDE' in rainfall.dims:
        dim_mapping['LATITUDE'] = 'lat'
    if 'longitude' in rainfall.dims:
        dim_mapping['longitude'] = 'lon'
    if 'LONGITUDE' in rainfall.dims:
        dim_mapping['LONGITUDE'] = 'lon'
    if 'TIME' in rainfall.dims:
        dim_mapping['TIME'] = 'time'
    
    if dim_mapping:
        rainfall = rainfall.rename(dim_mapping)
        print(f"Renamed dimensions: {dim_mapping}")
    else:
        print(f"No dimension renaming needed. Current dims: {list(rainfall.dims)}")
    
    return rainfall

# Function to detect observed onset dates based on rainfall threshold file
def detect_observed_onset(rainfall_ds, thresh_slice, year, mok=True):
    """Detect observed onset dates for a given year."""
    rain_slice = rainfall_ds
    window = 5 # 5-day wet spell window
    
    # Set start date based on mok flag
    if mok:
        start_date = datetime(year, 6, 2)  # MOK date: June 2nd
        date_label = "MOK date (June 2nd)"
    else:
        start_date = datetime(year, 5, 1)  # May 1st
        date_label = "May 1st"

    # Find start date index
    time_dates = pd.to_datetime(rain_slice.time.values)
    start_idx_candidates = np.where(time_dates > start_date)[0]
    
    if len(start_idx_candidates) == 0:
        print(f"Warning: {date_label} not found in data for year {year}")
        fallback_date = datetime(year, 4, 1)
        start_idx = np.where(time_dates >= fallback_date)[0][0]
        print(f"Using fallback date: April 1st")
    else:
        start_idx = start_idx_candidates[0]
        print(f"Using {date_label} as start date for onset detection")

    # Subset rain_slice from start date onward
    rain_subset = rain_slice.isel(time=slice(start_idx, None))

    # Create rolling 5-day sums
    rolling_sum = rain_subset.rolling(time=window, min_periods=window, center=False).sum()
    rolling_sum_aligned = rolling_sum.shift(time=-(window-1))

    # Create onset condition
    first_day_condition = rain_subset > 1
    sum_condition = rolling_sum_aligned > thresh_slice
    onset_condition = first_day_condition & sum_condition

    # Find first occurrence of onset condition for each grid point
    def find_first_true(arr):
        if arr.any():
            return int(np.argmax(arr))
        else:
            return -1

    onset_indices = xr.apply_ufunc(
        find_first_true,
        onset_condition,
        input_core_dims=[['time']],
        output_dtypes=[int],
        vectorize=True
    )

    # Convert indices to actual dates
    valid_mask = onset_indices.values >= 0
    time_coords = rain_subset.time.values
    onset_dates_array = np.full(onset_indices.shape, np.datetime64('NaT'), dtype='datetime64[ns]')

    for i in range(onset_indices.shape[0]):
        for j in range(onset_indices.shape[1]):
            if valid_mask[i, j]:
                idx = int(onset_indices[i, j].values)
                if 0 <= idx < len(time_coords):
                    onset_dates_array[i, j] = time_coords[idx]

    # Create final onset date DataArray
    onset_da = xr.DataArray(
        onset_dates_array,
        coords=[('lat', rain_slice.lat.values), ('lon', rain_slice.lon.values)],
        name='onset_date'
    )
    
    return onset_da

# Function to compute onset dates for all ensemble members and save it as DataFrame
def compute_onset_for_all_members(p_model, thresh_slice, onset_da, max_forecast_day=15, mok=True):
    """Compute onset dates for each ensemble member, initialization time, and grid point."""
    window = 5
    results_list = []
    
    # Get dimensions
    init_times = p_model.init_time.values
    members = p_model.member.values
    
    # Get the actual lat/lon coordinates from the data
    lats = p_model.lat.values  
    lons = p_model.lon.values
    
    # Create unique lat-lon pairs (no repetition)
    unique_pairs = list(zip(lons, lats))
    
    date_method = "MOK (June 2nd filter)" if mok else "no date filter"
    print(f"Processing {len(init_times)} init times x {len(unique_pairs)} unique locations x {len(members)} members...")
    print(f"Unique lat-lon pairs: {unique_pairs}")
    print(f"Using {date_method} for onset detection")
    
    max_steps_needed = max_forecast_day + window - 1
    
    # Track statistics
    total_potential_forecasts = 0
    valid_forecasts = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    
    # Loop over all combinations
    for t_idx, init_time in enumerate(init_times):
        if t_idx % 5 == 0:
            print(f"Processing init time {t_idx+1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}")
        
        init_date = pd.to_datetime(init_time)
        year = init_date.year
        mok_date = datetime(year, 6, 2)
        
        # Loop over unique lat-lon pairs only
        for loc_idx, (lon, lat) in enumerate(unique_pairs):
            
            total_potential_forecasts += len(members)
            
            # Get observed onset date for this location
            try:
                obs_onset = onset_da.isel(lat=loc_idx, lon=loc_idx).values
            except:
                skipped_no_obs += len(members)
                continue
            
            # Skip if no observed onset
            if pd.isna(obs_onset):
                skipped_no_obs += len(members)
                continue
            
            # Convert observed onset to datetime
            obs_onset_dt = pd.to_datetime(obs_onset)
            
            # Only process if forecast was initialized before observed onset
            if init_date >= obs_onset_dt:
                skipped_late_init += len(members)
                continue
            
            # Get threshold for this location
            thresh = thresh_slice.isel(lat=loc_idx, lon=loc_idx).values
            
            for m_idx, member in enumerate(members):
                
                valid_forecasts += 1
                
                try:
                    # Extract forecast time series for this member and location
                    forecast_series = p_model.isel(
                        init_time=t_idx,
                        lat=loc_idx, 
                        lon=loc_idx,
                        member=m_idx,
                        step=slice(0, max_steps_needed)
                    ).values
                    
                    if len(forecast_series) < max_steps_needed:
                        continue
                    
                    # Check for onset on each possible day
                    onset_day = None
                    
                    for day in range(1, max_forecast_day + 1):
                        start_idx = day - 1
                        end_idx = start_idx + window 
                        
                        if end_idx <= len(forecast_series):
                            window_series = forecast_series[start_idx:end_idx]
                            
                            # Check basic onset condition
                            if window_series[0] > 1 and np.nansum(window_series) > thresh:
                                
                                # Calculate the actual date this forecast day represents
                                forecast_date = init_date + pd.Timedelta(days=day)
                                
                                # If MOK flag is True, only count onset if it's on or after June 2nd
                                if mok:
                                    if forecast_date.date() > mok_date.date():
                                        onset_day = day
                                        break
                                else:
                                    onset_day = day
                                    break
                    
                    # Store result
                    result = {
                        'init_time': init_time,
                        'lat': lat,
                        'lon': lon, 
                        'member': member,
                        'onset_day': onset_day,
                        'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')
                    }
                    results_list.append(result)
                    
                except Exception as e:
                    print(f"Error at init_time {t_idx}, location ({lon}, {lat}), member {m_idx}: {e}")
                    continue
    
    # Convert to DataFrame
    onset_df = pd.DataFrame(results_list)
    
    print(f"\nProcessing Summary:")
    print(f"Total potential forecasts: {total_potential_forecasts}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid forecasts processed: {valid_forecasts}")
    print(f"Generated {len(onset_df)} member-forecast combinations")
    print(f"Found onset in {onset_df['onset_day'].notna().sum()} cases")
    print(f"Onset rate: {onset_df['onset_day'].notna().mean():.3f}")
    
    # Check for uniqueness
    unique_combinations = onset_df.groupby(['init_time', 'lat', 'lon', 'member']).size()
    if (unique_combinations > 1).any():
        print(f"Warning: Found {(unique_combinations > 1).sum()} duplicate combinations!")
    else:
        print("✓ All init_time-lat-lon-member combinations are unique")
    0
    return onset_df

# Function to find grid points inside a polygon (For core-monsoon zone analysis)
def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """
    Find grid points that are inside a polygon.
    
    Parameters:
    polygon_lon: array of polygon longitude vertices
    polygon_lat: array of polygon latitude vertices  
    grid_lons: array of grid longitude points
    grid_lats: array of grid latitude points
    
    Returns:
    inside_mask: boolean array indicating which points are inside
    inside_lons: longitude coordinates of points inside polygon
    inside_lats: latitude coordinates of points inside polygon
    """
    
    # Create polygon path
    polygon_vertices = np.column_stack((polygon_lon, polygon_lat))
    polygon_path = Path(polygon_vertices)
    
    # Create meshgrid if needed
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats
    
    # Flatten the grids to test each point
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    
    # Test which points are inside the polygon
    inside_mask = polygon_path.contains_points(points)
    inside_mask = inside_mask.reshape(lon_grid.shape)
    
    # Get coordinates of points inside polygon
    inside_lons = lon_grid[inside_mask]
    inside_lats = lat_grid[inside_mask]
    
    return inside_mask, inside_lons, inside_lats


# Function to create forecast-observation pairs with specified day bins for probabilistic verification
def create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, day_bins, max_forecast_day=15):
    """
    Create forecast-observation pairs using specified day bins, including a final bin for "after max_forecast_day".
    
    Parameters:
    -----------
    onset_all_members : DataFrame
        DataFrame with ensemble member onset predictions
    onset_da : xarray.DataArray
        Observed onset dates
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    max_forecast_day : int, default=15
        Maximum forecast day. Members without onset get assigned to "after day X" bin
    """
    results_list = []
    
    # Get unique combinations of init_time, lat, lon from the filtered forecast data
    forecast_groups = onset_all_members.groupby(['init_time', 'lat', 'lon'])
    
    # Add the "after max_forecast_day" bin
    extended_bins = day_bins + [(max_forecast_day + 1, float('inf'))]
    
    print(f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}")
    print(f"Including 'after day {max_forecast_day}' bin for members without onset in forecast window")
    
    for (init_time, lat, lon), group in forecast_groups:
        
        # Get observed onset for this location
        try:
            lat_idx = np.where(np.abs(onset_da.lat.values - lat) < 0.01)[0][0]
            lon_idx = np.where(np.abs(onset_da.lon.values - lon) < 0.01)[0][0]
            obs_date = onset_da.isel(lat=lat_idx, lon=lon_idx).values
        except:
            continue
        
        # Skip if no observed onset
        if pd.isna(obs_date):
            continue
            
        # Convert dates for comparison
        init_date = pd.to_datetime(init_time)
        obs_date_dt = pd.to_datetime(obs_date)
        
        # Double-check: Only use forecasts initialized before the observed onset
        if init_date >= obs_date_dt:
            continue
        
        # For each day bin (including the "after max_forecast_day" bin)
        for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
            
            # Handle the "after max_forecast_day" bin differently
            if bin_start > max_forecast_day:
                bin_label = f'After day {max_forecast_day}'
                
                # Check if observed onset occurs after max_forecast_day
                forecast_end_date = init_date + pd.Timedelta(days=max_forecast_day)
                observed_onset = int(obs_date_dt.date() > forecast_end_date.date())
                
                # Count members that didn't predict onset within forecast window
                members_with_onset_in_bin = 0
                total_members = len(group)
                
                for member_idx, member_row in group.iterrows():
                    member_onset_day = member_row['onset_day']
                    
                    # Member predicts "after day X" if onset_day is NaN or > max_forecast_day
                    if pd.isna(member_onset_day) or member_onset_day > max_forecast_day:
                        members_with_onset_in_bin += 1
                        
            else:
                # Regular bin within forecast window
                bin_label = f'Days {bin_start}-{bin_end}'
                
                # Calculate the date range for this bin
                bin_start_date = init_date + pd.Timedelta(days=bin_start)
                bin_end_date = init_date + pd.Timedelta(days=bin_end)
                
                # Check if observed onset falls within this day bin
                observed_onset = int(bin_start_date.date() <= obs_date_dt.date() <= bin_end_date.date())
                
                # Calculate ensemble probability for this day bin
                members_with_onset_in_bin = 0
                total_members = len(group)
                
                for member_idx, member_row in group.iterrows():
                    member_onset_day = member_row['onset_day']
                    
                    if pd.notna(member_onset_day) and bin_start <= member_onset_day <= bin_end:
                        members_with_onset_in_bin += 1
            
            # Calculate probability
            predicted_prob = members_with_onset_in_bin / total_members
            
            # Store result
            result = {
                'init_time': init_time,
                'lat': lat,
                'lon': lon,
                'bin_start': bin_start if bin_start <= max_forecast_day else max_forecast_day + 1,
                'bin_end': bin_end if bin_end <= max_forecast_day else float('inf'),
                'bin_label': bin_label,
                'predicted_prob': predicted_prob,
                'observed_onset': observed_onset,
                'members_with_onset': members_with_onset_in_bin,
                'total_members': total_members,
                'year': pd.to_datetime(init_time).year,
                'obs_onset_date': obs_date_dt.strftime('%Y-%m-%d'),
                'bin_index': bin_idx
            }
            results_list.append(result)
    
    # Convert to DataFrame
    forecast_obs_df = pd.DataFrame(results_list)
    
    print(f"Generated {len(forecast_obs_df)} forecast-observation pairs")
    print(f"Total bins per forecast: {len(extended_bins)}")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")
    
    # Show distribution across bins
    print(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean'
    }).round(3)
    print(bin_stats)
    
    return forecast_obs_df

# This function creates the observed forecast pairs for multiple years (core monsoon zone grids) and combines them
def multi_year_forecast_obs_pairs(years, model_forecast_dir, imd_folder, thres_file, mem_num, max_forecast_day, day_bins, mok=True, date_filter_year=2024, file_pattern = '{}.nc'):
    """Main function to perform multi-year reliability analysis."""
    
    print(f"Processing years: {years}")
    
    # Load threshold data (same for all years)
    thresh_ds = xr.open_dataset(thres_file)
    thresh_da = thresh_ds['MWmean']
    orig_lat = thresh_da.lat.values
    orig_lon = thresh_da.lon.values

    lat_diff = abs(orig_lat[1]-orig_lat[0])
    if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
        polygon1_lon = np.array([83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83])
        polygon1_lat = np.array([17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17])
        print("Using 2-degree CMZ polygon coordinates")
    elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
        polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
        polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
        print("Using 4-degree CMZ polygon coordinates")
    elif abs(lat_diff - 1.0) < 0.1:  # 1-degree resolution
        polygon1_lon = np.array([74, 85, 85, 86, 86, 87, 87, 88, 88, 88, 85, 85, 82, 82, 79, 79, 78, 78, 69, 69, 74, 74])
        polygon1_lat = np.array([18, 18, 19, 19, 20, 20, 21, 21, 21, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 21, 21, 18])
        print("Using 1-degree CMZ polygon coordinates")

    inside_mask, inside_lons, inside_lats = points_inside_polygon(polygon1_lon, polygon1_lat, orig_lon, orig_lat)
    thresh_slice = thresh_da.sel(lat=inside_lats, lon=inside_lons)

    # Initialize list to store all forecast-observation pairs
    all_forecast_obs_pairs = []
    
    # Process each year
    for year in years:
        print(f"\n{'='*50}")
        print(f"Processing year {year}")
        print(f"{'='*50}")
        
        try:
            # Load model and observation data
            print("Loading S2S model data...")
            p_model,_ = get_forecast_probabilistic_twice_weekly(year, model_forecast_dir, mem_num, date_filter_year, file_pattern)
            p_model_slice = p_model.sel(lat=inside_lats, lon=inside_lons)

            print("Loading IMD rainfall data...")
            rainfall_ds = load_imd_rainfall(year, imd_folder)
            rainfall_ds_slice = rainfall_ds.sel(lat=inside_lats, lon=inside_lons)
            print("Detecting observed onset...")
            onset_da = detect_observed_onset(rainfall_ds_slice, thresh_slice, year, mok)
            print(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")
            
            print("Computing onset for all ensemble members...")
            onset_all_members = compute_onset_for_all_members(p_model_slice, thresh_slice, onset_da, max_forecast_day=max_forecast_day, mok=True)
            print(f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases")
            
            print("Creating forecast-observation pairs...")
            forecast_obs_pairs = create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, day_bins, max_forecast_day=max_forecast_day)
            
            # Add to master list
            all_forecast_obs_pairs.append(forecast_obs_pairs)
            
            print(f"Year {year} completed: {len(forecast_obs_pairs)} forecast-observation pairs")
            
        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue
    
    # Combine all years
    print(f"\n{'='*50}")
    print("Combining all years")
    print(f"{'='*50}")
    
    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any year")
    
    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)
            
    # Print final summary statistics
    print(f"\nFinal Summary Statistics:")
    print(f"Years processed: {years}")    
    return combined_forecast_obs

# Function to calculate Brier Score and Fair Brier Score for the model forecasts (both overall and bin-wise)
def calculate_brier_score(forecast_obs_df):
    """
    Calculate Brier Score and Fair Brier Score for probabilistic forecasts.
    
    Brier Score = (1/n*m) * Σ(Y_ij - p_ij)²
    Fair Brier Score = (1/n*m) * Σ[(Y_ij - p_ij)² - p_ij(1-p_ij)/(ens-1)]
    
    where:
    - n = number of forecasts
    - m = number of bins per forecast  
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - ens = number of ensemble members
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members'
    
    Returns:
    --------
    dict with Brier score metrics
    """
    
    # Calculate squared differences
    squared_diffs = (forecast_obs_df['observed_onset'] - forecast_obs_df['predicted_prob'])**2
    
    # Calculate overall Brier Score
    brier_score = squared_diffs.mean()
    
    # Calculate Fair Brier Score correction term
    # ens-1 where ens is the number of ensemble members
    correction_term = (forecast_obs_df['predicted_prob'] * (1 - forecast_obs_df['predicted_prob'])) / (forecast_obs_df['total_members'] - 1)
    
    # Fair Brier Score
    fair_brier_components = squared_diffs - correction_term
    fair_brier_score = fair_brier_components.mean()
    # Calculate squared differences for bin-wise analysis
    forecast_obs_df['squared_diff'] = squared_diffs
    forecast_obs_df['fair_brier_component'] = fair_brier_components
    
    # Bin-wise Brier scores
    bin_brier_scores = forecast_obs_df.groupby('bin_label')['squared_diff'].mean()
    bin_fair_brier_scores = forecast_obs_df.groupby('bin_label')['fair_brier_component'].mean()
    
    brier_results = {
        'brier_score': brier_score,
        'fair_brier_score': fair_brier_score,
        'bin_brier_scores': bin_brier_scores.to_dict(),
        'bin_fair_brier_scores': bin_fair_brier_scores.to_dict(),
    }
    
    print(f"Brier Score: {brier_results['brier_score']:.4f}")
    print(f"Fair Brier Score: {brier_results['fair_brier_score']:.4f}")

    return brier_results

# Function to calculate Area Under the Curve (AUC) for the model forecasts (both overall and bin-wise)
def calculate_auc(forecast_obs_df):
    """
    Calculate Area Under the Curve (AUC) for probabilistic forecasts.
    
    AUC = Σ_{i,j,i',j'} Y_{ij}(1-Y_{i'j'}) · 1[p_{ij} > p_{i'j'}] / 
          [(Σ_{i,j} Y_{ij})(Σ_{i,j} (1-Y_{ij}))]
    
    where:
    - Y_{ij} = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_{ij} = predicted probability for bin j in forecast i
    - 1[p_{ij} > p_{i'j'}] = indicator function (1 if true, 0 if false)
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset'
    
    Returns:
    --------
    dict with AUC metrics
    """
    
    # Extract probabilities and observations
    p_ij = forecast_obs_df['predicted_prob'].values
    y_ij = forecast_obs_df['observed_onset'].values
    
    # Count total positive and negative cases
    n_positive = np.sum(y_ij)  # Σ Y_{ij}
    n_negative = np.sum(1 - y_ij)  # Σ (1-Y_{ij})
    
    if n_positive == 0 or n_negative == 0:
        print("Warning: Cannot calculate AUC - all cases are either positive or negative")
        return {
            'auc': np.nan,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'bin_auc_scores': {},
            'forecast_obs_df_with_ranks': forecast_obs_df
        }
    
    # Calculate AUC using the Mann-Whitney U statistic approach
    # This is equivalent to the formula but more computationally efficient
    
    # Separate positive and negative cases
    positive_probs = p_ij[y_ij == 1]
    negative_probs = p_ij[y_ij == 0]
    
    # Calculate Mann-Whitney U statistic
    u_statistic, _ = stats.mannwhitneyu(positive_probs, negative_probs, alternative='greater')
    
    # AUC is U statistic divided by (n_positive * n_negative)
    auc = u_statistic / (n_positive * n_negative)
    
    # Alternative direct calculation (less efficient for large datasets)
    # concordant_pairs = 0
    # for i in range(len(p_ij)):
    #     if y_ij[i] == 1:  # positive case
    #         for j in range(len(p_ij)):
    #             if y_ij[j] == 0:  # negative case
    #                 if p_ij[i] > p_ij[j]:
    #                     concordant_pairs += 1
    # auc_direct = concordant_pairs / (n_positive * n_negative)
    
    # Calculate AUC by bin
    bin_auc_scores = {}
    unique_bins = forecast_obs_df['bin_label'].unique()
    
    for bin_label in unique_bins:
        bin_data = forecast_obs_df[forecast_obs_df['bin_label'] == bin_label]
        
        if len(bin_data) > 0:
            bin_p = bin_data['predicted_prob'].values
            bin_y = bin_data['observed_onset'].values
            
            bin_n_positive = np.sum(bin_y)
            bin_n_negative = np.sum(1 - bin_y)
            
            if bin_n_positive > 0 and bin_n_negative > 0:
                bin_positive_probs = bin_p[bin_y == 1]
                bin_negative_probs = bin_p[bin_y == 0]
                
                bin_u_stat, _ = stats.mannwhitneyu(bin_positive_probs, bin_negative_probs, alternative='greater')
                bin_auc = bin_u_stat / (bin_n_positive * bin_n_negative)
            else:
                bin_auc = np.nan
            
            bin_auc_scores[bin_label] = bin_auc


    auc_results = {
        'auc': auc,
        'bin_auc_scores': bin_auc_scores,
    }
    
    return auc_results


# Function to calculate Ranked Probability Score (RPS) and Fair RPS for the model forecasts for (either 15 day or 30 day forecast)
def calculate_rps(forecast_obs_df):
    """
    Calculate Ranked Probability Score (RPS) and Fair RPS for probabilistic forecasts.
    
    RPS = (1/n*m) * Σ_i Σ_k (Σ_j≤k (Y_ij - p_ij))²
    Fair RPS = (1/n*m) * Σ_i Σ_k [(Σ_j≤k (Y_ij - p_ij))² - (Σ_j≤k p_ij)(1 - Σ_j≤k p_ij)/(ens-1)]
    
    where:
    - n = number of forecasts
    - m = number of bins per forecast  
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - k = cumulative index (1 to m)
    - ens = number of ensemble members
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members', 'bin_index'
    
    Returns:
    --------
    dict with RPS metrics
    """
    
    # Group by forecast (init_time, lat, lon) to get all bins for each forecast
    forecast_groups = forecast_obs_df.groupby(['init_time', 'lat', 'lon'])
    
    rps_values = []
    fair_rps_values = []
    
    for (init_time, lat, lon), group in forecast_groups:
        # Sort by bin_index to ensure proper ordering
        group_sorted = group.sort_values('bin_index')
        
        # Get predicted probabilities and observations for this forecast
        p_ij = group_sorted['predicted_prob'].values
        y_ij = group_sorted['observed_onset'].values
        total_members = group_sorted['total_members'].iloc[0]  # Same for all bins in forecast
        
        m = len(p_ij)  # Number of bins
        
        # Calculate RPS for this forecast
        rps_forecast = 0
        fair_rps_forecast = 0
        
        for k in range(1, m + 1):  # k from 1 to m
            # Cumulative sum up to bin k
            cum_p = np.sum(p_ij[:k])
            cum_y = np.sum(y_ij[:k])
            
            # RPS component
            diff_cum = cum_y - cum_p
            rps_component = diff_cum**2
            rps_forecast += rps_component
            
            # Fair RPS correction term
            fair_correction = (cum_p * (1 - cum_p)) / (total_members - 1)
            fair_rps_component = rps_component - fair_correction
            fair_rps_forecast += fair_rps_component
        
        rps_values.append(rps_forecast)
        fair_rps_values.append(fair_rps_forecast)
    
    # Calculate overall RPS (average over all forecasts)
    rps = np.mean(rps_values)
    fair_rps = np.mean(fair_rps_values)
    

    
    rps_results = {
        'rps': rps,
        'fair_rps': fair_rps,
        'n_forecasts': len(forecast_groups),

    }

    print(f"RPS: {rps_results['rps']:.4f}")
    print(f"Fair RPS: {rps_results['fair_rps']:.4f}")
    print(f"Number of forecasts: {rps_results['n_forecasts']}")
    
    return rps_results



## This function computes onset dates for all available years in IMD folder and creates a climatological onset dataset
def compute_climatological_onset_dataset(imd_folder, thresh_slice, years=None, mok=True):
    """
    Compute onset dates for all available years in IMD folder and create a climatological dataset.
    
    Parameters:
    -----------
    imd_folder : str
        Folder containing IMD NetCDF files
    thresh_slice : xarray.DataArray
        Rainfall threshold for each grid point
    years : list, optional
        Specific years to process. If None, will auto-detect available years
    mok : bool, default=True
        Whether to use MOK date filter (June 2nd)
    
    Returns:
    --------
    xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates
    """
    
    # Auto-detect available years if not specified
    if years is None:
        years = []
        # Look for files matching common IMD naming patterns
        file_patterns = ['data_*.nc', '*.nc']
        
        for pattern in file_patterns:
            files = glob.glob(os.path.join(imd_folder, pattern))
            for file in files:
                filename = os.path.basename(file)
                # Extract year from filename
                if filename.startswith('data_'):
                    year_str = filename.replace('data_', '').replace('.nc', '')
                else:
                    year_str = filename.replace('.nc', '')
                year = int(year_str)
                years.append(year)
            
        
        years = sorted(list(set(years)))  # Remove duplicates and sort
    
    if not years:
        raise ValueError(f"No valid years found in {imd_folder}")
    
    print(f"Processing {len(years)} years: {years}")
    
    # Initialize lists to store results
    onset_arrays = []
    valid_years = []
    
    # Process each year
    for year in years:
        print(f"\nProcessing year {year}...")
        
        try:
            # Load rainfall data for this year
            rainfall_ds = load_imd_rainfall(year, imd_folder)
            
            # Select the same spatial domain as thresh_slice
            rainfall_slice = rainfall_ds
            # Detect onset for this year
            onset_da = detect_observed_onset(rainfall_slice, thresh_slice, year, mok=mok)
            
            # Count valid onsets
            valid_onsets = (~pd.isna(onset_da.values)).sum()
            total_points = onset_da.size
            
            print(f"Year {year}: Found onset in {valid_onsets}/{total_points} grid points ({valid_onsets/total_points:.1%})")
            
            # Store the onset array
            onset_arrays.append(onset_da.values)
            valid_years.append(year)
            
        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue
    
    if not onset_arrays:
        raise ValueError("No years were successfully processed")
    
    # Stack all onset arrays into a 3D array
    onset_3d = np.stack(onset_arrays, axis=0)
    
    # Create the final DataArray
    climatological_onset_da = xr.DataArray(
        onset_3d,
        coords=[
            ('year', valid_years),
            ('lat', thresh_slice.lat.values),
            ('lon', thresh_slice.lon.values)
        ],
        name='climatological_onset_dates',
        attrs={
            'description': 'Onset dates for climatological ensemble',
            'method': 'MOK (June 2nd filter)' if mok else 'no date filter',
            'years_processed': valid_years,
            'total_years': len(valid_years)
        }
    )
    
    # Print summary statistics
    total_possible = len(valid_years) * thresh_slice.size
    total_valid = (~pd.isna(climatological_onset_da.values)).sum()
    
    print(f"\n{'='*60}")
    print(f"CLIMATOLOGICAL ONSET DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Years processed: {len(valid_years)} ({min(valid_years)}-{max(valid_years)})")
    print(f"Spatial domain: {len(thresh_slice.lat)} lats x {len(thresh_slice.lon)} lons")
    print(f"Total valid onsets: {total_valid:,}/{total_possible:,} ({total_valid/total_possible:.1%})")
    print(f"Method: {'MOK (June 2nd filter)' if mok else 'No date filter'}")
    
    # Show onset statistics by year
    print(f"\nOnset statistics by year:")
    for i, year in enumerate(valid_years):
        year_onsets = (~pd.isna(climatological_onset_da.isel(year=i).values)).sum()
        print(f"  {year}: {year_onsets}/{thresh_slice.size} ({year_onsets/thresh_slice.size:.1%})")
    
    return climatological_onset_da


## This function creates forecast-observation pairs using climatological ensemble where each year is a member
def create_climatological_forecast_obs_pairs(clim_onset, target_year, init_dates, day_bins, max_forecast_day=15, mok=True):
    """
    Create forecast-observation pairs using climatological ensemble where each year is a member.
    Uses day-of-year instead of calendar dates for onset comparison.
    
    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates for all years
    target_year : int
        The year to use as "truth" for observations
    init_dates : list or pandas.DatetimeIndex
        Initialization dates for forecasts
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    max_forecast_day : int, default=15
        Maximum forecast day
    mok : bool, default=True
        Whether to use MOK date filter (June 2nd)
    
    Returns:
    --------
    DataFrame with forecast-observation pairs
    """
    
    results_list = []
    
    # Get the observed onset for the target year
    if target_year not in clim_onset.year.values:
        raise ValueError(f"Target year {target_year} not found in climatological dataset")
    
    obs_onset_da = clim_onset.sel(year=target_year)
    
    # Use ALL years as ensemble members (including target year)
    ensemble_years = list(clim_onset.year.values)
    ensemble_onset_da = clim_onset.sel(year=ensemble_years)
    
    # Create extended bins including "before initialization" and "after max_forecast_day" bins
    extended_bins = [(-float('inf'), 0)] + day_bins + [(max_forecast_day + 1, float('inf'))]

    print(f"Creating climatological forecasts for target year {target_year}")
    print(f"Using {len(ensemble_years)} years as ensemble members: {ensemble_years}")
    print(f"Processing {len(init_dates)} initialization dates")
    print(f"Day bins: {day_bins}")
    print(f"Extended bins include: 'Before initialization' and 'After day {max_forecast_day}'")
    print(f"Using day-of-year method for onset comparison")
    
    # Get the actual lat/lon coordinates from the data
    lats = obs_onset_da.lat.values  
    lons = obs_onset_da.lon.values
    
    # Create unique lat-lon pairs (no repetition)
    unique_pairs = list(zip(lons, lats))
    
    print(f"Processing {len(unique_pairs)} unique lat-lon pairs")
    
    # Process each initialization date and location
    for init_date in init_dates:
        init_date = pd.to_datetime(init_date)
        init_doy = init_date.dayofyear  # Day of year for initialization
        
        # Loop over unique lat-lon pairs
        for pair_idx, (lon, lat) in enumerate(unique_pairs):
            
            # Get observed onset for this location and target year
            try:
                lat_idx = np.where(np.abs(obs_onset_da.lat.values - lat) < 0.01)[0][0]
                lon_idx = np.where(np.abs(obs_onset_da.lon.values - lon) < 0.01)[0][0]
                obs_onset = obs_onset_da.isel(lat=lat_idx, lon=lon_idx).values
            except:
                continue
            
            # Skip if no observed onset
            if pd.isna(obs_onset):
                continue
            
            obs_onset_dt = pd.to_datetime(obs_onset)
            obs_onset_doy = obs_onset_dt.dayofyear  # Day of year for observed onset
            
            # Only process forecasts that are initialized BEFORE the observed onset (by day of year)
            if init_doy >= obs_onset_doy:
                continue
            
            # Get ensemble member onsets for this location using the same indices
            ensemble_onsets = ensemble_onset_da.isel(lat=lat_idx, lon=lon_idx).values
            
            # Convert ensemble onsets to days from initialization using day-of-year
            ensemble_forecast_days = []
            ensemble_years_with_data = []
            
            for ens_idx, ens_onset in enumerate(ensemble_onsets):
                ens_year = ensemble_years[ens_idx]
                
                if pd.notna(ens_onset):
                    ens_onset_dt = pd.to_datetime(ens_onset)
                    ens_onset_doy = ens_onset_dt.dayofyear  # Day of year for ensemble onset
                    
                    # Calculate days from initialization using day-of-year difference
                    days_from_init = ens_onset_doy - init_doy                   
                    ensemble_forecast_days.append(days_from_init)
                    ensemble_years_with_data.append(ens_year)
                else:
                    # No onset predicted by this member
                    ensemble_forecast_days.append(None)
                    ensemble_years_with_data.append(ens_year)
            
            total_members = len(ensemble_years)
            
            # First pass: calculate total members with onset across all bins
            total_members_with_onset = 0
            bin_members_onset = []  # Store for second pass
            
            for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
                members_with_onset_in_bin = 0
                
                # Handle the "before initialization" bin
                if bin_start == -float('inf'):
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day <= 0:
                            members_with_onset_in_bin += 1
                            
                # Handle the "after max_forecast_day" bin
                elif bin_start > max_forecast_day:
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day > max_forecast_day:
                            members_with_onset_in_bin += 1
                            
                else:
                    # Regular bin within forecast window
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and bin_start <= member_onset_day <= bin_end:
                            members_with_onset_in_bin += 1
                
                bin_members_onset.append(members_with_onset_in_bin)
                total_members_with_onset += members_with_onset_in_bin
            
            # Skip if no members showed onset
            if total_members_with_onset == 0:
                continue
            
            # Second pass: For each day bin, calculate probabilities using total_members_with_onset
            for bin_idx, (bin_start, bin_end) in enumerate(extended_bins):
                
                members_with_onset_in_bin = bin_members_onset[bin_idx]
                
                # Track which years contribute to this bin
                contributing_years = []
                
                # Handle the "before initialization" bin
                if bin_start == -float('inf'):
                    bin_label = 'Before initialization'
                    
                    # Check if observed onset occurs before initialization (by day of year)
                    observed_onset = int(obs_onset_doy <= init_doy)
                    
                    # Get contributing years
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day <= 0:
                            contributing_years.append(ensemble_years_with_data[i])
                            
                # Handle the "after max_forecast_day" bin
                elif bin_start > max_forecast_day:
                    bin_label = f'After day {max_forecast_day}'
                    
                    # Check if observed onset occurs after max_forecast_day (by day of year)
                    obs_days_from_init = obs_onset_doy - init_doy
                    
                    observed_onset = int(obs_days_from_init > max_forecast_day)
                    
                    # Get contributing years
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and member_onset_day > max_forecast_day:
                            contributing_years.append(ensemble_years_with_data[i])
                            
                else:
                    # Regular bin within forecast window
                    bin_label = f'Days {bin_start}-{bin_end}'
                    
                    # Check if observed onset falls within this day bin (by day of year)
                    obs_days_from_init = obs_onset_doy - init_doy
                    observed_onset = int(bin_start <= obs_days_from_init <= bin_end)
                    
                    # Get contributing years
                    for i, member_onset_day in enumerate(ensemble_forecast_days):
                        if member_onset_day is not None and bin_start <= member_onset_day <= bin_end:
                            contributing_years.append(ensemble_years_with_data[i])
                
                # Calculate probability using only members that showed onset
                predicted_prob = members_with_onset_in_bin / total_members_with_onset
                
                # Convert contributing years to string for storage
                contributing_years_str = ','.join(map(str, sorted(contributing_years))) if contributing_years else ''
                
                # Store result
                result = {
                    'init_time': init_date.strftime('%Y-%m-%d'),
                    'lat': lat,
                    'lon': lon,
                    'bin_start': bin_start,
                    'bin_end': bin_end,
                    'bin_label': bin_label,
                    'predicted_prob': predicted_prob,
                    'observed_onset': observed_onset,
                    'members_with_onset': members_with_onset_in_bin,
                    'total_members': total_members,
                    'total_members_with_onset': total_members_with_onset,  # New field
                    'contributing_years': contributing_years_str,
                    'n_contributing_years': len(contributing_years),
                    'year': target_year,
                    'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d'),
                    'obs_onset_doy': obs_onset_doy,
                    'init_doy': init_doy,
                    'obs_days_from_init_doy': obs_days_from_init if 'obs_days_from_init' in locals() else (obs_onset_doy - init_doy),
                    'bin_index': bin_idx,
                    'forecast_type': 'climatological_doy'
                }
                results_list.append(result)
    
    # Convert to DataFrame
    forecast_obs_df = pd.DataFrame(results_list)
    
    if len(forecast_obs_df) == 0:
        print("Warning: No forecast-observation pairs generated")
        return forecast_obs_df
    
    print(f"Generated {len(forecast_obs_df)} climatological forecast-observation pairs")
    print(f"Unique lat-lon pairs processed: {len(unique_pairs)}")
    print(f"Total bins per forecast: {len(extended_bins)}")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")
    
    # Verify uniqueness
    unique_locations_in_output = len(forecast_obs_df[['lat', 'lon']].drop_duplicates())
    print(f"Unique locations in output: {unique_locations_in_output}")
    
    # Show distribution across bins
    print(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean',
        'n_contributing_years': 'mean',
        'total_members_with_onset': 'mean'
    }).round(3)
    print(bin_stats)
    
    return forecast_obs_df

def multi_year_climatological_forecast_obs_pairs(clim_onset, target_years, day_bins, mem_num, model_forecast_dir, date_filter_year=2024, file_pattern='tp_4p0_{}.nc', max_forecast_day=15, mok=True):
    """
    Create climatological forecast-observation pairs for multiple target years.
    
    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates
    target_years : list
        Years to use as truth for observations
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins
    max_forecast_day : int, default=15
        Maximum forecast day
    mok : bool, default=True
        Whether to use MOK date filter
    
    Returns:
    --------
    DataFrame with combined forecast-observation pairs from all target years
    """
    # Load threshold data (same for all years)

    orig_lat = clim_onset.lat.values
    orig_lon = clim_onset.lon.values

    lat_diff = abs(orig_lat[1]-orig_lat[0])
    if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
        polygon1_lon = np.array([83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83])
        polygon1_lat = np.array([17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17])
        print("Using 2-degree CMZ polygon coordinates")
    elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
        polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
        polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
        print("Using 4-degree CMZ polygon coordinates")
    elif abs(lat_diff - 1.0) < 0.1:  # 1-degree resolution
        polygon1_lon = np.array([74, 85, 85, 86, 86, 87, 87, 88, 88, 88, 85, 85, 82, 82, 79, 79, 78, 78, 69, 69, 74, 74])
        polygon1_lat = np.array([18, 18, 19, 19, 20, 20, 21, 21, 21, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 21, 21, 18])
        print("Using 1-degree CMZ polygon coordinates")

    inside_mask, inside_lons, inside_lats = points_inside_polygon(polygon1_lon, polygon1_lat, orig_lon, orig_lat)
    clim_onset_slice = clim_onset.sel(lat=inside_lats, lon=inside_lons)

    all_forecast_obs_pairs = []
    
    for target_year in target_years:
        print(f"\n{'='*50}")
        print(f"Processing target year {target_year}")
        print(f"{'='*50}")
        
        try:
            # Get initialization dates for this year
            _,init_dates = get_forecast_probabilistic_twice_weekly(target_year, model_forecast_dir, mem_num, date_filter_year, file_pattern)
            
            
            # Create forecast-observation pairs for this year
            forecast_obs_pairs = create_climatological_forecast_obs_pairs(
                clim_onset=clim_onset_slice,
                target_year=target_year,
                init_dates=init_dates,
                day_bins=day_bins,
                max_forecast_day=max_forecast_day,
                mok=mok
            )
            
            if len(forecast_obs_pairs) > 0:
                all_forecast_obs_pairs.append(forecast_obs_pairs)
                print(f"Target year {target_year} completed: {len(forecast_obs_pairs)} pairs")
            else:
                print(f"No pairs generated for target year {target_year}")
            
        except Exception as e:
            print(f"Error processing target year {target_year}: {e}")
            continue
    
    # Combine all years
    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any target year")
    
    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)
    
    print(f"\n{'='*50}")
    print("CLIMATOLOGICAL FORECAST SUMMARY")
    print(f"{'='*50}")
    print(f"Target years processed: {target_years}")
    print(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
    print(f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}")
    print(f"Overall observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}")
    
    return combined_forecast_obs


def calculate_brier_score_climatology(forecast_obs_df):
    """
    Calculate Brier Score and Fair Brier Score for probabilistic forecasts.
    
    Brier Score = (1/n*m) * Σ(Y_ij - p_ij)²
    Fair Brier Score = (1/n*m) * Σ[(Y_ij - p_ij)² - p_ij(1-p_ij)/(ens-1)]
    
    where:
    - n = number of forecasts
    - m = number of bins per forecast  
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - ens = number of ensemble members
    
    Note: Excludes "Before initialization" bin from calculations
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members', 'bin_label'
    
    Returns:
    --------
    dict with Brier score metrics
    """
    
    # Filter out "Before initialization" bin
    filtered_df = forecast_obs_df[forecast_obs_df['bin_label'] != 'Before initialization'].copy()
    
    if len(filtered_df) == 0:
        print("Warning: No data remaining after filtering out 'Before initialization' bin")
        return {
            'brier_score': np.nan,
            'fair_brier_score': np.nan,
            'bin_brier_scores': {},
            'bin_fair_brier_scores': {},
            'n_samples': 0,
            'filtered_bins': []
        }
    
    print(f"Calculating Brier Score excluding 'Before initialization' bin")
    print(f"Original samples: {len(forecast_obs_df)}, After filtering: {len(filtered_df)}")
    
    # Calculate squared differences
    squared_diffs = (filtered_df['observed_onset'] - filtered_df['predicted_prob'])**2
    
    # Calculate overall Brier Score
    brier_score = squared_diffs.mean()
    
    # Calculate Fair Brier Score correction term
    # ens-1 where ens is the number of ensemble members
    correction_term = (filtered_df['predicted_prob'] * (1 - filtered_df['predicted_prob'])) / (filtered_df['total_members_with_onset'] - 1)
    
    # Fair Brier Score
    fair_brier_components = squared_diffs - correction_term
    fair_brier_score = fair_brier_components.mean()
    
    # Calculate squared differences for bin-wise analysis
    filtered_df['squared_diff'] = squared_diffs
    filtered_df['fair_brier_component'] = fair_brier_components
    
    # Bin-wise Brier scores (excluding "Before initialization")
    bin_brier_scores = filtered_df.groupby('bin_label')['squared_diff'].mean()
    bin_fair_brier_scores = filtered_df.groupby('bin_label')['fair_brier_component'].mean()
    
    brier_results = {
        'brier_score': brier_score,
        'fair_brier_score': fair_brier_score,
        'bin_brier_scores': bin_brier_scores.to_dict(),
        'bin_fair_brier_scores': bin_fair_brier_scores.to_dict(),
        'n_samples': len(filtered_df),
        'filtered_bins': sorted(filtered_df['bin_label'].unique()),
        'excluded_bins': ['Before initialization']
    }
    
    print(f"Brier Score (excluding 'Before initialization'): {brier_results['brier_score']:.4f}")
    print(f"Fair Brier Score (excluding 'Before initialization'): {brier_results['fair_brier_score']:.4f}")
    print(f"Bins included in calculation: {brier_results['filtered_bins']}")

    return brier_results

def calculate_auc_climatology(forecast_obs_df):
    """
    Calculate Area Under the Curve (AUC) for probabilistic forecasts.
    
    AUC = Σ_{i,j,i',j'} Y_{ij}(1-Y_{i'j'}) · 1[p_{ij} > p_{i'j'}] / 
          [(Σ_{i,j} Y_{ij})(Σ_{i,j} (1-Y_{ij}))]
    
    where:
    - Y_{ij} = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_{ij} = predicted probability for bin j in forecast i
    - 1[p_{ij} > p_{i'j'}] = indicator function (1 if true, 0 if false)
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset'
    
    Returns:
    --------
    dict with AUC metrics
    """
    forecast_obs_df = forecast_obs_df[forecast_obs_df['bin_label'] != 'Before initialization'].copy()
    # Extract probabilities and observations
    p_ij = forecast_obs_df['predicted_prob'].values
    y_ij = forecast_obs_df['observed_onset'].values
    
    # Count total positive and negative cases
    n_positive = np.sum(y_ij)  # Σ Y_{ij}
    n_negative = np.sum(1 - y_ij)  # Σ (1-Y_{ij})
    
    if n_positive == 0 or n_negative == 0:
        print("Warning: Cannot calculate AUC - all cases are either positive or negative")
        return {
            'auc': np.nan,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'bin_auc_scores': {},
            'forecast_obs_df_with_ranks': forecast_obs_df
        }
    
    # Calculate AUC using the Mann-Whitney U statistic approach
    # This is equivalent to the formula but more computationally efficient
    
    # Separate positive and negative cases
    positive_probs = p_ij[y_ij == 1]
    negative_probs = p_ij[y_ij == 0]
    
    # Calculate Mann-Whitney U statistic
    u_statistic, _ = stats.mannwhitneyu(positive_probs, negative_probs, alternative='greater')
    
    # AUC is U statistic divided by (n_positive * n_negative)
    auc = u_statistic / (n_positive * n_negative)
    
    # Alternative direct calculation (less efficient for large datasets)
    # concordant_pairs = 0
    # for i in range(len(p_ij)):
    #     if y_ij[i] == 1:  # positive case
    #         for j in range(len(p_ij)):
    #             if y_ij[j] == 0:  # negative case
    #                 if p_ij[i] > p_ij[j]:
    #                     concordant_pairs += 1
    # auc_direct = concordant_pairs / (n_positive * n_negative)
    
    # Calculate AUC by bin
    bin_auc_scores = {}
    unique_bins = forecast_obs_df['bin_label'].unique()
    
    for bin_label in unique_bins:
        bin_data = forecast_obs_df[forecast_obs_df['bin_label'] == bin_label]
        
        if len(bin_data) > 0:
            bin_p = bin_data['predicted_prob'].values
            bin_y = bin_data['observed_onset'].values
            
            bin_n_positive = np.sum(bin_y)
            bin_n_negative = np.sum(1 - bin_y)
            
            if bin_n_positive > 0 and bin_n_negative > 0:
                bin_positive_probs = bin_p[bin_y == 1]
                bin_negative_probs = bin_p[bin_y == 0]
                
                bin_u_stat, _ = stats.mannwhitneyu(bin_positive_probs, bin_negative_probs, alternative='greater')
                bin_auc = bin_u_stat / (bin_n_positive * bin_n_negative)
            else:
                bin_auc = np.nan
            
            bin_auc_scores[bin_label] = bin_auc


    auc_results = {
        'auc': auc,
        'bin_auc_scores': bin_auc_scores,
    }
    
    return auc_results


def calculate_skill_scores(brier_forecast, rps_forecast, 
                          brier_climatology, rps_climatology):
    """
    Calculate skill scores for forecast model relative to climatology.
    
    Skill Score = 1 - (forecast_score / climatology_score)
    
    Parameters:
    -----------
    brier_forecast : dict
        Brier score results from forecast model
    rps_forecast : dict  
        RPS results from forecast model
    brier_climatology : dict
        Brier score results from climatology
    rps_climatology : dict
        RPS results from climatology
        
    Returns:
    --------
    dict with skill scores
    """
    
    skill_scores = {}
    
    print("="*60)
    print("SKILL SCORE CALCULATIONS")
    print("="*60)
    
    # Fair Brier Skill Score (1-15 day overall)
    fair_bss_overall = 1 - (brier_forecast['fair_brier_score'] / brier_climatology['fair_brier_score'])
    skill_scores['fair_brier_skill_score'] = fair_bss_overall
    
    print(f"Fair Brier Skill Score (1-15 day): {fair_bss_overall:.4f}")
    
    # Fair RPS Skill Score (1-15 day overall)
    fair_rpss_overall = 1 - (rps_forecast['fair_rps'] / rps_climatology['fair_rps'])
    skill_scores['fair_rps_skill_score'] = fair_rpss_overall
    
    print(f"Fair RPS Skill Score (1-15 day): {fair_rpss_overall:.4f}")
    
    # Automatically extract target bins from the data, excluding unwanted bins
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())
    
    # Get intersection of bins present in both forecast and climatology
    common_bins = all_forecast_bins.intersection(all_clim_bins)
    
    # Filter out unwanted bins and keep only "Days X-Y" format bins
    target_bins = []
    excluded_bins = []
    
    for bin_label in common_bins:
        # Include only bins that start with "Days " and don't contain "After" or "Before"
        if (bin_label.startswith('Days ') and 
            not bin_label.startswith('After') and 
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)
        else:
            excluded_bins.append(bin_label)
    
    # Sort bins by their day ranges
    def extract_day_range(bin_label):
        # Extract the start day from "Days X-Y" format
        if 'Days ' in bin_label:
            try:
                day_part = bin_label.replace('Days ', '').split('-')[0]
                return int(day_part)
            except:
                return 999  # Put unparseable bins at the end
        return 999
    
    target_bins = sorted(target_bins, key=extract_day_range)
    
    print(f"\nAutomatically detected target bins: {target_bins}")
    print(f"Excluded bins: {excluded_bins}")
    
    # Bin-wise Fair Brier Skill Scores
    bin_fair_bss = {}
    
    print(f"\nBin-wise Fair Brier Skill Scores:")
    for bin_label in target_bins:
        if bin_label in brier_forecast['bin_fair_brier_scores'] and bin_label in brier_climatology['bin_fair_brier_scores']:
            forecast_fair_brier_bin = brier_forecast['bin_fair_brier_scores'][bin_label]
            clim_fair_brier_bin = brier_climatology['bin_fair_brier_scores'][bin_label]
            
            fair_bss_bin = 1 - (forecast_fair_brier_bin / clim_fair_brier_bin)
            bin_fair_bss[bin_label] = fair_bss_bin
            
            print(f"  {bin_label}: Fair BSS = {fair_bss_bin:.4f}")
        else:
            bin_fair_bss[bin_label] = np.nan
            print(f"  {bin_label}: Fair BSS = NaN (missing data)")
    
    skill_scores['bin_fair_brier_skill_scores'] = bin_fair_bss
    
    # Dynamic table header based on detected bins
    header = f"{'Metric':<30} {'Overall (1-15 day)':<18}"
    for bin_name in target_bins:
        # Shorten bin names for table display
        short_name = bin_name.replace('Days ', '')
        header += f" {short_name:<12}"
    
    # Calculate table width
    table_width = 30 + 18 + 12 * len(target_bins)
    
    # Summary table
    print(f"\n" + "="*table_width)
    print("SKILL SCORE SUMMARY TABLE")
    print("="*table_width)
    print(header)
    print("-"*table_width)
    
    # Fair Brier Skill Score row
    fair_bss_row = f"{'Fair Brier Skill Score':<30} {fair_bss_overall:<18.4f}"
    for bin_name in target_bins:
        if bin_name in bin_fair_bss and not pd.isna(bin_fair_bss[bin_name]):
            fair_bss_row += f" {bin_fair_bss[bin_name]:<12.4f}"
        else:
            fair_bss_row += f" {'N/A':<12}"
    print(fair_bss_row)
    
    # Fair RPS Skill Score row
    fair_rpss_row = f"{'Fair RPS Skill Score':<30} {fair_rpss_overall:<18.4f}"
    for bin_name in target_bins:
        fair_rpss_row += f" {'N/A':<12}"  # RPS is overall only
    print(fair_rpss_row)
    
    print("-"*table_width)
    
    # Add interpretation guide
    print(f"\nInterpretation Guide:")
    print(f"• Positive skill scores indicate forecast is better than climatology")
    print(f"• Negative skill scores indicate forecast is worse than climatology") 
    print(f"• Skill score = 0 means forecast equals climatology")
    print(f"• Perfect score = 1.0")
    
    # Additional detailed results
    print(f"\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    print(f"\nForecast Fair Brier Score : {brier_forecast['fair_brier_score']:.4f}")
    print(f"Climatology Fair Brier Score : {brier_climatology['fair_brier_score']:.4f}")
    print(f"Fair Brier Skill Score: {fair_bss_overall:.4f}")
    
    print(f"\nForecast Fair RPS : {rps_forecast['fair_rps']:.4f}")
    print(f"Climatology Fair RPS : {rps_climatology['fair_rps']:.4f}")
    print(f"Fair RPS Skill Score: {fair_rpss_overall:.4f}")
    
    print(f"\nBin-wise Fair Brier Score Comparisons:")
    for bin_name in target_bins:
        if bin_name in brier_forecast['bin_fair_brier_scores'] and bin_name in brier_climatology['bin_fair_brier_scores']:
            forecast_val = brier_forecast['bin_fair_brier_scores'][bin_name]
            clim_val = brier_climatology['bin_fair_brier_scores'][bin_name]
            skill_val = bin_fair_bss[bin_name]
            print(f"  {bin_name}:")
            print(f"    Forecast: {forecast_val:.4f}")
            print(f"    Climatology: {clim_val:.4f}")
            print(f"    Skill Score: {skill_val:.4f}")
        else:
            print(f"  {bin_name}: Missing data")
    
    return skill_scores


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compute S2S Monsoon Onset Skill Scores')
    
    parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022, 2023, 2024],
                        help='Years to process (e.g., --years 2019 2020 2021)')
    parser.add_argument('--model_forecast_dir', type=str, required=True,
                        help='Directory containing S2S model data')
    parser.add_argument('--imd_folder', type=str, required=True,
                        help='Directory containing IMD rainfall data')
    parser.add_argument('--thres_file', type=str, required=True,
                        help='Path to threshold NetCDF file')
    parser.add_argument('--mem_num', type=int, required=True,
                        help='Number of ensemble members to use')
    parser.add_argument('--date_filter_year', type=int, default=2024,
                        help='Year to use for date filtering (default: 2024)')
    parser.add_argument('--file_pattern', type=str, default='{}.nc',
                        help='File pattern for S2S data (default: {}.nc)')
    parser.add_argument('--max_forecast_day', type=int, default=30,
                        help='Maximum forecast day (15 or 30, default: 30)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name for output files (e.g., gencast, fuxi)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save output CSV and PNG files (default: current directory)')
    parser.add_argument('--mok', action='store_true', default=True,
                        help='Use MOK date filter (June 2nd) for onset detection (default: True)')
    parser.add_argument('--no_mok', action='store_false', dest='mok',
                        help='Disable MOK date filter (use May 1st instead)')
    
    return parser.parse_args()

def get_target_bins(brier_forecast, brier_climatology):
    """Extract and sort target bins"""
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())
    common_bins = all_forecast_bins.intersection(all_clim_bins)
    
    target_bins = []
    for bin_label in common_bins:
        if (bin_label.startswith('Days ') and 
            not bin_label.startswith('After') and 
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)
    
    def extract_day_range(bin_label):
        if 'Days ' in bin_label:
            try:
                day_part = bin_label.replace('Days ', '').split('-')[0]
                return int(day_part)
            except:
                return 999
        return 999
    
    return sorted(target_bins, key=extract_day_range)

def save_results(auc_forecast, brier_forecast, skill_results, rps_forecast,
                auc_climatology, brier_climatology, model_name, max_forecast_day, save_dir=None):
    """
    Save overall and binned skill scores to CSV files
    
    Parameters:
    -----------
    auc_forecast : dict
        AUC results for forecast model
    brier_forecast : dict
        Brier score results for forecast model
    skill_results : dict
        Skill score results
    rps_forecast : dict
        RPS results for forecast model
    auc_climatology : dict
        AUC results for climatology
    brier_climatology : dict
        Brier score results for climatology
    model_name : str
        Name of the model for output filenames
    max_forecast_day : int
        Maximum forecast day (15 or 30)
    save_dir : str, optional
        Directory to save CSV files. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    """
    
    # Handle save directory
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved in: {save_dir}")
        
        # Ensure save_dir ends with a path separator for proper joining
        if not save_dir.endswith(os.sep):
            save_dir += os.sep
    else:
        save_dir = ""
        print("Results will be saved in current directory")
    
    # Save overall scores
    overall_scores = {
        'Metric': ['AUC', 'Fair Brier Score', 'Fair Brier Skill Score', 'Fair RPS', 'Fair RPS Skill Score'],
        'Score': [
            auc_forecast['auc'],
            brier_forecast['fair_brier_score'],
            skill_results['fair_brier_skill_score'],
            rps_forecast['fair_rps'],
            skill_results['fair_rps_skill_score']
        ]
    }
    
    overall_df = pd.DataFrame(overall_scores)
    overall_filename = f'{save_dir}overall_skill_scores_{model_name}_{max_forecast_day}day.csv'
    overall_df.to_csv(overall_filename, index=False)
    print(f"Saved overall scores to '{overall_filename}'")
    print(overall_df)
    
    # Get target bins
    target_bins = get_target_bins(brier_forecast, brier_climatology)
    
    # Save binned scores
    binned_data = {
        'Bin': target_bins,
        'Fair_Brier_Skill_Score': [skill_results['bin_fair_brier_skill_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'AUC': [auc_forecast['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Forecast': [brier_forecast['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Climatology': [brier_climatology['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    }
    
    binned_df = pd.DataFrame(binned_data)
    binned_filename = f'{save_dir}binned_skill_scores_{model_name}_{max_forecast_day}day.csv'
    binned_df.to_csv(binned_filename, index=False)
    print(f"Saved binned scores to '{binned_filename}'")
    print(binned_df)
    
    return overall_filename, binned_filename

def create_heatmap(skill_results, auc_forecast, auc_climatology, 
                  brier_forecast, brier_climatology, model_name, max_forecast_day, save_dir=None):
    """
    Create and save skill score heatmap
    
    Parameters:
    -----------
    ... (other parameters)
    save_dir : str, optional
        Directory to save the heatmap. If None, saves in current directory.
        If directory doesn't exist, it will be created.
    """
    
    # Handle save directory
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Ensure save_dir ends with a path separator for proper joining
        if not save_dir.endswith(os.sep):
            save_dir += os.sep
    else:
        save_dir = ""
    
    target_bins = get_target_bins(brier_forecast, brier_climatology)
    
    # Prepare data
    bss_values = [skill_results['bin_fair_brier_skill_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    auc_values = [auc_forecast['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    auc_clim_values = [auc_climatology['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    
    bin_labels_short = [bin_name.replace('Days ', '') for bin_name in target_bins]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    
    # Plot 1: BSS heatmap
    bss_data = np.array(bss_values).reshape(1, -1)
    sns.heatmap(bss_data*100, 
                annot=True, 
                fmt='.2g', 
                cmap='RdBu',
                vmin=-40, vmax=40,
                center=0,
                xticklabels=bin_labels_short,
                cbar=False,
                ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    ax1.set_ylabel('BSS (%)', fontsize=14)
    ax1.set_yticklabels([])
    
    # Plot 2: AUC heatmap
    auc_data = np.array(auc_values).reshape(1, -1)
    sns.heatmap(auc_data, 
                annot=False,
                cmap='Blues',
                vmin=0.7, vmax=1.0,
                xticklabels=bin_labels_short,
                cbar=False,
                ax=ax2)
    
    # Add custom annotations
    for i, (auc_val, auc_clim_val) in enumerate(zip(auc_values, auc_clim_values)):
        if not np.isnan(auc_val) and not np.isnan(auc_clim_val):
            ax2.text(i + 0.5, 0.5, f'{auc_val:.2g}', 
                    ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black')
            ax2.text(i + 0.5, 0.2, f'({auc_clim_val:.2g})', 
                    ha='center', va='center', 
                    fontsize=8, color='darkblue')
        elif not np.isnan(auc_val):
            ax2.text(i + 0.5, 0.5, f'{auc_val:.2g}', 
                    ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black')
    
    ax2.set_xlabel('Forecast Day Bins', fontsize=14)
    ax2.set_ylabel('AUC', fontsize=14)
    ax2.set_yticklabels([])
    
    plt.tight_layout()
    
    # Save with model name and forecast days
    figure_filename = f'{save_dir}skill_scores_heatmap_{model_name}_{max_forecast_day}day.png'
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved as '{figure_filename}'")
    
    return figure_filename

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Set day bins based on max_forecast_day
    if args.max_forecast_day == 15:
        day_bins = [(1, 5), (6, 10), (11, 15)]
    elif args.max_forecast_day == 30:
        day_bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    else:
        raise ValueError(f"Unsupported max_forecast_day: {args.max_forecast_day}")
    
    print("="*60)
    print("S2S MONSOON ONSET SKILL SCORE ANALYSIS")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Years: {args.years}")
    print(f"Max forecast day: {args.max_forecast_day}")
    print(f"Day bins: {day_bins}")
    print(f"MOK filter: {args.mok} ({'June 2nd' if args.mok else 'May 1st'})")
    if args.save_dir:
        print(f"Output directory: {args.save_dir}")
    else:
        print("Output directory: current directory")
    print("="*60)
    
    # Process forecast model
    print("\n1. Processing forecast model...")
    forecast_obs_df = multi_year_forecast_obs_pairs(
        args.years, args.model_forecast_dir, args.imd_folder, args.thres_file, args.mem_num,
        args.max_forecast_day, day_bins, 
        date_filter_year=args.date_filter_year, 
        file_pattern=args.file_pattern,
        mok=args.mok
    )
    
    print("\n2. Calculating forecast scores...")
    brier_forecast = calculate_brier_score(forecast_obs_df)
    rps_forecast = calculate_rps(forecast_obs_df)
    auc_forecast = calculate_auc(forecast_obs_df)
    
    # Process climatology
    print("\n3. Computing climatological dataset...")
    thresh_ds = xr.open_dataset(args.thres_file)
    thresh_slice = thresh_ds['MWmean']
    clim_onset = compute_climatological_onset_dataset(
        args.imd_folder, thresh_slice, years=None, mok=args.mok
    )
    
    print("\n4. Processing climatological forecasts...")
    climatology_obs_df = multi_year_climatological_forecast_obs_pairs(
        clim_onset, args.years, day_bins, args.mem_num, args.model_forecast_dir, 
        date_filter_year=args.date_filter_year, 
        file_pattern=args.file_pattern, 
        max_forecast_day=args.max_forecast_day, 
        mok=args.mok
    )
    
    print("\n5. Calculating climatology scores...")
    brier_climatology = calculate_brier_score_climatology(climatology_obs_df)
    rps_climatology = calculate_rps(climatology_obs_df)
    auc_climatology = calculate_auc_climatology(climatology_obs_df)
    
    # Calculate skill scores
    print("\n6. Calculating skill scores...")
    skill_results = calculate_skill_scores(
        brier_forecast, rps_forecast,
        brier_climatology, rps_climatology
    )
    
    # Save results
    print("\n7. Saving results...")
    overall_file, binned_file = save_results(
        auc_forecast, brier_forecast, skill_results, rps_forecast,
        auc_climatology, brier_climatology, args.model_name, args.max_forecast_day,
        save_dir=args.save_dir
    )
    
    # Create heatmap
    print("\n8. Creating heatmap...")
    heatmap_file = create_heatmap(
        skill_results, auc_forecast, auc_climatology,
        brier_forecast, brier_climatology, args.model_name, args.max_forecast_day,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Output files:")
    print(f"- {overall_file}")
    print(f"- {binned_file}")
    print(f"- {heatmap_file}")

if __name__ == "__main__":
    main()