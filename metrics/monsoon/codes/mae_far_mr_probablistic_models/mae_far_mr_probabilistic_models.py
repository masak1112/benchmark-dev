#!/usr/bin/env python3
"""
Monsoon Onset MAE, FAR, MR Analysis for Deterministic Models

This script computes Mean Absolute Error (MAE), False Alarm Rate (FAR), 
and Miss Rate (MR) for monsoon onset predictions from deterministic models.
"""

import argparse
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
import geopandas as gpd

def get_forecast_probabilistic_twice_weekly(yr, model_forecast_dir):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    Filters for Mondays and Thursdays in the specified year.
    The forecast file is expected to be named as '{year}.nc' in the model_forecast_dir with 
    variable "tp" being daily accumulated rainfall with dimensions (init_time, lat, lon, step, member).

    Parameters:
    yr: int, year to load data for
    
    Returns:
    p_model: ndarray, precipitation data 
    """
    fname = f'{yr}.nc'
    file_path = os.path.join(model_forecast_dir, fname)
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Filter for twice weekly data from daily for the specified year
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2024, 7, 31)
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
    # Find common dates between desired dates and available dates
    available_init_times = pd.to_datetime(ds.init_time.values)
    matching_times = available_init_times[available_init_times.isin(filtered_dates_yr)]
        
    if len(matching_times) == 0:
        raise ValueError(f"No matching initialization times found for year {yr}")
        
    # Select only the matching initialization times
    ds = ds.sel(init_time=matching_times)
    if 'day' in ds.dims:
        # Check if the first value of 'day' is 0, then slice to exclude it
        if ds['day'][0].values == 0:
            ds = ds.sel(day=slice(1, None))
    # Check if 'step' dimension exists and conditionally slice
    if 'step' in ds.dims:
        # Check if the first value of 'step' is 0, then slice to exclude it
        if ds['step'][0].values == 0:
            ds = ds.sel(step=slice(1, None)) 
    if 'day' in ds.dims:        
        ds = ds.rename({'day': 'step'})
    
    p_model = ds['tp']  # in mm    
    ds.close()
    return p_model

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
    # Check for latitude/longitude dimensions
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
    
    return rainfall

def detect_observed_onset(rainfall_ds, thresh_slice, year, mok=True):
    """Detect observed onset dates for a given year."""
    rain_slice = rainfall_ds
    window = 5
    
    if mok:
        start_date = datetime(year, 6, 2)  # MOK date: June 2nd
        date_label = "MOK date (June 2nd)"
    else:
        start_date = datetime(year, 5, 1)  # May 1st
        date_label = "May 1st"

    time_dates = pd.to_datetime(rain_slice.time.values)
    start_idx_candidates = np.where(time_dates > start_date)[0]
    
    if len(start_idx_candidates) == 0:
        print(f"Warning: {date_label} ({start_date.strftime('%Y-%m-%d')}) not found in data for year {year}")
        fallback_date = datetime(year, 4, 1)
        start_idx = np.where(time_dates >= fallback_date)[0][0]
        print(f"Using fallback date: April 1st")
    else:
        start_idx = start_idx_candidates[0]
        print(f"Using {date_label} ({start_date.strftime('%Y-%m-%d')}) as start date for onset detection")

    rain_subset = rain_slice.isel(time=slice(start_idx, None))
    rolling_sum = rain_subset.rolling(time=window, min_periods=window, center=False).sum()
    rolling_sum_aligned = rolling_sum.shift(time=-(window-1))

    first_day_condition = rain_subset > 1
    sum_condition = rolling_sum_aligned > thresh_slice
    onset_condition = first_day_condition & sum_condition

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

    valid_mask = onset_indices.values >= 0
    time_coords = rain_subset.time.values
    onset_dates_array = np.full(onset_indices.shape, np.datetime64('NaT'), dtype='datetime64[ns]')

    for i in range(onset_indices.shape[0]):
        for j in range(onset_indices.shape[1]):
            if valid_mask[i, j]:
                idx = int(onset_indices[i, j].values)
                if 0 <= idx < len(time_coords):
                    onset_dates_array[i, j] = time_coords[idx]

    onset_da = xr.DataArray(
        onset_dates_array,
        coords=[('lat', rain_slice.lat.values), ('lon', rain_slice.lon.values)],
        name='onset_date'
    )
    
    return onset_da

def compute_mean_onset_for_all_members(p_model, thresh_slice, onset_da, max_forecast_day=15, mok=True):
    """
    Compute onset dates for each ensemble member, initialization time, and grid point.
    Only processes forecasts initialized before the observed onset date.
    For each initialization, requires at least 50% of members to have onset.
    If threshold met, uses ceiling of mean onset day as the ensemble onset.
    
    Parameters:
    p_model: xarray DataArray with dims [init_time, step, lat, lon, member]
    thresh_slice: xarray DataArray with threshold values for each grid point
    onset_da: xarray DataArray with observed onset dates for filtering
    max_forecast_day: int, maximum forecast day to consider for onset (default 15)
    mok: bool, if True only count onset after June 2nd (MOK date), if False use all forecasts
    
    Returns:
    pandas DataFrame with columns: init_time, lat, lon, onset_day, member_onset_count, total_members
    """
    
    window = 5
    results_list = []
    
    # Get dimensions
    init_times = p_model.init_time.values
    lats = p_model.lat.values  
    lons = p_model.lon.values
    members = p_model.member.values
    
    date_method = "MOK (June 2nd filter)" if mok else "no date filter"
    print(f"Processing {len(init_times)} init times x {len(lats)} lats x {len(lons)} lons...")
    print(f"Using {date_method} for onset detection")
    print(f"Only processing forecasts initialized before observed onset dates")
    print(f"Requiring ≥50% of {len(members)} members to have onset for ensemble onset")
    
    # We need first 19 days to check onset up to day 15 (because of 5-day window)
    max_steps_needed = max_forecast_day + window - 1
    
    # Track statistics
    total_potential_inits = 0
    valid_inits = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    ensemble_onsets_found = 0
    
    # Loop over all combinations
    for t_idx, init_time in enumerate(init_times):
        if t_idx % 5 == 0:  # Print progress every 5 init times
            print(f"Processing init time {t_idx+1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}")
        
        # Get init date for MOK filtering and onset comparison
        init_date = pd.to_datetime(init_time)
        year = init_date.year
        mok_date = datetime(year, 6, 2)  # June 2nd of the same year
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                
                total_potential_inits += 1
                
                # Get observed onset date for this grid point
                try:
                    obs_onset = onset_da.isel(lat=i, lon=j).values
                except:
                    skipped_no_obs += 1
                    continue
                
                # Skip if no observed onset
                if pd.isna(obs_onset):
                    skipped_no_obs += 1
                    continue
                
                # Convert observed onset to datetime
                obs_onset_dt = pd.to_datetime(obs_onset)
                
                # Only process if forecast was initialized before observed onset
                if init_date >= obs_onset_dt:
                    skipped_late_init += 1
                    continue
                
                valid_inits += 1
                
                # Get threshold for this grid point
                thresh = thresh_slice.isel(lat=i, lon=j).values
                
                # Collect onset days for all members at this init/location
                member_onset_days = []
                
                for m_idx, member in enumerate(members):
                    try:
                        # Extract forecast time series for this member
                        forecast_series = p_model.isel(
                            init_time=t_idx,
                            lat=i, 
                            lon=j,
                            member=m_idx,
                        ).sel(step=slice(1, max_steps_needed)).values
                        
                        if len(forecast_series) < max_steps_needed:
                            member_onset_days.append(None)
                            continue
                        
                        # Check for onset on each possible day
                        member_onset_day = None
                        
                        for day in range(1, max_forecast_day + 1):
                            start_idx = day - 1
                            end_idx = start_idx + window 

                            if end_idx <= len(forecast_series):
                                window_series = forecast_series[start_idx:end_idx]
                                
                                # Check basic onset condition: first day > 1mm AND 5-day sum > threshold
                                if window_series[0] > 1 and np.nansum(window_series) > thresh:
                                    
                                    # Calculate the actual date this forecast day represents
                                    forecast_date = init_date + pd.Timedelta(days=day)
                                    
                                    # If MOK flag is True, only count onset if it's on or after June 2nd
                                    if mok:
                                        if forecast_date.date() > mok_date.date():
                                            member_onset_day = day
                                            break  # Found valid onset after MOK date
                                        # else: continue checking later days
                                    else:
                                        # No MOK filtering, count this onset
                                        member_onset_day = day
                                        break
                        
                        member_onset_days.append(member_onset_day)
                        
                    except Exception as e:
                        print(f"Error at init_time {t_idx}, lat {i}, lon {j}, member {m_idx}: {e}")
                        member_onset_days.append(None)
                
                # Now check if at least 50% of members have onset
                valid_onsets = [day for day in member_onset_days if day is not None]
                onset_count = len(valid_onsets)
                total_members = len(member_onset_days)
                onset_percentage = onset_count / total_members if total_members > 0 else 0
                
                # Determine ensemble onset day
                ensemble_onset_day = None
                ensemble_onset_date = None
                if onset_percentage >= 0.5:  # At least 50% of members have onset
                    # Use rounding of mean onset day
                    mean_onset = np.mean(valid_onsets)
                    ensemble_onset_day = int(round(mean_onset))
                    ensemble_onsets_found += 1
                    ensemble_onset_date = init_date + pd.Timedelta(days=ensemble_onset_day)
                
                # Store result
                result = {
                    'init_time': init_time,
                    'lat': lat,
                    'lon': lon,
                    'onset_day': ensemble_onset_day,  # None if <50% members have onset
                    'onset_date': ensemble_onset_date.strftime('%Y-%m-%d') if ensemble_onset_date is not None else None,
                    'member_onset_count': onset_count,
                    'total_members': total_members,
                    'onset_percentage': onset_percentage,
                    'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')  # Store observed onset for reference
                }
                results_list.append(result)
    
    # Convert to DataFrame
    onset_df = pd.DataFrame(results_list)
    
    print(f"\nProcessing Summary:")
    print(f"Total potential initializations: {total_potential_inits}")
    print(f"Skipped (no observed onset): {skipped_no_obs}")
    print(f"Skipped (initialized after observed onset): {skipped_late_init}")
    print(f"Valid initializations processed: {valid_inits}")
    print(f"Ensemble onsets found (≥50% members): {ensemble_onsets_found}")
    print(f"Ensemble onset rate: {ensemble_onsets_found/valid_inits:.3f}" if valid_inits > 0 else "Ensemble onset rate: 0.000")
    
    if mok:
        print(f"Note: Only onsets on or after June 2nd were counted due to MOK flag")
    
    return onset_df

def compute_onset_metrics_with_windows(onset_df, tolerance_days=3, verification_window=1, forecast_days=15):
    """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows."""
    print(f"Computing onset metrics with tolerance = {tolerance_days} days")
    print(f"Verification window starts {verification_window} days after initialization")
    print(f"Forecast window length: {forecast_days} days")
    
    results_list = []
    unique_locations = onset_df[['lat', 'lon']].drop_duplicates()
    
    print(f"Processing {len(unique_locations)} unique grid points...")
    
    for idx, (_, row) in enumerate(unique_locations.iterrows()):
        lat, lon = row['lat'], row['lon']
        
        if idx % 10 == 0:
            print(f"Processing grid point {idx+1}/{len(unique_locations)}: lat={lat:.2f}, lon={lon:.2f}")
        
        grid_data = onset_df[(onset_df['lat'] == lat) & (onset_df['lon'] == lon)].copy()
        
        grid_data['obs_onset_dt'] = pd.to_datetime(grid_data['obs_onset_date'])
        grid_data['model_onset_dt'] = pd.to_datetime(grid_data['onset_date'])
        grid_data['init_dt'] = pd.to_datetime(grid_data['init_time'])
        
        TP = 0
        FP = 0 
        FN = 0
        TN = 0
        num_onset = 0
        num_no_onset = 0
        mae_tp = []
        mae_fp = []
        
        gt_grd = grid_data['obs_onset_dt'].iloc[0]
        
        true_onset_window_start = gt_grd - pd.Timedelta(days=tolerance_days)
        true_onset_window_end = gt_grd + pd.Timedelta(days=tolerance_days)
        
        for _, init_row in grid_data.iterrows():
            t_init = init_row['init_dt']
            model_onset = init_row['model_onset_dt']
            
            valid_window_start = t_init + pd.Timedelta(days=verification_window)
            valid_window_end = valid_window_start + pd.Timedelta(days=14)
            
            whole_forecast_window_start = t_init + pd.Timedelta(days=1)
            whole_forecast_window_end = t_init + pd.Timedelta(days=forecast_days)
            
            is_onset_in_whole_window = whole_forecast_window_start <= gt_grd <= whole_forecast_window_end
            if is_onset_in_whole_window:
                num_onset += 1
            else:
                num_no_onset += 1
            
            has_model_onset = not pd.isna(model_onset)
            
            if has_model_onset:
                is_model_in_valid_window = valid_window_start <= model_onset <= valid_window_end
                
                if is_model_in_valid_window:
                    abs_diff_days = abs((model_onset - gt_grd).days)
                    
                    if abs_diff_days <= tolerance_days:
                        TP += 1
                        mae_tp.append(abs_diff_days)
                    else:
                        FP += 1
                        mae_fp.append(abs_diff_days)
                        
            else:
                if is_onset_in_whole_window:
                    FN += 1
                else:
                    TN += 1
        
        total_forecasts = len(grid_data)
        
        mae_combined = mae_tp + mae_fp
        mae = np.mean(mae_combined) if len(mae_combined) > 0 else np.nan
        mae_tp_only = np.mean(mae_tp) if len(mae_tp) > 0 else np.nan
        
        result = {
            'lat': lat,
            'lon': lon,
            'total_forecasts': total_forecasts,
            'true_positive': TP,
            'true_negative': TN,
            'false_positive': FP,
            'false_negative': FN,
            'num_onset': num_onset,
            'num_no_onset': num_no_onset,
            'mae_combined': mae,
            'mae_tp_only': mae_tp_only,
            'num_tp_errors': len(mae_tp),
            'num_fp_errors': len(mae_fp),
            'tolerance_days': tolerance_days,
            'verification_window': verification_window,
            'forecast_days': forecast_days
        }
        results_list.append(result)
    
    metrics_df = pd.DataFrame(results_list)
    
    summary_stats = {
        'total_grid_points': len(metrics_df),
        'total_forecasts': metrics_df['total_forecasts'].sum(),
        'overall_true_positive': metrics_df['true_positive'].sum(),
        'overall_true_negative': metrics_df['true_negative'].sum(),
        'overall_false_positive': metrics_df['false_positive'].sum(),
        'overall_false_negative': metrics_df['false_negative'].sum(),
        'overall_num_onset': metrics_df['num_onset'].sum(),
        'overall_num_no_onset': metrics_df['num_no_onset'].sum(),
        'overall_mae_combined': metrics_df['mae_combined'].mean(),
        'overall_mae_tp_only': metrics_df['mae_tp_only'].mean(),
        'tolerance_days': tolerance_days,
        'verification_window': verification_window,
        'forecast_days': forecast_days
    }
    
    return metrics_df, summary_stats

def compute_metrics_multiple_years(years, model_forecast_dir, imd_folder, thres_file, 
                                 tolerance_days=3, verification_window=1, forecast_days=15, 
                                 max_forecast_day=15, mok=True):
    """Compute onset metrics for multiple years."""
    metrics_df_dict = {}
    onset_da_dict = {}
    
    thresh_ds = xr.open_dataset(thres_file)
    thres_da = thresh_ds['MWmean']
    
    for year in years:
        print(f"\n{'='*50}")
        print(f"Processing year {year}")
        print(f"{'='*50}")
        
        p_model = get_forecast_probabilistic_twice_weekly(year, model_forecast_dir)
        imd = load_imd_rainfall(year, imd_folder)
        onset_da = detect_observed_onset(imd, thres_da, year, mok=mok)
        
        onset_df = compute_mean_onset_for_all_members(
            p_model, thres_da, onset_da, 
            max_forecast_day=max_forecast_day, mok=mok
        )
        
        metrics_df, summary_stats = compute_onset_metrics_with_windows(
            onset_df, 
            tolerance_days=tolerance_days, 
            verification_window=verification_window, 
            forecast_days=forecast_days
        )
        
        metrics_df_dict[year] = metrics_df
        onset_da_dict[year] = onset_da
        
        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
    
    return metrics_df_dict, onset_da_dict

def create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict):
    """Create spatial maps of False Alarm Rate, Miss Rate, yearly MAE, and mean MAE across years."""
    first_year = list(onset_da_dict.keys())[0]
    lats = onset_da_dict[first_year].lat.values
    lons = onset_da_dict[first_year].lon.values
    
    print(f"Creating spatial FAR, Miss Rate, yearly MAE, and mean MAE maps...")
    print(f"Grid dimensions: {len(lats)} lats x {len(lons)} lons")
    print(f"Years: {list(metrics_df_dict.keys())}")
    
    spatial_metrics = {}
    
    false_alarm_rate_map = np.full((len(lats), len(lons)), np.nan)
    miss_rate_map = np.full((len(lats), len(lons)), np.nan)
    mean_mae_map = np.full((len(lats), len(lons)), np.nan)
    
    yearly_mae_maps = {}
    for year in metrics_df_dict.keys():
        yearly_mae_maps[year] = np.full((len(lats), len(lons)), np.nan)
    
    for i, lat_val in enumerate(lats):
        for j, lon_val in enumerate(lons):
            
            total_FP = 0
            total_TN = 0
            total_FN = 0
            total_num_onset = 0
            
            mae_values = []
            has_any_valid_data = False
            
            for year, metrics_df in metrics_df_dict.items():
                obs_onset_val = onset_da_dict[year].isel(lat=i, lon=j).values
                
                if pd.isna(obs_onset_val):
                    continue
                
                grid_data = metrics_df[(metrics_df['lat'] == lat_val) & (metrics_df['lon'] == lon_val)]
                
                if len(grid_data) > 0:
                    has_any_valid_data = True
                    row = grid_data.iloc[0]
                    
                    total_FP += row['false_positive']
                    total_TN += row['true_negative'] 
                    total_FN += row['false_negative']
                    total_num_onset += row['num_onset']
                    
                    mae_val = row['mae_combined']
                    if not pd.isna(mae_val):
                        yearly_mae_maps[year][i, j] = mae_val
                        mae_values.append(mae_val)
            
            if has_any_valid_data:
                if (total_FP + total_TN) > 0:
                    false_alarm_rate_map[i, j] = total_FP / (total_FP + total_TN)
                else:
                    false_alarm_rate_map[i, j] = 0
                
                if total_num_onset > 0:
                    miss_rate_map[i, j] = total_FN / total_num_onset
                else:
                    miss_rate_map[i, j] = 0
                
                if len(mae_values) > 0:
                    mean_mae_map[i, j] = np.mean(mae_values)
    
    spatial_metrics['false_alarm_rate'] = xr.DataArray(
        false_alarm_rate_map, 
        coords=[('lat', lats), ('lon', lons)], 
        name='false_alarm_rate',
        attrs={'description': 'False Alarm Rate = sum(FP) / sum(FP + TN) across all valid years'}
    )
    
    spatial_metrics['miss_rate'] = xr.DataArray(
        miss_rate_map, 
        coords=[('lat', lats), ('lon', lons)], 
        name='miss_rate',
        attrs={'description': 'Miss Rate = sum(FN) / sum(total_onsets) across all valid years'}
    )
    
    spatial_metrics['mean_mae'] = xr.DataArray(
        mean_mae_map, 
        coords=[('lat', lats), ('lon', lons)], 
        name='mean_mae',
        attrs={'description': 'Mean MAE across all valid years (omitting NaN values)'}
    )
    
    for year, mae_map in yearly_mae_maps.items():
        spatial_metrics[f'mae_{year}'] = xr.DataArray(
            mae_map, 
            coords=[('lat', lats), ('lon', lons)], 
            name=f'mae_{year}',
            attrs={'description': f'Mean Absolute Error for year {year}'}
        )

    return spatial_metrics

def get_india_outline(shp_file_path):
    """Get India outline coordinates from shapefile."""
    import geopandas as gpd
    india_gdf = gpd.read_file(shp_file_path)
        
    boundaries = []
    for geom in india_gdf.geometry:
        if hasattr(geom, 'exterior'):
            coords = list(geom.exterior.coords)
            lon_coords = [coord[0] for coord in coords]
            lat_coords = [coord[1] for coord in coords]
            boundaries.append((lon_coords, lat_coords))
        elif hasattr(geom, 'geoms'):
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, 'exterior'):
                    coords = list(sub_geom.exterior.coords)
                    lon_coords = [coord[0] for coord in coords]
                    lat_coords = [coord[1] for coord in coords]
                    boundaries.append((lon_coords, lat_coords))
    return boundaries

def plot_spatial_metrics(spatial_metrics, shpfile_path, figsize=(18, 6), save_path=None):
    """
    Plot spatial maps of Mean MAE, False Alarm Rate, and Miss Rate in a 1x3 subplot
    with India outline, CMZ polygon, grid values displayed, and CMZ averages.
    """
    
    # Extract data
    mean_mae = spatial_metrics['mean_mae']
    far = spatial_metrics['false_alarm_rate'] * 100  # Convert to percentage
    miss_rate = spatial_metrics['miss_rate'] * 100   # Convert to percentage
    
    # Get coordinates
    lats = mean_mae.lat.values
    lons = mean_mae.lon.values
    
    # Detect resolution from latitude spacing
    lat_diff = abs(lats[1] - lats[0])
    print(f"Detected resolution: {lat_diff:.1f} degrees")
    
    # Define Core Monsoon Zone bounding polygon coordinates based on resolution 
    polygon_defined = False
    if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
        polygon1_lon = np.array([83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83])
        polygon1_lat = np.array([17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17])
        polygon_defined = True
        print("Using 2-degree CMZ polygon coordinates")
    elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
        polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
        polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
        polygon_defined = True
        print("Using 4-degree CMZ polygon coordinates")
    elif abs(lat_diff - 1.0) < 0.1:  # 1-degree resolution
        polygon1_lon = np.array([74, 85, 85, 86, 86, 87, 87, 88, 88, 88, 85, 85, 82, 82, 79, 79, 78, 78, 69, 69, 74, 74])
        polygon1_lat = np.array([18, 18, 19, 19, 20, 20, 21, 21, 21, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 21, 21, 18])
        polygon_defined = True
        print("Using 1-degree CMZ polygon coordinates")        
    else:
        print(f"Resolution {lat_diff:.1f} degrees not supported for CMZ polygon. Plotting without polygon and CMZ averages.")
        polygon_defined = False

    def calculate_cmz_averages(data_array, lons, lats, polygon_lon, polygon_lat):
        """Calculate spatial average within the CMZ polygon"""
        from matplotlib.path import Path
        polygon_path = Path(list(zip(polygon_lon, polygon_lat)))
        
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
        inside_polygon = polygon_path.contains_points(points).reshape(lon_grid.shape)
        
        values_inside = data_array.values[inside_polygon]
        
        if len(values_inside) > 0:
            return np.nanmean(values_inside)
        else:
            return np.nan
    
    def calculate_mae_stats_across_years(spatial_metrics, lons, lats, polygon_lon, polygon_lat):
        """Calculate MAE statistics: spatial average for each year, then mean ± SE across years"""
        yearly_mae_keys = [key for key in spatial_metrics.keys() if key.startswith('mae_') and key != 'mae_combined']
        
        if not yearly_mae_keys:
            print("Warning: No yearly MAE maps found")
            return np.nan, np.nan, np.nan, np.nan
        
        cmz_yearly_averages = []
        overall_yearly_averages = []
        
        for mae_key in yearly_mae_keys:
            year_mae_map = spatial_metrics[mae_key]
            
            if polygon_defined and polygon_lon is not None:
                cmz_avg = calculate_cmz_averages(year_mae_map, lons, lats, polygon_lon, polygon_lat)
                if not np.isnan(cmz_avg):
                    cmz_yearly_averages.append(cmz_avg)
            
            overall_avg = np.nanmean(year_mae_map.values)
            if not np.isnan(overall_avg):
                overall_yearly_averages.append(overall_avg)
        
        if len(cmz_yearly_averages) > 0 and polygon_defined:
            cmz_mean = np.mean(cmz_yearly_averages)
            cmz_se = np.std(cmz_yearly_averages, ddof=1) / np.sqrt(len(cmz_yearly_averages)) if len(cmz_yearly_averages) > 1 else 0
        else:
            cmz_mean, cmz_se = np.nan, np.nan
        
        if len(overall_yearly_averages) > 0:
            overall_mean = np.mean(overall_yearly_averages)
            overall_se = np.std(overall_yearly_averages, ddof=1) / np.sqrt(len(overall_yearly_averages)) if len(overall_yearly_averages) > 1 else 0
        else:
            overall_mean, overall_se = np.nan, np.nan
        
        return cmz_mean, cmz_se, overall_mean, overall_se
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Calculate statistics (only calculate CMZ stats if polygon is defined)
    if polygon_defined:
        cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = calculate_mae_stats_across_years(
            spatial_metrics, lons, lats, polygon1_lon, polygon1_lat
        )
        
        cmz_far = calculate_cmz_averages(spatial_metrics['false_alarm_rate'] * 100, lons, lats, polygon1_lon, polygon1_lat)
        cmz_mr = calculate_cmz_averages(spatial_metrics['miss_rate'] * 100, lons, lats, polygon1_lon, polygon1_lat)
    else:
        cmz_mae_mean, cmz_mae_se, overall_mae_mean, overall_mae_se = calculate_mae_stats_across_years(
            spatial_metrics, lons, lats, None, None
        )
        cmz_far = np.nan
        cmz_mr = np.nan
    
    # Create edges for pcolormesh (cell boundaries)
    lon_edges = np.concatenate([lons - (lons[1]-lons[0])/2, [lons[-1] + (lons[1]-lons[0])/2]])
    lat_edges = np.concatenate([lats - (lats[1]-lats[0])/2, [lats[-1] + (lats[1]-lats[0])/2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)
    
    # Plot parameters
    map_lw = 0.75
    polygon_lw = 1.25
    panel_linewidth = 0.5
    tick_length = 3
    tick_width = 0.8
    if abs(lat_diff - 2.0) < 0.1:
        txt_fsize = 8
    elif abs(lat_diff - 4.0) < 0.1:
        txt_fsize = 10
    elif abs(lat_diff - 1.0) < 0.1:
        txt_fsize = 6
    else:
        txt_fsize = 8
        
    # Panel 1: Mean MAE
    masked_mae = np.ma.masked_invalid(mean_mae.values)
    im1 = axes[0].pcolormesh(LON_edges, LAT_edges, masked_mae, 
                             cmap='OrRd', vmin=0, vmax=15, shading='flat')
    
    # Add India outline
    india_boundaries = get_india_outline(shpfile_path)
    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        axes[0].plot(india_lon, india_lat, color='black', linewidth=map_lw)
    
    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                         fill=False, edgecolor='black', linewidth=polygon_lw)
        axes[0].add_patch(polygon)
    
    # Add text annotations for MAE values
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = mean_mae.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > 7.5 else 'black'
                axes[0].text(lon, lat, f'{value:.1f}', 
                           ha='center', va='center',
                           color=text_color, fontsize=txt_fsize, fontweight='normal')
    
    # Add CMZ average text with mean ± SE across years (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_mae_mean):
        if cmz_mae_se > 0:
            cmz_text = f'MAE: {cmz_mae_mean:.1f}±{cmz_mae_se:.1f} days'
        else:
            cmz_text = f'MAE: {cmz_mae_mean:.1f} days'
        
        axes[0].text(0.98, 0.02, cmz_text, transform=axes[0].transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')

    axes[0].text(0.98, 0.98, 'MAE (in days)', transform=axes[0].transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')
    axes[0].set_xlabel('Longitude', fontsize=12)
    axes[0].set_ylabel('Latitude', fontsize=12)
    
    # Panel 2: False Alarm Rate
    masked_far = np.ma.masked_invalid(far.values)
    im2 = axes[1].pcolormesh(LON_edges, LAT_edges, masked_far, 
                             cmap='Reds', vmin=0, vmax=100, shading='flat')
    
    # Add India outline
    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        axes[1].plot(india_lon, india_lat, color='black', linewidth=map_lw)
    
    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                         fill=False, edgecolor='black', linewidth=polygon_lw)
        axes[1].add_patch(polygon)
    
    # Add text annotations for FAR values
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = far.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > 50 else 'black'
                axes[1].text(lon, lat, f'{value:.0f}', 
                           ha='center', va='center',
                           color=text_color, fontsize=txt_fsize, fontweight='normal')
    
    # Add CMZ average text (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_far):
        cmz_text = f'FAR: {cmz_far:.1f}%'
        axes[1].text(0.98, 0.02, cmz_text, transform=axes[1].transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')

    axes[1].text(0.98, 0.98, 'False Alarm Rate (%)', transform=axes[1].transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')
    axes[1].set_xlabel('Longitude', fontsize=12)
    
    # Panel 3: Miss Rate
    masked_mr = np.ma.masked_invalid(miss_rate.values)
    im3 = axes[2].pcolormesh(LON_edges, LAT_edges, masked_mr, 
                             cmap='Blues', vmin=0, vmax=100, shading='flat')
    
    # Add India outline
    for boundary in india_boundaries:
        india_lon, india_lat = boundary
        axes[2].plot(india_lon, india_lat, color='black', linewidth=map_lw)
    
    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                         fill=False, edgecolor='black', linewidth=polygon_lw)
        axes[2].add_patch(polygon)
    
    # Add text annotations for Miss Rate values
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = miss_rate.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > 50 else 'black'
                axes[2].text(lon, lat, f'{value:.0f}', 
                           ha='center', va='center',
                           color=text_color, fontsize=txt_fsize, fontweight='normal')
    
    # Add CMZ average text (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_mr):
        cmz_text = f'MR: {cmz_mr:.1f}%'
        axes[2].text(0.98, 0.02, cmz_text, transform=axes[2].transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')
    
    axes[2].text(0.98, 0.98, 'Miss Rate (%)', transform=axes[2].transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')

    axes[2].set_xlabel('Longitude', fontsize=12)
    
    # Set consistent axis limits and styling for all panels
    for i, ax in enumerate(axes):
        ax.set_xlim([lons.min()-2, lons.max()+2])
        ax.set_ylim([lats.min()-2, lats.max()+2])
        
        xticks = np.arange(lons.min(), lons.max()+1, 8)
        xticklabels = [f"{int(x)}°E" for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        
        if i == 0:
            yticks = np.arange(lats.min(), lats.max()+1, 4)
            yticklabels = [f"{int(y)}°N" for y in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        
        ax.tick_params(axis='both', which='major', labelsize=10, 
                      length=tick_length, width=tick_width)
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_linewidth(panel_linewidth)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Only print CMZ averages if polygon is defined
    if polygon_defined:
        print(f"\n=== CORE MONSOON ZONE (CMZ) AVERAGES ===")
        
        if not np.isnan(cmz_mae_mean):
            print(f"CMZ Mean MAE (avg across years): {cmz_mae_mean:.2f} ± {cmz_mae_se:.2f} days")
        else:
            print(f"CMZ Mean MAE: N/A")
        
        print(f"CMZ False Alarm Rate: {cmz_far:.1f} %")
        print(f"CMZ Miss Rate: {cmz_mr:.1f} %")
    else:
        print(f"\nNote: CMZ averages not calculated (resolution {lat_diff:.1f}° not supported)")
    
    return fig, axes

def main():
    parser = argparse.ArgumentParser(description='Compute MAE, FAR, MR for monsoon onset predictions')
    
    parser.add_argument('--years', nargs='+', type=int, required=True,
                        help='Years to process (e.g., 2019 2020 2021)')
    parser.add_argument('--model_forecast_dir', type=str, required=True,
                        help='Directory containing S2S model data')
    parser.add_argument('--imd_folder', type=str, required=True,
                        help='Directory containing IMD rainfall data')
    parser.add_argument('--thres_file', type=str, required=True,
                        help='Path to threshold NetCDF file')
    parser.add_argument('--shpfile_path', type=str, required=True,
                        help='Path to India shapefile')
    parser.add_argument('--tolerance_days', type=int, default=3,
                        help='Tolerance in days for onset prediction (default: 3)')
    parser.add_argument('--verification_window', type=int, default=1,
                        help='Days after init to start validation window (default: 1)')
    parser.add_argument('--forecast_days', type=int, default=15,
                        help='Length of forecast window in days (default: 15)')
    parser.add_argument('--max_forecast_day', type=int, default=15,
                        help='Maximum forecast day to consider for onset (default: 15)')
    parser.add_argument('--mok', action='store_true',
                        help='Use MOK date filter (June 2nd) for onset detection')
    parser.add_argument('--output_file', type=str, default='spatial_metrics.nc',
                        help='Output NetCDF file name (default: spatial_metrics.nc)')
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='Directory to save plot PNG file (optional)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[18, 6],
                        help='Figure size in inches [width height] (default: 18 6)')
    
    args = parser.parse_args()
    
    print(f"Processing years: {args.years}")
    print(f"S2S data directory: {args.model_forecast_dir}")
    print(f"IMD folder: {args.imd_folder}")
    print(f"Threshold file: {args.thres_file}")
    print(f"Shapefile path: {args.shpfile_path}")
    print(f"Tolerance days: {args.tolerance_days}")
    print(f"Verification window: {args.verification_window}")
    print(f"Forecast days: {args.forecast_days}")
    print(f"Max forecast day: {args.max_forecast_day}")
    print(f"MOK filter: {args.mok}")
    print(f"Output file: {args.output_file}")
    print(f"Plot directory: {args.plot_dir}")
    print(f"Figure size: {args.figsize}")
    
    # Compute metrics for multiple years
    metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(
        args.years, args.model_forecast_dir, args.imd_folder, args.thres_file,
        tolerance_days=args.tolerance_days, 
        verification_window=args.verification_window, 
        forecast_days=args.forecast_days,
        max_forecast_day=args.max_forecast_day, 
        mok=args.mok
    )
    
    # Create spatial metrics
    spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
    
    # Convert to xarray Dataset for saving
    # Convert to xarray Dataset for saving
    ds = xr.Dataset(spatial_metrics)
    
    # Add global attributes
    ds.attrs['title'] = 'Monsoon Onset MAE, FAR, MR Analysis'
    ds.attrs['description'] = 'Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate for monsoon onset predictions'
    ds.attrs['years'] = str(args.years)
    ds.attrs['tolerance_days'] = args.tolerance_days
    ds.attrs['verification_window'] = args.verification_window
    ds.attrs['forecast_days'] = args.forecast_days
    ds.attrs['max_forecast_day'] = args.max_forecast_day
    ds.attrs['mok_filter'] = int(args.mok)  # Convert boolean to integer (0 or 1)
    
    # Save to NetCDF
    ds.to_netcdf(args.output_file)
    print(f"\nSpatial metrics saved to: {args.output_file}")
    
    # Generate and save plot if plot_dir is specified
    if args.plot_dir:
        # Create plot directory if it doesn't exist
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # Generate plot filename
        years_str = f"{min(args.years)}-{max(args.years)}"
        window_str = f"{args.verification_window}-{args.forecast_days}day"
        mok_str = "MOK" if args.mok else "noMOK"
        plot_filename = f"spatial_metrics_{years_str}_{window_str}_{mok_str}.png"
        plot_path = os.path.join(args.plot_dir, plot_filename)
        
        print(f"\nGenerating spatial plot...")
        
        # Create the plot
        fig, axes = plot_spatial_metrics(
            spatial_metrics, 
            args.shpfile_path,
            figsize=tuple(args.figsize), 
            save_path=plot_path
        )
        
        # Close the figure to free memory
        plt.close(fig)
        
        print(f"Plot saved to: {plot_path}")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Mean MAE: {float(ds['mean_mae'].mean().values):.2f} days")
    print(f"Mean FAR: {float(ds['false_alarm_rate'].mean().values)*100:.1f}%")
    print(f"Mean Miss Rate: {float(ds['miss_rate'].mean().values)*100:.1f}%")

if __name__ == "__main__":
    main()