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
from matplotlib.path import Path as MplPath
import argparse


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

def detect_observed_onset(rainfall_ds, thresh_slice, year, mok=True):
    """Detect observed onset dates for a given year."""
    rain_slice = rainfall_ds
    window = 5
    
    if mok:
        start_date = datetime(year, 6, 2)
        date_label = "MOK date (June 2nd)"
    else:
        start_date = datetime(year, 5, 1)
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

def compute_onset_for_all_members(p_model, thresh_slice, onset_da, max_forecast_day=15, mok=True):
    """Compute onset dates for each ensemble member, initialization time, and grid point."""
    window = 5
    results_list = []
    
    init_times = p_model.init_time.values
    members = p_model.member.values
    lats = p_model.lat.values  
    lons = p_model.lon.values
    unique_pairs = list(zip(lons, lats))
    
    date_method = "MOK (June 2nd filter)" if mok else "no date filter"
    print(f"Processing {len(init_times)} init times x {len(unique_pairs)} unique locations x {len(members)} members...")
    print(f"Using {date_method} for onset detection")
    
    max_steps_needed = max_forecast_day + window - 1
    
    total_potential_forecasts = 0
    valid_forecasts = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    
    for t_idx, init_time in enumerate(init_times):
        if t_idx % 5 == 0:
            print(f"Processing init time {t_idx+1}/{len(init_times)}: {pd.to_datetime(init_time).strftime('%Y-%m-%d')}")
        
        init_date = pd.to_datetime(init_time)
        year = init_date.year
        mok_date = datetime(year, 6, 2)
        
        for loc_idx, (lon, lat) in enumerate(unique_pairs):
            total_potential_forecasts += len(members)
            
            try:
                obs_onset = onset_da.isel(lat=loc_idx, lon=loc_idx).values
            except:
                skipped_no_obs += len(members)
                continue
            
            if pd.isna(obs_onset):
                skipped_no_obs += len(members)
                continue
            
            obs_onset_dt = pd.to_datetime(obs_onset)
            
            if init_date >= obs_onset_dt:
                skipped_late_init += len(members)
                continue
            
            thresh = thresh_slice.isel(lat=loc_idx, lon=loc_idx).values
            
            for m_idx, member in enumerate(members):
                valid_forecasts += 1
                
                try:
                    forecast_series = p_model.isel(
                        init_time=t_idx,
                        lat=loc_idx, 
                        lon=loc_idx,
                        member=m_idx,
                        step=slice(0, max_steps_needed)
                    ).values
                    
                    if len(forecast_series) < max_steps_needed:
                        continue
                    
                    onset_day = None
                    
                    for day in range(1, max_forecast_day + 1):
                        start_idx = day - 1
                        end_idx = start_idx + window 
                        
                        if end_idx <= len(forecast_series):
                            window_series = forecast_series[start_idx:end_idx]
                            
                            if window_series[0] > 1 and np.nansum(window_series) > thresh:
                                forecast_date = init_date + pd.Timedelta(days=day)
                                
                                if mok:
                                    if forecast_date.date() > mok_date.date():
                                        onset_day = day
                                        break
                                else:
                                    onset_day = day
                                    break
                    
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
    
    onset_df = pd.DataFrame(results_list)
    return onset_df

def create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, day_bins):
    """Create forecast-observation pairs using specified day bins."""
    results_list = []
    forecast_groups = onset_all_members.groupby(['init_time', 'lat', 'lon'])
    
    print(f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}...")
    
    for (init_time, lat, lon), group in forecast_groups:
        try:
            lat_idx = np.where(np.abs(onset_da.lat.values - lat) < 0.01)[0][0]
            lon_idx = np.where(np.abs(onset_da.lon.values - lon) < 0.01)[0][0]
            obs_date = onset_da.isel(lat=lat_idx, lon=lon_idx).values
        except:
            continue
        
        if pd.isna(obs_date):
            continue
            
        init_date = pd.to_datetime(init_time)
        obs_date_dt = pd.to_datetime(obs_date)
        
        if init_date >= obs_date_dt:
            continue
        
        for bin_start, bin_end in day_bins:
            bin_start_date = init_date + pd.Timedelta(days=bin_start)
            bin_end_date = init_date + pd.Timedelta(days=bin_end)
            
            observed_onset = int(bin_start_date.date() <= obs_date_dt.date() <= bin_end_date.date())
            
            members_with_onset_in_bin = 0
            total_members = len(group)
            
            for member_idx, member_row in group.iterrows():
                member_onset_day = member_row['onset_day']
                
                if pd.notna(member_onset_day) and bin_start <= member_onset_day <= bin_end:
                    members_with_onset_in_bin += 1
            
            predicted_prob = members_with_onset_in_bin / total_members
            
            result = {
                'init_time': init_time,
                'lat': lat,
                'lon': lon,
                'bin_start': bin_start,
                'bin_end': bin_end,
                'bin_label': f'Days {bin_start}-{bin_end}',
                'predicted_prob': predicted_prob,
                'observed_onset': observed_onset,
                'members_with_onset': members_with_onset_in_bin,
                'total_members': total_members,
                'year': pd.to_datetime(init_time).year,
                'obs_onset_date': obs_date_dt.strftime('%Y-%m-%d')
            }
            results_list.append(result)
    
    forecast_obs_df = pd.DataFrame(results_list)
    
    print(f"Generated {len(forecast_obs_df)} forecast-observation pairs")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")
    
    return forecast_obs_df

def points_inside_polygon(polygon_lon, polygon_lat, grid_lons, grid_lats):
    """Find grid points that are inside a polygon."""
    polygon_vertices = np.column_stack((polygon_lon, polygon_lat))
    polygon_path = MplPath(polygon_vertices)
    
    if grid_lons.ndim == 1 and grid_lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    else:
        lon_grid, lat_grid = grid_lons, grid_lats
    
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    inside_mask = polygon_path.contains_points(points)
    inside_mask = inside_mask.reshape(lon_grid.shape)
    
    inside_lons = lon_grid[inside_mask]
    inside_lats = lat_grid[inside_mask]
    
    return inside_mask, inside_lons, inside_lats

def multi_year_reliability_analysis(years, model_forecast_dir, imd_folder, thres_file, mem_num, max_forecast_day, day_bins, mok=True, date_filter_year=2024, file_pattern='{}.nc'):
    """Main function to perform multi-year reliability analysis."""
    
    print(f"Processing years: {years}")
    
    # Load threshold data
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

    all_forecast_obs_pairs = []
    
    for year in years:
        print(f"\n{'='*50}")
        print(f"Processing year {year}")
        print(f"{'='*50}")
        
        try:
            print("Loading model forecast data...")
            p_model,_ = get_forecast_probabilistic_twice_weekly(year, model_forecast_dir, mem_num, date_filter_year, file_pattern)
            p_model_slice = p_model.sel(lat=inside_lats, lon=inside_lons)

            print("Loading IMD rainfall data...")
            rainfall_ds = load_imd_rainfall(year, imd_folder)
            rainfall_ds_slice = rainfall_ds.sel(lat=inside_lats, lon=inside_lons)
            
            print("Detecting observed onset...")
            onset_da = detect_observed_onset(rainfall_ds_slice, thresh_slice, year, mok=True)
            print(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")
            
            print("Computing onset for all ensemble members...")
            onset_all_members = compute_onset_for_all_members(p_model_slice, thresh_slice, onset_da, max_forecast_day=max_forecast_day, mok=True)
            print(f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases")
            
            print("Creating forecast-observation pairs...")
            forecast_obs_pairs = create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, day_bins)
            
            all_forecast_obs_pairs.append(forecast_obs_pairs)
            
            print(f"Year {year} completed: {len(forecast_obs_pairs)} forecast-observation pairs")
            
        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Combining all years")
    print(f"{'='*50}")
    
    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any year")
    
    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)
    
    print(f"\nFinal Summary Statistics:")
    print(f"Years processed: {years}")    
    return combined_forecast_obs

def plot_reliability_diagram(forecast_obs_pairs_multi, years, max_forecast_day, save_path=None):
    """Plot reliability diagram from forecast-observation pairs."""
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    reliability_y = np.zeros(n_bins)
    mean_forecast_prob = np.zeros(n_bins)
    frequency = np.zeros(n_bins)
    n_forecasts_array = np.zeros(n_bins)

    print("\nReliability Analysis:")
    print("Bin Range\t\tN_Forecasts\tMean_Forecast_Prob\tReliability\tFrequency\tError_Bar")
    print("-" * 90)

    results_for_csv = []

    for i in range(n_bins):
        if i == 0:
            in_bin = ((forecast_obs_pairs_multi['predicted_prob'] >= bin_edges[i]) & 
                    (forecast_obs_pairs_multi['predicted_prob'] <= bin_edges[i+1]))
        else:
            in_bin = ((forecast_obs_pairs_multi['predicted_prob'] > bin_edges[i]) & 
                    (forecast_obs_pairs_multi['predicted_prob'] <= bin_edges[i+1]))
        
        n_forecasts = in_bin.sum()
        n_forecasts_array[i] = n_forecasts
        
        if n_forecasts > 0:
            mean_forecast_prob[i] = forecast_obs_pairs_multi.loc[in_bin, 'predicted_prob'].mean()
            reliability_y[i] = forecast_obs_pairs_multi.loc[in_bin, 'observed_onset'].mean()
            frequency[i] = n_forecasts / len(forecast_obs_pairs_multi)
            error_bar = np.sqrt(reliability_y[i] * (1 - reliability_y[i]) / n_forecasts)
        else:
            mean_forecast_prob[i] = np.nan
            reliability_y[i] = np.nan
            frequency[i] = 0
            error_bar = np.nan
        
        bin_range = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        
        print(f"{bin_range}\t\t{n_forecasts}\t\t{mean_forecast_prob[i]:.3f}\t\t\t{reliability_y[i]:.3f}\t\t{frequency[i]:.3f}\t\t{error_bar:.3f}")
        
        results_for_csv.append({
            'Bin_Range': bin_range,
            'N_Forecasts': n_forecasts,
            'Mean_Forecast_Prob': round(mean_forecast_prob[i], 3) if not np.isnan(mean_forecast_prob[i]) else np.nan,
            'Observed_Frequency': round(reliability_y[i], 3) if not np.isnan(reliability_y[i]) else np.nan,
            'Frequency': round(frequency[i], 3),
            'Error_Bar': round(error_bar, 3) if not np.isnan(error_bar) else np.nan
        })

    results_df = pd.DataFrame(results_for_csv)

    error_bars = np.sqrt(reliability_y * (1 - reliability_y) / n_forecasts_array)
    error_bars = np.where(n_forecasts_array > 0, error_bars, 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    valid_bins = ~np.isnan(reliability_y) & ~np.isnan(mean_forecast_prob)
    ax.errorbar(mean_forecast_prob[valid_bins], reliability_y[valid_bins], 
                yerr=error_bars[valid_bins], fmt='o-', 
                color='blue', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='Reliability')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Reliability')

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.bar(bin_centers, frequency, width=0.08, alpha=0.3, color='gray', label='Frequency')
    max_freq = max(frequency)
    min_freq = min([f for f in frequency if f > 0]) if any(f > 0 for f in frequency) else 1e-4
    ax2.set_ylim(min_freq * 0.5, max_freq * 2)
    ax2.set_ylabel('Forecast frequency', fontsize=12)

    ax.set_xlabel('Forecast Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)

    if len(years) > 1:
        year_str = f"{min(years)}-{max(years)}"
    else:
        year_str = str(years[0])

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Save figure if save_path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_save_path = os.path.join(save_path, f'reliability_{max_forecast_day}day.png')
        fig.savefig(fig_save_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {fig_save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, results_df

def main():
    parser = argparse.ArgumentParser(description='Monsoon Onset Reliability Analysis')
    parser.add_argument('--model_forecast_dir', required=True, help='Directory containing model forecast data')
    parser.add_argument('--imd_folder', required=True, help='Directory containing IMD rainfall data')
    parser.add_argument('--mem_num', type=int, required=True, help='Number of ensemble members to use')
    parser.add_argument('--date_filter_year', type=int, default=2024,
                        help='Year to use for date filtering (default: 2024)')
    parser.add_argument('--thres_file', required=True, help='Path to threshold NetCDF file')
    parser.add_argument('--max_forecast_day', type=int, default=15, help='Maximum forecast day (default: 15)')
    parser.add_argument('--save_path', required=True, help='Directory to save outputs')
    parser.add_argument('--years', nargs='+', type=int, required=True, help='Years to process (e.g., 2019 2020 2021)')
    parser.add_argument('--file_pattern', default='{}.nc', help='File pattern for model forecast data (default: {}.nc)')
    parser.add_argument('--mok', action='store_true', default=True, help='Enable MOK filter (default: True)')
    parser.add_argument('--no-mok', dest='mok', action='store_false', help='Disable MOK filter')

    args = parser.parse_args()
    
    # Set day bins based on max_forecast_day
    if args.max_forecast_day == 15:
        day_bins = [(1, 5), (6, 10), (11, 15)]
    elif args.max_forecast_day == 30:
        day_bins = [(16, 20), (21, 25), (26, 30)]
    else:
        raise ValueError("max_forecast_day must be either 15 or 30")
    
    print(f"Starting reliability analysis with parameters:")
    print(f"Model forecast data directory: {args.model_forecast_dir}")
    print(f"IMD folder: {args.imd_folder}")
    print(f"Threshold file: {args.thres_file}")
    print(f"Max forecast day: {args.max_forecast_day}")
    print(f"Years: {args.years}")
    print(f"File pattern: {args.file_pattern}")
    print(f"Save path: {args.save_path}")
    print(f"Day bins: {day_bins}")
    
    # Run the analysis
    forecast_obs_df = multi_year_reliability_analysis(
        args.years, 
        args.model_forecast_dir, 
        args.imd_folder, 
        args.thres_file,
        args.mem_num, 
        args.max_forecast_day, 
        day_bins,
        date_filter_year=args.date_filter_year, 
        mok=args.mok,
        file_pattern=args.file_pattern
    )
    
    # Plot reliability diagram and get results
    fig, ax, reliability_df = plot_reliability_diagram(
        forecast_obs_df, 
        args.years, 
        args.max_forecast_day,
        args.save_path
    )
    if args.mok:
        mok_suffix = '_mok'
    else:
        mok_suffix = '_no_mok'
    # Save reliability DataFrame as CSV
    os.makedirs(args.save_path, exist_ok=True)
    csv_path = os.path.join(args.save_path, f'reliability_results_{args.max_forecast_day}day{mok_suffix}.csv')
    reliability_df.to_csv(csv_path, index=False)
    print(f"Reliability results saved to: {csv_path}")
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()