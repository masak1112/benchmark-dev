import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings

def merge_precipitation_files(input_dir, output_file, search_string):
    """
    Merge multiple NetCDF files containing precipitation data into a single file.

    Parameters:
    input_dir (str): Directory containing NetCDF files.
    output_file (str): Path to save the merged file.
    search_string (str): String to filter files by name.
    """
    # Get all NetCDF files containing the search string
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
             if f.endswith('.nc') and search_string in f]
    
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in directory: {input_dir} with string '{search_string}'")
    
    # Open and merge all selected files
    datasets = []
    for file in files:
        print(f"Processing file: {file}")
        ds = xr.open_dataset(file)
        # Extract precipitation variable and add to the list
        datasets.append(ds['total_precipitation_24hr'])
    
    # Concatenate all datasets along the time dimension
    merged_precip = xr.concat(datasets, dim='time')
    
    # Save the merged dataset to the output file
    merged_precip.to_netcdf(output_file)
    print(f"Merged precipitation data saved to: {output_file}")


#### functions to load model forecast data ####
def get_s2s_prob_twice_weekly(yr, data_dir):
    """
    Loads model precip data for twice-weekly initializations from May to July.
    Filters for Mondays and Thursdays in the specified year.
    
    Parameters:
    yr: int, year to load data for
    
    Returns:
    t_model: pandas DatetimeIndex, initialization times
    p_model: ndarray, precipitation data
    lon: array, longitude coordinates
    lat: array, latitude coordinates
    """
    
    fname = f'merged_precip_{yr}.nc'
    file_path = os.path.join(data_dir, fname)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Filter for twice weekly data from daily for the specified year
    # Define date range from May 1 to July 31 of 2024
    # start_date = datetime(2024, 5, 1)
    # end_date = datetime(2024, 7, 31)
    '''
    for testing the code, we changed the date range to january to march
    '''
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Find Mondays (weekday=0) and Thursdays (weekday=3) in pandas
    is_monday = date_range.weekday == 0
    is_thursday = date_range.weekday == 3
    filtered_dates = date_range[is_monday | is_thursday]

    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{yr}-%m-%d'))
    # Load data using xarray
    ds = xr.open_dataset(file_path)
    ds = ds.sel(init_time=filtered_dates_yr)
    ds = ds.sel(step=slice(1, None))
    p_model = ds['precip'] * 1000  # Convert to mm (multiply by 1000)
    
    # Close the dataset
    ds.close()
 
    
    return p_model


#### functions to load IMD rainfall data ####
def load_imd_rainfall(year, imd_folder):
    """
    Load IMD daily rainfall NetCDF for a given year.

    Parameters
    ----------
    imd_folder : str
        Folder containing IMD NetCDF files named like 'data_YYYY.nc'.
    year : int
        Year to load.
    lat_range : list, optional
        [min_lat, max_lat] to subset the data.
    lon_range : list, optional
        [min_lon, max_lon] to subset the data.

    Returns
    -------
    rainfall_ds : xarray.DataArray
        IMD rainfall data with dims ['lon','lat','time'].
    """
    imd_file = f"{imd_folder}/data_{year}.nc"
    ds = xr.open_dataset(imd_file)
    rainfall = ds['RAINFALL']


    # Assume rainfall variable is named 'RAINFALL'
    # Only rename dimensions if year is not 2024 (2024 already has lat/lon naming)
    if year != 2024:
        rainfall = rainfall.rename({
            'latitude': 'lat',
            'longitude': 'lon', 
            'TIME': 'time'
        })

    return rainfall


#### functions to detect observed onset dates ####
def detect_observed_onset(rainfall_ds, thresh_slice, year, mok=False):
    """
    Detect observed onset dates for a given year.
    
    Parameters:
    rainfall_ds: xarray DataArray with rainfall data
    thresh_slice: xarray DataArray with threshold values
    year: int, year to process
    mok: bool, if True use June 2nd as start date (MOK), if False use May 1st
    
    Returns:
    onset_da: xarray DataArray with onset dates
    """
    # Subset rainfall data
    rain_slice = rainfall_ds.sel(lat=slice(10, 30), lon=slice(70, 90))
    
    # Parameters
    window = 5
    
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
        print(f"Warning: {date_label} ({start_date.strftime('%Y-%m-%d')}) not found in data for year {year}")
        # Fallback to April 1st if start date not available
        fallback_date = datetime(year, 4, 1)
        start_idx = np.where(time_dates >= fallback_date)[0][0]
        print(f"Using fallback date: April 1st")
    else:
        start_idx = start_idx_candidates[0]
        print(f"Using {date_label} ({start_date.strftime('%Y-%m-%d')}) as start date for onset detection")

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



#### functions to detect model onset dates for all members ####
def compute_onset_for_all_members(p_model, thresh_slice, onset_da, max_forecast_day=15, mok=False):
    """
    Compute onset dates for each ensemble member, initialization time, and grid point.
    Only processes forecasts initialized before the observed onset date.
    
    Parameters:
    p_model: xarray DataArray with dims [init_time, step, lat, lon, member]
    thresh_slice: xarray DataArray with threshold values for each grid point
    onset_da: xarray DataArray with observed onset dates for filtering
    max_forecast_day: int, maximum forecast day to consider for onset (default 15)
    mok: bool, if True only count onset after June 2nd (MOK date), if False use all forecasts
    
    Returns:
    pandas DataFrame with columns: init_time, lat, lon, member, onset_day
    """
    
    window = 5
    results_list = []
    
    # Get dimensions
    init_times = p_model.init_time.values
    lats = p_model.lat.values  
    lons = p_model.lon.values
    members = p_model.member.values
    
    date_method = "MOK (June 2nd filter)" if mok else "no date filter"
    print(f"Processing {len(init_times)} init times x {len(lats)} lats x {len(lons)} lons x {len(members)} members...")
    print(f"Using {date_method} for onset detection")
    print(f"Only processing forecasts initialized before observed onset dates")
    
    # We need first 19 days to check onset up to day 15 (because of 5-day window)
    max_steps_needed = max_forecast_day + window - 1
    
    # Track statistics
    total_potential_forecasts = 0
    valid_forecasts = 0
    skipped_no_obs = 0
    skipped_late_init = 0
    
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
                
                total_potential_forecasts += len(members)
                
                # Get observed onset date for this grid point
                try:
                    obs_onset = onset_da.isel(lat=i, lon=j).values
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
                
                # Get threshold for this grid point
                thresh = thresh_slice.isel(lat=i, lon=j).values
                
                for m_idx, member in enumerate(members):
                    
                    valid_forecasts += 1
                    
                    try:
                        # Extract forecast time series for this member
                        forecast_series = p_model.isel(
                            init_time=t_idx,
                            lat=i, 
                            lon=j,
                            member=m_idx,
                            step=slice(0, max_steps_needed )  # steps are 0-indexed
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
                                
                                # Check basic onset condition: first day > 1mm AND 5-day sum > threshold
                                if window_series[0] > 1 and np.nansum(window_series) > thresh:
                                    
                                    # Calculate the actual date this forecast day represents
                                    forecast_date = init_date + pd.Timedelta(days=day)
                                    
                                    # If MOK flag is True, only count onset if it's on or after June 2nd
                                    if mok:
                                        if forecast_date.date() >= mok_date.date():
                                            onset_day = day
                                            break  # Found valid onset after MOK date
                                        # else: continue checking later days
                                    else:
                                        # No MOK filtering, count this onset
                                        onset_day = day
                                        break
                        
                        # Store result
                        result = {
                            'init_time': init_time,
                            'lat': lat,
                            'lon': lon, 
                            'member': member,
                            'onset_day': onset_day,  # None if no onset found (or no valid onset after MOK)
                            'obs_onset_date': obs_onset_dt.strftime('%Y-%m-%d')  # Store observed onset for reference
                        }
                        results_list.append(result)
                        
                    except Exception as e:
                        print(f"Error at init_time {t_idx}, lat {i}, lon {j}, member {m_idx}: {e}")
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
    
    if mok:
        print(f"Note: Only onsets on or after June 2nd were counted due to MOK flag")
    
    return onset_df


#### functions to create forecast-observation pairs with day bins ####
def create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, day_bins):
    """
    Create forecast-observation pairs using specified day bins.
    Only uses forecasts that were initialized before the observed onset.
    
    Parameters:
    onset_all_members: DataFrame with individual member onset forecasts (filtered by observed onset)
    onset_da: xarray DataArray with observed onset dates
    day_bins: list of tuples defining the bins, e.g., [(1, 5), (6, 10), (11, 15)]
    
    Returns:
    forecast_obs_df: DataFrame with forecast-observation pairs
    """
    results_list = []
    
    # Get unique combinations of init_time, lat, lon from the filtered forecast data
    forecast_groups = onset_all_members.groupby(['init_time', 'lat', 'lon'])
    
    print(f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}...")
    
    for (init_time, lat, lon), group in forecast_groups:
        
        # Get observed onset for this location
        try:
            lat_idx = np.where(np.abs(onset_da.lat.values - lat) < 0.01)[0][0]
            lon_idx = np.where(np.abs(onset_da.lon.values - lon) < 0.01)[0][0]
            obs_date = onset_da.isel(lat=lat_idx, lon=lon_idx).values
        except:
            continue
        
        # Skip if no observed onset (this should already be filtered out)
        if pd.isna(obs_date):
            continue
            
        # Convert dates for comparison
        init_date = pd.to_datetime(init_time)
        obs_date_dt = pd.to_datetime(obs_date)
        
        # Double-check: Only use forecasts initialized before the observed onset
        if init_date >= obs_date_dt:
            continue
        
        # For each day bin
        for bin_start, bin_end in day_bins:
            
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
                'bin_start': bin_start,
                'bin_end': bin_end,
                'bin_label': f'Days {bin_start}-{bin_end}',
                'predicted_prob': predicted_prob,
                'observed_onset': observed_onset,
                'members_with_onset': members_with_onset_in_bin,
                'total_members': total_members,
                'year': pd.to_datetime(init_time).year,
                'obs_onset_date': obs_date_dt.strftime('%Y-%m-%d')  # Include observed onset for reference
            }
            results_list.append(result)
    
    # Convert to DataFrame
    forecast_obs_df = pd.DataFrame(results_list)
    
    print(f"Generated {len(forecast_obs_df)} forecast-observation pairs")
    print(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")
    
    return forecast_obs_df


#### main function to perform multi-year reliability analysis ####
def multi_year_reliability_analysis(years, s2s_data_dir, imd_folder, thres_file, max_forecast_day, day_bins=None):
    """
    Main function to perform multi-year reliability analysis.
    
    Parameters:
    years: list of years to process
    s2s_data_dir: directory containing S2S model data
    imd_folder: directory containing IMD rainfall data
    thres_file: path to threshold file
    day_bins: list of tuples defining forecast bins, e.g., [(1, 5), (6, 10), (11, 15)]
    
    Returns:
    combined_forecast_obs: DataFrame with all forecast-observation pairs
    """
    
    print(f"Processing years: {years}")
    
    # Load threshold data (same for all years)
    thresh_ds = xr.open_dataset(thres_file)
    thresh_da = thresh_ds['MWmean']
    thresh_slice = thresh_da.sel(lat=slice(10, 30), lon=slice(70, 90))
    
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
            p_model = get_s2s_prob_twice_weekly(year, s2s_data_dir)
            p_model_slice = p_model.sel(lat=slice(10, 30), lon=slice(70, 90))
            
            print("Loading IMD rainfall data...")
            rainfall_ds = load_imd_rainfall(year, imd_folder)
            
            print("Detecting observed onset...")
            onset_da = detect_observed_onset(rainfall_ds, thresh_slice, year, mok=True)
            print(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")
            
            print("Computing onset for all ensemble members...")
            onset_all_members = compute_onset_for_all_members(p_model_slice, thresh_slice, onset_da, max_forecast_day=max_forecast_day, mok=True)
            print(f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases")
            
            print("Creating forecast-observation pairs...")
            forecast_obs_pairs = create_forecast_observation_pairs_with_bins(onset_all_members, onset_da, day_bins)
            
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
    
    print(f"Combined dataset: {len(combined_forecast_obs)} total forecast-observation pairs")
    print(f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}")
    print(f"Observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}")
    print(f"Non-zero probabilities: {(combined_forecast_obs['predicted_prob'] > 0).sum()}")
    
    # Show breakdown by year
    print(f"\nBreakdown by year:")
    for year in years:
        year_data = combined_forecast_obs[combined_forecast_obs['year'] == year]
        if len(year_data) > 0:
            print(f"  {year}: {len(year_data)} pairs, observed rate: {year_data['observed_onset'].mean():.3f}")
    
    # Show breakdown by bin
    print(f"\nBreakdown by forecast bin:")
    for bin_label in sorted(combined_forecast_obs['bin_label'].unique()):
        bin_data = combined_forecast_obs[combined_forecast_obs['bin_label'] == bin_label]
        print(f"  {bin_label}: {len(bin_data)} pairs, mean prob: {bin_data['predicted_prob'].mean():.3f}, observed rate: {bin_data['observed_onset'].mean():.3f}")
    
    # Print final summary statistics
    print(f"\nFinal Summary Statistics:")
    print(f"Years processed: {years}")
    print(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
    print(f"Base rate (climatological frequency): {combined_forecast_obs['observed_onset'].mean():.3f}")
    print(f"Mean forecast probability: {combined_forecast_obs['predicted_prob'].mean():.3f}")
    brier_score = np.mean((combined_forecast_obs['predicted_prob'] - combined_forecast_obs['observed_onset'])**2)
    print(f"Brier Score: {brier_score:.3f}")
    
    return combined_forecast_obs


#### functions to plot reliability diagram ####
def plot_reliability_diagram(forecast_obs_df, years=None, n_bins=10, title_suffix="", save_path=None):
    """
    Plot reliability diagram from forecast-observation pairs.
    
    Parameters:
    forecast_obs_df: DataFrame with forecast-observation pairs
    years: list of years (for title), if None will be inferred
    n_bins: number of probability bins
    title_suffix: additional text for title
    save_path: path to save figure, if None will display only
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    
    # Infer years if not provided
    if years is None:
        if 'year' in forecast_obs_df.columns:
            years = sorted(forecast_obs_df['year'].unique())
        else:
            years = ["Unknown"]
    
    # Create probability bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initialize arrays for reliability calculation
    reliability_y = np.zeros(n_bins)
    frequency = np.zeros(n_bins)
    n_forecasts_per_bin = np.zeros(n_bins)
    
    print("\nReliability Analysis:")
    print("Bin Range\t\tN_Forecasts\tReliability\tFrequency")
    print("-" * 60)
    
    # Calculate reliability for each bin
    for i in range(n_bins):
        # Find forecasts in this probability bin
        if i == n_bins - 1:  # Include right edge for last bin
            in_bin = ((forecast_obs_df['predicted_prob'] >= bin_edges[i]) & 
                      (forecast_obs_df['predicted_prob'] <= bin_edges[i+1]))
        else:
            in_bin = ((forecast_obs_df['predicted_prob'] >= bin_edges[i]) & 
                      (forecast_obs_df['predicted_prob'] < bin_edges[i+1]))
        
        n_forecasts = in_bin.sum()
        n_forecasts_per_bin[i] = n_forecasts
        
        if n_forecasts > 0:
            # Calculate conditional frequency (reliability)
            reliability_y[i] = forecast_obs_df.loc[in_bin, 'observed_onset'].mean()
            frequency[i] = n_forecasts / len(forecast_obs_df)
        else:
            reliability_y[i] = np.nan
            frequency[i] = 0
        
        print(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}\t\t{n_forecasts}\t\t{reliability_y[i]:.3f}\t\t{frequency[i]:.3f}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot reliability curve
    valid_bins = ~np.isnan(reliability_y)
    ax.plot(bin_centers[valid_bins], reliability_y[valid_bins], 'o-', 
            color='blue', linewidth=2, markersize=8)
    
    # Plot perfect reliability lines
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    # Add frequency histogram
    ax2 = ax.twinx()
    ax2.bar(bin_centers, frequency, width=0.08, alpha=0.3, color='gray', label='Frequency')
    #ax2.set_ylabel('Frequency of Use', fontsize=12)
    ax2.set_ylim(0, max(frequency) * 1.2 if max(frequency) > 0 else 0.1)
    
    # Formatting
    ax.set_xlabel('Forecast Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    
    # Create title
    if len(years) > 1:
        year_str = f"{min(years)}-{max(years)}"
    else:
        year_str = str(years[0])
    
    # Get bin information for title
    if 'bin_label' in forecast_obs_df.columns:
        bin_labels = sorted(forecast_obs_df['bin_label'].unique())
        bin_info = f"({', '.join(bin_labels)})"
    else:
        bin_info = ""
    
    title = f'Reliability Diagram - Monsoon Onset Forecasts {year_str}'.strip()
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add legends
    ax2.legend(loc='upper right')
    
    # Calculate and display Brier Score
    brier_score = np.mean((forecast_obs_df['predicted_prob'] - forecast_obs_df['observed_onset'])**2)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax

# merge precipitation files first
input_directory = '/glade/derecho/scratch/weidong/predictions/'  # Input directory
output_filepath = '/glade/derecho/scratch/weidong/merged_precip_2018.nc'  # Output file path
search_string = 'ens_0'
merge_precipitation_files(input_directory, output_filepath, search_string)

# then run the multi-year reliability analysis
years = [2018, 2019, 2020, 2021, 2022, 2023]
s2s_data_dir = '/Users/Rajat/Library/CloudStorage/Box-Box/UChicago_postdoc/onset_benchmark_paper/ngcm51/twice_weekly_0z/tp_2p0'
imd_folder = '/glade/derecho/scratch/rajatm/IMD_2deg'
thres_file = './mwset2x2.nc4'
max_forecast_day = 15
day_bins = [(1, 5), (6, 10), (11, 15)]
forecast_obs_df = multi_year_reliability_analysis(years, s2s_data_dir, imd_folder, thres_file, max_forecast_day, day_bins)

fig,ax = plot_reliability_diagram(forecast_obs_df, years=years, n_bins=10, title_suffix="", save_path=f'reliability_{max_forecast_day}.png')


