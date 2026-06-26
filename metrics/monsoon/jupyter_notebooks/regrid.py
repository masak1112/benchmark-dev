import os
import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime
import argparse
from pathlib import Path
import subprocess

def run_cmd(cmd, verbose=True):
    """Run a command and handle errors"""
    if verbose:
        print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("❌ Error:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result

def extract_precipitation_files(input_dir, output_dir, months, year=2018, file_pattern=None, regrid=False, target_grid_file=None):
    """
    Extract total_precipitation_24hr variable from forecast files for specified months.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the forecast files
    output_dir : str
        Directory to save extracted precipitation files
    months : list
        List of months to process (1-12)
    year : int
        Year to process (default: 2018)
    file_pattern : str
        File pattern to match (if None, uses default pattern)
    regrid : bool
        Whether to regrid the files using CDO
    target_grid_file : str
        Path to target grid file for regridding
    """
    
    if file_pattern is None:
        file_pattern = f"pangu_plasim_1_24h_45step_{year}*_ens_*.nc"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert months to set for faster lookup
    target_months = set(months)
    
    # Find all files matching the pattern
    pattern = os.path.join(input_dir, file_pattern)
    files = sorted(glob(pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    if regrid and not target_grid_file:
        print("❌ Error: regrid=True but no target_grid_file provided")
        return
    
    processed_count = 0
    error_count = 0
    
    # Create temporary files for regridding process
    temp_dir = os.path.join(output_dir, "temp_regrid")
    if regrid:
        os.makedirs(temp_dir, exist_ok=True)
        temp_nc = os.path.join(temp_dir, "temp.nc")
        orig_grid = os.path.join(temp_dir, "orig_grid.txt")
        fixed_grid = os.path.join(temp_dir, "fixed_grid.txt")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        try:
            # Parse filename to extract date
            # Example: pangu_plasim_1_24h_45step_2018010800_ens_14.nc
            parts = filename.replace(".nc", "").split("_")
            
            # Find the date part (should be 10 digits: YYYYMMDDHH)
            date_str = None
            for part in parts:
                if len(part) == 10 and part.isdigit():
                    date_str = part
                    break
            
            if date_str is None:
                print(f"⚠️ Could not parse date from filename: {filename}. Skipping.")
                error_count += 1
                continue
            
            # Extract month from date (characters 4-6: MM)
            file_month = int(date_str[4:6])
            
            # Check if this file's month is in our target months
            if file_month not in target_months:
                continue
            
            # Open the file and extract precipitation variable
            print(f"Processing: {filename}")
            
            with xr.open_dataset(file_path) as ds:
                if "total_precipitation_24hr" not in ds.data_vars:
                    print(f"⚠️ Variable 'total_precipitation_24hr' not found in {filename}. Available variables: {list(ds.data_vars.keys())}")
                    error_count += 1
                    continue
                
                # Extract only the precipitation variable
                precip_ds = ds[["total_precipitation_24hr"]].copy()
                
                # Create output filename
                if regrid:
                    output_filename = f"regridded_precip_{filename}"
                else:
                    output_filename = f"precip_{filename}"
                
                output_path = os.path.join(output_dir, output_filename)
                
                if regrid:
                    # Save temporary file first
                    temp_precip_file = os.path.join(temp_dir, f"temp_precip_{filename}")
                    precip_ds.to_netcdf(temp_precip_file)
                    
                    try:
                        # 1. Extract original grid
                        grid_out = subprocess.check_output(["cdo", "griddes", temp_precip_file], text=True)
                        with open(orig_grid, "w") as grid_f:
                            grid_f.write(grid_out)

                        # 2. Fix grid (replace 'generic' with 'lonlat')
                        with open(orig_grid, "r") as fin, open(fixed_grid, "w") as fout:
                            for line in fin:
                                fout.write(line.replace("generic", "lonlat"))

                        # 3. Set grid and regrid
                        run_cmd(["cdo", f"setgrid,{fixed_grid}", temp_precip_file, temp_nc], verbose=False)
                        run_cmd(["cdo", f"remapcon,{target_grid_file}", temp_nc, output_path], verbose=False)
                        
                        # Clean up temporary files
                        os.remove(temp_precip_file)
                        if os.path.exists(temp_nc):
                            os.remove(temp_nc)
                        
                        print(f"✅ Regridded and saved: {output_filename}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"❌ CDO error for {filename}: {e}")
                        error_count += 1
                        # Clean up on error
                        for temp_file in [temp_precip_file, temp_nc]:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        continue
                        
                else:
                    # Save without regridding
                    precip_ds.to_netcdf(output_path)
                    print(f"✅ Saved: {output_filename}")
                
                processed_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            error_count += 1
            continue
    
    # Clean up regridding temporary directory
    if regrid:
        for f in [orig_grid, fixed_grid]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                print(f"⚠️ Could not remove temp directory {temp_dir} (may not be empty)")
    
    print(f"\n📊 Summary:")
    print(f"   Processed: {processed_count} files")
    print(f"   Errors: {error_count} files")
    print(f"   Output directory: {output_dir}")
    if regrid:
        print(f"   Regridded to: {target_grid_file}")

def extract_precipitation_by_month_range(input_dir, output_dir, start_month, end_month, year=2018, regrid=False, target_grid_file=None):
    """
    Extract precipitation files for a range of months.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the forecast files
    output_dir : str
        Directory to save extracted precipitation files
    start_month : int
        Starting month (1-12)
    end_month : int
        Ending month (1-12)
    year : int
        Year to process (default: 2018)
    regrid : bool
        Whether to regrid the files using CDO
    target_grid_file : str
        Path to target grid file for regridding
    """
    
    if start_month <= end_month:
        months = list(range(start_month, end_month + 1))
    else:
        # Handle wrap-around (e.g., Nov to Feb)
        months = list(range(start_month, 13)) + list(range(1, end_month + 1))
    
    print(f"Processing months: {months}")
    extract_precipitation_files(input_dir, output_dir, months, year, regrid=regrid, target_grid_file=target_grid_file)



# Extract and regrid precipitation files for January 2018
year = 2019
extract_precipitation_files(
    input_dir="/glade/campaign/univ/uchi0014/bing/predictions",
    output_dir="/glade/work/bgong/benchmark-dev/metrics/monsoon/jupyter_notebooks/test_regrid", 
    months=[5, 6, 7],
    year=year,
    file_pattern=f"pangu_plasim_2_24h_45step_{year}*00_ens_*.nc",
    regrid=True,
    target_grid_file="/glade/work/bgong/benchmark-dev/metrics/monsoon/data_files/grid_2deg_india.txt"
)


def merge_regridded_files(input_dir, output_file, file_pattern="regridded_precip_*.nc"):
    """
    Merge all regridded precipitation files into a single file with dimensions:
    - init_time: initialization time
    - member: ensemble member
    - day: forecast lead day (renamed from time)
    
    Parameters:
    -----------
    input_dir : str
        Directory containing regridded precipitation files
    output_file : str
        Path for the merged output file
    file_pattern : str
        Pattern to match regridded files
    """
    
    # Find all regridded files
    pattern = os.path.join(input_dir, file_pattern)
    files = sorted(glob(pattern))
    print(f"Found {len(files)} regridded files to merge")
    
    if not files:
        print("❌ No regridded files found!")
        return
    
    # Dictionary to organize files by init_time and member
    data_dict = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        try:
            # Parse filename to extract init_time and member
            # Example: regridded_precip_pangu_plasim_1_24h_45step_2018010800_ens_14.nc
            parts = filename.replace("regridded_precip_", "").replace(".nc", "").split("_")
            
            # Find date part (10 digits: YYYYMMDDHH)
            date_str = None
            member_str = None
            
            for i, part in enumerate(parts):
                if len(part) == 10 and part.isdigit():
                    date_str = part
                elif part == "ens" and i + 1 < len(parts):
                    member_str = parts[i + 1]
                    break
            
            if date_str is None or member_str is None:
                print(f"⚠️ Could not parse date/member from {filename}")
                continue
            
            # Convert to datetime and member number
            init_time = datetime.strptime(date_str, "%Y%m%d%H")
            member = int(member_str)
            
            # Open dataset and rename time to day
            ds = xr.open_dataset(file_path)
            ds = ds.rename({"time": "day"})
            
            # Add coordinates
            ds = ds.assign_coords({
                "day": np.arange(len(ds.day), dtype="int64")  # 0, 1, 2, ... for forecast days
            })
            
            # Add to dictionary
            init_time_key = np.datetime64(init_time)
            if init_time_key not in data_dict:
                data_dict[init_time_key] = {}
            
            data_dict[init_time_key][member] = ds
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    # Convert to list of datasets for concatenation
    all_datasets = []
    
    for init_time, member_dict in sorted(data_dict.items()):
        member_datasets = []
        
        for member in sorted(member_dict.keys()):
            ds = member_dict[member]
            # Add member dimension
            ds = ds.expand_dims(dim={"member": [member]})
            member_datasets.append(ds)
        
        if member_datasets:
            # Combine all members for this init_time
            members_combined = xr.concat(member_datasets, dim="member")
            # Add init_time dimension
            members_combined = members_combined.expand_dims(dim={"init_time": [init_time]})
            all_datasets.append(members_combined)
    
    # Final merge across all init_times
    if all_datasets:
        final_ds = xr.concat(all_datasets, dim="init_time")
        
        # Reorder dimensions for clarity
        final_ds = final_ds.transpose("init_time", "member", "day", "lat", "lon")
        
        # Save to file
        print(f"Saving merged dataset with shape: {dict(final_ds.dims)}")
        final_ds.to_netcdf(output_file)
        print(f"✅ Merged file saved to: {output_file}")
        
        # Print summary
        print(f"\n📊 Merged Dataset Summary:")
        print(f"   Init times: {len(final_ds.init_time)}")
        print(f"   Members: {len(final_ds.member)}")
        print(f"   Forecast days: {len(final_ds.day)}")
        print(f"   Variables: {list(final_ds.data_vars.keys())}")
        
        return final_ds
    else:
        print("❌ No valid datasets found for merging")
        return None

def merge_regridded_files_by_month(input_dir, output_dir, months, year=2018):
    """
    Merge regridded files by month(s) to create separate merged files.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing regridded precipitation files
    output_dir : str
        Directory to save merged files
    months : list
        List of months to process
    year : int
        Year to process
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for month in months:
        print(f"\n🔄 Processing month {month:02d}")
        
        # Create month-specific pattern
        month_pattern = f"regridded_precip_*{year}{month:02d}*_ens_*.nc"
        
        # Output file for this month
        output_file = os.path.join(output_dir, f"merged_precip_{year}_{month:02d}.nc")
        
        # Merge files for this month
        merge_regridded_files(
            input_dir=input_dir,
            output_file=output_file,
            file_pattern=month_pattern
        )
        
# Merge all regridded files from January 2018
year = 2019
merge_regridded_files(
    input_dir="/glade/work/bgong/benchmark-dev/metrics/monsoon/jupyter_notebooks/test_regrid",
    output_file=f"/glade/work/bgong/benchmark-dev/metrics/monsoon/jupyter_notebooks/merged_precip_{year}.nc",
    file_pattern=f"regridded_precip_*{year}*.nc"
)    