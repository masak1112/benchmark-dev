#!/usr/bin/env python3
"""
Parallel version of precipitation processing script
"""

import os
import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime
import argparse
from pathlib import Path
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools


def run_cmd(cmd, verbose=True):
    """Run a command and handle errors"""
    if verbose:
        print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("❌ Error:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def process_single_file(args):
    """Process a single file - designed for multiprocessing"""
    file_path, output_dir, target_months, regrid, target_grid_file = args
    
    filename = os.path.basename(file_path)
    
    try:
        # Parse filename to extract date
        parts = filename.replace(".nc", "").split("_")
        
        # Find the date part (should be 10 digits: YYYYMMDDHH)
        date_str = None
        for part in parts:
            if len(part) == 10 and part.isdigit():
                date_str = part
                break
        
        if date_str is None:
            return {"status": "error", "message": f"Could not parse date from filename: {filename}"}
        
        # Extract month from date (characters 4-6: MM)
        file_month = int(date_str[4:6])
        
        # Check if this file's month is in our target months
        if file_month not in target_months:
            return {"status": "skipped", "message": f"Month {file_month} not in target months"}
        
        # Process the file
        with xr.open_dataset(file_path) as ds:
            if "total_precipitation_24hr" not in ds.data_vars:
                return {"status": "error", "message": f"Variable not found in {filename}"}
            
            # Extract only the precipitation variable
            precip_ds = ds[["total_precipitation_24hr"]].copy()
            
            # Create output filename
            if regrid:
                output_filename = f"regridded_precip_{filename}"
            else:
                output_filename = f"precip_{filename}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            if regrid:
                # Create process-specific temp directory to avoid conflicts
                pid = os.getpid()
                temp_dir = os.path.join(output_dir, f"temp_regrid_{pid}")
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_precip_file = os.path.join(temp_dir, f"temp_precip_{filename}")
                temp_nc = os.path.join(temp_dir, "temp.nc")
                orig_grid = os.path.join(temp_dir, "orig_grid.txt")
                fixed_grid = os.path.join(temp_dir, "fixed_grid.txt")
                
                try:
                    # Save temporary file first
                    precip_ds.to_netcdf(temp_precip_file)
                    
                    # Regridding steps
                    grid_out = subprocess.check_output(["cdo", "griddes", temp_precip_file], text=True)
                    with open(orig_grid, "w") as grid_f:
                        grid_f.write(grid_out)
                    
                    with open(orig_grid, "r") as fin, open(fixed_grid, "w") as fout:
                        for line in fin:
                            fout.write(line.replace("generic", "lonlat"))
                    
                    run_cmd(["cdo", f"setgrid,{fixed_grid}", temp_precip_file, temp_nc], verbose=False)
                    run_cmd(["cdo", f"remapcon,{target_grid_file}", temp_nc, output_path], verbose=False)
                    
                    # Clean up
                    for f in [temp_precip_file, temp_nc, orig_grid, fixed_grid]:
                        if os.path.exists(f):
                            os.remove(f)
                    os.rmdir(temp_dir)
                    
                    return {"status": "success", "message": f"Regridded and saved: {output_filename}"}
                    
                except Exception as e:
                    # Clean up on error
                    for f in [temp_precip_file, temp_nc, orig_grid, fixed_grid]:
                        if os.path.exists(f):
                            os.remove(f)
                    if os.path.exists(temp_dir):
                        try:
                            os.rmdir(temp_dir)
                        except:
                            pass
                    return {"status": "error", "message": f"CDO error for {filename}: {e}"}
            else:
                # Save without regridding
                precip_ds.to_netcdf(output_path)
                return {"status": "success", "message": f"Saved: {output_filename}"}
                
    except Exception as e:
        return {"status": "error", "message": f"Error processing {filename}: {e}"}


def extract_precipitation_files_parallel(input_dir, output_dir, months, year=2018, 
                                        file_pattern=None, regrid=False, target_grid_file=None, 
                                        max_workers=None):
    """
    Parallel version of extract_precipitation_files
    """
    if file_pattern is None:
        file_pattern = "pangu_plasim_1_24h_45step_{year}*_ens_*.nc"
    
    file_pattern = file_pattern.format(year=year)
    os.makedirs(output_dir, exist_ok=True)
    target_months = set(months)
    
    # Find all files matching the pattern
    pattern = os.path.join(input_dir, file_pattern)
    files = sorted(glob(pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    if regrid and not target_grid_file:
        print("❌ Error: regrid=True but no target_grid_file provided")
        return
    
    if max_workers is None:
        max_workers = min(len(files), mp.cpu_count())
    
    # Prepare arguments for parallel processing
    file_args = [(file_path, output_dir, target_months, regrid, target_grid_file) 
                 for file_path in files]
    
    processed_count = 0
    error_count = 0
    
    print(f"Processing {len(files)} files using {max_workers} workers...")
    
    # Use ProcessPoolExecutor for CPU-intensive tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, args): args[0] 
                         for args in file_args}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                
                if result["status"] == "success":
                    processed_count += 1
                    print(f"✅ {result['message']}")
                elif result["status"] == "error":
                    error_count += 1
                    print(f"❌ {result['message']}")
                # Skip logging for "skipped" status to reduce noise
                    
            except Exception as exc:
                error_count += 1
                print(f"❌ {file_path} generated an exception: {exc}")
    
    print(f"\n📊 Summary:")
    print(f"   Processed: {processed_count} files")
    print(f"   Errors: {error_count} files")
    print(f"   Output directory: {output_dir}")


def merge_regridded_files(input_dir, output_file, file_pattern="regridded_precip_*.nc"):
    """
    Merge all regridded precipitation files into a single file with dimensions:
    - init_time: initialization time
    - member: ensemble member
    - day: forecast lead day (renamed from time)
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
                "day": np.arange(len(ds.day), dtype="int64")
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


def process_year(year, input_dir, output_dir, target_grid_file, months, file_pattern, 
                output_filename_template, max_workers=None):
    """
    Process a single year: extract, regrid, and merge precipitation data.
    
    Parameters:
    -----------
    year : int
        Year to process
    input_dir : str
        Input directory containing forecast files
    output_dir : str
        Output directory for processed files
    target_grid_file : str
        Path to target grid file for regridding
    months : list
        List of months to process
    file_pattern : str
        File pattern for input files (use {year} as placeholder)
    output_filename_template : str
        Template for output filename (use {year} as placeholder)
    max_workers : int, optional
        Maximum number of parallel workers
    """
    
    print(f"\n{'='*60}")
    print(f"🔄 Processing Year {year}")
    print(f"{'='*60}")
    
    # Create year-specific output directory
    year_output_dir = os.path.join(output_dir, f"year_{year}")
    
    # Step 1: Extract and regrid precipitation files
    print(f"\n📊 Step 1: Extracting and regridding precipitation for months {months}")
    extract_precipitation_files_parallel(
        input_dir=input_dir,
        output_dir=year_output_dir,
        months=months,
        year=year,
        file_pattern=file_pattern,
        regrid=True,
        target_grid_file=target_grid_file,
        max_workers=max_workers
    )
    
    # Step 2: Merge regridded files
    print(f"\n📊 Step 2: Merging regridded files")
    output_filename = output_filename_template.format(year=year)
    output_file = os.path.join(year_output_dir, output_filename)
    
    merge_regridded_files(
        input_dir=year_output_dir,
        output_file=output_file
    )
    
    print(f"\n✅ Completed processing for year {year}")
    return output_file


def process_years_parallel(years, input_dir, output_dir, target_grid_file, months, 
                          file_pattern, output_filename_template, max_workers_per_year=None, 
                          max_years_parallel=None):
    """
    Process multiple years in parallel
    """
    if max_years_parallel is None:
        max_years_parallel = min(len(years), mp.cpu_count() // 4)  # Reserve cores per year
    
    def process_year_wrapper(year):
        return process_year(
            year=year,
            input_dir=input_dir,
            output_dir=output_dir,
            target_grid_file=target_grid_file,
            months=months,
            file_pattern=file_pattern,
            output_filename_template=output_filename_template,
            max_workers=max_workers_per_year
        )
    
    processed_files = []
    
    print(f"Processing {len(years)} years using {max_years_parallel} parallel processes")
    
    with ProcessPoolExecutor(max_workers=max_years_parallel) as executor:
        future_to_year = {executor.submit(process_year_wrapper, year): year for year in years}
        
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                output_file = future.result()
                processed_files.append(output_file)
                print(f"✅ Completed year {year}")
            except Exception as exc:
                print(f"❌ Year {year} generated an exception: {exc}")
    
    return processed_files


def main():
    """Main function to process multiple years"""
    
    parser = argparse.ArgumentParser(description="Extract, regrid, and merge precipitation data for multiple years")
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020, 2021, 2022, 2023],
                       help="Years to process (default: 2018-2023)")
    parser.add_argument("--months", nargs="+", type=int, default=[5, 6, 7],
                       help="Months to process (default: 5,6,7 for monsoon)")
    parser.add_argument("--input-dir", type=str, 
                       default="/scratch/10786/bgong1/PanguWeather/v2.0/HPC_scripts/results/S2S/2/predictions",
                       help="Input directory containing forecast files")
    parser.add_argument("--output-dir", type=str, 
                       default="/scratch/10441/rmasiwal/s2s_model_out/regrid_v1pt1pt1",
                       help="Output directory for processed files")
    parser.add_argument("--grid-file", type=str, 
                       default="/scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/cdo_grid_config/2p0_grid_ind.txt",
                       help="Target grid file for regridding")
    parser.add_argument("--file-pattern", type=str,
                       default="pangu_plasim_2_24h_45step_{year}*00_ens_*.nc",
                       help="File pattern for input files. Use {year} as placeholder for year.")
    parser.add_argument("--output-filename", type=str,
                       default="{year}_merged.nc",
                       help="Output filename template. Use {year} as placeholder for year.")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of worker processes per year (default: auto)")
    parser.add_argument("--max-years-parallel", type=int, default=None,
                       help="Maximum number of years to process in parallel (default: auto)")
    parser.add_argument("--parallel-mode", choices=["files", "years", "both"], default="both",
                       help="Parallelization strategy: files only, years only, or both")
    
    args = parser.parse_args()
    
    print("🚀 Starting multi-year precipitation processing")
    print(f"Years: {args.years}")
    print(f"Months: {args.months}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Grid file: {args.grid_file}")
    print(f"Parallel mode: {args.parallel_mode}")
    print(f"Max workers per year: {args.max_workers}")
    print(f"Max years parallel: {args.max_years_parallel}")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each year based on parallel mode
    processed_files = []
    
    if args.parallel_mode == "years":
        # Process years in parallel, files sequentially within each year
        processed_files = process_years_parallel(
            years=args.years,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_grid_file=args.grid_file,
            months=args.months,
            file_pattern=args.file_pattern,
            output_filename_template=args.output_filename,
            max_workers_per_year=1,  # Sequential file processing
            max_years_parallel=args.max_years_parallel
        )
    elif args.parallel_mode == "files":
        # Process years sequentially, files in parallel within each year
        for year in args.years:
            try:
                output_file = process_year(
                    year=year,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    target_grid_file=args.grid_file,
                    months=args.months,
                    file_pattern=args.file_pattern,
                    output_filename_template=args.output_filename,
                    max_workers=args.max_workers
                )
                processed_files.append(output_file)
            except Exception as e:
                print(f"❌ Error processing year {year}: {e}")
                continue
    else:  # both
        # Process years in parallel AND files in parallel within each year
        processed_files = process_years_parallel(
            years=args.years,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_grid_file=args.grid_file,
            months=args.months,
            file_pattern=args.file_pattern,
            output_filename_template=args.output_filename,
            max_workers_per_year=args.max_workers,
            max_years_parallel=args.max_years_parallel
        )
    
    print(f"\n{'='*60}")
    print("🎉 All years processing completed!")
    print(f"{'='*60}")
    print(f"Processed {len(processed_files)} years successfully:")
    for file_path in processed_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()