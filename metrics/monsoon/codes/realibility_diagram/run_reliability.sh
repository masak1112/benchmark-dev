
# With MOK filter (default)
python onset_reliability_diagram_cmz.py \
    --model_forecast_dir "/glade/derecho/scratch/rajatm/monsoon_onset_benchmarking_data_codes/model_forecast_data/gencast52/tp_lsm_2p0" \
    --imd_folder "/glade/derecho/scratch/rajatm/monsoon_onset_benchmarking_data_codes/imd_rainfall_data/2p0" \
    --thres_file "/glade/derecho/scratch/rajatm/monsoon_onset_benchmarking_data_codes/imd_onset_threshold/mwset2x2.nc4" \
    --max_forecast_day 30 \
    --save_path "./output/gencast/" \
    --years 2019 2020 2021 2022 2023 2024 \
    --file_pattern "{}.nc" \
    --mok