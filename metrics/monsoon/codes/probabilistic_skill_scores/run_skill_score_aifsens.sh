python binned_skill_score_cmz.py \
    --model_forecast_dir "/glade/derecho/scratch/rajatm/monsoon_onset_benchmarking_data_codes/model_forecast_data/aifs_ens/output_twice_weekly/tp_4p0_lsm" \
    --imd_folder "/glade/derecho/scratch/rajatm/monsoon_onset_benchmarking_data_codes/imd_rainfall_data/4p0" \
    --thres_file "/glade/derecho/scratch/rajatm/monsoon_onset_benchmarking_data_codes/imd_onset_threshold/mwset4x4.nc4" \
    --max_forecast_day 15 \
    --years 2019 2020 2021 2022 2023 2024 \
    --file_pattern "{}.nc" \
    --date_filter_year 2024 \
    --model_name "aifsens" \s
    --mok