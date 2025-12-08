#!/bin/bash

# Model configuration variables
MODEL_NAME="ngcm_qmapped"
MODEL_FORECAST_DIR="/scratch/10441/rmasiwal/onset_benchmark_paper/qmapping_output"

# Common parameters
YEARS="2019 2020 2021 2022 2023 2024"
FILE_PATTERN='{}_qm_ngcm.nc'
mem_num="51"
# Common paths
IMD_FOLDER="/scratch/10441/rmasiwal/monsoon_onset_benchmarking_data_codes/imd_rainfall_data/4p0"
THRES_FILE="/scratch/10441/rmasiwal/monsoon_onset_benchmarking_data_codes/imd_onset_threshold/mwset4x4.nc4"
SHPFILE_PATH="/scratch/10441/rmasiwal/monsoon_onset_benchmarking_data_codes/ind_map_shpfile/india_shapefile.shp"

# python /scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/mae_far_mr_probablistic_models/mae_far_mr_probabilistic_models.py \
#     --years ${YEARS} \
#     --model_forecast_dir "${MODEL_FORECAST_DIR}" \
#     --imd_folder "${IMD_FOLDER}" \
#     --thres_file "${THRES_FILE}" \
#     --mem_num ${mem_num} \
#     --file_pattern "${FILE_PATTERN}" \
#     --shpfile_path "${SHPFILE_PATH}" \
#     --tolerance_days 3 \
#     --verification_window 1 \
#     --forecast_days 15 \
#     --max_forecast_day 15 \
#     --mok \
#     --output_file "results_${MODEL_NAME}_1_15_day_MOK.nc" \
#     --plot_dir "./output/plots/${MODEL_NAME}"

# python /scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/mae_far_mr_probablistic_models/mae_far_mr_probabilistic_models.py \
#     --years ${YEARS} \
#     --model_forecast_dir "${MODEL_FORECAST_DIR}" \
#     --imd_folder "${IMD_FOLDER}" \
#     --thres_file "${THRES_FILE}" \
#     --mem_num ${mem_num} \
#     --file_pattern "${FILE_PATTERN}" \
#     --shpfile_path "${SHPFILE_PATH}" \
#     --tolerance_days 5 \
#     --verification_window 16 \
#     --forecast_days 30 \
#     --max_forecast_day 30 \
#     --mok \
#     --output_file "results_${MODEL_NAME}_16_30_day_MOK.nc" \
#     --plot_dir "./output/plots/${MODEL_NAME}"

# # With MOK filter (default)
# python /scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/realibility_diagram/onset_reliability_diagram_cmz.py \
#     --model_forecast_dir "${MODEL_FORECAST_DIR}" \
#     --imd_folder "${IMD_FOLDER}" \
#     --thres_file "${THRES_FILE}" \
#     --mem_num ${mem_num} \
#     --max_forecast_day 15 \
#     --save_path "./output/${MODEL_NAME}/" \
#     --years ${YEARS} \
#     --file_pattern "${FILE_PATTERN}" \
#     --mok

# python /scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/realibility_diagram/onset_reliability_diagram_cmz.py \
#     --model_forecast_dir "${MODEL_FORECAST_DIR}" \
#     --imd_folder "${IMD_FOLDER}" \
#     --thres_file "${THRES_FILE}" \
#     --mem_num ${mem_num} \
#     --max_forecast_day 30 \
#     --save_path "./output/${MODEL_NAME}/" \
#     --years ${YEARS} \
#     --file_pattern "${FILE_PATTERN}" \
#     --mok

python /scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/probabilistic_skill_scores/binned_skill_score_cmz.py \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${mem_num} \
    --max_forecast_day 15 \
    --years ${YEARS} \
    --file_pattern "${FILE_PATTERN}" \
    --date_filter_year 2024 \
    --model_name "${MODEL_NAME}" \
    --mok

# python /scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes/probabilistic_skill_scores/binned_skill_score_cmz.py \
#     --model_forecast_dir "${MODEL_FORECAST_DIR}" \
#     --imd_folder "${IMD_FOLDER}" \
#     --thres_file "${THRES_FILE}" \
#     --mem_num ${mem_num} \
#     --max_forecast_day 30 \
#     --years ${YEARS} \
#     --file_pattern "${FILE_PATTERN}" \
#     --date_filter_year 2024 \
#     --model_name "${MODEL_NAME}" \
#     --mok