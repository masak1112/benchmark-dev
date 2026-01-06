#!/bin/bash

# =============================================================================
# Monsoon Onset Benchmarking Pipeline
# =============================================================================
# This script runs a complete pipeline for monsoon onset benchmarking analysis
# Make sure to update the configuration section below before running
# =============================================================================

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# =============================================================================

# Model configuration [Change this for a different model version or forecast dataset]
MODEL_NAME="${MODEL_NAME:-s2sens_3step}"
MODEL_FORECAST_DIR="${MODEL_FORECAST_DIR:-/scratch/10441/rmasiwal/s2s_model_out/regrid_rollout_step3}"

# Analysis parameters
YEARS="2019 2020 2021 2022 2023 2024"
MEM_NUM="${MEM_NUM:-50}"
FILE_PATTERN="{}.nc"

# Input data paths
IMD_FOLDER="${IMD_FOLDER:-/scratch/10441/rmasiwal/monsoon_onset_benchmarking_data_codes/imd_rainfall_data/2p0}"
THRES_FILE="${THRES_FILE:-/scratch/10441/rmasiwal/monsoon_onset_benchmarking_data_codes/imd_onset_threshold/mwset2x2.nc4}"
SHPFILE_PATH="${SHPFILE_PATH:-/scratch/10441/rmasiwal/monsoon_onset_benchmarking_data_codes/ind_map_shpfile/india_shapefile.shp}"

# Code paths
BENCHMARK_CODE_DIR="${BENCHMARK_CODE_DIR:-/scratch/10441/rmasiwal/benchmark-dev/metrics/monsoon/codes}"
MAE_SCRIPT="${MAE_SCRIPT:-${BENCHMARK_CODE_DIR}/mae_far_mr_probablistic_models/mae_far_mr_probabilistic_models.py}"
RELIABILITY_SCRIPT="${RELIABILITY_SCRIPT:-${BENCHMARK_CODE_DIR}/realibility_diagram/onset_reliability_diagram_cmz.py}"
SKILL_SCORE_SCRIPT="${SKILL_SCORE_SCRIPT:-${BENCHMARK_CODE_DIR}/probabilistic_skill_scores/binned_skill_score_cmz.py}"

# Output paths
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}}"

# =============================================================================
# VALIDATION - CHECK IF REQUIRED VARIABLES ARE SET
# =============================================================================

check_variable() {
    if [[ -z "${!1}" || "${!1}" == "/path/to/"* || "${!1}" == "your_model_name" ]]; then
        echo "ERROR: $1 is not properly configured. Please update the configuration section."
        exit 1
    fi
}

echo "Validating configuration..."
check_variable "MODEL_NAME"
check_variable "MODEL_FORECAST_DIR"
check_variable "IMD_FOLDER"
check_variable "THRES_FILE"
check_variable "SHPFILE_PATH"
check_variable "BENCHMARK_CODE_DIR"

# Create output directories
mkdir -p "${OUTPUT_DIR}"

echo "Configuration validated. Starting pipeline for model: ${MODEL_NAME}"

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

echo "Step 1: Running MAE/FAR/MR analysis for 1-15 day forecasts..."
python "${MAE_SCRIPT}" \
    --years ${YEARS} \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${MEM_NUM} \
    --file_pattern "${FILE_PATTERN}" \
    --shpfile_path "${SHPFILE_PATH}" \
    --tolerance_days 3 \
    --verification_window 1 \
    --forecast_days 15 \
    --max_forecast_day 15 \
    --mok \
    --output_file "${OUTPUT_DIR}/results_${MODEL_NAME}_1_15_day_MOK.nc" \
    --plot_dir "${OUTPUT_DIR}"

echo "Step 2: Running MAE/FAR/MR analysis for 16-30 day forecasts..."
python "${MAE_SCRIPT}" \
    --years ${YEARS} \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${MEM_NUM} \
    --file_pattern "${FILE_PATTERN}" \
    --shpfile_path "${SHPFILE_PATH}" \
    --tolerance_days 5 \
    --verification_window 16 \
    --forecast_days 30 \
    --max_forecast_day 30 \
    --mok \
    --output_file "${OUTPUT_DIR}/results_${MODEL_NAME}_16_30_day_MOK.nc" \
    --plot_dir "${OUTPUT_DIR}"

echo "Step 3: Generating reliability diagrams for 15-day forecasts..."
python "${RELIABILITY_SCRIPT}" \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${MEM_NUM} \
    --max_forecast_day 15 \
    --save_path "${OUTPUT_DIR}" \
    --years ${YEARS} \
    --file_pattern "${FILE_PATTERN}" \
    --mok

echo "Step 4: Generating reliability diagrams for 30-day forecasts..."
python "${RELIABILITY_SCRIPT}" \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${MEM_NUM} \
    --max_forecast_day 30 \
    --save_path "${OUTPUT_DIR}" \
    --years ${YEARS} \
    --file_pattern "${FILE_PATTERN}" \
    --mok

echo "Step 5: Calculating probabilistic skill scores..."
python "${SKILL_SCORE_SCRIPT}" \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${MEM_NUM} \
    --max_forecast_day 30 \
    --years ${YEARS} \
    --file_pattern "${FILE_PATTERN}" \
    --date_filter_year 2024 \
    --model_name "${MODEL_NAME}" \
    --save_dir "${OUTPUT_DIR}" \
    --mok

echo "Pipeline completed successfully! Results saved in: ${OUTPUT_DIR}"