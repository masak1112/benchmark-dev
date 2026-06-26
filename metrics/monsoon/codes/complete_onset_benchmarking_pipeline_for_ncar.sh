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

# Model configuration
# Model configuration
BASE_MODEL_NAME="${BASE_MODEL_NAME:-gencast}"  # Base name for the model, will be combined with period or custom tag
MEM_NUM="${MEM_NUM:-51}"
DATE_FILTER_YEAR="${DATE_FILTER_YEAR:-2024}"  # Year to use for date filtering, change for FuXi-S2S to 2022

PERIOD="${PERIOD:-extended}"  # Options: recent, extended, common, full, custom
CUSTOM_YEARS="${CUSTOM_YEARS:-2019 2020 2021 2022}"  # Space-separated list of years for custom period

# Set years based on period or custom input
if [[ "${PERIOD}" == "custom" ]]; then
    if [[ -z "${CUSTOM_YEARS}" ]]; then
        echo "ERROR: CUSTOM_YEARS must be specified when PERIOD=custom"
        echo "Example: PERIOD=custom CUSTOM_YEARS=\"2010 2011 2015 2020 2021\" ./script.sh"
        exit 1
    fi
    YEARS="${CUSTOM_YEARS}"
    MODEL_NAME="${BASE_MODEL_NAME}_customperiod"
else
    case "${PERIOD}" in
        "recent")
            YEARS="$(seq -s ' ' 2019 2024)"
            ;;
        "extended")
            YEARS="$(seq -s ' ' 1965 1978) $(seq -s ' ' 2019 2024)"
            ;;
        "common")
            YEARS="$(seq -s ' ' 2004 2021)"
            ;;
        "full")
            YEARS="$(seq -s ' ' 1965 2024)"
            ;;
        *)
            echo "ERROR: Invalid PERIOD '${PERIOD}'. Valid options: recent, extended, common, full, custom"
            exit 1
            ;;
    esac
    MODEL_NAME="${BASE_MODEL_NAME}_${PERIOD}period"
fi

# MODEL_NAME="${MODEL_NAME:-ifss2s}"
MODEL_FORECAST_DIR="${MODEL_FORECAST_DIR:-/glade/work/rajatm/benchmark-dev/metrics/monsoon/data_files/model_forecast_data/gencast52/tp_lsm_2p0}"

# Analysis parameters
#YEARS="2019 2020 2021 2022 2023 2024"
FILE_PATTERN="{}.nc"

# Input data paths
IMD_FOLDER="${IMD_FOLDER:-/glade/work/rajatm/benchmark-dev/metrics/monsoon/data_files/IMD_2by2}"
THRES_FILE="${THRES_FILE:-/glade/work/rajatm/benchmark-dev/metrics/monsoon/data_files/mwset2x2.nc4}"
SHPFILE_PATH="${SHPFILE_PATH:-/glade/work/rajatm/benchmark-dev/metrics/monsoon/data_files/ind_map_shpfile/india_shapefile.shp}"

# Code paths
BENCHMARK_CODE_DIR="${BENCHMARK_CODE_DIR:-/glade/work/rajatm/benchmark-dev/metrics/monsoon}"
MAE_SCRIPT="${MAE_SCRIPT:-${BENCHMARK_CODE_DIR}/codes/mae_far_mr_probablistic_models/mae_far_mr_probabilistic_models.py}"
RELIABILITY_SCRIPT="${RELIABILITY_SCRIPT:-${BENCHMARK_CODE_DIR}/codes/realibility_diagram/onset_reliability_diagram_cmz.py}"
SKILL_SCORE_SCRIPT="${SKILL_SCORE_SCRIPT:-${BENCHMARK_CODE_DIR}/codes/probabilistic_skill_scores/binned_skill_score_cmz.py}"

# Output paths
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}}"
#PLOT_DIR="${PLOT_DIR:-${OUTPUT_DIR}/plots/${MODEL_NAME}}"
#CSV_OUTPUT_DIR="${CSV_OUTPUT_DIR:-${OUTPUT_DIR}/csv_results/${MODEL_NAME}}"

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
check_variable "BASE_MODEL_NAME"
check_variable "MODEL_FORECAST_DIR"
check_variable "IMD_FOLDER"
check_variable "THRES_FILE"
check_variable "SHPFILE_PATH"
check_variable "BENCHMARK_CODE_DIR"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
#mkdir -p "${PLOT_DIR}"

echo "Configuration validated. Starting pipeline for model: ${MODEL_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Period: ${PERIOD}"
echo "Years: ${YEARS}"
echo "Starting pipeline..."

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
    --date_filter_year "${DATE_FILTER_YEAR}" \
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
    --date_filter_year "${DATE_FILTER_YEAR}" \
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
    --date_filter_year "${DATE_FILTER_YEAR}" \
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
    --date_filter_year "${DATE_FILTER_YEAR}" \
    --mok

echo "Step 5: Calculating probabilistic skill scores...1-15 day"
python "${SKILL_SCORE_SCRIPT}" \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${MEM_NUM} \
    --max_forecast_day 15 \
    --years ${YEARS} \
    --file_pattern "${FILE_PATTERN}" \
    --date_filter_year "${DATE_FILTER_YEAR}" \
    --model_name "${MODEL_NAME}" \
    --save_dir "${OUTPUT_DIR}" \
    --mok


echo "Step 6: Calculating probabilistic skill scores...1-30 day"
python "${SKILL_SCORE_SCRIPT}" \
    --model_forecast_dir "${MODEL_FORECAST_DIR}" \
    --imd_folder "${IMD_FOLDER}" \
    --thres_file "${THRES_FILE}" \
    --mem_num ${MEM_NUM} \
    --max_forecast_day 30 \
    --years ${YEARS} \
    --file_pattern "${FILE_PATTERN}" \
    --date_filter_year "${DATE_FILTER_YEAR}" \
    --model_name "${MODEL_NAME}" \
    --save_dir "${OUTPUT_DIR}" \
    --mok

echo "Pipeline completed successfully! Results saved in: ${OUTPUT_DIR}/${MODEL_NAME}"