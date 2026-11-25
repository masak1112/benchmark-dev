#!/bin/bash
# Simple run script for crps_ssr_main.py (no PBS, for brev environment)
#
# Usage:
#   bash submit_crps_ssr_brev.sh                             # all models, all vars
#   MODEL=ours_si bash submit_crps_ssr_brev.sh               # single model, all vars
#   MODEL=finetune_rollout_crps VAR=total_precipitation_24hr bash submit_crps_ssr_brev.sh

TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job started at: $TSTAMP"
echo "Running on host: $(hostname)"

source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate s2s_env

echo "Python: $(which python)"
echo "Python version: $(python --version)"

SCRIPT=/data/bing/benchmark-dev/metrics/CRPS/crps_ssr_main.py
CONFIG=/data/bing/benchmark-dev/metrics/CRPS/configuration/config_brev.yaml

# Build argument list from optional environment variables
ARGS="--config $CONFIG"
[ -n "$MODEL" ] && ARGS="$ARGS --model $MODEL"
[ -n "$VAR"   ] && ARGS="$ARGS --var $VAR"

echo "Running: python $SCRIPT $ARGS"
python $SCRIPT $ARGS

EXIT_CODE=$?
TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job finished at: $TSTAMP with exit code $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: script exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
