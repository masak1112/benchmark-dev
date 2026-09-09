#!/bin/bash
# PBS submission script for crps_ssr_main.py
#
# Usage:
#   qsub submit_crps_ssr.sh                             # all models, all vars
#   qsub -v MODEL=ours_si submit_crps_ssr.sh            # single model, all vars
#   qsub -v MODEL=amip_s2s_train_scratch,VAR=geopotential_500 submit_crps_ssr.sh
#
# Override job name at submission:
#   qsub -N my_job_name submit_crps_ssr.sh

#PBS -A UCHI0018
#PBS -N crps_ssr
#PBS -q develop
#PBS -l walltime=05:40:00
#PBS -l select=1:ncpus=4:mem=100GB
#PBS -o /glade/derecho/scratch/bgong/tmp/crps_ssr_v2wqs.log
#PBS -e /glade/derecho/scratch/bgong/tmp/crps_ssr.err
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job started at: $TSTAMP"
echo "Running on host: $(hostname)"

module load conda
conda activate /glade/work/bgong/conda-envs/my_env

echo "Python: $(which python)"
echo "Python version: $(python --version)"

SCRIPT=/glade/work/bgong/benchmark-dev/metrics/CRPS/crps_ssr_main.py
CONFIG=/glade/work/bgong/benchmark-dev/metrics/CRPS/configuration/config.yaml

# Build argument list from optional PBS variables
ARGS="--config $CONFIG"
[ -n "$MODEL" ] && ARGS="$ARGS --model $MODEL"
[ -n "$VAR"   ] && ARGS="$ARGS --var $VAR"

echo "Running: python $SCRIPT $ARGS"
python $SCRIPT $ARGS

EXIT_CODE=$?
TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job finished at: $TSTAMP with exit code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Output files ==="
    ls -lh /glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/
else
    echo "ERROR: script exited with code $EXIT_CODE"
fi

exit $EXIT_CODE


