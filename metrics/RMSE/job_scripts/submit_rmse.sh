#!/bin/bash
# PBS submission script for rmse_main.py
#
# Usage:
#   qsub submit_rmse.sh                                          # all models, all vars
#   qsub -v MODEL=uc_s2s_v2 submit_rmse.sh                      # single model
#   qsub -v MODEL=uc_s2s_v2,VAR=sea_surface_temperature submit_rmse.sh

#PBS -A UCHI0018
#PBS -N rmse_eval
#PBS -q develop
#PBS -l walltime=05:40:00
#PBS -l select=1:ncpus=4:mem=100GB
#PBS -o /glade/derecho/scratch/bgong/tmp/rmse_eval.log
#PBS -e /glade/derecho/scratch/bgong/tmp/rmse_eval.err
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

echo "Job started: $(date)"
echo "Host: $(hostname)"

module load conda
conda activate /glade/work/bgong/conda-envs/my_env

SCRIPT=/glade/work/bgong/benchmark-dev/metrics/RMSE/rmse_main.py
CONFIG=/glade/work/bgong/benchmark-dev/metrics/RMSE/configuration/config.yaml

ARGS="--config $CONFIG"
[ -n "$MODEL" ] && ARGS="$ARGS --model $MODEL"
[ -n "$VAR"   ] && ARGS="$ARGS --var $VAR"

echo "Running: python $SCRIPT $ARGS"
python $SCRIPT $ARGS

EXIT_CODE=$?
echo "Job finished: $(date) | exit code $EXIT_CODE"
exit $EXIT_CODE
