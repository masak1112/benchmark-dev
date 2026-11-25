#!/bin/bash
# PBS job script: CRPS/SSR for ERA5 climatological baseline (1979-2021 as ensemble members)
#
# Usage:
#   qsub submit_era5_clim_crps.sh
#   qsub -v VAR=2m_temperature submit_era5_clim_crps.sh
#   qsub -v VAR=geopotential_500 submit_era5_clim_crps.sh

#PBS -A UCHI0018
#PBS -N era5_clim_crps
#PBS -q develop
#PBS -l walltime=05:40:00
#PBS -l select=1:ncpus=8:mem=150GB
#PBS -o /glade/derecho/scratch/bgong/tmp/era5_clim_crps.log
#PBS -e /glade/derecho/scratch/bgong/tmp/era5_clim_crps.err
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

ARGS="--config $CONFIG --model era5_climatology"
[ -n "$VAR" ] && ARGS="$ARGS --var $VAR"

echo "Running: python $SCRIPT $ARGS"
python $SCRIPT $ARGS

EXIT_CODE=$?
TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job finished at: $TSTAMP with exit code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Output files ==="
    ls -lh /glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/era5_climatology/
fi

exit $EXIT_CODE
