#!/bin/bash
#PBS -A UCHI0018
#PBS -N amip_crps_ssr
#PBS -q develop
#PBS -l walltime=05:40:00
#PBS -l select=1:ncpus=4:mem=100GB
#PBS -o /glade/derecho/scratch/bgong/tmp/amip_crps_ssr.log
#PBS -e /glade/derecho/scratch/bgong/tmp/amip_crps_ssr.err
#PBS -j oe

# Scratch for temporary/cache files
export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job started at: $TSTAMP"
echo "Running on host: $(hostname)"

# Load conda and activate environment
module load conda
conda activate /glade/work/bgong/conda-envs/my_env

echo "Python: $(which python)"
echo "Python version: $(python --version)"

SCRIPT=/glade/work/bgong/benchmark-dev/metrics/CRPS/amip_crps_ssr.py

# Run all variables (pass a variable name as argument to run just one, e.g.:
#   python $SCRIPT 2m_temperature
#   python $SCRIPT geopotential_500
#   python $SCRIPT sea_surface_temperature
#   python $SCRIPT total_precipitation_24hr
python $SCRIPT total_precipitation_24hr

EXIT_CODE=$?
TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job finished at: $TSTAMP with exit code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Output files ==="
    ls -lh /glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/amip_results_2019_2022/
else
    echo "ERROR: script exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
