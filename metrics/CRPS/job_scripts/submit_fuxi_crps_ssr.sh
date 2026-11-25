#!/bin/bash
# PBS submission script for fuxi_s2s CRPS/SSR evaluation.
#
# Each ~3 GB solid .7z archive must be fully decompressed per variable.
# FUXI_CACHE_DIR reuses extracted archives across the 3 variable runs,
# so each archive is decompressed only once total (~317 archives x 1-2 min).
#
# Usage:
#   qsub job_scripts/submit_fuxi_crps_ssr.sh
#   qsub -v VAR=2m_temperature job_scripts/submit_fuxi_crps_ssr.sh

#PBS -A UCHI0018
#PBS -N fuxi_crps_ssr
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=100GB
#PBS -o /glade/derecho/scratch/bgong/tmp/fuxi_crps_ssr.log
#PBS -e /glade/derecho/scratch/bgong/tmp/fuxi_crps_ssr.err
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

# Persistent cache: extracted archives are reused across variable calls.
# Directory is cleaned up at the end of this job.
export FUXI_CACHE_DIR=/glade/derecho/scratch/$USER/tmp/fuxi_extract_cache
mkdir -p $FUXI_CACHE_DIR

TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job started at: $TSTAMP"
echo "Running on host: $(hostname)"
echo "FUXI_CACHE_DIR: $FUXI_CACHE_DIR"

module load conda
conda activate /glade/work/bgong/conda-envs/my_env

echo "Python: $(which python)"
echo "Python version: $(python --version)"

SCRIPT=/glade/work/bgong/benchmark-dev/metrics/CRPS/crps_ssr_main.py
CONFIG=/glade/work/bgong/benchmark-dev/metrics/CRPS/configuration/config.yaml

ARGS="--config $CONFIG --model fuxi_s2s"
[ -n "$VAR" ] && ARGS="$ARGS --var $VAR"

echo "Running: python $SCRIPT $ARGS"
python $SCRIPT $ARGS

EXIT_CODE=$?
TSTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job finished at: $TSTAMP with exit code $EXIT_CODE"

# Clean up extracted cache to free scratch space
echo "Cleaning up cache dir: $FUXI_CACHE_DIR"
rm -rf "$FUXI_CACHE_DIR"

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Output files ==="
    ls -lh /glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/fuxi_s2s/ 2>/dev/null || echo "(output dir not yet created)"
else
    echo "ERROR: script exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
