#!/bin/bash
#PBS -A UCHI0014
#PBS -N clim_dm
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=100GB
#PBS -o /glade/derecho/scratch/bgong/tmp/clim_dailymean.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
python -u /glade/work/bgong/benchmark-dev/metrics/TCC/compute_clim_dailymean.py
EXIT_CODE=$?
echo "Job finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
