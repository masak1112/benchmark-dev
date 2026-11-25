#!/bin/bash
# ERA5 DJF NAO EOF + daily index 1979-2024 (reads ERA5 6-hourly Z500)
#PBS -A UCHI0014
#PBS -N nao_eof
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mem=60GB
#PBS -o /glade/derecho/scratch/bgong/tmp/nao_eof.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
cd /glade/work/bgong/benchmark-dev/metrics/NAO
set -e
python -u nao_eof.py --workers 16
echo "Job finished at: $(date)"
