#!/bin/bash
# UC-S2S V2 on its 151 extracted inits: aggregate + month-block bootstrap
#PBS -A UCHI0014
#PBS -N tcc_ucall
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ucall_followup.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
python -u tcc_weekly.py --model ucs2s_v2 --stage aggregate --dates-file dates_ucs2sv2_cached151.txt --tag cached151
python -u tcc_bootstrap.py --models ucs2s_v2 --dates-file dates_ucs2sv2_cached151.txt --tag cached151 --block month
echo "Job finished at: $(date) with exit code 0"
