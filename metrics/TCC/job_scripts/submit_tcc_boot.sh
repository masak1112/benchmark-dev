#!/bin/bash
# Block-bootstrap CIs for weekly TCC (reads the tcc_cache written by submit_tcc.sh STAGE=extract)
#PBS -A UCHI0014
#PBS -N tcc_boot
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_bootstrap.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC

set -e
# paired UC-S2S V2 vs FuXi on the 86 common inits: month blocks (main result) and iid (for reference)
python -u tcc_bootstrap.py --models ucs2s_v2 fuxi --tag common86 --block month
python -u tcc_bootstrap.py --models ucs2s_v2 fuxi --tag common86 --block none
# FuXi alone on 2017-2021 (paper test period)
python -u tcc_bootstrap.py --models fuxi --dates-file dates_fuxi_2017_2021.txt --tag y2017_2021 --block month

echo "Job finished at: $(date) with exit code 0"
