#!/bin/bash
# ECMWF S2S reforecasts (model year 2018) -> per-member daily NAO index, Nov-Mar inits
#PBS -A UCHI0014
#PBS -N nao_ecmwf
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=32:mem=100GB
#PBS -j oe
exec > /glade/derecho/scratch/bgong/tmp/nao_extract_ecmwf_rf2018.log 2>&1
echo "Job started at: $(date) on $(hostname)"
cd /glade/work/bgong/benchmark-dev/metrics/NAO
/glade/work/bgong/.venv/bin/python -u nao_extract.py --model ecmwf_rf2018 --months 11 12 01 02 03 --workers 32
echo "Job finished at: $(date) with exit code $?"
