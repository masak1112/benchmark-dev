#!/bin/bash
# Soil-moisture TCC (land): UC-S2S V2 swvl1 vs ERA5 swvl1 (00Z); ECMWF hindcast sm20 vs ERA5 0-20 cm (daily mean)
#PBS -A UCHI0014
#PBS -N tcc_soil
#PBS -q develop
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_soil.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
/glade/work/bgong/.venv/bin/python -u tcc_soil.py --stage all --workers 32
EXIT_CODE=$?
echo "Job finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
