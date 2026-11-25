#!/bin/bash
# ECMWF S2S in the TCC: (1) fix the cached FuXi/NeuralGCM tp truth (ERA5 tp24 at 00 UTC of day L+1),
# (2) extract ECMWF 2019-2021 at 20 members (GRIB via cfgrib -> runs in /glade/work/bgong/.venv)
#PBS -A UCHI0014
#PBS -N tcc_ec_prep
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ecmwf_prep.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export TMPDIR=/glade/derecho/scratch/$USER/tmp
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
$PY -u fix_tp_truth.py 32
$PY -u tcc_weekly.py --model ecmwf --stage extract --workers 32 --dates-file dates_ecmwf_2019_2021.txt --tag ec1921 --n-members 20
echo "Job finished at: $(date) with exit code 0"
