#!/bin/bash
# FuXi with all 51 members on its 2019-2021 inits (312): aggregate + bootstrap, as is and month mean removed
#PBS -A UCHI0014
#PBS -N tcc_fx_m51
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_fuxi_m51.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export OMP_NUM_THREADS=32
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_fuxi_2019_2021.txt
$PY -u tcc_weekly.py --model fuxi --stage aggregate --dates-file $D --tag fx1921
$PY -u tcc_weekly.py --model fuxi --stage aggregate --dates-file $D --tag fx1921_mm --monthly-mean
$PY -u tcc_bootstrap.py --models fuxi --dates-file $D --tag fx1921    --block month
$PY -u tcc_bootstrap.py --models fuxi --dates-file $D --tag fx1921_mm --block month --monthly-mean
for suf in _m20 _m20_mm; do
    $PY -u plot_TCC_paper.py --own-record --suffix $suf --fuxi-m51
done
echo "Job finished at: $(date) with exit code 0"
