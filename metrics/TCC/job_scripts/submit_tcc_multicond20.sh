#!/bin/bash
# UC-S2S multicond (20 members) vs UC-S2S V2 (20 members) on their common inits: bootstrap + paper bar plot
# (the bootstrap needs ~8+ GB and is killed on the login nodes)
#PBS -A UCHI0014
#PBS -N tcc_mc20
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_multicond20.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
C=common_dates_v2_multicond20.txt; T=v2mc20_m20
$PY -u tcc_bootstrap.py --block none --dates-file $C --tag $T --n-members 20 --models ucs2s_v2 ucs2s_multicond20
$PY -u plot_TCC_paper.py --block none --models ucs2s_v2 ucs2s_multicond20 --tag $T
$PY -u plot_TCC_paper.py --block none --models ucs2s_v2 ucs2s_multicond20 --tag $T --vars t2m --region land
echo "Job finished at: $(date) with exit code 0"
