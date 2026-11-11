#!/bin/bash
# multicond vs ECMWF hindcast, 2019 only, paired on their shared inits (Jan - Aug 2019)
#PBS -A UCHI0014
#PBS -N tcc_mcec19
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_mcec2019.log
#PBS -j oe
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_mcec2019_common.txt; T=mcec2019_m20
$PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 20 --models ucs2s_multicond_all ecmwf_rf
for R in "--t2m-land" "" "--vars t2m --region land"; do
    case "$R" in "") U=_t2m_global;; --vars*) U=_t2m_land;; *) U=;; esac
    $PY -u plot_TCC_paper.py --block none --name multicond_vs_ecmwf_rf_2019_common$U $R --series \
        "ucs2s_multicond_all:$T:UC-S2S multicond" "ecmwf_rf:$T:ECMWF hindcast"
done
echo "Job finished at: $(date) with exit code 0"
