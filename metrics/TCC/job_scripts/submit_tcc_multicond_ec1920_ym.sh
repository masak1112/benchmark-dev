#!/bin/bash
# multicond (20) vs ECMWF hindcast (11), paired on the 15 inits they share in the 2019 + 2020 windows
#PBS -A UCHI0014
#PBS -N tcc_ymec1920
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_multicond_ec1920_ym.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
# paired multicond vs ECMWF hindcast on the 15 inits both have exactly (V2 on the same labels, nearest-init paired)
D=dates_mc_ec_1920_common.txt; T=mcec1920_m20_ym
for m in ucs2s_multicond1920 ucs2s_v2_mcpair ecmwf_rf; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag $T --n-members 20 --yearly-mean; done
$PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 20 --yearly-mean --models ucs2s_multicond1920 ucs2s_v2_mcpair ecmwf_rf
for R in "--t2m-land" "" "--vars t2m --region land"; do
    N=multicond_vs_ecmwf_rf_common15_2019_2020_ym
    case "$R" in "") N=${N}_t2m_global;; --vars*) N=${N}_t2m_land;; esac
    $PY -u plot_TCC_paper.py --block none --name $N $R --series \
        "ucs2s_multicond1920:$T:UC-S2S multicond" "ecmwf_rf:$T:ECMWF hindcast" "ucs2s_v2_mcpair:$T:UC-S2S V2"
done
echo "Job finished at: $(date) with exit code 0"
