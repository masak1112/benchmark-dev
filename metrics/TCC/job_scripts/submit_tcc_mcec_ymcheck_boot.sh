#!/bin/bash
# bootstraps + plots for the multicond vs ECMWF pooled checks (2019+2020 plain/ym, 2019-2021 ym)
#PBS -A UCHI0014
#PBS -N tcc_ymboot
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=100GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ymcheck_boot.log
#PBS -j oe
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
for p in "dates_mcec1920_v3.txt chk1920_m20 multicond_vs_ecmwf_rf_2019_2020_common" \
         "dates_mcec1920_v3.txt chk1920_m20_ym multicond_vs_ecmwf_rf_2019_2020_common_ym --yearly-mean" \
         "dates_mcall_ec_common.txt chkall_m20_ym multicond_vs_ecmwf_rf_2019_2021_common_ym --yearly-mean"; do
    set -- $p
    $PY -u tcc_bootstrap.py --block none --dates-file $1 --tag $2 --n-members 20 $4 --models ucs2s_multicond_all ecmwf_rf
    for R in "--t2m-land" "" "--vars t2m --region land"; do
        case "$R" in "") U=_t2m_global;; --vars*) U=_t2m_land;; *) U=;; esac
        $PY -u plot_TCC_paper.py --block none --name $3$U $R --series "ucs2s_multicond_all:$2:UC-S2S multicond" "ecmwf_rf:$2:ECMWF hindcast"
    done
done
echo "Job finished at: $(date) with exit code 0"
