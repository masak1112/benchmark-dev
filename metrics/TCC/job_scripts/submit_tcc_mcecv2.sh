#!/bin/bash
# multicond vs ECMWF hindcast vs UC-S2S V2 (nearest init, <=1 day), paired on inits all three share:
# 2019 only (plain) and 2019-2021 (plain + --yearly-mean). Stars vs multicond.
#PBS -A UCHI0014
#PBS -N tcc_mcecv2
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=120GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_mcecv2.log
#PBS -j oe
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
M="ucs2s_multicond_all ecmwf_rf ucs2s_v2_mcpair3"
run() {  # dates tag name extra
    for m in $M; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $1 --tag $2 --n-members 20 $4; done
    $PY -u tcc_bootstrap.py --block none --dates-file $1 --tag $2 --n-members 20 $4 --models $M
    for R in "--t2m-land" "" "--vars t2m --region land"; do
        case "$R" in "") U=_t2m_global;; --vars*) U=_t2m_land;; *) U=;; esac
        $PY -u plot_TCC_paper.py --block none --name $3$U $R --series \
            "ucs2s_multicond_all:$2:UC-S2S multicond" "ecmwf_rf:$2:ECMWF hindcast" "ucs2s_v2_mcpair3:$2:UC-S2S V2"
    done
}
run dates_mcecv2_2019.txt mcecv2_2019_m20 multicond_vs_ecmwf_rf_v2_2019_common ""
run dates_mcecv2_all.txt mcecv2_all_m20 multicond_vs_ecmwf_rf_v2_all_common ""
run dates_mcecv2_all.txt mcecv2_all_m20_ym multicond_vs_ecmwf_rf_v2_all_common_ym "--yearly-mean"
echo "Job finished at: $(date) with exit code 0"
