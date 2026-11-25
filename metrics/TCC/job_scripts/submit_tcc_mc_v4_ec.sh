#!/bin/bash
# multicond v4 vs ECMWF hindcast vs UC-S2S V2 (nearest init <=1 day), paired on inits all three share:
# 2019, 2020, 2021, 2019-2021 (plain + --yearly-mean). Plus multicond vs ECMWF on all shared inits (no V2 needed).
# Stars vs multicond. T2M global.
#PBS -A UCHI0014
#PBS -N tcc_mc4e
#PBS -q develop
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=8:mem=230GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_mc_v4_ec.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
run() {  # dates tag name extra models...
    local F=$1 T=$2 N=$3 X=$4; shift 4
    for m in "$@"; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20 $X; done
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models "$@"
    local SER=("ucs2s_multicond_all:$T:UC-S2S multicond" "ecmwf_rf:$T:ECMWF hindcast")
    [ $# -eq 3 ] && SER+=("ucs2s_v2_mcpair3:$T:UC-S2S V2")
    $PY -u plot_TCC_paper.py --block none --name $N --series "${SER[@]}"
}
M3="ucs2s_multicond_all ecmwf_rf ucs2s_v2_mcpair3"; M2="ucs2s_multicond_all ecmwf_rf"
run dates_mcecv2_v4_all.txt mcecv2_v4_all "multicond_v4_vs_ecmwf_rf_v2_all_common" "" $M3
run dates_mcecv2_v4_all.txt mcecv2_v4_all_ym "multicond_v4_vs_ecmwf_rf_v2_all_common_ym" "--yearly-mean" $M3
for y in 2019 2020 2021; do
    run dates_mcecv2_v4_$y.txt mcecv2_v4_$y "multicond_v4_vs_ecmwf_rf_v2_${y}_common" "" $M3
done
run dates_mcec_v4.txt mcec_v4_all "multicond_v4_vs_ecmwf_rf_all_common" "" $M2
run dates_mcec_v4.txt mcec_v4_all_ym "multicond_v4_vs_ecmwf_rf_all_common_ym" "--yearly-mean" $M2
echo "Job finished at: $(date) with exit code 0"
