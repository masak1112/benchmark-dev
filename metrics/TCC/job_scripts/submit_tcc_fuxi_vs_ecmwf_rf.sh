#!/bin/bash
# FuXi vs ECMWF 2025-model hindcast on their 162 shared inits (2019-2021): FuXi with 20 and with all 51 members
# (ECMWF hindcast always 11), as is and month mean removed; aggregates, paired bootstrap, paper figures
#PBS -A UCHI0014
#PBS -N tcc_fx_ecrf
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_fuxi_vs_ecmwf_rf.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export OMP_NUM_THREADS=32
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
C=common_dates_fuxi_ecmwf_rf.txt
for mem in m20 m51; do
    N=""; [ $mem = m20 ] && N="--n-members 20"       # m51: no cap -> FuXi 51, ECMWF hindcast 11
    for suf in "" _mm; do
        O="$N"; [ "$suf" = "_mm" ] && O="$O --monthly-mean"
        T=fxecrf_$mem$suf
        for m in fuxi ecmwf_rf; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $C --tag $T $O; done
        $PY -u tcc_bootstrap.py --models fuxi ecmwf_rf --dates-file $C --tag $T --block month $O
        $PY -u plot_TCC_paper.py --models fuxi ecmwf_rf --tag $T
    done
done
echo "Job finished at: $(date) with exit code 0"
