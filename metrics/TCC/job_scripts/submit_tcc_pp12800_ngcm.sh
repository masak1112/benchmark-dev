#!/bin/bash
# multicond pp12800 (20) vs ECMWF hindcast (11) vs NeuralGCM (20, nearest init <=2 days; pairs_ngcm_pp12800.txt)
# on the 63 multicond/ECMWF shared inits. NeuralGCM has no t2m. Plain + --yearly-mean; global / NH / tropics.
#PBS -A UCHI0014
#PBS -N tcc_ppng
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_pp12800_ngcm.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
F=dates_pp12800_ec_m20.txt; MC=ucs2s_mc_pp12800f; NG=ngcm_pp12800pair
for X in "" "--yearly-mean"; do
    T=pp20_ngec; N=multicond_pp12800_vs_ecmwf_rf_ngcm; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
    for m in $MC ecmwf_rf $NG; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20 $X; done
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $MC ecmwf_rf $NG
    SER=("$MC:$T:UC-S2S multicond" "$NG:$T:NeuralGCM" "ecmwf_rf:$T:ECMWF hindcast")
    for R in global NH tropics; do S=""; [ $R != global ] && S=_$R
        $PY -u plot_TCC_paper.py --block none --region $R --name $N$S --series "${SER[@]}"
        $PY -u plot_TCC_paper.py --block none --region $R --vars u850 v850 --name ${N}_uv850$S --series "${SER[@]}"; done
done
echo "Job finished at: $(date) with exit code 0"
