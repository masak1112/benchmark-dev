#!/bin/bash
# multicond v4 vs FuXi-S2S (20, paired on mc & FuXi) and ECMWF hindcast (own inits in mc windows; and all three
# paired on mc & FuXi & ECMWF). Plain + --yearly-mean. T2M global.
#PBS -A UCHI0014
#PBS -N tcc_mc4f
#PBS -q develop
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=8:mem=230GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_mc_v4_fx.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
MC=ucs2s_multicond_all; D=dates_mcfx_v4.txt; E=dates_ecmwf_rf_mcwin_v4.txt; X=dates_mcfxec_v4.txt
for YM in "" "--yearly-mean"; do
    S=""; [ -n "$YM" ] && S=_ym
    T=mcfx_v4$S; TE=ec_mcwin_v4$S; TX=mcfxec_v4$S
    for m in $MC fuxi; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag $T --n-members 20 $YM; done
    $PY -u tcc_weekly.py --model ecmwf_rf --stage aggregate --dates-file $E --tag $TE --n-members 20 $YM
    for m in $MC fuxi ecmwf_rf; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $X --tag $TX --n-members 20 $YM; done
    $PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 20 $YM --models $MC fuxi
    $PY -u tcc_bootstrap.py --block none --dates-file $E --tag $TE --n-members 20 $YM --models ecmwf_rf
    $PY -u tcc_bootstrap.py --block none --dates-file $X --tag $TX --n-members 20 $YM --models $MC fuxi ecmwf_rf
    $PY -u plot_TCC_paper.py --block none --name multicond_v4_vs_fuxi_ecmwf_rf$S --series \
        "$MC:$T:UC-S2S multicond" "fuxi:$T:FuXi-S2S" "ecmwf_rf:$TE:ECMWF hindcast"
    $PY -u plot_TCC_paper.py --block none --name multicond_v4_vs_fuxi_ecmwf_rf_common$S --series \
        "$MC:$TX:UC-S2S multicond" "fuxi:$TX:FuXi-S2S" "ecmwf_rf:$TX:ECMWF hindcast"
done
echo "Job finished at: $(date) with exit code 0"
