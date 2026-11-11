#!/bin/bash
# SSTanom / V2 / FuXi / NeuralGCM / SI-S2S on the 73 inits shared by all (dates_sst6_m20.txt); partners paired to nearest init.
# 20 members each. No ECMWF. Stars vs SSTanom. Plain anomalies (ERA5 1979-2018 DOY clim, centered TCC).
#PBS -A UCHI0014
#PBS -N tcc_sst6fx
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst6fx.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/u/apps/opt/conda/envs/npl-2025b/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
F=dates_sst6_m20.txt; T=sst6fx; N=sstanom_vs_v2_fuxi_ngcm_si
SST=ucs2s_sstanom3y; V2=ucs2s_v2_mc3ypair; FX=fuxi_mc3ypair; NG=ngcm_mc3ypair; SI=si_s2s_mc3ypair
ALL="$SST $V2 $FX $NG $SI"
for m in $ALL; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20; done
$PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 --models $ALL
SER=("$SST:$T:UC-S2S SSTanom" "$V2:$T:UC-S2S V2" "$FX:$T:FuXi" "$NG:$T:NeuralGCM" "$SI:$T:SI-S2S")
for R in global tropics extratropics; do S=""; [ $R != global ] && S=_$R
    $PY -u plot_TCC_paper.py --block none --region $R --vars tp z500 u850 v850 --name ${N}_tp_z500_uv850$S --series "${SER[@]}"
    $PY -u plot_TCC_paper.py --block none --region $R --vars t2m --name ${N}_t2m$S --series "${SER[@]}"; done
$PY -u plot_TCC_paper.py --block none --region land_interior --vars t2m --name ${N}_t2m_land_interior --series "${SER[@]}"
echo "Job finished at: $(date) with exit code 0"
