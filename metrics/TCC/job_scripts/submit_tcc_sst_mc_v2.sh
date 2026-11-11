#!/bin/bash
# UC-S2S SSTanom (updated transfer 2026-11-05) vs multicond vs V2 vs ECMWF hindcast (11 members), 20 members each, on their common inits
# (dates_sst_mc_v2_m20.txt; V2 paired by nearest init, pairs_ucs2s_v2_mc3y.txt). Stars vs SSTanom.
# Plain + --yearly-mean; global / NH / tropics / extratropics; TP+Z500+UV850 (2x2) and T2M.
#PBS -A UCHI0014
#PBS -N tcc_sst3w
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst_mc_v2.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/u/apps/opt/conda/envs/npl-2025b/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
P=/glade/derecho/scratch/bgong/pp_si_infer_sstanom/predictions
SST=ucs2s_sstanom3y; MC=ucs2s_mc_pp12800f; V2=ucs2s_v2_mc3ypair
# caches built by the first run of this job (8078757)
F=dates_sst_mc_v2_m20.txt; ALL="$SST $MC $V2 ecmwf_rf"
for X in "" "--yearly-mean"; do
    T=sst3w; N=sstanom_vs_multicond_v2_ecmwf_rf; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
    for m in $ALL; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20 $X; done
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $ALL
    SER=("$SST:$T:UC-S2S SSTanom" "$MC:$T:UC-S2S multicond" "$V2:$T:UC-S2S V2" "ecmwf_rf:$T:ECMWF hindcast")
    for R in global NH tropics extratropics; do S=""; [ $R != global ] && S=_$R
        $PY -u plot_TCC_paper.py --block none --region $R --vars tp z500 u850 v850 --name ${N}_tp_z500_uv850$S --series "${SER[@]}"
        $PY -u plot_TCC_paper.py --block none --region $R --vars t2m --name ${N}_t2m$S --series "${SER[@]}"; done
done
echo "Job finished at: $(date) with exit code 0"
