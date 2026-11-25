#!/bin/bash
# Add UC-S2S SSTanom member 0 only (ucs2s_sstanom3y_e0) to the SSTanom comparison (48 inits, dates_sst3y_all7_m20.txt).
# Re-bootstrap with all 8 models (tags sst3y7, sst3y7_ym; same seed); plots without FuXi; global/NH/tropics/extratropics.
#PBS -A UCHI0014
#PBS -N tcc_sste0
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst3y_e0.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/u/apps/opt/conda/envs/npl-2025b/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
F=dates_sst3y_all7_m20.txt; SST=ucs2s_sstanom3y; E0=ucs2s_sstanom3y_e0; MC=ucs2s_mc_pp12800f
$PY -u tcc_weekly.py --model $E0 --stage extract --dates-file $F --n-members 1 --workers 8
ALL="$SST $E0 $MC ucs2s_v2_mc3ypair si_s2s_mc3ypair ngcm_mc3ypair fuxi_mc3ypair ecmwf_rf"
for X in "" "--yearly-mean"; do
    T=sst3y7; N=sstanom3y_e0_vs_mc_v2_si_ngcm_ecmwf_rf; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
    $PY -u tcc_weekly.py --model $E0 --stage aggregate --dates-file $F --tag $T --n-members 20 $X
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $ALL
    SER=("$SST:$T:UC-S2S SSTanom" "$E0:$T:UC-S2S SSTanom (ens_0)" "$MC:$T:UC-S2S multicond" "ucs2s_v2_mc3ypair:$T:UC-S2S V2" "si_s2s_mc3ypair:$T:SI-S2S" "ngcm_mc3ypair:$T:NeuralGCM" "ecmwf_rf:$T:ECMWF hindcast")
    for R in global NH tropics extratropics; do S=""; [ $R != global ] && S=_$R
        $PY -u plot_TCC_paper.py --block none --region $R --vars tp z500 u850 v850 --name ${N}_tp_z500_uv850$S --series "${SER[@]}"
        $PY -u plot_TCC_paper.py --block none --region $R --vars t2m --name ${N}_t2m$S --series "${SER[@]}"; done
done
echo "Job finished at: $(date) with exit code 0"
