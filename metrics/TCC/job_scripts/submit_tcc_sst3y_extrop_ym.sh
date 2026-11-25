#!/bin/bash
# Re-bootstrap the SSTanom 7-model comparison (tags sst3y7, sst3y7_ym; 48 inits) so the pickles include the new
# 'extratropics' region (|lat| >= 30), then plot tropics + extratropics without FuXi. Same seed -> other regions unchanged.
#PBS -A UCHI0014
#PBS -N tcc_sstx_ym
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst3y_extrop_ym.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/u/apps/opt/conda/envs/npl-2025b/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
F=dates_sst3y_all7_m20.txt; SST=ucs2s_sstanom3y; MC=ucs2s_mc_pp12800f
ALL="$SST $MC ucs2s_v2_mc3ypair si_s2s_mc3ypair ngcm_mc3ypair fuxi_mc3ypair ecmwf_rf"
for X in "--yearly-mean"; do
    T=sst3y7; N=sstanom3y_vs_mc_v2_si_ngcm_ecmwf_rf; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $ALL
    SER=("$SST:$T:UC-S2S SSTanom" "$MC:$T:UC-S2S multicond" "ucs2s_v2_mc3ypair:$T:UC-S2S V2" "si_s2s_mc3ypair:$T:SI-S2S" "ngcm_mc3ypair:$T:NeuralGCM" "ecmwf_rf:$T:ECMWF hindcast")
    for R in tropics extratropics; do
        $PY -u plot_TCC_paper.py --block none --region $R --vars tp z500 u850 v850 --name ${N}_tp_z500_uv850_$R --series "${SER[@]}"
        $PY -u plot_TCC_paper.py --block none --region $R --vars t2m --name ${N}_t2m_$R --series "${SER[@]}"; done
done
echo "Job finished at: $(date) with exit code 0"
