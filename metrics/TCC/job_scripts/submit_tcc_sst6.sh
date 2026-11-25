#!/bin/bash
# (A) SSTanom / multicond / V2 / SI-S2S / NeuralGCM / ECMWF hindcast on the 73 inits shared by all six (dates_sst6_m20.txt;
#     SI-S2S limited to inits with >=20 members). (B) same without SI-S2S on the 128-init set (dates_sst_mc_v2_m20.txt).
# Partners paired by nearest init (pairs_*_mc3y.txt). No FuXi. Stars vs SSTanom. Plain + --yearly-mean.
# Regions global / NH / tropics / extratropics; T2M also over interior land.
#PBS -A UCHI0014
#PBS -N tcc_sst6
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst6.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/u/apps/opt/conda/envs/npl-2025b/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
SST=ucs2s_sstanom3y; MC=ucs2s_mc_pp12800f; V2=ucs2s_v2_mc3ypair; SI=si_s2s_mc3ypair; NG=ngcm_mc3ypair
run() {  # dates tag name models...
    local F=$1 T0=$2 N0=$3; shift 3; local ALL="$*"
    for X in "" "--yearly-mean"; do
        T=$T0; N=$N0; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
        for m in $ALL; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20 $X; done
        $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $ALL
        SER=()
        for m in $ALL; do case $m in
            $SST) SER+=("$m:$T:UC-S2S SSTanom");; $MC) SER+=("$m:$T:UC-S2S multicond");; $V2) SER+=("$m:$T:UC-S2S V2");;
            $SI) SER+=("$m:$T:SI-S2S");; $NG) SER+=("$m:$T:NeuralGCM");; ecmwf_rf) SER+=("$m:$T:ECMWF hindcast");; esac; done
        for R in global NH tropics extratropics; do S=""; [ $R != global ] && S=_$R
            $PY -u plot_TCC_paper.py --block none --region $R --vars tp z500 u850 v850 --name ${N}_tp_z500_uv850$S --series "${SER[@]}"
            $PY -u plot_TCC_paper.py --block none --region $R --vars t2m --name ${N}_t2m$S --series "${SER[@]}"; done
        $PY -u plot_TCC_paper.py --block none --region land_interior --vars t2m --name ${N}_t2m_land_interior --series "${SER[@]}"
    done
}
run dates_sst6_m20.txt sst6 sstanom_vs_mc_v2_si_ngcm_ecmwf_rf $SST $MC $V2 $SI $NG ecmwf_rf
run dates_sst_mc_v2_m20.txt sst5n sstanom_vs_mc_v2_ngcm_ecmwf_rf_128 $SST $MC $V2 $NG ecmwf_rf
echo "Job finished at: $(date) with exit code 0"
