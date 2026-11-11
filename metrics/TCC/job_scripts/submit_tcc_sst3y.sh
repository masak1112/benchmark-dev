#!/bin/bash
# UC-S2S SSTanom 3-year rerun (pp_si_infer_sstanom/predictions, 2026-11-04), 20 members, added to the 6-model
# comparison (multicond, V2, SI-S2S, NeuralGCM, FuXi, ECMWF hindcast; partners paired by nearest init, pairs_*_mc3y.txt)
# on the inits shared by all seven: dates_sst3y_all7_m20.txt. Stars vs SSTanom (first series). Plain + --yearly-mean.
#PBS -A UCHI0014
#PBS -N tcc_sst3y
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst3y.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
# .venv broke 2026-11-04 (/usr/bin/python3.10 removed from derecho); use NCAR npl-2025b
PY=/glade/u/apps/opt/conda/envs/npl-2025b/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
P=/glade/derecho/scratch/bgong/pp_si_infer_sstanom/predictions
L=/glade/derecho/scratch/bgong/sstanom3y_links; rm -rf $L; mkdir -p $L
for d in $(cat dates_sst3y_m20.txt); do for e in $(seq 0 19); do
    ln -s $P/pangu_plasim_pp_si_infer_sstanom_${d:0:4}_24h_45step_${d}00_ens_$e.nc $L/pangu_plasim_pp_si_infer_sstanom3y_24h_45step_${d}00_ens_$e.nc; done; done
echo "linked inits: $(ls $L | sed 's/.*_45step_\(........\).*/\1/' | sort -u | wc -l)"
SST=ucs2s_sstanom3y; MC=ucs2s_mc_pp12800f
$PY -u tcc_weekly.py --model $SST --stage extract --dates-file dates_sst3y_m20.txt --n-members 20 --workers 8
F=dates_sst3y_all7_m20.txt
ALL="$SST $MC ucs2s_v2_mc3ypair si_s2s_mc3ypair ngcm_mc3ypair fuxi_mc3ypair ecmwf_rf"
for X in "" "--yearly-mean"; do
    T=sst3y7; N=sstanom3y_vs_mc_v2_si_ngcm_fuxi_ecmwf_rf; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
    for m in $ALL; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20 $X; done
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $ALL
    SER=("$SST:$T:UC-S2S SSTanom" "$MC:$T:UC-S2S multicond" "ucs2s_v2_mc3ypair:$T:UC-S2S V2" "si_s2s_mc3ypair:$T:SI-S2S" "ngcm_mc3ypair:$T:NeuralGCM" "fuxi_mc3ypair:$T:FuXi-S2S" "ecmwf_rf:$T:ECMWF hindcast")
    for R in global NH tropics; do S=""; [ $R != global ] && S=_$R
        $PY -u plot_TCC_paper.py --block none --region $R --vars tp z500 u850 v850 --name ${N}_tp_z500_uv850$S --series "${SER[@]}"
        $PY -u plot_TCC_paper.py --block none --region $R --vars t2m --name ${N}_t2m$S --series "${SER[@]}"; done
done
echo "Job finished at: $(date) with exit code 0"
