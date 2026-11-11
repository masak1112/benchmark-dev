#!/bin/bash
# Re-bootstrap SSTanom / multicond / V2 / ECMWF (tags sst3w, sst3w_ym; 128 inits; same seed) so the pickles include the
# 'land_interior' region (non-coastal land, ~64% of land area), then plot T2M over interior land. Stars vs SSTanom.
#PBS -A UCHI0014
#PBS -N tcc_landin
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst_landin_plain.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/u/apps/opt/conda/envs/npl-2025b/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
F=dates_sst_mc_v2_m20.txt; SST=ucs2s_sstanom3y; MC=ucs2s_mc_pp12800f; V2=ucs2s_v2_mc3ypair
ALL="$SST $MC $V2 ecmwf_rf"
# the _ym bootstrap of job 8078952 started after land_interior was added -> only the plain one is redone here
for X in ""; do   # plain only; the _ym land_interior plot is made after job 8078952
    T=sst3w; N=sstanom_vs_multicond_v2_ecmwf_rf; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
    [ -z "$X" ] && $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $ALL
    SER=("$SST:$T:UC-S2S SSTanom" "$MC:$T:UC-S2S multicond" "$V2:$T:UC-S2S V2" "ecmwf_rf:$T:ECMWF hindcast")
    $PY -u plot_TCC_paper.py --block none --region land_interior --vars t2m --name ${N}_t2m_land_interior --series "${SER[@]}"
done
echo "Job finished at: $(date) with exit code 0"
