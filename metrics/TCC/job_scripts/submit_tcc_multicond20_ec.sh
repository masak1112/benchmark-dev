#!/bin/bash
# UC-S2S multicond (20) vs UC-S2S V2 (20) on their common inits (paired, stars) + ECMWF hindcast (11 members)
# on all its own inits in the same window. Bootstraps need >8 GB -> compute node.
#PBS -A UCHI0014
#PBS -N tcc_mc20ec
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_multicond20_ec.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
E=dates_ecmwf_rf_mc20win.txt; TE=ecmc20win_m20; T=v2mc20_m20
$PY -u tcc_weekly.py --model ecmwf_rf --stage aggregate --dates-file $E --tag $TE --n-members 20
$PY -u tcc_bootstrap.py --block none --dates-file $E --tag $TE --n-members 20 --models ecmwf_rf
for R in "--t2m-land" "--vars t2m --region land"; do
    N=multicond20_vs_v2_ecmwf_rf_201901_02; [ "$R" != "--t2m-land" ] && N=${N}_t2m_land
    $PY -u plot_TCC_paper.py --block none --name $N $R --series \
        "ucs2s_v2:$T:UC-S2S V2" "ucs2s_multicond20:$T:UC-S2S multicond" "ecmwf_rf:$TE:ECMWF hindcast"
done
echo "Job finished at: $(date) with exit code 0"
