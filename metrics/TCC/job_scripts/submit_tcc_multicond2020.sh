#!/bin/bash
# 2020 multicond (20 members, FuXi init dates Jan 3 - Feb 3 2020) vs FuXi (20) on the same 10 inits (paired, stars),
# UC-S2S V2 (20) on its 10 nearest Mon/Thu inits (<=1 day apart), ECMWF hindcast (11) on its own inits in the window.
#PBS -A UCHI0014
#PBS -N tcc_mc2020
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_multicond2020.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_mcfx2020.txt; T=mcfx2020_m20
V=dates_ucs2sv2_2020_mc.txt; TV=v2y2020mc_m20
E=dates_ecmwf_rf_mc2020win.txt; TE=ecmc2020win_m20
for m in ucs2s_multicond2020 fuxi; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag $T --n-members 20; done
$PY -u tcc_weekly.py --model ucs2s_v2 --stage aggregate --dates-file $V --tag $TV --n-members 20
$PY -u tcc_weekly.py --model ecmwf_rf --stage aggregate --dates-file $E --tag $TE --n-members 20
$PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 20 --models ucs2s_multicond2020 fuxi
$PY -u tcc_bootstrap.py --block none --dates-file $V --tag $TV --n-members 20 --models ucs2s_v2
$PY -u tcc_bootstrap.py --block none --dates-file $E --tag $TE --n-members 20 --models ecmwf_rf
for R in "--t2m-land" "" "--vars t2m --region land"; do
    N=multicond2020_vs_v2_fuxi_ecmwf_rf_202001_02
    case "$R" in "") N=${N}_t2m_global;; --vars*) N=${N}_t2m_land;; esac
    $PY -u plot_TCC_paper.py --block none --name $N $R --series \
        "ucs2s_multicond2020:$T:UC-S2S multicond" "ucs2s_v2:$TV:UC-S2S V2" "fuxi:$T:FuXi-S2S" "ecmwf_rf:$TE:ECMWF hindcast"
done
echo "Job finished at: $(date) with exit code 0"
