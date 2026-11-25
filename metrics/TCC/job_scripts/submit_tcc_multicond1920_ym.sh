#!/bin/bash
# multicond pooled over Jan 3 - Feb 28 2019 (17 inits) + Jan 3 - Feb 3 2020 (10 inits), 20 members, vs UC-S2S V2 (20)
# paired init-by-init (2020: V2's nearest Mon/Thu init, <=1 day) -> stars; FuXi (20) and ECMWF hindcast (11)
# on their own inits in the same two windows (27 / 46).
#PBS -A UCHI0014
#PBS -N tcc_ym1920
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_multicond1920_ym.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_mcv2_1920.txt; T=mcv2_1920_m20_ym
F=dates_fuxi_mc1920win.txt; TF=fxmc1920win_m20_ym
E=dates_ecmwf_rf_mc1920win.txt; TE=ecmc1920win_m20_ym
for m in ucs2s_multicond1920 ucs2s_v2_mcpair; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag $T --n-members 20 --yearly-mean; done
$PY -u tcc_weekly.py --model fuxi --stage aggregate --dates-file $F --tag $TF --n-members 20 --yearly-mean
$PY -u tcc_weekly.py --model ecmwf_rf --stage aggregate --dates-file $E --tag $TE --n-members 20 --yearly-mean
$PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 20 --yearly-mean --models ucs2s_multicond1920 ucs2s_v2_mcpair
$PY -u tcc_bootstrap.py --block none --dates-file $F --tag $TF --n-members 20 --yearly-mean --models fuxi
$PY -u tcc_bootstrap.py --block none --dates-file $E --tag $TE --n-members 20 --yearly-mean --models ecmwf_rf
for R in "--t2m-land" "" "--vars t2m --region land"; do
    N=multicond_2019_2020_vs_v2_fuxi_ecmwf_rf_ym
    case "$R" in "") N=${N}_t2m_global;; --vars*) N=${N}_t2m_land;; esac
    $PY -u plot_TCC_paper.py --block none --name $N $R --series \
        "ucs2s_multicond1920:$T:UC-S2S multicond" "ucs2s_v2_mcpair:$T:UC-S2S V2" "fuxi:$TF:FuXi-S2S" "ecmwf_rf:$TE:ECMWF hindcast"
done
echo "Job finished at: $(date) with exit code 0"
