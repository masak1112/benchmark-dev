#!/bin/bash
# New 2019 multicond run (FuXi dates, Jan 3 - May 23 2019, 41 inits) + 2020 run (Jan 3 - Feb 3, 10) = 51 inits, 20 members.
# A: multicond vs FuXi (same dates) and V2 (nearest Mon/Thu, <=1 day) paired on 51; ECMWF hindcast on its own inits in the windows.
# B: all four paired on the inits ECMWF shares exactly. Each with plain centered TCC and with --yearly-mean (_ym).
#PBS -A UCHI0014
#PBS -N tcc_mc1920n
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=100GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_multicond1920n.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
C=/glade/derecho/scratch/bgong/tcc_cache
$PY -u tcc_weekly.py --model ucs2s_multicond2019n --stage extract --dates-file dates_mc2019n.txt --n-members 20 --workers 8
mkdir -p $C/ucs2s_multicond1920n
for d in $(cat dates_mc2019n.txt); do ln -sf $C/ucs2s_multicond2019n/$d.npz $C/ucs2s_multicond1920n/; done
for d in $(cat dates_ucs2s_multicond2020.txt); do ln -sf $C/ucs2s_multicond2020/$d.npz $C/ucs2s_multicond1920n/; done
$PY -u check_year_bias.py
D=dates_mc1920n.txt; E=dates_ecmwf_rf_mc1920nwin.txt; X=dates_mc1920n_ec_common.txt
MC=ucs2s_multicond1920n; V2=ucs2s_v2_mcpair2
for YM in "" "--yearly-mean"; do
    S=""; [ -n "$YM" ] && S=_ym
    T=mcfxv2_1920n_m20$S; TE=ecmc1920nwin_m20$S; TX=mcall_1920n_ec_m20$S
    for m in $MC $V2 fuxi; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag $T --n-members 20 $YM; done
    $PY -u tcc_weekly.py --model ecmwf_rf --stage aggregate --dates-file $E --tag $TE --n-members 20 $YM
    for m in $MC $V2 fuxi ecmwf_rf; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $X --tag $TX --n-members 20 $YM; done
    $PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 20 $YM --models $MC $V2 fuxi
    $PY -u tcc_bootstrap.py --block none --dates-file $E --tag $TE --n-members 20 $YM --models ecmwf_rf
    $PY -u tcc_bootstrap.py --block none --dates-file $X --tag $TX --n-members 20 $YM --models $MC $V2 fuxi ecmwf_rf
    for R in "--t2m-land" "" "--vars t2m --region land"; do
        case "$R" in "") U=_t2m_global;; --vars*) U=_t2m_land;; *) U=;; esac
        $PY -u plot_TCC_paper.py --block none --name multicond_1920n_vs_v2_fuxi_ecmwf_rf$S$U $R --series \
            "$MC:$T:UC-S2S multicond" "$V2:$T:UC-S2S V2" "fuxi:$T:FuXi-S2S" "ecmwf_rf:$TE:ECMWF hindcast"
        $PY -u plot_TCC_paper.py --block none --name multicond_1920n_vs_all_common_ec$S$U $R --series \
            "$MC:$TX:UC-S2S multicond" "$V2:$TX:UC-S2S V2" "fuxi:$TX:FuXi-S2S" "ecmwf_rf:$TX:ECMWF hindcast"
    done
done
echo "Job finished at: $(date) with exit code 0"
