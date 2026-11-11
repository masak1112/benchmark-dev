#!/bin/bash
# multicond rerun (2019 + 2020 + 2021, FuXi init dates, 20 members) vs FuXi-S2S (20, same dates -> paired) and
# ECMWF hindcast (11): (A) ECMWF on its own inits in the same months; (B) all three paired on the inits ECMWF shares.
# Plain centered TCC and --yearly-mean (_ym). Date files are rebuilt here from complete inits (20 full-size members).
#PBS -A UCHI0014
#PBS -N tcc_mcall
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_multicond_all.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
C=/glade/derecho/scratch/bgong/tcc_cache
P=/glade/derecho/scratch/bgong/pp_si_infer_multicond_2019/predictions
for y in 2019 2020 2021; do
    ls -l $P/pangu_plasim_pp_si_infer_multicond_${y}_24h_45step_${y}*_ens_*.nc | awk '$5==1144652886 {split($9,a,"_45step_"); c[substr(a[2],1,8)]++} END{for(k in c) if(c[k]==20) print k}' | sort > dates_mc${y}_v3.txt
    echo "$y: $(wc -l < dates_mc${y}_v3.txt) complete inits"
done
$PY -u tcc_weekly.py --model ucs2s_multicond2019n --stage extract --dates-file dates_mc2019_v3.txt --n-members 20 --workers 8
$PY -u tcc_weekly.py --model ucs2s_multicond2020 --stage extract --dates-file dates_mc2020_v3.txt --n-members 20 --workers 8
$PY -u tcc_weekly.py --model ucs2s_multicond2021 --stage extract --dates-file dates_mc2021_v3.txt --n-members 20 --workers 8
rm -rf $C/ucs2s_multicond_all; mkdir -p $C/ucs2s_multicond_all
for y in 2019 2020 2021; do k=ucs2s_multicond$y; [ $y = 2019 ] && k=ucs2s_multicond2019n
    for d in $(cat dates_mc${y}_v3.txt); do ln -s $C/$k/$d.npz $C/ucs2s_multicond_all/; done; done
cat dates_mc20{19,20,21}_v3.txt > dates_mc_all_v3.txt
# ECMWF: own inits in the months multicond covers in each year; and the exact shared inits
: > dates_ecmwf_rf_mcallwin.txt
for y in 2019 2020 2021; do lo=$(head -1 dates_mc${y}_v3.txt); hi=$(tail -1 dates_mc${y}_v3.txt)
    ls $C/ecmwf_rf | sed 's/.npz//' | awk -v lo=$lo -v hi=$hi '$1>=lo && $1<=hi' >> dates_ecmwf_rf_mcallwin.txt; done
comm -12 dates_mc_all_v3.txt <(sort dates_ecmwf_rf_mcallwin.txt) > dates_mcall_ec_common.txt
wc -l dates_mc_all_v3.txt dates_ecmwf_rf_mcallwin.txt dates_mcall_ec_common.txt
MC=ucs2s_multicond_all; D=dates_mc_all_v3.txt; E=dates_ecmwf_rf_mcallwin.txt; X=dates_mcall_ec_common.txt
for YM in "" "--yearly-mean"; do
    S=""; [ -n "$YM" ] && S=_ym
    T=mcfx_all_m20$S; TE=ec_mcallwin_m20$S; TX=mcfxec_all_m20$S
    for m in $MC fuxi; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag $T --n-members 20 $YM; done
    $PY -u tcc_weekly.py --model ecmwf_rf --stage aggregate --dates-file $E --tag $TE --n-members 20 $YM
    for m in $MC fuxi ecmwf_rf; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $X --tag $TX --n-members 20 $YM; done
    $PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 20 $YM --models $MC fuxi
    $PY -u tcc_bootstrap.py --block none --dates-file $E --tag $TE --n-members 20 $YM --models ecmwf_rf
    $PY -u tcc_bootstrap.py --block none --dates-file $X --tag $TX --n-members 20 $YM --models $MC fuxi ecmwf_rf
    for R in "--t2m-land" "" "--vars t2m --region land"; do
        case "$R" in "") U=_t2m_global;; --vars*) U=_t2m_land;; *) U=;; esac
        $PY -u plot_TCC_paper.py --block none --name multicond_all_vs_fuxi_ecmwf_rf$S$U $R --series \
            "$MC:$T:UC-S2S multicond" "fuxi:$T:FuXi-S2S" "ecmwf_rf:$TE:ECMWF hindcast"
        $PY -u plot_TCC_paper.py --block none --name multicond_all_vs_fuxi_ecmwf_rf_common$S$U $R --series \
            "$MC:$TX:UC-S2S multicond" "fuxi:$TX:FuXi-S2S" "ecmwf_rf:$TX:ECMWF hindcast"
    done
done
echo "Job finished at: $(date) with exit code 0"
