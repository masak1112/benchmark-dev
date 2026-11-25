#!/bin/bash
# Regenerated multicond pp12800 (3 years, 2026-11-02), 20 members, vs UC-S2S V2 (20), SI-S2S (20), NeuralGCM (20), FuXi (20),
# ECMWF hindcast (11). Only inits with all 20 members; all are ECMWF dates. V2 / SI-S2S / NGCM / FuXi use their nearest
# init (one-to-one, <=2 days; pairs_*_mc3y.txt; SI-S2S only inits with >=20 members). Common set: dates_mc3y_all_m20.txt
# (76 inits). Plain + --yearly-mean. Figures: TP+Z500+U850+V850 (2x2) and T2M alone. Set EXTRACT=1 to rebuild caches.
#PBS -A UCHI0014
#PBS -N tcc_mc3y
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_mc3y.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
P=/glade/derecho/scratch/bgong/pp_si_infer_multicond_pp12800/predictions
C=/glade/derecho/scratch/bgong/tcc_cache
MC=ucs2s_mc_pp12800f
if [ "${EXTRACT:-0}" = 1 ]; then
L=/glade/derecho/scratch/bgong/multicond_pp12800_links; rm -rf $L; mkdir -p $L
for d in $(cat dates_mc3y_ec_m20.txt); do for e in $(seq 0 19); do
    ln -s $P/pangu_plasim_pp_si_infer_multicond12800_${d:0:4}_24h_45step_${d}00_ens_$e.nc $L/pangu_plasim_pp_si_infer_multicond_pp12800f_24h_45step_${d}00_ens_$e.nc; done; done
echo "linked inits: $(ls $L | sed 's/.*_45step_\(........\).*/\1/' | sort -u | wc -l)"
$PY -u tcc_weekly.py --model $MC --stage extract --dates-file dates_mc3y_ec_m20.txt --n-members 20 --workers 8
fi
# SI-S2S partner inits not yet cached at 20 members
cut -d' ' -f2 pairs_si_s2s_mc3y.txt | sort -u | comm -23 - <(ls $C/si_s2s_m20 | sed 's/.npz//' | sort) > dates_si_s2s_mc3y_new.txt
echo "SI-S2S inits to extract: $(wc -l < dates_si_s2s_mc3y_new.txt)"
[ -s dates_si_s2s_mc3y_new.txt ] && $PY -u tcc_weekly.py --model si_s2s --stage extract --dates-file dates_si_s2s_mc3y_new.txt --n-members 20 --workers 8
F=dates_mc3y_all_m20.txt
ALL="$MC ucs2s_v2_mc3ypair si_s2s_mc3ypair ngcm_mc3ypair fuxi_mc3ypair ecmwf_rf"
for X in "" "--yearly-mean"; do
    T=mc3yfx; N=multicond3y_vs_v2_si_ngcm_fuxi_ecmwf_rf; [ -n "$X" ] && T=${T}_ym && N=${N}_ym
    for m in $ALL; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20 $X; done
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models $ALL
    SER=("$MC:$T:UC-S2S multicond" "ucs2s_v2_mc3ypair:$T:UC-S2S V2" "si_s2s_mc3ypair:$T:SI-S2S" "ngcm_mc3ypair:$T:NeuralGCM" "fuxi_mc3ypair:$T:FuXi-S2S" "ecmwf_rf:$T:ECMWF hindcast")
    for R in global NH tropics; do S=""; [ $R != global ] && S=_$R
        $PY -u plot_TCC_paper.py --block none --region $R --vars tp z500 u850 v850 --name ${N}_tp_z500_uv850$S --series "${SER[@]}"
        $PY -u plot_TCC_paper.py --block none --region $R --vars t2m --name ${N}_t2m$S --series "${SER[@]}"; done
done
echo "Job finished at: $(date) with exit code 0"
