#!/bin/bash
# multicond v4 (2026-10-30 data): rebuild date lists from complete inits (20 full-size members), extract new inits,
# relink ucs2s_multicond_all, re-pair V2. Date sets: D = mc & FuXi, E = ECMWF own inits in mc windows,
# X = mc & ECMWF, V = X & V2-paired (all + per year).
#PBS -A UCHI0014
#PBS -N tcc_mc4p
#PBS -q develop
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_mc_v4_prep.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
C=/glade/derecho/scratch/bgong/tcc_cache
P=/glade/derecho/scratch/bgong/pp_si_infer_multicond_2019/predictions
for y in 2019 2020 2021; do
    ls -l $P/pangu_plasim_pp_si_infer_multicond_${y}_24h_45step_${y}*_ens_*.nc | awk '$5==1144652886 {split($9,a,"_45step_"); c[substr(a[2],1,8)]++} END{for(k in c) if(c[k]==20) print k}' | sort > dates_mc${y}_v4.txt
    k=ucs2s_multicond$y; [ $y = 2019 ] && k=ucs2s_multicond2019n
    $PY -u tcc_weekly.py --model $k --stage extract --dates-file dates_mc${y}_v4.txt --n-members 20 --workers 8
done
rm -rf $C/ucs2s_multicond_all; mkdir -p $C/ucs2s_multicond_all
for y in 2019 2020 2021; do k=ucs2s_multicond$y; [ $y = 2019 ] && k=ucs2s_multicond2019n
    for d in $(cat dates_mc${y}_v4.txt); do ln -s $C/$k/$d.npz $C/ucs2s_multicond_all/; done; done
cat dates_mc20{19,20,21}_v4.txt > dates_mc_all_v4.txt
ls $C/fuxi_m20 | sed 's/.npz//' | sort > /tmp/fx_$$.txt
comm -12 dates_mc_all_v4.txt /tmp/fx_$$.txt > dates_mcfx_v4.txt
: > dates_ecmwf_rf_mcwin_v4.txt
for y in 2019 2020 2021; do lo=$(head -1 dates_mc${y}_v4.txt); hi=$(tail -1 dates_mc${y}_v4.txt)
    ls $C/ecmwf_rf | sed 's/.npz//' | awk -v lo=$lo -v hi=$hi '$1>=lo && $1<=hi' >> dates_ecmwf_rf_mcwin_v4.txt; done
comm -12 dates_mc_all_v4.txt <(sort dates_ecmwf_rf_mcwin_v4.txt) > dates_mcec_v4.txt
comm -12 dates_mcfx_v4.txt dates_mcec_v4.txt > dates_mcfxec_v4.txt
$PY make_pairs_v2.py dates_mc_all_v4.txt pairs_v2_mcall.txt
comm -12 dates_mcec_v4.txt <(cut -d' ' -f1 pairs_v2_mcall.txt | sort) > dates_mcecv2_v4_all.txt
for y in 2019 2020 2021; do grep "^$y" dates_mcec_v4.txt > dates_mcec_v4_$y.txt || true
    grep "^$y" dates_mcecv2_v4_all.txt > dates_mcecv2_v4_$y.txt || true; done
wc -l dates_mc20{19,20,21}_v4.txt dates_mc_all_v4.txt dates_mcfx_v4.txt dates_ecmwf_rf_mcwin_v4.txt dates_mcec_v4*.txt dates_mcfxec_v4.txt dates_mcecv2_v4_*.txt
echo "Job finished at: $(date) with exit code 0"
