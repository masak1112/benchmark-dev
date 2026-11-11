#!/bin/bash
# UC-S2S V2-fbcorr (pp_si_fbcorr_allvars, 20 members): extract, TCC on its 2019-2021 inits and on the dates shared
# with UC-S2S V2 (as is and month mean removed), bootstraps, paper figures
#PBS -A UCHI0014
#PBS -N tcc_fbcorr
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_fbcorr.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export OMP_NUM_THREADS=32
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
$PY -u tcc_weekly.py --model ucs2s_v2_fbcorr --stage extract --workers 32 --dates-file dates_ucs2sv2_fbcorr_2019_2021.txt --tag fb1921 --n-members 20
# inits that really have 20 members (cached) -> own record and the set shared with UC-S2S V2
ls /glade/derecho/scratch/bgong/tcc_cache/ucs2s_v2_fbcorr | grep -oE '^[0-9]{8}' | sort | grep -E '^20(19|20|21)' > dates_ucs2sv2_fbcorr_2019_2021_m20.txt
comm -12 dates_ucs2sv2_fbcorr_2019_2021_m20.txt dates_ucs2sv2_2019_2021.txt > common_dates_ucs2sv2_fbcorr.txt
echo "fbcorr inits with 20 members: $(wc -l < dates_ucs2sv2_fbcorr_2019_2021_m20.txt)   shared with UC-S2S V2: $(wc -l < common_dates_ucs2sv2_fbcorr.txt)"
D=dates_ucs2sv2_fbcorr_2019_2021_m20.txt; C=common_dates_ucs2sv2_fbcorr.txt
for suf in "" _mm; do
    O="--n-members 20"; [ "$suf" = "_mm" ] && O="$O --monthly-mean"
    $PY -u tcc_weekly.py --model ucs2s_v2_fbcorr --stage aggregate --dates-file $D --tag fb1921_m20$suf $O
    for m in ucs2s_v2 ucs2s_v2_fbcorr; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $C --tag ucfb_m20$suf $O; done
    $PY -u tcc_bootstrap.py --models ucs2s_v2_fbcorr          --dates-file $D --tag fb1921_m20$suf --block month $O
    $PY -u tcc_bootstrap.py --models ucs2s_v2 ucs2s_v2_fbcorr --dates-file $C --tag ucfb_m20$suf   --block month $O
    $PY -u plot_TCC_paper.py --models ucs2s_v2 ucs2s_v2_fbcorr --tag ucfb_m20$suf
    $PY -u plot_TCC_paper.py --own-record --suffix _m20$suf --models ucs2s_v2 ucs2s_v2_fbcorr fuxi ngcm ecmwf_rf
done
echo "Job finished at: $(date) with exit code 0"
