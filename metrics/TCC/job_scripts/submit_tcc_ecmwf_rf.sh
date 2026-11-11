#!/bin/bash
# TCC for the ECMWF 2025-model HINDCASTS (11 members, hindcast years 2019-2021) and comparison with the other models
# (20 members). Uses whatever hindcast files exist when the job starts; rerun after the download completes.
#PBS -A UCHI0014
#PBS -N tcc_ecrf
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ecmwf_rf.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export TMPDIR=/glade/derecho/scratch/$USER/tmp
export OMP_NUM_THREADS=32
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
# date lists from the files present now
$PY -c "import tcc_weekly as t; open('dates_ecmwf_rf_2019_2021.txt','w').write('\n'.join(t.list_dates('ecmwf_rf'))+'\n')"
comm -12 dates_ucs2sv2_2019_2021.txt <(sort dates_ecmwf_rf_2019_2021.txt) > common_dates_ucs2sv2_ecmwf_rf.txt
echo "hindcast inits: $(wc -l < dates_ecmwf_rf_2019_2021.txt)   shared with UC-S2S V2: $(wc -l < common_dates_ucs2sv2_ecmwf_rf.txt)"
D=dates_ecmwf_rf_2019_2021.txt; C=common_dates_ucs2sv2_ecmwf_rf.txt
# 11 members (native); --n-members 20 elsewhere leaves 11-member models at their full ensemble
$PY -u tcc_weekly.py --model ecmwf_rf --stage extract --workers 32 --dates-file $D --tag ecrf1921 --n-members 11
for suf in "" _mm; do
    O="--n-members 20"; [ "$suf" = "_mm" ] && O="$O --monthly-mean"
    $PY -u tcc_weekly.py --model ecmwf_rf --stage aggregate --dates-file $D --tag ecrf1921_m20$suf $O
    for m in ucs2s_v2 ecmwf_rf; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $C --tag ucecrf_m20$suf $O; done
    $PY -u tcc_bootstrap.py --models ecmwf_rf          --dates-file $D --tag ecrf1921_m20$suf --block month $O
    $PY -u tcc_bootstrap.py --models ucs2s_v2 ecmwf_rf --dates-file $C --tag ucecrf_m20$suf   --block month $O
    $PY -u plot_TCC_paper.py --own-record --suffix _m20$suf --models ucs2s_v2 fuxi si_s2s ngcm ecmwf_rf
    $PY -u plot_TCC_paper.py --own-record --suffix _m20$suf --models ucs2s_v2 fuxi ngcm ecmwf_rf
    # same with FuXi's full 51-member ensemble next to its 20-member bar
    $PY -u plot_TCC_paper.py --own-record --suffix _m20$suf --models ucs2s_v2 fuxi si_s2s ngcm ecmwf_rf --fuxi-m51
    $PY -u plot_TCC_paper.py --own-record --suffix _m20$suf --models ucs2s_v2 fuxi ngcm ecmwf_rf --fuxi-m51
    $PY -u plot_TCC_paper.py --models ucs2s_v2 ecmwf_rf --tag ucecrf_m20$suf
done
echo "Job finished at: $(date) with exit code 0"
