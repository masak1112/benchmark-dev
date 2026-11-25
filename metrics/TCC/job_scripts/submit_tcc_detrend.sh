#!/bin/bash
# Detrended TCC (FuXi-S2S paper method, observation side): 2002-2016 ERA5 linear-trend climatology,
# then aggregates, bootstraps and bar plots for the main comparisons (tags *_dt)
#PBS -A UCHI0014
#PBS -N tcc_detrend
#PBS -q develop
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_detrend.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
python -u compute_trend_clim.py

M="--n-members 20 --detrend"
agg()  { python -u tcc_weekly.py --model $1 --stage aggregate --dates-file $2 --tag $3 $4; }
boot() { d=$1; t=$2; o=$3; shift 3; python -u tcc_bootstrap.py --models "$@" --dates-file $d --tag $t --block month $o; }

# FuXi-S2S paper setup: FuXi, 51 members, 2017-2021
agg fuxi dates_fuxi_2017_2021.txt y2017_2021_dt --detrend
boot dates_fuxi_2017_2021.txt y2017_2021_dt --detrend fuxi

# 2019-2021 own records, 20 members
agg ucs2s_v2 dates_ucs2sv2_2019_2021.txt ucall1921_m20_dt "$M"
agg fuxi     dates_fuxi_2019_2021.txt    fx1921_m20_dt    "$M"
agg ngcm     dates_ngcm_2019_2021.txt    y2019_2021_m20_dt "$M"
for m in ucs2s_v2 si_s2s; do agg $m common_dates_ucs2sv2_sis2s_2019_2021.txt si1921_m20_dt "$M"; done
# same-date sets, 20 members
for m in ucs2s_v2 fuxi;             do agg $m common_dates_ucs2sv2_fuxi.txt           common86_m20_dt   "$M"; done
for m in ucs2s_v2 fuxi ngcm;        do agg $m common_dates_ucs2sv2_fuxi_ngcm.txt      common72_m20_dt   "$M"; done
for m in ucs2s_v2 fuxi si_s2s ngcm; do agg $m common_dates_ucs2sv2_fuxi_sis2s_ngcm.txt common39x4_m20_dt "$M"; done

boot dates_ucs2sv2_2019_2021.txt ucall1921_m20_dt "$M" ucs2s_v2
boot dates_fuxi_2019_2021.txt    fx1921_m20_dt    "$M" fuxi
boot dates_ngcm_2019_2021.txt    y2019_2021_m20_dt "$M" ngcm
boot common_dates_ucs2sv2_sis2s_2019_2021.txt si1921_m20_dt "$M" ucs2s_v2 si_s2s
boot common_dates_ucs2sv2_fuxi.txt            common86_m20_dt   "$M" ucs2s_v2 fuxi
boot common_dates_ucs2sv2_fuxi_ngcm.txt       common72_m20_dt   "$M" ucs2s_v2 fuxi ngcm
boot common_dates_ucs2sv2_fuxi_sis2s_ngcm.txt common39x4_m20_dt "$M" ucs2s_v2 fuxi si_s2s ngcm

for m in "ucs2s_v2 fuxi si_s2s ngcm" "ucs2s_v2 fuxi ngcm" "ucs2s_v2 si_s2s ngcm"; do
    python -u plot_TCC_bars_own_record.py --models $m --mem-suffix _m20_dt --period 2019-2021
done
python -u plot_TCC.py --models ucs2s_v2 fuxi             --tag common86   --mem-suffix _m20_dt --bars-only --no-own-record
python -u plot_TCC.py --models ucs2s_v2 fuxi ngcm        --tag common72   --mem-suffix _m20_dt --bars-only --no-own-record
python -u plot_TCC.py --models ucs2s_v2 fuxi si_s2s ngcm --tag common39x4 --mem-suffix _m20_dt --bars-only --no-own-record
python -u plot_TCC.py --models ucs2s_v2 si_s2s           --tag si1921     --mem-suffix _m20_dt --bars-only --no-own-record
python -u plot_TCC.py --models ucs2s_v2 fuxi             --tag common86   --mem-suffix _m20_dt
echo "Job finished at: $(date) with exit code 0"
