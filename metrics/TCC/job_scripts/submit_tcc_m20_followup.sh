#!/bin/bash
# 20-member versions of every TCC comparison: aggregates, bootstraps, plots (tags get _m20)
#PBS -A UCHI0014
#PBS -N tcc_m20
#PBS -q develop
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_m20_followup.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
M="--n-members 20"
agg() { python -u tcc_weekly.py --model $1 --stage aggregate --dates-file $2 --tag ${3}_m20 $M; }
boot() { d=$1; t=$2; shift 2; python -u tcc_bootstrap.py --models "$@" --dates-file $d --tag ${t}_m20 --block month $M; }

# common-date sets
for m in ucs2s_v2 fuxi;        do agg $m common_dates_ucs2sv2_fuxi.txt        common86; done
for m in ucs2s_v2 fuxi ngcm;   do agg $m common_dates_ucs2sv2_fuxi_ngcm.txt   common72; done
for m in ucs2s_v2 fuxi si_s2s; do agg $m common_dates_ucs2sv2_fuxi_sis2s.txt  common39; done
for m in ucs2s_v2 si_s2s;      do agg $m common_dates_ucs2sv2_sis2s.txt       si104;    done
# each model's own record
agg ucs2s_v2 dates_ucs2sv2_cached151.txt cached151
agg fuxi     dates_fuxi_2017_2021.txt    y2017_2021
agg ngcm     dates_ngcm_2019_2021.txt    y2019_2021

boot common_dates_ucs2sv2_fuxi.txt       common86  ucs2s_v2 fuxi
boot common_dates_ucs2sv2_fuxi_ngcm.txt  common72  ucs2s_v2 fuxi ngcm
boot common_dates_ucs2sv2_fuxi_sis2s.txt common39  ucs2s_v2 fuxi si_s2s
boot common_dates_ucs2sv2_sis2s.txt      si104     ucs2s_v2 si_s2s
boot dates_ucs2sv2_cached151.txt         cached151 ucs2s_v2
boot dates_fuxi_2017_2021.txt            y2017_2021 fuxi
boot dates_ngcm_2019_2021.txt            y2019_2021 ngcm

python -u plot_TCC.py --models ucs2s_v2 fuxi        --tag common86 --mem-suffix _m20
python -u plot_TCC.py --models ucs2s_v2 fuxi ngcm   --tag common72 --mem-suffix _m20
python -u plot_TCC.py --models ucs2s_v2 fuxi si_s2s --tag common39 --mem-suffix _m20
python -u plot_TCC.py --models ucs2s_v2 si_s2s      --tag si104    --mem-suffix _m20
python -u plot_TCC_bars_own_record.py --models ucs2s_v2 fuxi ngcm   --mem-suffix _m20
python -u plot_TCC_bars_own_record.py --models ucs2s_v2 si_s2s ngcm --mem-suffix _m20
echo "Job finished at: $(date) with exit code 0"
