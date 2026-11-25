#!/bin/bash
# 20-member TCC with every model restricted to 2019-2021 inits: aggregates, bootstraps, bar plots
#PBS -A UCHI0014
#PBS -N tcc_m20_1921
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_m20_1921.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
M="--n-members 20"
python -u tcc_weekly.py --model ucs2s_v2 --stage aggregate --dates-file dates_ucs2sv2_cached_2019_2021.txt --tag uc1921_m20 $M
python -u tcc_weekly.py --model fuxi     --stage aggregate --dates-file dates_fuxi_2019_2021.txt          --tag fx1921_m20 $M
for m in ucs2s_v2 si_s2s; do
    python -u tcc_weekly.py --model $m --stage aggregate --dates-file common_dates_ucs2sv2_sis2s_2019_2021.txt --tag si1921_m20 $M
done
python -u tcc_bootstrap.py --models ucs2s_v2 --dates-file dates_ucs2sv2_cached_2019_2021.txt --tag uc1921_m20 --block month $M
python -u tcc_bootstrap.py --models fuxi --dates-file dates_fuxi_2019_2021.txt --tag fx1921_m20 --block month $M
python -u tcc_bootstrap.py --models ucs2s_v2 si_s2s --dates-file common_dates_ucs2sv2_sis2s_2019_2021.txt --tag si1921_m20 --block month $M
# NGCM own record (2019-2021) was already done: tag y2019_2021_m20
for m in "ucs2s_v2 fuxi si_s2s ngcm" "ucs2s_v2 fuxi ngcm" "ucs2s_v2 si_s2s ngcm"; do
    python -u plot_TCC_bars_own_record.py --models $m --mem-suffix _m20 --period 2019-2021
done
python -u plot_TCC.py --models ucs2s_v2 si_s2s --tag si1921 --mem-suffix _m20 --bars-only --no-own-record
echo "Job finished at: $(date) with exit code 0"
