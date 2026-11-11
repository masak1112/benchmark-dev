#!/bin/bash
# 20-member results that do not need FuXi: UC-S2S V2 vs SI-S2S (104 inits) + own-record bars (UC, SI, NGCM)
#PBS -A UCHI0014
#PBS -N tcc_m20_part
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_m20_partial.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
M="--n-members 20"
for m in ucs2s_v2 si_s2s; do python -u tcc_weekly.py --model $m --stage aggregate --dates-file common_dates_ucs2sv2_sis2s.txt --tag si104_m20 $M; done
python -u tcc_weekly.py --model ucs2s_v2 --stage aggregate --dates-file dates_ucs2sv2_cached151.txt --tag cached151_m20 $M
python -u tcc_weekly.py --model ngcm     --stage aggregate --dates-file dates_ngcm_2019_2021.txt    --tag y2019_2021_m20 $M
python -u tcc_bootstrap.py --models ucs2s_v2 si_s2s --dates-file common_dates_ucs2sv2_sis2s.txt --tag si104_m20 --block month $M
python -u tcc_bootstrap.py --models ucs2s_v2 --dates-file dates_ucs2sv2_cached151.txt --tag cached151_m20 --block month $M
python -u tcc_bootstrap.py --models ngcm --dates-file dates_ngcm_2019_2021.txt --tag y2019_2021_m20 --block month $M
python -u plot_TCC_bars_own_record.py --models ucs2s_v2 si_s2s ngcm --mem-suffix _m20
python -u plot_TCC.py --models ucs2s_v2 si_s2s --tag si104 --mem-suffix _m20
echo "Job finished at: $(date) with exit code 0"
