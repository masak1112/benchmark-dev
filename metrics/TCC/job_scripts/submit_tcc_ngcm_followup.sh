#!/bin/bash
# After NGCM extraction: NGCM on its full record, and UC-S2S V2 / FuXi / NGCM on their 72 common inits
#PBS -A UCHI0014
#PBS -N tcc_ngcm_agg
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ngcm_followup.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D3=common_dates_ucs2sv2_fuxi_ngcm.txt
python -u tcc_weekly.py --model ngcm --stage aggregate --dates-file dates_ngcm_2019_2021.txt --tag y2019_2021
for m in ucs2s_v2 fuxi ngcm; do python -u tcc_weekly.py --model $m --stage aggregate --dates-file $D3 --tag common72; done
python -u tcc_bootstrap.py --models ucs2s_v2 fuxi ngcm --dates-file $D3 --tag common72 --block month
python -u tcc_bootstrap.py --models ngcm --dates-file dates_ngcm_2019_2021.txt --tag y2019_2021 --block month
python -u plot_TCC.py --models ucs2s_v2 fuxi ngcm --tag common72
echo "Job finished at: $(date) with exit code 0"
