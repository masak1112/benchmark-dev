#!/bin/bash
# UC-S2S V2 on all 302 of its 2019-2021 inits (20 members = native): aggregate, bootstrap, 2019-2021 bar plots
#PBS -A UCHI0014
#PBS -N tcc_ucall1921
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ucall1921.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_ucs2sv2_2019_2021.txt
python -u tcc_weekly.py --model ucs2s_v2 --stage aggregate --dates-file $D --tag ucall1921_m20 --n-members 20
python -u tcc_bootstrap.py --models ucs2s_v2 --dates-file $D --tag ucall1921_m20 --block month --n-members 20
for m in "ucs2s_v2 fuxi si_s2s ngcm" "ucs2s_v2 fuxi ngcm" "ucs2s_v2 si_s2s ngcm"; do
    python -u plot_TCC_bars_own_record.py --models $m --mem-suffix _m20 --period 2019-2021
done
echo "Job finished at: $(date) with exit code 0"
