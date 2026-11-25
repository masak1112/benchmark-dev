#!/bin/bash
# All four models at 20 members on their 39 common inits (May-Jul 2019-2021): aggregates, bootstrap, plots
#PBS -A UCHI0014
#PBS -N tcc_m20_4mod
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_m20_4models.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=common_dates_ucs2sv2_fuxi_sis2s_ngcm.txt
for m in ucs2s_v2 fuxi si_s2s ngcm; do
    python -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag common39x4_m20 --n-members 20
done
python -u tcc_bootstrap.py --models ucs2s_v2 fuxi si_s2s ngcm --dates-file $D --tag common39x4_m20 --block month --n-members 20
python -u plot_TCC.py --models ucs2s_v2 fuxi si_s2s ngcm --tag common39x4 --mem-suffix _m20 --bars-only --no-own-record
python -u plot_TCC.py --models ucs2s_v2 fuxi si_s2s ngcm --tag common39x4 --mem-suffix _m20
echo "Job finished at: $(date) with exit code 0"
