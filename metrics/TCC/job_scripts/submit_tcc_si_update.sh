#!/bin/bash
# Updated SI-S2S (129 inits with 20 members in 2019-2021): aggregate + bootstrap (as is and month-mean removed),
# then redraw the 2019-2021 own-record bar charts
#PBS -A UCHI0014
#PBS -N tcc_si_upd
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_si_update.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_sis2s_20mem_2019_2021.txt
python -u tcc_weekly.py --model si_s2s --stage aggregate --dates-file $D --tag si20m1921_m20 --n-members 20
python -u tcc_weekly.py --model si_s2s --stage aggregate --dates-file $D --tag si20m1921_m20_mm --n-members 20 --monthly-mean
python -u tcc_bootstrap.py --models si_s2s --dates-file $D --tag si20m1921_m20 --block month --n-members 20
python -u tcc_bootstrap.py --models si_s2s --dates-file $D --tag si20m1921_m20_mm --block month --n-members 20 --monthly-mean
for suf in _m20 _m20_mm; do
    for m in "ucs2s_v2 fuxi si_s2s ngcm" "ucs2s_v2 si_s2s ngcm"; do
        python -u plot_TCC_bars_own_record.py --models $m --mem-suffix $suf --period 2019-2021
    done
done
echo "Job finished at: $(date) with exit code 0"
