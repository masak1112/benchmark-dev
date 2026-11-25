#!/bin/bash
# After SI-S2S extraction: aggregates + bootstraps for the 3-model (39 inits) and UC/SI (104 inits) sets
#PBS -A UCHI0014
#PBS -N tcc_si_agg
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_si_followup.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
export OMP_NUM_THREADS=32
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D3=common_dates_ucs2sv2_fuxi_sis2s.txt
D2=common_dates_ucs2sv2_sis2s.txt
for m in ucs2s_v2 fuxi si_s2s; do python -u tcc_weekly.py --model $m --stage aggregate --dates-file $D3 --tag common39; done
for m in ucs2s_v2 si_s2s;      do python -u tcc_weekly.py --model $m --stage aggregate --dates-file $D2 --tag si104; done
python -u tcc_bootstrap.py --models ucs2s_v2 fuxi si_s2s --dates-file $D3 --tag common39 --block month
python -u tcc_bootstrap.py --models ucs2s_v2 si_s2s      --dates-file $D2 --tag si104    --block month
echo "Job finished at: $(date) with exit code 0"
