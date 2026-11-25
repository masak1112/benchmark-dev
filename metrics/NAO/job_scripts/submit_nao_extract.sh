#!/bin/bash
# Per-member daily NAO index for every DJF init:  qsub -v MODEL=fuxi[,DATES=<file>] submit_nao_extract.sh
#PBS -A UCHI0014
#PBS -N nao_extract
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR
exec > /glade/derecho/scratch/bgong/tmp/nao_extract_${MODEL}.log 2>&1
echo "Job started at: $(date) on $(hostname)  MODEL=$MODEL"
module load conda
conda activate /glade/work/bgong/conda-envs/my_env
cd /glade/work/bgong/benchmark-dev/metrics/NAO
python -u nao_extract.py --model $MODEL --workers 32 ${DATES:+--dates-file $DATES}
echo "Job finished at: $(date) with exit code $?"
