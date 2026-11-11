#!/bin/bash
# qsub -v MODEL=fuxi,STAGE=extract     submit_tcc.sh
# qsub -v MODEL=fuxi,STAGE=aggregate   submit_tcc.sh
# optional: DATES=<file> TAG=<name>  (default: common_dates_ucs2sv2_fuxi.txt / common86)
#           NMEM=<N>  use only the first N ensemble members
#PBS -A UCHI0014
#PBS -N tcc_weekly
#PBS -q develop
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR
DATES=${DATES:-common_dates_ucs2sv2_fuxi.txt}
TAG=${TAG:-common86}
NMEM_ARG=${NMEM:+--n-members $NMEM}
LOG=/glade/derecho/scratch/bgong/tmp/tcc_${MODEL}_${STAGE}_${TAG}${NMEM:+_m$NMEM}.log
exec > $LOG 2>&1

echo "Job started at: $(date) on $(hostname)  MODEL=$MODEL STAGE=$STAGE"

module load conda
conda activate /glade/work/bgong/conda-envs/my_env

cd /glade/work/bgong/benchmark-dev/metrics/TCC
python -u tcc_weekly.py --model $MODEL --stage $STAGE --workers 32 --dates-file $DATES --tag $TAG $NMEM_ARG

EXIT_CODE=$?
echo "Job finished at: $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
