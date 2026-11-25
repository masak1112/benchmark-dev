#!/bin/bash
# PBS submission script for rank_histogram_main.py
#
# Usage:
#   qsub -v MODEL=ours_si_postprocess_v2_0903,VAR=geopotential_500 submit_rank_hist.sh
#   qsub -v MODEL=amip_s2s_train_scratch,VAR=geopotential_500    submit_rank_hist.sh

#PBS -A UCHI0014
#PBS -N rank_hist
#PBS -q develop
#PBS -l walltime=05:40:00
#PBS -l select=1:ncpus=8:mem=220GB
#PBS -o /glade/derecho/scratch/bgong/tmp/rank_hist.log
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running on host: $(hostname)"

module load conda
conda activate /glade/work/bgong/conda-envs/my_env

SCRIPT=/glade/work/bgong/benchmark-dev/metrics/rank_histogram/rank_histogram_main.py
CONFIG=/glade/work/bgong/benchmark-dev/metrics/rank_histogram/configuration/config.yaml

ARGS="--config $CONFIG"
[ -n "$MODEL" ] && ARGS="$ARGS --model $MODEL"
[ -n "$VAR"   ] && ARGS="$ARGS --var $VAR"
[ -n "$OVERWRITE" ] && ARGS="$ARGS --overwrite"
[ -n "$DATES_FILE" ] && ARGS="$ARGS --dates-file $DATES_FILE"
[ -n "$DATE_TAG" ]   && ARGS="$ARGS --date-tag $DATE_TAG"
[ -n "$WEEKLY" ]     && ARGS="$ARGS --weekly"
[ -n "$YEARS" ]      && ARGS="$ARGS --years $YEARS"
[ -n "$REGION" ]     && ARGS="$ARGS --region $REGION"

echo "Running: python $SCRIPT $ARGS"
python $SCRIPT $ARGS
EXIT_CODE=$?

echo "Job finished at: $(date '+%Y-%m-%d %H:%M:%S') with exit code $EXIT_CODE"
exit $EXIT_CODE
