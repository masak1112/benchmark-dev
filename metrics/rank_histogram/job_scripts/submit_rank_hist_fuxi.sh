#!/bin/bash
# PBS submission script for rank_histogram_main.py
#
# Usage:
#   qsub -v MODEL=ours_si_postprocess_v2_0903,VAR=geopotential_500 submit_rank_hist.sh
#   qsub -v MODEL=amip_s2s_train_scratch,VAR=geopotential_500    submit_rank_hist.sh

#PBS -A UCHI0014
#PBS -N rank_hist
#PBS -q main
#PBS -l walltime=11:50:00
#PBS -l select=1:ncpus=8:mem=220GB
#PBS -o /glade/derecho/scratch/bgong/tmp/rank_hist.log
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running on host: $(hostname)"

export FUXI_CACHE_DIR=/glade/derecho/scratch/bgong/tmp/fuxi_cache
mkdir -p $FUXI_CACHE_DIR
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

# Run the variables SEQUENTIALLY in one job. The 7z archives are solid-
# compressed and shared via FUXI_CACHE_DIR; concurrent jobs would race on a
# half-extracted directory. Sequential also means each archive extracts once
# for the whole run instead of once per variable.
EXIT_CODE=0
for V in ${VARS:-2m_temperature geopotential_500 geopotential_250}; do
  echo "=== VARIABLE: $V ==="
  python $SCRIPT --config $CONFIG --model fuxi_s2s --var $V ${YEARS:+--years $YEARS} ${WEEKLY:+--weekly}
  rc=$?
  [ $rc -ne 0 ] && EXIT_CODE=$rc
done

echo "Job finished at: $(date '+%Y-%m-%d %H:%M:%S') with exit code $EXIT_CODE"
exit $EXIT_CODE
