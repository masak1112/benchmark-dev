#!/bin/bash
# PBS submission script for rmse_map.py + plot_rmse_map.py
#
# Computes gridpoint RMSE maps (vs ERA5) and renders the panel figure.
#
# Usage:
#   qsub -v MODEL=uc_s2s_v2,VAR=2m_temperature,YEARS=2019 submit_rmse_map.sh
#   qsub -v MODEL=uc_s2s_v2,VAR=2m_temperature,YEARS="2019 2020" submit_rmse_map.sh
#   qsub -v MODEL=uc_s2s_v2,VAR=2m_temperature,YEARS=2019,MAXDATES=5 submit_rmse_map.sh   # quick test
#
# Watch it:   qstat -u $USER
#             tail -f /glade/derecho/scratch/bgong/tmp/rmse_map.log

#PBS -A UCHI0018
#PBS -N rmse_map
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=60GB
#PBS -o /glade/derecho/scratch/bgong/tmp/rmse_map.log
#PBS -j oe

export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

echo "Job started: $(date)"
echo "Host: $(hostname)"

module load conda
conda activate /glade/work/bgong/conda-envs/my_env

BASE=/glade/work/bgong/benchmark-dev/metrics/RMSE
CONFIG=$BASE/configuration/config.yaml

# Defaults so a bare `qsub submit_rmse_map.sh` still does something sensible.
MODEL=${MODEL:-uc_s2s_v2}
VAR=${VAR:-2m_temperature}
YEARS=${YEARS:-2019}

ARGS="--config $CONFIG --model $MODEL --var $VAR --years $YEARS"
[ -n "$MAXDATES" ] && ARGS="$ARGS --max-dates $MAXDATES"

echo "Running: python $BASE/rmse_map.py $ARGS"
python $BASE/rmse_map.py $ARGS
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "rmse_map.py failed (exit $EXIT_CODE) - skipping plot"
    exit $EXIT_CODE
fi

# Tag matches what rmse_map.py derives from --years (space -> underscore).
TAG=$(echo $YEARS | tr ' ' '_')

echo "Running: plot_rmse_map.py --tag $TAG"
python $BASE/plot_rmse_map.py --config $CONFIG --model $MODEL --var $VAR --tag $TAG
EXIT_CODE=$?

echo "Job finished: $(date) | exit code $EXIT_CODE"
exit $EXIT_CODE
