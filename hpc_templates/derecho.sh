#!/bin/bash
#PBS -A uric0009
#PBS -N metrics
#PBS -q develop
#PBS -l walltime=00:10:00 
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -e ucar_metrics_error.txt
#PBS -o ucar_metrics.out
#PBS -l gpu_type=a100
#export WORLD_SIZE=$((PBS_NUM_NODES * PBS_NUM_PPN))
#echo "Total tasks: $WORLD_SIZE"

ml conda
conda activate /glade/work/bgong/conda-envs/myenv
 python /glade/work/bgong/benchmark-dev/metrics/CRPS/crps.py