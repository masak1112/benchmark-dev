#!/bin/bash
#PBS -A UCHI0014
#PBS -N train_exp03
#PBS -q main
#PBS -l walltime=00:30:00 
#PBS -l select=1:ncpus=64
#PBS -e ucar.txt
#PBS -o ucar.out
##PBS -l gpu_type=a100
#export WORLD_SIZE=$((PBS_NUM_NODES * PBS_NUM_PPN))
#echo "Total tasks: $WORLD_SIZE"

cd /glade/work/bgong/benchmark-dev/metrics/monsoon/jupyter_notebooks
module load conda
conda activate /glade/work/bgong/conda-envs/myenv
python regrid.py
