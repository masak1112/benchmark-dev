#!/bin/bash
# S41 UC-S2S debug job: one rank per tile; node n runs its own task farm as job "<PBS_JOBID>_farm<n>" on ONE shared pool
# (two farms = the 2-job claim test inside one debug job; debug allows one running job per user).
R=/lus/flare/projects/S2Storm/s2storm
G=${PALS_RANKID:-0}; L=${PALS_LOCAL_RANKID:-0}
export ZE_AFFINITY_MASK=$(( L % 12 ))
export PBS_JOBID="${PBS_JOBID%%.*}_farm$(( G / 12 ))"
# PALS_*/PMI* stay set: the farm reads its rank from them and strips them from every child (MPI_ENV_PREFIXES)
source $R/runs/bing/env/ucs2s-xpu/bin/activate
export PALS_SIZE=1 S2STORM_CODE=$C UCS2S_REPO=$R/ext/S2S UCS2S_CLIMATOLOGY=$R/data/placeholder/climatology_placeholder.nc S2STORM_CLIM=
export D=$R/runs/bing/debug S2STORM_STATE=${STATE:-$R/runs/bing/debug/state} S2STORM_OUT=$R/runs/bing/debug/out
export PYTHONPATH=$C OMP_NUM_THREADS=8 HDF5_USE_FILE_LOCKING=FALSE PYTHONPYCACHEPREFIX=$R/runs/bing/pycache
cd $C
python -m s2storm.aurora.taskfarm --tasks ${TASKS:-$D/tasks.jsonl} --state ${STATE:-$D/state} --min_left 600
