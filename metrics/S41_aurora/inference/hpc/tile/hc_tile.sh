#!/bin/bash
# S41 UC-S2S production, one rank per tile. Local rank 0 copies both checkpoint bundles (16-stripe copies) to /dev/shm once per node;
# the other ranks wait for READY. PALS_* stay set: the farm reads its rank from them and strips them from every child.
R=/lus/flare/projects/S2Storm/s2storm; H=$R/runs/bing/hc
G=${PALS_RANKID:-0}; L=${PALS_LOCAL_RANKID:-0}
export ZE_AFFINITY_MASK=$(( L % 12 ))
S=/dev/shm/s41
if [ $L = 0 ]; then
  rm -rf $S; mkdir -p $S
  for m in sstanom plain; do cp -r $R/aurora_bundle_v5/checkpoints_${m}_c16 $S/checkpoints_${m}; done
  touch $S/READY
else
  w=0; while [ ! -f $S/READY ] && [ $w -lt 900 ]; do sleep 5; w=$((w+5)); done
fi
source $R/runs/bing/env/ucs2s-xpu/bin/activate
export S2STORM_CODE=$C UCS2S_REPO=$R/ext/S2S UCS2S_CLIMATOLOGY=$R/data/placeholder/climatology_placeholder.nc S2STORM_CLIM=
export S2STORM_STATE=$H/state PYTHONPATH=$C OMP_NUM_THREADS=8 HDF5_USE_FILE_LOCKING=FALSE PYTHONPYCACHEPREFIX=$R/runs/bing/pycache
mkdir -p $H/state/timing
cd $C
python -m s2storm.aurora.taskfarm --tasks ${TASKS:-$H/tasks.jsonl} --state $H/state --min_left ${MIN_LEFT:-1250}
