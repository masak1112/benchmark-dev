#!/bin/bash
# S41 fix test (Bing 7ceed2a boundary off-by-one): same as hc/hc_tile.sh, plain checkpoints only, UCS2S_REPO = new worktree, own state dir.
R=/lus/flare/projects/S2Storm/s2storm; H=$R/runs/bing/hc; F=$R/runs/bing/hcfix
G=${PALS_RANKID:-0}; L=${PALS_LOCAL_RANKID:-0}
export ZE_AFFINITY_MASK=$(( L % 12 ))
S=/dev/shm/s41
if [ $L = 0 ]; then
  rm -rf $S; mkdir -p $S
  cp -r $R/aurora_bundle_v5/checkpoints_plain_c16 $S/checkpoints_plain
  touch $S/READY
else
  w=0; while [ ! -f $S/READY ] && [ $w -lt 900 ]; do sleep 5; w=$((w+5)); done
fi
source $R/runs/bing/env/ucs2s-xpu/bin/activate
export S2STORM_CODE=$C UCS2S_REPO=$R/runs/bing/ext/S2S_fix_7ceed2a UCS2S_CLIMATOLOGY=$R/data/placeholder/climatology_placeholder.nc S2STORM_CLIM=
export S2STORM_STATE=$F/state PYTHONPATH=$C OMP_NUM_THREADS=8 HDF5_USE_FILE_LOCKING=FALSE PYTHONPYCACHEPREFIX=$R/runs/bing/pycache_fix
mkdir -p $F/state/timing
cd $C
python -m s2storm.aurora.taskfarm --tasks $TASKS --state $F/state --min_left ${MIN_LEFT:-1250}
