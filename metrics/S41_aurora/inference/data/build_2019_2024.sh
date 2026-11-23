#!/bin/bash
# UC-S2S daily 00 UTC inputs 2019-01-01..2024-12-31 from ARCO (s2storm.data.era5_h5, hybrid regrid), 6 shards, resumable.
# Rerun this script to resume: --skip_existing skips finished files; partial files are written as .h5.partial then renamed.
B=/lus/flare/projects/S2Storm/s2storm/aurora_bundle_v5
source $B/logs/env.sh
cd $B/logs
N=6
for k in $(seq 0 $((N-1))); do
  ( for try in 1 2 3 4 5 6 7 8; do
      nice -n 10 python -u -m s2storm.data.era5_h5 --start 2019-01-01 --end 2024-12-31 --hours 0 --task $k --ntasks $N \
        --skip_existing --out $B/h5data >> build_$k.log 2>&1 && { echo "shard $k ok $(date -u)" >> build_$k.log; break; }
      echo "shard $k try $try failed $(date -u)" >> build_$k.log; sleep 60
    done ) &
done
wait
echo BUILD_DONE $(date -u) $(ls $B/h5data/*.h5 | wc -l) files > $B/logs/build.done
