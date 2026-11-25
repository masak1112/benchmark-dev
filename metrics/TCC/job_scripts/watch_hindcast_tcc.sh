#!/bin/bash
# Wait until the ECMWF hindcast download has all 1488 files (or its supervisor has finished), then rerun the TCC job.
until [ $(ls /glade/derecho/scratch/bgong/ec_s2s_model_2025_hindcast | grep -c '\.grib$') -ge 1488 ] || grep -q "supervisor finished" /glade/derecho/scratch/bgong/ec_s2s_model_2025_hindcast/download.log; do sleep 900; done
sleep 600   # let the last files close
cd /glade/work/bgong/benchmark-dev/metrics/TCC/job_scripts
echo "$(date) watch_hindcast_tcc: $(ls /glade/derecho/scratch/bgong/ec_s2s_model_2025_hindcast | grep -c '\.grib$') files, submitting $(qsub submit_tcc_ecmwf_rf.sh)" >> /glade/derecho/scratch/bgong/tmp/watch_hindcast_tcc.log
