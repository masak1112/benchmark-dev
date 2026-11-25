#!/bin/bash
# quick check: pooled multicond vs ECMWF (shared inits) with and without --yearly-mean, 2019-2021 and 2019+2020
#PBS -A UCHI0014
#PBS -N tcc_ymchk
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ymcheck.log
#PBS -j oe
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
for D in dates_mcall_ec_common.txt:chkall dates_mcec1920_v3.txt:chk1920; do f=${D%%:*}; t=${D##*:}
  for YM in "" "--yearly-mean"; do S=""; [ -n "$YM" ] && S=_ym
    for m in ucs2s_multicond_all ecmwf_rf; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $f --tag ${t}_m20$S --n-members 20 $YM; done
  done
done
echo "Job finished at: $(date)"
