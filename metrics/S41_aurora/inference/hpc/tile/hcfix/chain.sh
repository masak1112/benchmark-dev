#!/bin/bash
# resubmit the fix-test bulk job (debug, 2 nodes) after each one ends, until 120 tasks are done (max 4 jobs total). Only this brief's jobs.
cd /lus/flare/projects/S2Storm/s2storm/runs/bing/hcfix
n=1
n=1; last=9097212
while true; do
  while qstat $last >/dev/null 2>&1; do sleep 30; done
  d=$(ls state/*.done 2>/dev/null | wc -l)
  echo "$(date -u) job $last ended; done=$d" >> chain.log
  [ $d -ge 120 ] && break
  [ $n -ge 4 ] && break
  n=$((n+1))
  last=$(qsub -N s2storm_S41_fx_b$n -l select=2 -l walltime=01:00:00 -v TASKS=$PWD/tasks_full.jsonl fix.pbs | cut -d. -f1)
  echo "$(date -u) submitted $last" >> chain.log
done
