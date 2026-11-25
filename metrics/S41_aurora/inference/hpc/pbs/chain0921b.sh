#!/bin/bash
# resubmit the debug job until all 60 tasks of 2026-09-21 are done (max 3 jobs)
H=/lus/flare/projects/S2Storm/s2storm/runs/bing/hc; P=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python3; J=
for n in 1 2 3; do
  while qstat $J >/dev/null 2>&1 && ! qstat -x $J | grep -q " F "; do sleep 60; done
  left=$($P -c "import json,os; T=[json.loads(l) for l in open('$H/tasks_20260921.jsonl')]; print(sum(not os.path.exists('$H/state/'+t['id']+'.done') for t in T))")
  echo "$(date -u) job $J ended; left $left" >> $H/chain0921.log; [ "$left" -gt 0 ] || exit 0
  $P $H/../guard/guard.py check uc 2 1 >> $H/chain0921.log 2>&1 || exit 1
  until J2=$(qsub $H/s41_hc0921d.pbs 2>>$H/chain0921.log); do sleep 300; done; J=$(echo $J2 | cut -d. -f1)
  echo "$(date -u) submitted $J2" >> $H/chain0921.log; echo "$(date -u +%FT%TZ) $J2 s41_hc0921d debug 2x1:00 (2026-09-21 leftovers)" >> $H/../guard/submitted.txt
done
