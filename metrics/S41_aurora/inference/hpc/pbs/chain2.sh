#!/bin/bash
# S41: when 9075362 (UC-S2S 2019-24 part 1) has finished, submit part 2 on the same pool if tasks remain and the guard allows it.
H=/lus/flare/projects/S2Storm/s2storm/runs/bing/hc; P=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python3
while qstat 9075362 >/dev/null 2>&1 && ! qstat -x 9075362 | grep -q " F "; do sleep 120; done
left=$($P -c "
import json,os
T=[json.loads(l) for l in open('$H/tasks_2019-24.jsonl')]
print(sum(not os.path.exists('$H/state/'+t['id']+'.done') for t in T))")
echo "$(date -u) part 1 ended; tasks left $left" >> $H/chain2.log
[ "$left" -gt 0 ] || exit 0
if $P $H/../guard/guard.py check uc 256 1.0833 >> $H/chain2.log 2>&1; then
  J=$(qsub $H/s41_hc2.pbs); echo "$(date -u) submitted $J" >> $H/chain2.log
  echo "$(date -u +%FT%TZ) $J s41_hc2 prod 256x1:05 (UC-S2S 2019-24 part 2, chain2.sh)" >> $H/../guard/submitted.txt
else echo "$(date -u) guard refused part 2" >> $H/chain2.log; fi
