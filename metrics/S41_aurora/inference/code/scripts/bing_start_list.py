"""Weekly start list for Bing's UC-S2S hindcasts (DECISIONS 2026-11-09): for each week, the IFS init date closest to its Monday;
the week's Monday where IFS has no date. Week = Monday-3 .. Monday+3 (Fri..Thu), so windows partition time and no IFS date is used
twice; tie (Fri vs Thu) -> the earlier. Mondays from the first Monday >= 2019-01-01 to the last Monday <= 2026-09-30.

    python scripts/bing_start_list.py --ifs <date list> [<date list> ...] --out <dir>
Writes starts.csv (monday, start, source=ifs|monday, offset_days, ifs_list) and starts.txt (YYYY-MM-DD).
"""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ifs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--first', default='2019-01-01')
    ap.add_argument('--last', default='2026-09-30')
    a = ap.parse_args()
    ifs = {}
    for f in a.ifs:
        for x in Path(f).read_text().split():
            d = date(int(x[:4]), int(x[4:6]), int(x[6:8]))
            ifs.setdefault(d, Path(f).name)
    d0 = date.fromisoformat(a.first)
    mon = d0 + timedelta(days=(7 - d0.weekday()) % 7)
    last = date.fromisoformat(a.last)
    rows = []
    while mon <= last:
        cands = sorted((abs(k), k) for k in range(-3, 4) if mon + timedelta(days=k) in ifs)
        if cands:
            k = cands[0][1]
            s = mon + timedelta(days=k)
            rows.append(dict(monday=mon.isoformat(), start=s.isoformat(), source='ifs', offset_days=k, ifs_list=ifs[s]))
        else:
            rows.append(dict(monday=mon.isoformat(), start=mon.isoformat(), source='monday', offset_days=0, ifs_list=''))
        mon += timedelta(days=7)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'starts.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / 'starts.txt').write_text(''.join(r['start'] + '\n' for r in rows))
    from collections import Counter
    print('weeks', len(rows), 'ifs', sum(r['source'] == 'ifs' for r in rows), 'monday-only', sum(r['source'] == 'monday' for r in rows))
    print('offsets', Counter(r['offset_days'] for r in rows if r['source'] == 'ifs'))
    by = Counter((r['start'][:4], r['source']) for r in rows)
    for y in sorted({r['start'][:4] for r in rows}):
        print(y, 'ifs', by[(y, 'ifs')], 'monday', by[(y, 'monday')])
    wd = Counter(date.fromisoformat(d).strftime('%a') for d in [k.isoformat() for k in ifs])
    print('IFS list weekdays', dict(wd), 'n', len(ifs), min(ifs), max(ifs))


if __name__ == '__main__':
    main()
