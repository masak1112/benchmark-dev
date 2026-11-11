"""Pair each init in a date file with the nearest cached init of another model, one-to-one: same-day matches first,
then +1, -1, +2, -2 ... days up to --max-days, each partner init used at most once.
usage: python make_pairs_v2.py dates_in.txt pairs_out.txt [--cache ucs2s_v2] [--max-days 1]"""
import os, argparse, datetime as dt
ap = argparse.ArgumentParser()
ap.add_argument('dates_in'); ap.add_argument('pairs_out')
ap.add_argument('--cache', default='ucs2s_v2', help='cache dir under tcc_cache holding the partner inits')
ap.add_argument('--max-days', type=int, default=1)
ap.add_argument('--have-file', default=None, help='partner inits from this date file instead of the cache dir')
a = ap.parse_args()
have = ({l.strip() for l in open(a.have_file) if l.strip()} if a.have_file else
        {f[:8] for f in os.listdir(os.path.join('/glade/derecho/scratch/bgong/tcc_cache', a.cache)) if f.endswith('.npz')})
dates = [l.strip() for l in open(a.dates_in) if l.strip()]
pairs = {}
for k in [0] + [s * n for n in range(1, a.max_days + 1) for s in (1, -1)]:
    for d in dates:
        r = (dt.datetime.strptime(d, '%Y%m%d') + dt.timedelta(days=k)).strftime('%Y%m%d')
        if d not in pairs and r in have and r not in pairs.values():
            pairs[d] = r
out = [f'{d} {pairs[d]}' for d in dates if d in pairs]
open(a.pairs_out, 'w').write('\n'.join(out) + '\n')
print(f'{len(out)} pairs -> {a.pairs_out}')
