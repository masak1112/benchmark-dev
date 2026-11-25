#!/bin/bash
# ECMWF with 11 members (control + perturbed 1-10, like the reforecasts in the FuXi-S2S paper), 2019-2021
#PBS -A UCHI0014
#PBS -N tcc_ec_m11
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_ecmwf_m11.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export TMPDIR=/glade/derecho/scratch/$USER/tmp
export OMP_NUM_THREADS=32
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_ecmwf_2019_2021.txt
$PY -u tcc_weekly.py --model ecmwf --stage extract   --workers 32 --dates-file $D --tag ec1921 --n-members 11
$PY -u tcc_weekly.py --model ecmwf --stage aggregate --dates-file $D --tag ec1921_m11    --n-members 11
$PY -u tcc_weekly.py --model ecmwf --stage aggregate --dates-file $D --tag ec1921_m11_mm --n-members 11 --monthly-mean
$PY -u tcc_bootstrap.py --models ecmwf --dates-file $D --tag ec1921_m11    --block month --n-members 11
$PY -u tcc_bootstrap.py --models ecmwf --dates-file $D --tag ec1921_m11_mm --block month --n-members 11 --monthly-mean
$PY - <<'PYEOF'
import pickle
wk = ['week1', 'week2', 'week3', 'week4', 'week5', 'week6', 'week3-4', 'week5-6']
L = lambda t: pickle.load(open(f'results/tcc_weekly_ecmwf_{t}.pkl', 'rb'))
print('\nECMWF global TCC, 313 inits 2019-2021   ' + ' '.join(f'{w:>7s}' for w in wk))
for v in ['tp', 't2m', 'z500', 'u850']:
    for t in ['ec1921_m11', 'ec1921_m20', 'ec1921_m11_mm', 'ec1921_m20_mm']:
        r = L(t); g = r['regional'][v]['global']
        print(f'{v:5s} {t:16s}                  ' + ' '.join(f'{g[r["weeks"].index(w)]:7.3f}' for w in wk))
PYEOF
echo "Job finished at: $(date) with exit code 0"
