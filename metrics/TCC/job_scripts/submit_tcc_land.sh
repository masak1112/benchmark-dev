#!/bin/bash
# Re-run the own-record 2019-2021 bootstraps so they also hold the land / ocean regions
# (ERA5 lsm >= 0.5), check the old regions are unchanged, then plot T2M over land only.
# Old pickles: results/backup_pre_land/
#PBS -A UCHI0014
#PBS -N tcc_land
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_land.log
#PBS -j oe

echo "Job started at: $(date) on $(hostname)"
export TMPDIR=/glade/derecho/scratch/$USER/tmp
export OMP_NUM_THREADS=32
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
for suf in "" _mm; do
    O="--n-members 20"; [ "$suf" = "_mm" ] && O="$O --monthly-mean"
    $PY -u tcc_bootstrap.py --models ucs2s_v2 --dates-file dates_ucs2sv2_2019_2021.txt --tag ucall1921_m20$suf  --block month $O
    $PY -u tcc_bootstrap.py --models fuxi     --dates-file dates_fuxi_2019_2021.txt    --tag fx1921_m20$suf     --block month $O
    $PY -u tcc_bootstrap.py --models ngcm     --dates-file dates_ngcm_2019_2021.txt    --tag y2019_2021_m20$suf --block month $O
    $PY -u tcc_bootstrap.py --models ecmwf_rf --dates-file dates_ecmwf_rf_2019_2021.txt --tag ecrf1921_m20$suf  --block month $O
done
# the four latitude-band regions must match the previous run (same seed, same inits)
$PY - <<'EOF'
import pickle, numpy as np
for t in ['ucall1921', 'fx1921', 'y2019_2021', 'ecrf1921']:
    for suf in ['', '_mm']:
        f = f'tcc_bootstrap_{t}_m20{suf}_month.pkl'
        old, new = (pickle.load(open(p + f, 'rb')) for p in ('results/backup_pre_land/', 'results/'))
        assert old['init_dates'] == new['init_dates'], f
        d = max(np.nanmax(np.abs(old['boot'][m][l] - new['boot'][m][l][..., :len(old['regions'])]))
                for m in old['boot'] for l in old['boot'][m] if not np.isnan(old['boot'][m][l]).all())
        print(f'{f}: n={len(new["init_dates"])}, max |old-new| = {d:.2e}')
EOF
for suf in "" _mm; do
    $PY -u plot_TCC_paper.py --own-record --suffix _m20$suf --models ucs2s_v2 ngcm fuxi ecmwf_rf --vars t2m --region land
done
echo "Job finished at: $(date) with exit code 0"
