#!/bin/bash
# UC-S2S SSTanom vs multicond (new 2019 run) vs V2, all 4 members, paired on their 17 shared inits (Thu, Jan 3 - Apr 25 2019)
#PBS -A UCHI0014
#PBS -N tcc_sstmc
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_sst_vs_mc2019n.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
D=dates_sst_mc2019n_common.txt; T=sstmc17_m4
for m in ucs2s_sstanom ucs2s_multicond2019n ucs2s_v2; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $D --tag $T --n-members 4; done
$PY -u tcc_bootstrap.py --block none --dates-file $D --tag $T --n-members 4 --models ucs2s_sstanom ucs2s_multicond2019n ucs2s_v2
for R in "--t2m-land" "" "--vars t2m --region land"; do
    case "$R" in "") U=_t2m_global;; --vars*) U=_t2m_land;; *) U=;; esac
    $PY -u plot_TCC_paper.py --block none --name sstanom_vs_multicond_v2_2019_common17_m4$U $R --series \
        "ucs2s_sstanom:$T:UC-S2S SSTanom" "ucs2s_multicond2019n:$T:UC-S2S multicond" "ucs2s_v2:$T:UC-S2S V2"
done
echo "Job finished at: $(date) with exit code 0"
