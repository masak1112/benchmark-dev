#!/bin/bash
# ECMWF TCC results (20 members): aggregates, bootstraps, plots. SUF='' (as is) or SUF=_mm (month mean removed)
#   qsub -v SUF=     submit_tcc_ecmwf_results.sh
#   qsub -v SUF=_mm  submit_tcc_ecmwf_results.sh
#PBS -A UCHI0014
#PBS -N tcc_ec_res
#PBS -q develop
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=32:mem=200GB
#PBS -j oe

LOG=/glade/derecho/scratch/bgong/tmp/tcc_ecmwf_results${SUF}.log
exec > $LOG 2>&1
echo "Job started at: $(date) on $(hostname)  SUF='$SUF'"
export OMP_NUM_THREADS=32
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
OPT="--n-members 20"; [ "$SUF" = "_mm" ] && OPT="$OPT --monthly-mean"
T="_m20$SUF"
agg()  { $PY -u tcc_weekly.py --model $1 --stage aggregate --dates-file $2 --tag $3$T $OPT; }
boot() { d=$1; t=$2; shift 2; $PY -u tcc_bootstrap.py --models "$@" --dates-file $d --tag $t$T --block month $OPT; }

# --- ECMWF sets
agg ecmwf dates_ecmwf_2019_2021.txt ec1921
for m in ucs2s_v2 ecmwf;           do agg $m common_dates_ucs2sv2_ecmwf.txt           ucec263;    done
for m in ucs2s_v2 fuxi ecmwf;      do agg $m common_dates_ucs2sv2_fuxi_ecmwf.txt      ucfxec60;   done
for m in ucs2s_v2 fuxi ngcm ecmwf; do agg $m common_dates_ucs2sv2_fuxi_ngcm_ecmwf.txt ucfxngec46; done
# --- FuXi / NeuralGCM sets redone with the corrected tp truth
agg fuxi dates_fuxi_2019_2021.txt fx1921
agg ngcm dates_ngcm_2019_2021.txt y2019_2021
agg fuxi common_dates_ucs2sv2_fuxi.txt common86
for m in fuxi ngcm; do agg $m common_dates_ucs2sv2_fuxi_ngcm.txt   common72;   done
for m in fuxi ngcm; do agg $m common_dates_4models_v2.txt          common39x4; done

boot dates_ecmwf_2019_2021.txt            ec1921     ecmwf
boot common_dates_ucs2sv2_ecmwf.txt       ucec263    ucs2s_v2 ecmwf
boot common_dates_ucs2sv2_fuxi_ecmwf.txt  ucfxec60   ucs2s_v2 fuxi ecmwf
boot common_dates_ucs2sv2_fuxi_ngcm_ecmwf.txt ucfxngec46 ucs2s_v2 fuxi ngcm ecmwf
boot dates_fuxi_2019_2021.txt             fx1921     fuxi
boot dates_ngcm_2019_2021.txt             y2019_2021 ngcm
boot common_dates_ucs2sv2_fuxi.txt        common86   ucs2s_v2 fuxi
boot common_dates_ucs2sv2_fuxi_ngcm.txt   common72   ucs2s_v2 fuxi ngcm
boot common_dates_4models_v2.txt          common39x4 ucs2s_v2 fuxi si_s2s ngcm

# --- plots
for m in "ucs2s_v2 fuxi si_s2s ngcm ecmwf" "ucs2s_v2 fuxi ngcm ecmwf" "ucs2s_v2 ecmwf" \
         "ucs2s_v2 fuxi si_s2s ngcm" "ucs2s_v2 fuxi ngcm" "ucs2s_v2 si_s2s ngcm"; do
    $PY -u plot_TCC_bars_own_record.py --models $m --mem-suffix $T --period 2019-2021
done
B="--bars-only --no-own-record --mem-suffix $T"
$PY -u plot_TCC.py --models ucs2s_v2 ecmwf                 --tag ucec263    $B
$PY -u plot_TCC.py --models ucs2s_v2 fuxi ecmwf            --tag ucfxec60   $B
$PY -u plot_TCC.py --models ucs2s_v2 fuxi ngcm ecmwf       --tag ucfxngec46 $B
$PY -u plot_TCC.py --models ucs2s_v2 fuxi                  --tag common86   $B
$PY -u plot_TCC.py --models ucs2s_v2 fuxi ngcm             --tag common72   $B
$PY -u plot_TCC.py --models ucs2s_v2 fuxi si_s2s ngcm      --tag common39x4 $B
# line plots, maps and the significance figure for the main ECMWF head-to-head
$PY -u plot_TCC.py --models ucs2s_v2 ecmwf --tag ucec263 --mem-suffix $T
echo "Job finished at: $(date) with exit code 0"
