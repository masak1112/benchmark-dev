#!/bin/bash
# multicond rerun with postprocess SI pp_si_ckpt_12800 (all years), 20 members, vs ECMWF hindcast (11) and FuXi-S2S (20):
# FuXi is paired to each multicond init by the nearest FuXi init (one-to-one, <=2 days; pairs_fuxi_pp12800.txt).
# (A) multicond vs ECMWF on all shared inits, (B) multicond vs ECMWF vs FuXi (nearest init) on the same inits.
# Plain + --yearly-mean; global / NH / tropics. Stars vs multicond.
#PBS -A UCHI0014
#PBS -N tcc_pp20
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_pp12800_m20.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
P=/glade/derecho/scratch/bgong/pp_si_infer_multicond_pp12800/predictions
C=/glade/derecho/scratch/bgong/tcc_cache
# The folder holds two pp12800 runs: per-year multicond12800_{2019,2020,2021} (preferred) and the earlier
# multicond_pp12800. Per init take members 0..19 from ONE run (full-size files only), linked under a single prefix.
L=/glade/derecho/scratch/bgong/multicond_pp12800_links; rm -rf $L; mkdir -p $L
NEWP='pangu_plasim_pp_si_infer_multicond12800_20[0-9][0-9]_24h_45step_'; OLDP='pangu_plasim_pp_si_infer_multicond_pp12800_24h_45step_'
complete() {  # prefix regex -> inits with members 0..19 full size
    ls -l $P | awk -v pre="^$1" '$5>=1144652880 && $9 ~ pre {split($9,a,"_45step_"); split(a[2],b,"_ens_"); m=b[2]+0;
        if (m<=19) c[substr(b[1],1,8)]++} END {for (k in c) if (c[k]==20) print k}' | sort
}
complete "$NEWP" > /tmp/pp_new_$$.txt; complete "$OLDP" > /tmp/pp_old_$$.txt
for d in $(cat /tmp/pp_new_$$.txt); do for e in $(seq 0 19); do
    ln -s $P/pangu_plasim_pp_si_infer_multicond12800_${d:0:4}_24h_45step_${d}00_ens_$e.nc $L/pangu_plasim_pp_si_infer_multicond_pp12800f_24h_45step_${d}00_ens_$e.nc; done; done
for d in $(comm -13 /tmp/pp_new_$$.txt /tmp/pp_old_$$.txt); do for e in $(seq 0 19); do
    ln -s $P/${OLDP}${d}00_ens_$e.nc $L/pangu_plasim_pp_si_infer_multicond_pp12800f_24h_45step_${d}00_ens_$e.nc; done; done
ls $L | sed 's/.*_45step_\(........\).*/\1/' | sort -u > dates_pp12800_m20.txt
echo "complete inits (20 members): $(wc -l < dates_pp12800_m20.txt) = $(wc -l < /tmp/pp_new_$$.txt) per-year run + $(comm -13 /tmp/pp_new_$$.txt /tmp/pp_old_$$.txt | wc -l) earlier run"
cut -c1-4 dates_pp12800_m20.txt | uniq -c
$PY -u tcc_weekly.py --model ucs2s_mc_pp12800f --stage extract --dates-file dates_pp12800_m20.txt --n-members 20 --workers 8
comm -12 dates_pp12800_m20.txt <(ls $C/ecmwf_rf | sed 's/.npz//' | sort) > dates_pp12800_ec_m20.txt
$PY make_pairs_v2.py dates_pp12800_ec_m20.txt pairs_fuxi_pp12800.txt --cache fuxi_m20 --max-days 2
cut -d' ' -f1 pairs_fuxi_pp12800.txt | sort > dates_pp12800_fxec_m20.txt
wc -l dates_pp12800_m20.txt dates_pp12800_ec_m20.txt dates_pp12800_fxec_m20.txt
MC=ucs2s_mc_pp12800f; FX=fuxi_pp12800pair
run() {  # dates tag name extra models...
    local F=$1 T=$2 N=$3 X=$4; shift 4
    for m in "$@"; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members 20 $X; done
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members 20 $X --models "$@"
    local SER=("$MC:$T:UC-S2S multicond")
    [[ " $* " == *" $FX "* ]] && SER+=("$FX:$T:FuXi-S2S")
    [[ " $* " == *" ecmwf_rf "* ]] && SER+=("ecmwf_rf:$T:ECMWF hindcast")
    for R in global NH tropics; do S=""; [ $R != global ] && S=_$R
        $PY -u plot_TCC_paper.py --block none --region $R --name $N$S --series "${SER[@]}"; done
}
run dates_pp12800_ec_m20.txt pp20_ec multicond_pp12800_vs_ecmwf_rf "" $MC ecmwf_rf
run dates_pp12800_ec_m20.txt pp20_ec_ym multicond_pp12800_vs_ecmwf_rf_ym "--yearly-mean" $MC ecmwf_rf
run dates_pp12800_fxec_m20.txt pp20_fxec multicond_pp12800_vs_ecmwf_rf_fuxi "" $MC ecmwf_rf $FX
run dates_pp12800_fxec_m20.txt pp20_fxec_ym multicond_pp12800_vs_ecmwf_rf_fuxi_ym "--yearly-mean" $MC ecmwf_rf $FX
echo "Job finished at: $(date) with exit code 0"
