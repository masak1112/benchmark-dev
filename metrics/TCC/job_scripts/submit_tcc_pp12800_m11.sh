#!/bin/bash
# multicond rerun with postprocess SI pp_si_ckpt_12800 in every year vs UC-S2S V2 (nearest init, one-to-one, <=1 day),
# both 11 members (and multicond 20 members vs ECMWF); plus ECMWF hindcast (11) on the inits all three share, and multicond vs ECMWF on all shared inits. Plain + --yearly-mean, global / NH / tropics.
#PBS -A UCHI0014
#PBS -N tcc_pp12k
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=200GB
#PBS -o /glade/derecho/scratch/bgong/tmp/tcc_pp12800_m11.log
#PBS -j oe
echo "Job started at: $(date) on $(hostname)"
PY=/glade/work/bgong/.venv/bin/python
cd /glade/work/bgong/benchmark-dev/metrics/TCC
set -e
P=/glade/derecho/scratch/bgong/pp_si_infer_multicond_pp12800/predictions
C=/glade/derecho/scratch/bgong/tcc_cache
# inits whose members 0..10 are all full-size files (1144652889 or ...895 bytes: header text differs)
ls -l $P | awk '$5>=1144652880 {split($9,a,"_45step_"); split(a[2],b,"_ens_"); m=b[2]+0; if (m<=10) c[substr(b[1],1,8)]++}
    END {for (k in c) if (c[k]==11) print k}' | sort > dates_pp12800_m11.txt
echo "complete inits (11 members): $(wc -l < dates_pp12800_m11.txt)"; cut -c1-4 dates_pp12800_m11.txt | uniq -c
$PY make_pairs_v2.py dates_pp12800_m11.txt pairs_v2_pp12800.txt
cut -d' ' -f2 pairs_v2_pp12800.txt | sort > dates_v2_pp12800_real.txt
$PY -u tcc_weekly.py --model ucs2s_mc_pp12800f --stage extract --dates-file dates_pp12800_m11.txt --n-members 11 --workers 8
$PY -u tcc_weekly.py --model ucs2s_v2 --stage extract --dates-file dates_v2_pp12800_real.txt --n-members 11 --workers 8
ls -l $P | awk '$5>=1144652880 {split($9,a,"_45step_"); split(a[2],b,"_ens_"); m=b[2]+0; if (m<=19) c[substr(b[1],1,8)]++}
    END {for (k in c) if (c[k]==20) print k}' | sort > dates_pp12800_m20.txt
$PY -u tcc_weekly.py --model ucs2s_mc_pp12800f --stage extract --dates-file dates_pp12800_m20.txt --n-members 20 --workers 8
comm -12 dates_pp12800_m20.txt <(ls $C/ecmwf_rf | sed 's/.npz//' | sort) > dates_pp12800_ec_m20.txt
cut -d' ' -f1 pairs_v2_pp12800.txt | sort > dates_pp12800_v2.txt
comm -12 dates_pp12800_v2.txt <(ls $C/ecmwf_rf | sed 's/.npz//' | sort) > dates_pp12800_v2_ec.txt
comm -12 dates_pp12800_m11.txt <(ls $C/ecmwf_rf | sed 's/.npz//' | sort) > dates_pp12800_ec.txt
wc -l dates_pp12800_m20.txt dates_pp12800_ec_m20.txt dates_pp12800_m11.txt dates_pp12800_v2.txt dates_pp12800_v2_ec.txt dates_pp12800_ec.txt
MC=ucs2s_mc_pp12800f; V2=ucs2s_v2_pp12800pair; NM=11
run() {  # dates tag name extra models...
    local F=$1 T=$2 N=$3 X=$4; shift 4
    for m in "$@"; do $PY -u tcc_weekly.py --model $m --stage aggregate --dates-file $F --tag $T --n-members $NM $X; done
    $PY -u tcc_bootstrap.py --block none --dates-file $F --tag $T --n-members $NM $X --models "$@"
    local SER=("$MC:$T:UC-S2S multicond")
    [[ " $* " == *" $V2 "* ]] && SER+=("$V2:$T:UC-S2S V2")
    [[ " $* " == *" ecmwf_rf "* ]] && SER+=("ecmwf_rf:$T:ECMWF hindcast")
    for R in global NH tropics; do S=""; [ $R != global ] && S=_$R
        $PY -u plot_TCC_paper.py --block none --region $R --name $N$S --series "${SER[@]}"; done
}
NM=20
run dates_pp12800_ec_m20.txt pp12k_ec_m20 multicond_pp12800_m20_vs_ecmwf_rf "" $MC ecmwf_rf
run dates_pp12800_ec_m20.txt pp12k_ec_m20_ym multicond_pp12800_m20_vs_ecmwf_rf_ym "--yearly-mean" $MC ecmwf_rf
NM=11
run dates_pp12800_ec.txt pp12k_ec_m11 multicond_pp12800_vs_ecmwf_rf_m11 "" $MC ecmwf_rf
run dates_pp12800_ec.txt pp12k_ec_m11_ym multicond_pp12800_vs_ecmwf_rf_m11_ym "--yearly-mean" $MC ecmwf_rf
run dates_pp12800_v2.txt pp12k_v2_m11 multicond_pp12800_vs_v2_m11 "" $MC $V2
run dates_pp12800_v2_ec.txt pp12k_v2ec_m11 multicond_pp12800_vs_v2_ecmwf_rf_m11 "" $MC $V2 ecmwf_rf
run dates_pp12800_v2.txt pp12k_v2_m11_ym multicond_pp12800_vs_v2_m11_ym "--yearly-mean" $MC $V2
echo "Job finished at: $(date) with exit code 0"
