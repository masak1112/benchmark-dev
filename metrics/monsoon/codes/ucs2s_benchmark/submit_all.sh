#!/bin/bash
# Submit the full UC-S2S monsoon benchmark chain on Derecho.
#   postprocess ─┬─> bias_correct ─┬─> onset_core ──┬─> plots
#   era5_indices ┘                 └─> raw_metrics ─┘
# Skip stages whose outputs already exist with e.g.  SKIP_PP=1 SKIP_ERA5=1 SKIP_BC=1 ./submit_all.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
dep() { local d=""; for j in "$@"; do [ -n "$j" ] && d="${d:+$d:}$j"; done; [ -n "$d" ] && echo "-W depend=afterok:$d"; }
PP=""; E5=""
[ -z "${SKIP_PP:-}" ]   && PP=$(qsub pbs/postprocess.pbs)   && echo "postprocess  $PP"
[ -z "${SKIP_ERA5:-}" ] && E5=$(qsub pbs/era5_indices.pbs)  && echo "era5_indices $E5"
BC=""; [ -z "${SKIP_BC:-}" ] && BC=$(qsub $(dep $PP $E5) pbs/bias_correct.pbs) && echo "bias_correct $BC"
ON=$(qsub $(dep $PP $BC) pbs/onset_core.pbs);       echo "onset_core   $ON"
RM=$(qsub $(dep $PP $E5 $BC) pbs/raw_metrics.pbs);  echo "raw_metrics  $RM"
PL=$(qsub $(dep $ON $RM) pbs/plots.pbs);         echo "plots        $PL"
