# Provenance of `metrics/S41_aurora/` (copied 2026-11-12)

Self-contained copy of the S41 evaluation code from the s2storm repository. Imports were rewritten mechanically
(`s2storm.metrics.bing_eval` → `s41eval.bing_eval`, `s2storm.metrics.<mod>` → `s41eval.<mod>`, `s2storm.sci` → `s41eval.sci`,
`s2storm.models.ucs2s_hindcast` → `s41eval.hindcast_grid`); no other change. Production-only scripts (hindcast task lists, ensemble
statistics, start list) were left out. `loaders/format_s41.py`, `README.md`, this file and `environment.yml` are new.

| File here | Source repo | Branch | Commit | Source file | Imports rewritten |
|---|---|---|---|---|---|
| `metrics/S41_aurora/s41eval/bing_eval/__init__.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/__init__.py` | no |
| `metrics/S41_aurora/s41eval/bing_eval/core.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/core.py` | yes |
| `metrics/S41_aurora/s41eval/bing_eval/data.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/data.py` | yes |
| `metrics/S41_aurora/s41eval/bing_eval/diffmaps.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/diffmaps.py` | yes |
| `metrics/S41_aurora/s41eval/bing_eval/extremes.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/extremes.py` | yes |
| `metrics/S41_aurora/s41eval/bing_eval/pipeline.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/pipeline.py` | yes |
| `metrics/S41_aurora/s41eval/bing_eval/report.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/report.py` | yes |
| `metrics/S41_aurora/s41eval/bing_eval/scorecard.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/scorecard.py` | yes |
| `metrics/S41_aurora/s41eval/bing_eval/stats.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_eval/stats.py` | yes |
| `metrics/S41_aurora/s41eval/bing_weak.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/bing_weak.py` | yes |
| `metrics/S41_aurora/s41eval/biascorr.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/metrics/biascorr.py` | yes |
| `metrics/S41_aurora/s41eval/bing_modes.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `s2storm/metrics/bing_modes.py` | yes |
| `metrics/S41_aurora/s41eval/bing_scores.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `s2storm/metrics/bing_scores.py` | yes |
| `metrics/S41_aurora/s41eval/bing_waves.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `s2storm/metrics/bing_waves.py` | yes |
| `metrics/S41_aurora/s41eval/bing_helperc.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `s2storm/metrics/bing_helperc.py` | no |
| `metrics/S41_aurora/s41eval/sci/clim_trend.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/sci/clim_trend.py` | yes |
| `metrics/S41_aurora/s41eval/sci/helperB_style.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/sci/helperB_style.py` | no |
| `metrics/S41_aurora/s41eval/sci/helperc_style.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `s2storm/sci/helperc_style.py` | no |
| `metrics/S41_aurora/s41eval/hindcast_grid.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `s2storm/models/ucs2s_hindcast.py` | yes |
| `metrics/S41_aurora/scripts/bing_ablation_compare.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_ablation_compare.py` | yes |
| `metrics/S41_aurora/scripts/bing_ablation_mjo.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_ablation_mjo.py` | yes |
| `metrics/S41_aurora/scripts/bing_ablation_response.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_ablation_response.py` | yes |
| `metrics/S41_aurora/scripts/bing_eval.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_eval.py` | yes |
| `metrics/S41_aurora/scripts/bing_eval_stage.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_eval_stage.py` | yes |
| `metrics/S41_aurora/scripts/bing_headline_mjo.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_headline_mjo.py` | no |
| `metrics/S41_aurora/scripts/bing_helperA.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_helperA.py` | yes |
| `metrics/S41_aurora/scripts/bing_helperA_figs.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_helperA_figs.py` | yes |
| `metrics/S41_aurora/scripts/bing_helperA_mjo.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_helperA_mjo.py` | no |
| `metrics/S41_aurora/scripts/bing_reseed_compare.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_reseed_compare.py` | yes |
| `metrics/S41_aurora/scripts/bing_response_test.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_response_test.py` | no |
| `metrics/S41_aurora/scripts/bing_scores.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_scores.py` | yes |
| `metrics/S41_aurora/scripts/bing_slides_main.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_slides_main.py` | no |
| `metrics/S41_aurora/scripts/bing_stage_era5.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_stage_era5.py` | yes |
| `metrics/S41_aurora/scripts/bing_w7_extras.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_w7_extras.py` | no |
| `metrics/S41_aurora/scripts/bing_weak.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_weak.py` | yes |
| `metrics/S41_aurora/scripts/bing_weak_figs.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `scripts/bing_weak_figs.py` | yes |
| `metrics/S41_aurora/scripts/bing_modes.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/bing_modes.py` | yes |
| `metrics/S41_aurora/scripts/bing_modes_figs.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/bing_modes_figs.py` | yes |
| `metrics/S41_aurora/scripts/bing_modes_score.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/bing_modes_score.py` | yes |
| `metrics/S41_aurora/scripts/bing_modes_stage.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/bing_modes_stage.py` | yes |
| `metrics/S41_aurora/scripts/bing_modes_summary.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/bing_modes_summary.py` | yes |
| `metrics/S41_aurora/scripts/bing_waves.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/bing_waves.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c01_mjo.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c01_mjo.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c02_hov.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c02_hov.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c03_waves.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c03_waves.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c04_tele.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c04_tele.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c05_strat.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c05_strat.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c06_enso.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c06_enso.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c07_era5exp.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c07_era5exp.py` | yes |
| `metrics/S41_aurora/scripts/helperc_c07_windows.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_c07_windows.py` | yes |
| `metrics/S41_aurora/scripts/helperc_figs.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `scripts/helperc_figs.py` | yes |
| `metrics/S41_aurora/tests/test_bing_eval.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `tests/test_bing_eval.py` | yes |
| `metrics/S41_aurora/tests/test_bing_helperA.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `tests/test_bing_helperA.py` | yes |
| `metrics/S41_aurora/tests/test_bing_scores.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `tests/test_bing_scores.py` | yes |
| `metrics/S41_aurora/tests/test_bing_weak.py` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `tests/test_bing_weak.py` | yes |
| `metrics/S41_aurora/tests/test_bing_modes.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `tests/test_bing_modes.py` | yes |
| `metrics/S41_aurora/tests/test_bing_waves.py` | github.com/jhayron-perez/s2storm | bing-modes-c | `e6e086500a` | `tests/test_bing_waves.py` | yes |
| `metrics/S41_aurora/configs/bing_eval/aurora.yaml` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `configs/bing_eval/aurora.yaml` | yes |
| `metrics/S41_aurora/configs/bing_eval/derecho.yaml` | github.com/jhayron-perez/s2storm | bing-eval | `3cdf4a9763` | `configs/bing_eval/derecho.yaml` | yes |

New files: `loaders/format_s41.py`, `metrics/S41_aurora/README.md`, `metrics/S41_aurora/PROVENANCE.md`, `metrics/S41_aurora/environment.yml`,
and empty `__init__.py` files for `s41eval/`, `s41eval/bing_eval/`, `s41eval/sci/` where the source had none.
