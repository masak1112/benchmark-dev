"""S41 evaluation package for Bing's hindcasts (UC-S2S SST-anomaly / plain, FuXi-S2S, IFS): brief
``research-os/projects/s2storm/briefs/2026-11-11_aurora_bing-evaluation.md`` and its plan (§1 "our version", §2 design, §3 A/D).

Modules
  core      grids (common conservative 1.5 deg g15, 3 deg g30), regions, windows, calendar, pure score functions
  data      readers (one per system file format, registry ``FORMATS``), ERA5 verification store, the two climatological references
  pipeline  config (YAML per site), resumable tasks (reduce / verif / score), collect (scores.csv, fail_summary.csv, provenance.json)
  stats     moving-block bootstrap, paired differences, grading rule (robust / suggestive / noise), headline helper
  report    summary.md tables and figures
CLI: ``scripts/bing_eval.py`` (see its docstring). Configs: ``configs/bing_eval/<site>.yaml``.
"""
