# Provenance (one row per file)

Root on Aurora: `<root>` = `/lus/flare/projects/S2Storm/s2storm`. "s2storm" = `github.com/jhayron-perez/s2storm`. All `code/` files are
`git show 0b3e213:<path>` (byte-identical to what the production jobs ran; the worktree `<root>/code/s41/run_hc` was clean). No file was
edited when copying. Secret scan before the push: no keys, tokens or passwords. Paths are the Aurora project tree, the DSI project tree
(`/net/spaces/monsoon/jhayron/s2storm`, defaults in `paths.py` / `ucs2s.py` not used on Aurora), Bing's DSI checkpoint/data dirs as
launcher defaults, Bing's original training paths inside her configs/MANIFESTs, and ALCF system paths in logs.

## Our code (s2storm 0b3e213)
| file | source path | last commit touching it (history) | what it changes relative to upstream |
|---|---|---|---|
| `code/s2storm/models/ucs2s.py` | `s2storm/models/ucs2s.py` | d7cede4 (d7cede4 a32940a 018a20c c12b92c 16cfadd 77ef79e e5fa76a b95c53d 567e2de e574657 6b14e8e) | launcher. No upstream file edited; behaviour changed by subclass/monkeypatch: **OLR sign fix d7cede4** (`_install_olr_fix`, `save_prediction`), **strict-load report a32940a** (`_strict_report`, env `S2STORM_STRICT_REPORT`), **`sst_clim_path` from `<ckpt_dir>/config` and `checkpoint_path_diff` from the bundle's fallback folder 018a20c**, hindcast save mode c12b92c/a32940a, init-date calendar patch 77ef79e (`dates_all` = the requested dates, returned on upstream's 2019 loop iteration), checkpoint/data paths on `params`, own `dist.init_process_group`, timings (XPU sync e5fa76a) |
| `code/s2storm/models/ucs2s_xpu.py` | `s2storm/models/ucs2s_xpu.py` | 3e5ad58 (3e5ad58 af57c6f 40a9fd0 d1a413a) | XPU monkeypatches (torch.cuda.* → xpu, torch.load map_location, DDP device ids, autocast, YParams transformer_engine off, Trainer.device), world size from mpi4py under PALS (af57c6f), `--mode dsi` + `S2STORM_MEMBER_SHARD` |
| `code/s2storm/models/ucs2s_hindcast.py` | same | a32940a (a32940a c12b92c) | 27-field writer (`*_h25.nc`): new output file, upstream file untouched |
| `code/s2storm/models/ucs2s_ckpt.py` | same | 567e2de | `--check_ckpts` only (not used in production) |
| `code/s2storm/aurora/taskfarm.py` | same | **566fdf0** (566fdf0 c16ef2e af57c6f 77a7619) | farm; **claim fix 566fdf0** (heartbeat claims, locked takeover, retry limit, lost-claim kill); c16ef2e strips PMIx/PALS env from tasks + per-task timeout |
| `code/s2storm/data/era5_h5.py` | same | 6110388 (6110388 ee24a53) | ERA5 ARCO → UC-S2S h5 builder (hybrid regrid ee24a53, range sharding 6110388). Blob fe7d3b0 = the file in `code/wt-bing` that built 2019–2024 |
| `code/s2storm/{__init__,device,paths}.py`, `models/__init__.py` | same | ddc4786 | helpers (device selection, `xccl` backend name, data root) |
| `code/s2storm/aurora/__init__.py` | same | 77a7619 | empty |
| `code/s2storm/data/__init__.py` | same | ee24a53 | docstring only |
| `code/scripts/bing_hc_tasks.py` | same | 0b3e213 (0b3e213 bb29926) | task-list generator; `--ckpt_root` (0b3e213) = per-node /dev/shm checkpoints |
| `code/scripts/bing_start_list.py` | same | 791545c | start-list generator |
| `code/tests/test_taskfarm_claims.py` | same | 566fdf0 | CPU tests of the claims |
Not a commit: **`PYTHONPYCACHEPREFIX`** is an env var in `hpc/tile/hc_tile.sh` (`<root>/runs/bing/pycache`). It was added on 2026-11-11 so
that no stale `__pycache__` on the hung Flare OST 9 is read. It does not change the code.
Later commits on `bing-hindcasts` (ablation hooks 3b22f49/2b2f607, reseed `--seed_offset` 0685220/95ee0d9) were **not** used by production.

## Upstream (not copied, used unmodified)
| what | where | commit |
|---|---|---|
| UC-S2S code | `<root>/ext/S2S` (clone of `masak1112/S2S`, branch `uc-s2s_v3_multicond`, clean) | db18364 |
| fix-test variant | `<root>/runs/bing/ext/S2S_fix_7ceed2a` (git worktree; db18364 + `cherry-pick -x 7ceed2a`) | 6982d54 |

## Job scripts and HPC files (`hpc/`)
| file | source path | notes |
|---|---|---|
| `hpc/pbs/s41_hc.pbs`, `s41_hc2.pbs`, `s41_hc0921d.pbs`, `chain2.sh`, `chain0921b.sh` | `<root>/runs/bing/hc/` | as run |
| `hpc/tile/hc_tile.sh` | `<root>/runs/bing/hc/hc_tile.sh` | as run (production) |
| `hpc/tile/hcfix/{hc_tile_fix.sh,fix.pbs,chain.sh}` | `<root>/runs/bing/hcfix/` | fix test, as run |
| `hpc/tile/debug/{uc_tile.sh,s41_dbg.pbs}` | `<root>/runs/bing/debug/` | debug job 9074094, as run |
| `hpc/env/install_runs-bing-env.sh` | `<root>/runs/bing/env/install_uc.sh` | renamed only |
| `hpc/env/runs-bing-env-ucs2s-xpu.freeze.txt` | `<root>/runs/bing/env/ucs2s-xpu.freeze.txt` | written at build (2026-11-11) |
| `hpc/env/envs-ucs2s-xpu.freeze.txt` | `pip freeze` of `<root>/envs/ucs2s-xpu`, taken 2026-11-23 | not the production env |
| `hpc/env/envs-ucs2s-xpu.pip_install.log` | `<root>/logs/pip_ucs2s.log` | build log 2026-10-31 |
| `hpc/env/constraints_frameworks.txt` | `<root>/envs/constraints_frameworks.txt` | |
| `hpc/jobs/logs/<job>.OU` | `<root>/logs/S41/<job>.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov.OU` | renamed only |
| `hpc/jobs/logs/S41u_*_20250106_m00.{log,state.json,timing.json}` | `<root>/runs/bing/hc/state/{logs/<id>.log,<id>.json,timing/<id>/rank0.json}` | |
| `hpc/jobs/logs/failed_attempts.jsonl` | `cat <root>/runs/bing/hc/state/*.failed` | 9 lines |

## Configs, checkpoints, data, starts, tasks
| file | source path | notes |
|---|---|---|
| `configs/exp16_nvidia_v3.yaml`, `configs/exp16_midway_sstanom.yaml` | `<root>/aurora_bundle_v5/checkpoints_{plain,sstanom}/config/` | Bing's backups; sha256 = `SHA256SUMS` = the `_c16` copies the jobs used |
| `configs/upstream_db18364/*.yaml` | `git -C <root>/ext/S2S show db18364:v2.0/config/<file>` | |
| `configs/*.diff_vs_db18364.txt`, `resolved_section_diff.txt` | generated 2026-11-23 (`diff -u`; PyYAML SafeLoader, anchors merged) | |
| `checkpoints/MANIFEST_*.md`, `SHA256SUMS_*` | `<root>/aurora_bundle_v5/checkpoints_{plain,sstanom}/` | Bing's text |
| `data/build_2019_2024.sh`, `data/env.sh` | `<root>/aurora_bundle_v5/logs/` | as run |
| `data/compare_hybrid.json` | `<root>/aurora_bundle_v5/test/compare_hybrid.json` | check vs Bing's DSI file 2020_0664.h5 |
| `data/h5_keys.txt`, `data/normalisation_sha256.txt` | generated 2026-11-23 from `<root>/aurora_bundle_v5/h5data` | |
| `starts/starts.csv`, `starts/ecmwf_hindcast_dates_*.yyyymmdd` | `<root>/runs/bing/starts/` | date lists from Bing's `intial_dates.zip` |
| `tasks/tasks_sample.jsonl` | first member-0 line per model from `tasks.jsonl`, `tasks_2019-24.jsonl`, `tasks_20260921.jsonl` in `<root>/runs/bing/hc/` (key `_file` added) | the full pools (25,080 lines) are not copied; sha256 in `tasks/task_files_sha256.txt` |

## Written for this package
`README.md`, `PROVENANCE.md`, `KNOWN_ISSUES.md`, `hpc/README.md`, `selftest/selftest.pbs`, `selftest/compare_selftest.py` (2026-11-23).
