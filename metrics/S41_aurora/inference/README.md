# S41 Aurora UC-S2S hindcasts — inference package

Read-only copy of the code, job scripts, commands and input recipe used for the production UC-S2S hindcasts run on
ALCF Aurora on 2026-11-11 (project S2Storm), for reproducing the runs and for looking into the +1.1 K day-1 SST jump
(`KNOWN_ISSUES.md`). Nothing here edits upstream `masak1112/S2S`. The upstream code is used unmodified from a clone; all
changes are wrappers and monkeypatches in our package (`code/s2storm/`). No data or checkpoints are included, only their
paths and sha256 sums.

## What was run
| | plain UC-S2S | SST-anomaly UC-S2S |
|---|---|---|
| config / section | `exp16_nvidia_v3.yaml` / `S2S` | `exp16_midway_sstanom.yaml` / `S2S_SSTANOM` |
| base checkpoint | `rollout_ckpt_1000.tar` (epoch 12, iters 1000) | `rollout_ckpt_650.tar` (epoch 2, iters 650) |
| postprocess SI | `pp_si_ckpt_10800.tar` | `pp_si_ckpt_11400.tar` |
| starts | 403 of the 404 in `starts/starts.csv` (2019-01-07 → 2026-09-21; 2026-09-28 not run, ERA5 not yet available) | same 403 |
| members | 30 per start (0–29), one task per (model, start, member) | same |
| leads | 45 daily steps (`--inference_steps 45`), init 00 UTC, output lead 0…45 | same |

- Count from the farm's state files (`runs/bing/hc/state/S41u_<model>_<YYYYMMDD>_m<MM>.done`): 403 starts × 30 members per model,
  no partial starts. Earlier notes say "402 of 404". That was before 2026-09-21 was run (job 9076613). See KNOWN_ISSUES §6 for that start.
- Starts: weekly. For each week (Fri–Thu around a Monday) the ECMWF reforecast date closest to the Monday is used, or the
  Monday itself if there is none. All of 2025–26 are Mondays. Generator `code/scripts/bing_start_list.py` (s2storm 791545c), input
  `starts/ecmwf_hindcast_dates_<2019..2024>.yyyymmdd` (from Bing's `intial_dates.zip`), output `starts/starts.csv`.

## Upstream code
- Repo `github.com/masak1112/S2S`, branch `uc-s2s_v3_multicond`, commit **db18364** (`db18364a9019a1c56caa1ad4910233a1d8a0096b`,
  2026-10-30 "Add DSI multicond inference scripts and ECMWF hindcast date sets"). The Aurora clone is `/lus/flare/projects/S2Storm/s2storm/ext/S2S`,
  clean working tree (`git status` empty). At the last fetch db18364 is still the head of `origin/uc-s2s_v3_multicond`.
- Model script: `v2.0/inference_postprocess_si.py`, class `PostprocessInference`. It is imported (not run as `__main__`) and
  subclassed by our launcher. Its argument parser is reused verbatim (`code/s2storm/models/ucs2s.py:265-280`).

## Call chain (production)
```
qsub hpc/pbs/s41_hc.pbs  (or s41_hc2.pbs / s41_hc0921d.pbs; job scripts below)
 └─ mpiexec -n NODES*12 --ppn 12 --cpu-bind=list:1-8:9-16:...:93-100   bash hpc/tile/hc_tile.sh        # one rank per XPU tile
     └─ hc_tile.sh: ZE_AFFINITY_MASK = local rank; local rank 0 copies both checkpoint bundles to /dev/shm/s41 once per node;
        activates the venv, sets env; python -m s2storm.aurora.taskfarm --tasks <pool>.jsonl --state runs/bing/hc/state --min_left 1250
         └─ taskfarm (code/s2storm/aurora/taskfarm.py): claims one task (heartbeat claim file), strips PALS_/PMI*/PMIX_ env,
            runs the task's `cmd` through bash, writes <id>.done on exit 0
             └─ S2STORM_MEMBER_SHARD=<m>/30 python -m s2storm.models.ucs2s_xpu --mode dsi ... -- <launcher args> -- <upstream args>
                 └─ ucs2s_xpu.run_dsi (code/s2storm/models/ucs2s_xpu.py:696-729): XPU monkeypatches, sets upstream module
                    globals world_rank=m, world_size=30, calls s2storm.models.ucs2s.main
                     └─ ucs2s.main (code/s2storm/models/ucs2s.py:122-262): sets paths on params, patches the init-date calendar,
                        builds S2StormInference(PostprocessInference) and calls upstream predict()
                         └─ upstream v2.0/inference_postprocess_si.py PostprocessInference.predict (lines 489-631):
                            member list range(world_rank, num_members, world_size) = [m]; seed 100000*batch_idx + m;
                            45 x (_base_step -> _apply_correction); save_prediction (overridden by us: OLR sign fix + 27-field file)
```

## Software versions (Aurora; details and precision settings in `hpc/README.md`)
| item | version |
|---|---|
| module | `frameworks/2026.1.0` |
| oneAPI | 2026.1.0 (`oneapi/release/2026.1.0`); GPU driver module `intel_gpu_umd_aicoe/2026.06.19`; MPICH 5.0.0 (aurora_test.87e2045) |
| Python | 3.12.12 |
| torch | 2.13.0a0+gitcf30153 (native XPU backend; `torch.version.xpu` 20260100) |
| intel-extension-for-pytorch | not installed / not used |
| process group | xccl, world size 1 per task |
| numpy / scipy / h5py | 2.3.5 / 1.18.0 / 3.16.0 |
| timm / xarray / netCDF4 | 1.0.14 / 2026.9.0 / 1.7.5.1.post1 |
| precision | fp32, no autocast; upstream `train.py` sets `torch.set_float32_matmul_precision('high')` at import (effect on XPU not tested) |
| device | Intel Data Center GPU Max 1550, 1 tile (64 GiB) per member process, 12 per node |

## Our code (copied byte-identical from s2storm commit **0b3e213**)
All jobs ran code commit **0b3e213** (`bing_hc_tasks: --ckpt_root`). Each job log prints it:
`logs/S41/<job>.OU: "job … code 0b3e213"` for 9074289, 9075362, 9075522, 9076613. The jobs ran the detached worktree
`code/s41/run_hc` (clean) of `jhayron-perez/s2storm`. 0b3e213 is on branch `bing-hindcasts` (later commits on that branch are
ablation/reseed hooks that production did not use). Files and their history: `PROVENANCE.md`.

| file | role |
|---|---|
| `code/s2storm/models/ucs2s_xpu.py` | XPU entry point; device monkeypatches (docstring lists every CUDA use patched); `--mode dsi` + `S2STORM_MEMBER_SHARD` |
| `code/s2storm/models/ucs2s.py` | launcher: checkpoint/config paths, calendar patch, runner subclass (OLR fix, strict-load report, hindcast save mode, timings) |
| `code/s2storm/models/ucs2s_hindcast.py` | the 27-field writer (`*_h25.nc`; native 1° in production, `--hc_grid native`, zlib 4, lossless) |
| `code/s2storm/models/ucs2s_ckpt.py` | `--check_ckpts` (not used in production; imported only with that flag) |
| `code/s2storm/aurora/taskfarm.py` | task farm with heartbeat claims (566fdf0) |
| `code/s2storm/{__init__,device,paths}.py`, `models/__init__.py`, `aurora/__init__.py`, `data/__init__.py` | imported helpers |
| `code/s2storm/data/era5_h5.py` | input builder (ERA5 ARCO → UC-S2S `.h5`) |
| `code/scripts/bing_hc_tasks.py` | task-list generator (one task per model/start/member; input checks; OST pinning) |
| `code/scripts/bing_start_list.py` | start-list generator |
| `code/tests/test_taskfarm_claims.py` | CPU tests of the claim logic |

## Job scripts (`hpc/pbs/`, `hpc/tile/`, copied from `runs/bing/hc/` as run; details `hpc/README.md`)
| job | script | queue | nodes × walltime | elapsed | node-h | task pool | result |
|---|---|---|---|---|---|---|---|
| 9074289 | `hpc/pbs/s41_hc.pbs` | prod (→ small) | 256 × 0:45 | 00:38:37 | 164.8 | `tasks.jsonl` (6,240: 2025–26 both models, 89 starts × 30, + 900 SST-anomaly 2019 members) | 6,144 done |
| 9075362 | `hpc/pbs/s41_hc2.pbs` | prod | 256 × 1:05 | 00:58:12 | 248.3 | `tasks_2019-24.jsonl` (18,780 = 313 starts × 2 × 30) | part 1 |
| 9075522 | `hpc/pbs/s41_hc2.pbs` (submitted by `chain2.sh`) | prod | 256 × 1:05 | 00:57:41 | 246.1 | same pool | 18,780/18,780 |
| 9076613 | `hpc/pbs/s41_hc0921d.pbs` (submitted by `chain0921b.sh`) | debug | 2 × 1:00 | 00:53:19 | 1.78 | `tasks_20260921.jsonl` (60) | 60/60 |

Elapsed and node-h are from the S41 experiment record (`experiments/S41_bing-hindcasts/aurora.md` in research-os, taken from PBS).
Per-node and per-tile settings (from the scripts):
- `#PBS -A S2Storm -l filesystems=home:flare -j oe`; `module avail frameworks; module load frameworks/2026.1.0` (oneAPI 2026.1.0,
  Python 3.12.12, torch 2.13.0a0+gitcf30153 XPU); venv `runs/bing/env/ucs2s-xpu` (`--system-site-packages`; `hpc/env/install_runs-bing-env.sh`,
  `hpc/env/runs-bing-env-ucs2s-xpu.freeze.txt`, constraints `hpc/env/constraints_frameworks.txt`).
- 12 ranks per node, one per tile: `ZE_AFFINITY_MASK=<local rank>`; CPU binding 8 cores per rank
  (`--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100`); `OMP_NUM_THREADS=8`.
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` come from the frameworks module.
- Env: `S2STORM_CODE=$C` (`code/s41/run_hc`), `PYTHONPATH=$C`, `UCS2S_REPO=<root>/ext/S2S`,
  `UCS2S_CLIMATOLOGY=<root>/data/placeholder/climatology_placeholder.nc` (upstream opens `climatology_file` but inference never reads
  it), `S2STORM_CLIM=` (empty), `S2STORM_STATE=<root>/runs/bing/hc/state`, `HDF5_USE_FILE_LOCKING=FALSE`,
  `PYTHONPYCACHEPREFIX=<root>/runs/bing/pycache` (Lustre OST outage workaround: stale `__pycache__` files on a hung OST blocked
  imports), `S2STORM_DEADLINE` = job start + walltime − 120 s (the farm starts no task with < 1,250 s left).
- Checkpoints: per node, local rank 0 copies `aurora_bundle_v5/checkpoints_{sstanom,plain}_c16` (16-stripe Lustre copies of the
  bundles) to `/dev/shm/s41/checkpoints_{sstanom,plain}`; the task commands point `--ckpt_dir` and `--yaml` there.

## Exact commands (from the task files; `tasks/tasks_sample.jsonl` has real lines, `tasks/task_files_sha256.txt` the pools)
Only `<YYYYMMDD>`, `<YYYY-MM-DD>`, `<m>` and `<MM>` vary between the 24,000+ tasks of a model. The template was checked over all task files.

Plain (`S2S`):
```
S2STORM_MEMBER_SHARD=<m>/30 python -m s2storm.models.ucs2s_xpu --mode dsi --out $S2STORM_STATE/S41u_plain_<YYYYMMDD>_m<MM>.json -- \
  --run_name S41u_plain_<YYYYMMDD> --init_dates <YYYY-MM-DD> --save_mode hindcast --full_members 0 1 2 ... 29 \
  --yaml /dev/shm/s41/checkpoints_plain/config/exp16_nvidia_v3.yaml --section S2S \
  --base_ckpt_rel loaded/3_base_full_state_rollout_finetuned/rollout_ckpt_1000.tar --pp_ckpt_rel loaded/4_postprocess_si/pp_si_ckpt_10800.tar \
  --ckpt_dir /dev/shm/s41/checkpoints_plain --data_dir /lus/flare/projects/S2Storm/s2storm/aurora_bundle_v5/h5data \
  --out_root /lus/flare/projects/S2Storm/s2storm/runs/bing/hc/plain --timing_json $S2STORM_STATE/timing/S41u_plain_<YYYYMMDD>_m<MM> -- \
  --num_members 30 --inference_steps 45 --blend 1.0 --feedback base --base_latent si
```
SST-anomaly (`S2S_SSTANOM`): same with `sstanom`, `--yaml …/checkpoints_sstanom/config/exp16_midway_sstanom.yaml --section S2S_SSTANOM
--base_ckpt_rel loaded/3_base_full_state_rollout_finetuned_sstanom/rollout_ckpt_650.tar --pp_ckpt_rel loaded/4_postprocess_si/pp_si_ckpt_11400.tar
--ckpt_dir /dev/shm/s41/checkpoints_sstanom`, and the upstream args end with **`--sst_cond_mode persistent`** (passed explicitly).

Upstream flags not passed take upstream defaults (`inference_postprocess_si.py` parser): plain gets `--sst_cond_mode persistent`
(default), which does nothing for it because `sst_cond_anom` is not set in `S2S`. Neither model passes `--mc_dropout`, so it defaults
to `params.infer_mc_dropout` (absent in both sections) → **True** for `--base_latent si` (lines 117-120). `--latent_cond sst` (default),
no `--years`, `--max_dates`, `--save_vars` or `--no_correction`, no `--overwrite`. Base temperature = `base_temperature: 1.0` (config),
`pp_sample_steps: 20`, pp blend 1.0. Production passed `--full_members 0…29`, so every member also wrote the full upstream file. The
ensemble-statistics jobs later deleted the full files of members 1–29 (9075219, 9076383). Member 0's full file is kept.

### How a member id becomes a seed (and the repeat across starts)
- The farm gives each member its own process: `S2STORM_MEMBER_SHARD=m/30`. `ucs2s_xpu.run_dsi` sets upstream's module globals
  `world_rank = m`, `world_size = 30` (`ucs2s_xpu.py:703-713`), so upstream's `my_members = range(world_rank, num_members, world_size)`
  = `[m]` (`inference_postprocess_si.py:514`).
- Each process runs ONE start (`--init_dates <date>`; our calendar patch `ucs2s.py:103-119` makes `dates_all` = that one date), so the
  upstream date index `batch_idx` is **0** for every start (`inference_postprocess_si.py:522`).
- Seed = `100_000 * batch_idx + ens_id` = **m** (`inference_postprocess_si.py:565-567`): `torch.manual_seed(m)` and
  `torch.cuda.manual_seed_all(m)` (our patch redirects the latter to `torch.xpu.manual_seed_all`). It is set once per member before its
  45-step rollout. All later draws (latent SI, MC dropout, postprocess SI) consume that RNG stream.
- **Seeds therefore repeat across starts**: member m has seed m at every start, in both models. Bing's own runs use one date list per year
  (`--years`), so `batch_idx` differs per start there. Evidence and effect: `KNOWN_ISSUES.md` §2.

## Input data (`data/`)
- Files: `aurora_bundle_v5/h5data/<YYYY>_<NNNN>.h5`, NNNN = 6-hourly index from Jan 1 00 UTC; daily 00 UTC; one group `input` with
  **107 datasets** (106 fields + `time`; list `data/h5_keys.txt`); 180 × 360, 1°, lat −89.5…89.5; SST NaN on land (21,972 cells), filled
  to 270 K by the upstream loader (`mask_fill`). One `--data_dir` also holds the normalisation files (`data/normalisation_sha256.txt`).
- Builder: `code/s2storm/data/era5_h5.py` (last change 6110388; identical blob `fe7d3b0` in the build checkout and in 0b3e213), source
  ERA5 on Google ARCO `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (anonymous), 0.25° → 1° with
  `--regrid hybrid` (the default; point sample for land_sea_mask, geopotential_at_surface, SST, sea ice, swvl1, stl1; trapezoid area mean
  ×cos lat for all other fields; tp24 = sum of the 24 hourly accumulations ending at t).
- Command (driver `data/build_2019_2024.sh`, env `data/env.sh`, 6 strided shards on the login node, 2026-11-10 01:49 → 07:25 UTC):
  `python -u -m s2storm.data.era5_h5 --start 2019-01-01 --end 2024-12-31 --hours 0 --task <k> --ntasks 6 --skip_existing --out aurora_bundle_v5/h5data`
  (`--regrid hybrid` is the default and was not typed; the S41 report records the recipe as `--hours 0 --regrid hybrid --skip_existing`).
- Check against Bing's DSI file `2020_0664.h5` (2020-06-15 00 UTC) (`data/compare_hybrid.json`): 106 fields compared, **0 NaN-pattern mismatches**;
  the six point-sampled fields incl. **SST identical (max |Δ| = 0)**; area-mean fields within ≤ 0.19 % relative RMS (worst v at 5 hPa,
  0.194 %; t2m 0.015 %, skt 0.016 %); 19/106 pass `allclose`.
- Where each year came from (identified by inode of the hard links in `aurora_bundle_v5/h5data`):
  | years | files | origin |
  |---|---|---|
  | 2019–2024 | 2,192 | built on Aurora by the command above (`aurora_bundle_v5`) |
  | 2025 | 365 | `aurora_bundle_v3` = DSI build, S16 (same `era5_h5.py`, hybrid, `slurm/S16_build_h5.sbatch`, DSI jobs 2284627 / 2285675 / 2285676 [inferred: job-to-file mapping not checked per file]); transferred by the PI 2026-11-01, `sha256sum -c` OK |
  | 2026-03-01 … 10-25 (239) | 239 | `aurora_bundle_v2` = DSI build, S16 (DSI jobs 2284600 / 2284624 [inferred: same caveat]); same transfer and check |
  | 2026-01-01 … 02-28, 10-26 … 10-31 | 65 | built on Aurora from ARCO, same builder, into `aurora_bundle_v4` (2026-11-09) |
  | 2026-11-01 … 11-03 | 3 | built on Aurora into `aurora_bundle_v5` (`logs/build_2026nov.log`) |
  | 2026-11-04, 11-05 | (2) | built 2026-11-11 ≈ 18:4x UTC, **later quarantined** (11-04 all zeros, 11-05 tp24 incomplete; ARCO gap); used by the 2026-09-21 start only, see KNOWN_ISSUES §6 |
- Normalisation (`pangu_s2s_1979-2018_{mean,std,surface_mean,surface_std}.nc`, `lat.npy`, `lon.npy`): hard links to `aurora_bundle/h5data`,
  file dates 2025-11-05, from Bing's DSI store `/net/spaces/monsoon/S2S/h5data` [inferred: the copy step is not recorded]. sha256 in
  `data/normalisation_sha256.txt`.

## Checkpoints (`checkpoints/`: Bing's MANIFEST.md and SHA256SUMS of each bundle)
Bundles on Aurora: `/lus/flare/projects/S2Storm/s2storm/aurora_bundle_v5/checkpoints_{plain,sstanom}` (hard links of Bing's DSI
backups; `_c16` = 16-stripe copies, the ones the jobs copied to `/dev/shm`). sha256 of every file in the `_c16` copies recomputed on
2026-11-23. **All 15 match `SHA256SUMS`.** Strict-load report from debug job 9074094 (`S2STORM_STRICT_REPORT`, `ucs2s.py:340-367`):
571/571 keys, 0 missing, 0 unexpected, 0 shape mismatches, all tensors equal after load, PASS for both.

| model | file (under the bundle) | config key / flag | sha256 (first 16 hex) | notes |
|---|---|---|---|---|
| plain | `loaded/1_vae_c1/ckpt.tar` | `checkpoint_path_vae_c1` | `daf8205658ba325c` | SST VAE encoder (component 8 = SST); strict load |
| plain | `loaded/2_pangu_deterministic/ckpt.tar` | `checkpoint_path_det` | `8966afaf12fffd1e` | Pangu encoder/decoder; its model_det is overridden by the base checkpoint |
| plain | `loaded/3_base_full_state_rollout_finetuned/rollout_ckpt_1000.tar` | `checkpoint_path_finetuned` (`--base_ckpt_rel`) | `a23dd7ecf3435a72` | epoch 12, iters 1000; 571 keys; strict=False, loaded last |
| plain | `loaded/4_postprocess_si/pp_si_ckpt_10800.tar` | `--pp_ckpt_rel` | `6a296950ff64af80` | iters 10800, 96 fields, max_lead_steps 10 (MANIFEST) |
| plain | `fallback_not_loaded/latent_si_c1_full_state/last_ckpt.tar` | `checkpoint_path_diff` | `29968d7db62bd0b0` | latent-SI fallback, read only if the base checkpoint is missing. **Not in the `_c16` copy**, so in production `checkpoint_path_diff` pointed at a non-existent file (`ucs2s.py:160-161`; never read) |
| plain | `config/exp16_nvidia_v3.yaml` | `--yaml` | `7fed08f539721c82` | section `S2S` |
| sstanom | `loaded/1_vae_c1/ckpt.tar` | `checkpoint_path_vae_c1` | `daf8205658ba325c` | same file as plain; loaded (strict) but bypassed: `sst_cond` is given (`diffusion.py:780-781`) |
| sstanom | `loaded/2_pangu_deterministic/ckpt.tar` | `checkpoint_path_det` | `8966afaf12fffd1e` | same file as plain |
| sstanom | `loaded/3_base_full_state_rollout_finetuned_sstanom/rollout_ckpt_650.tar` | `checkpoint_path_finetuned` (`--base_ckpt_rel`) | `bd7913d2e0fe77a5` | epoch 2, iters 650; 571 keys |
| sstanom | `loaded/4_postprocess_si/pp_si_ckpt_11400.tar` | `--pp_ckpt_rel` | `9ac9646da8118f6c` | iters 11400, warm-started from pp_si_ckpt_10800 (MANIFEST) |
| sstanom | `fallback_not_loaded/latent_si_sstanom_stage1/last_ckpt.tar` | `checkpoint_path_diff` | `0230cb67afebb163` | stage-1 latent SI; not in `_c16`; not read |
| sstanom | `config/clim_00z_sst_std.npy` | `sst_clim_path` (set by `ucs2s.py:162-164`) | `08b5cbfca4d36edb` | 366 × 180 × 360 daily SST climatology in standardized units, NaN on land (MANIFEST) |
| sstanom | `config/exp16_midway_sstanom.yaml` | `--yaml` | `d4f9f99df0fcc280` | section `S2S_SSTANOM`: `sst_cond_anom: True`, `sst_anom_scale: 0.0583`, `sst_cond_size: [23, 45]`, `vae_cond_mode: mean` |

Full hashes: `checkpoints/SHA256SUMS_<model>` (`_c16` recomputation identical).

## Configs (`configs/`)
- `exp16_nvidia_v3.yaml`, `exp16_midway_sstanom.yaml`: the bundle copies used (`--yaml`).
- `upstream_db18364/`: the same files at db18364. `*.diff_vs_db18364.txt`: full-file `diff -u`. `resolved_section_diff.txt`: the
  run's section with YAML anchors merged, key by key.
  - `S2S`: only `checkpoint_path_diff` / `checkpoint_paht_diff` (typo key upstream), `checkpoint_path_finetuned` (both overridden by
    the launcher, `ucs2s.py:155-165`) and **`train_year_start` 1984 (bundle) vs 1979 (upstream)**. Inference reads
    `train_year_start` only through the training data loader, which inference does not build [inferred from `train.py:271-300`, not tested].
    Upstream also has `S2S_SSTCOND` / `S2S_SSTANOM` sections in this file; the bundle copy does not.
  - `S2S_SSTANOM`: no differences (upstream additionally has `S2S_MULTICOND`).
- The launcher overrides on top of the yaml (`ucs2s.py:147-187`): `checkpoint_path_vae_c1`, `checkpoint_path_det`,
  `checkpoint_path_diff`, `checkpoint_path_finetuned`, `sst_clim_path` (SST-anomaly only), `cond_anom_dir`, `data_dir`, `climatology_file`
  (placeholder), `inference_steps`, `init_date_set = None`, `exp_dir`, `run_num`, `world_size`, `local_rank`, `log_to_wandb = False`.

## Self-test (`selftest/`, not run)
`selftest/selftest.pbs` runs 1 start × 1 member of the plain model on one debug node (≤ 0:15 walltime = ≤ 0.25 node-h) from the
packaged `code/`, against the same upstream clone, checkpoints and inputs. `selftest/compare_selftest.py` then checks the 27 fields
against production member 0 of 2025-01-06 (expected bitwise identical: same seed, same device type). Not run yet. It needs a
debug-node slot.

## Other files
`hpc/` (tile and job scripts, environments, job records and two full member logs; `hpc/README.md`), `PROVENANCE.md` (every file: source path, commit, differences from upstream), `KNOWN_ISSUES.md` (facts with evidence, incl. where
step 0 gets its SST). Evaluation code: `../` (`metrics/S41_aurora/`, commit f433b3d).
