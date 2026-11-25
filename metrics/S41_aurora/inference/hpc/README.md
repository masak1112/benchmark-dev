# HPC layer as run (Aurora, ALCF), copied without cleanup

## Versions (for cross-server comparison)
| item | version | source |
|---|---|---|
| module | `frameworks/2026.1.0` (loaded after `module avail frameworks`; bare `module load frameworks` loaded nothing on login nodes) | job scripts |
| oneAPI | `oneapi/release/2026.1.0`, `intel-oneapi-runtime/2026.1.0`; GPU driver module `intel_gpu_umd_aicoe/2026.06.19`; `mpich/prd/5.0.0.aurora_test.87e2045`; `hdf5/1.14.6` | `module list` / `$LOADEDMODULES` after loading (login node, 2026-11-23) |
| Python | 3.12.12 (frameworks miniforge) | venv `pyvenv.cfg`, job logs |
| torch | **2.13.0a0+gitcf30153** (ALCF build "pytorch_2.13.0_patched_08_02_2026"), `torch.version.xpu` = 20260100 | job logs ("Torch version: 2.13.0a0+gitcf30153"), import on login node |
| intel-extension-for-pytorch | **not installed / not used** (native torch XPU backend) | import test |
| oneccl_bindings_for_pytorch | not installed; process group backend **xccl** (single process per task, world size 1) | import test; state json `"backend": "xccl"` |
| triton-xpu | 3.7.2 | import test |
| numpy / scipy / h5py / mpi4py | 2.3.5 / 1.18.0 / 3.16.0 / 4.1.2 | import test |
| venv extras (timm, xarray, netCDF4, …) | `env/runs-bing-env-ucs2s-xpu.freeze.txt` (timm 1.0.14, xarray 2026.9.0, netCDF4 1.7.5.1.post1, cftime 1.6.6.1, cf_xarray 0.11.3, zarr 3.4.2) | `pip freeze` written at build |
| device | Intel Data Center GPU Max 1550, one tile per process (64 GiB), `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | state json `device_name` |

## Precision / backend settings in the inference path
- dtype: fp32 throughout. `torch.get_default_dtype()` = float32. Upstream casts inputs to float32 (`inference_postprocess_si.py:530-532`).
  No autocast in `inference_postprocess_si.py`. Our autocast remap (`ucs2s_xpu.py`, autocast cuda → xpu) only matters for training code.
- `train.py:47-49` (imported by upstream) runs `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`
  and **`torch.set_float32_matmul_precision('high')`** at import. The first two are CUDA-only. Whether 'high' changes fp32 matmul on XPU
  in this torch build was not tested [inferred: it may allow TF32-class math on XPU].
- `ucs2s.py:197`: `torch.backends.cudnn.benchmark = True` (`--deterministic` not passed). It is CUDA-only.
- `torch.use_deterministic_algorithms` was not set in production. Same seed → bitwise identical outputs across processes/tiles was
  observed for UC-S2S on this stack (control rerun job 9077592 = production member 0, 27 fields bitwise).
- XPU / oneAPI env (set by the frameworks module, seen on the login node 2026-11-23; not dumped inside the jobs):
  `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`, `ZE_ENABLE_API_TRACING=1`,
  `CCL_PROCESS_LAUNCHER=pmix`, `CCL_CONFIGURATION=cpu_gpu_dpcpp`, `OMP_TARGET_OFFLOAD=MANDATORY`, `TORCH_CPP_LOG_LEVEL=ERROR`.
  Set by our scripts: `ZE_AFFINITY_MASK=<local rank % 12>`, `OMP_NUM_THREADS=8`, `HDF5_USE_FILE_LOCKING=FALSE`. `MKL_NUM_THREADS` and
  other MKL variables were **not set**.
- CUDA → XPU monkeypatches (no upstream edits): list in the docstring of `../code/s2storm/models/ucs2s_xpu.py` and in each task's state json
  (`jobs/logs/*.state.json`, key `patches`).

## Tile / farm scripts (`tile/`, `pbs/`)
| file | used by | what |
|---|---|---|
| `tile/hc_tile.sh` | production jobs 9074289, 9075362, 9075522, 9076613 | one rank per tile: `ZE_AFFINITY_MASK = PALS_LOCAL_RANKID % 12`. Local rank 0 copies both checkpoint bundles to `/dev/shm/s41` and the other ranks wait for `READY` (≤ 900 s). Then venv, env vars, and `python -m s2storm.aurora.taskfarm`. `PALS_*` stay set for the farm, which reads its rank from them; the farm strips `PALS_ PMI_ PMIX_ PMI MPI_LOCAL OMPI_` from every task (`taskfarm.py:37-39`) |
| `pbs/s41_hc.pbs`, `pbs/s41_hc2.pbs`, `pbs/s41_hc0921d.pbs` | the production jobs | `mpiexec -n NODES*12 --ppn 12 --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100`: 12 processes per node, 8 physical cores each, cores 49-52 and 101+ unused. `S2STORM_DEADLINE` from the walltime |
| `pbs/chain2.sh`, `pbs/chain0921b.sh` | login-node chains that submitted 9075522 and 9076613 | wait, count tasks left, budget check (guard script not included), `qsub` |
| `tile/hcfix/hc_tile_fix.sh`, `fix.pbs`, `chain.sh` | fix test (jobs 9097167, 9097212, 9097295) | same as `hc_tile.sh` with plain checkpoints only, `UCS2S_REPO = runs/bing/ext/S2S_fix_7ceed2a` (db18364 + 7ceed2a = 6982d54), own state and pycache dirs |
| `tile/debug/uc_tile.sh`, `s41_dbg.pbs` | debug job 9074094 (strict-load, claim test) | same pinning; one farm per node on a shared pool; `PALS_SIZE=1` |

Processes per node: 12 (one per tile). Each task (one model × start × member) is one Python process on one tile. Per member, from
`jobs/logs/*.timing.json`: plain 45 base steps × 20.6 s + 45 pp steps × 1.90 s; 1,031 s `predict()`; peak 7.2 GB on the tile.
SST-anomaly: 20.8 s + 1.88 s per step, 1,037 s.

## Environments (`env/`)
Both are `python -m venv --system-site-packages` on top of frameworks/2026.1.0, so torch, numpy etc. are the frameworks builds.
Installs use `-c constraints_frameworks.txt` (frameworks pins written as `name==version`).
- **`runs/bing/env/ucs2s-xpu` = the one production used** (`hc_tile.sh` activates it). Built 2026-11-11 during the Flare OST 9/13
  outage, with `--without-pip`, installed with the frameworks pip: `env/install_runs-bing-env.sh` (its first line installs openvino
  into the FuXi env and does not concern UC-S2S). Freeze `env/runs-bing-env-ucs2s-xpu.freeze.txt`. Our package is not pip-installed
  here; it comes in through `PYTHONPATH=$C`.
- `envs/ucs2s-xpu` = the earlier env (2026-10-31, same package versions; log `env/envs-ucs2s-xpu.pip_install.log`). Freeze taken
  2026-11-23: `env/envs-ucs2s-xpu.freeze.txt`. Differences from the production env: no gcsfs/google-*/h5netcdf/grpcio-status, fsspec
  2026.6.0 vs 2026.9.0, grpcio 1.83.0 vs 1.85.1, and an editable install of `jhayron-perez/s2storm`. Production did not use this env.

## Job records (`jobs/`)
PBS history (`qstat -xf`) has expired for these jobs ("Unknown Job Id" on 2026-11-23). Values below come from the job scripts, the job
logs (`jobs/logs/<job>.OU`: code commit, node count, start/end, tasks done) and the S41 experiment record (elapsed and node-h, taken
from PBS on 2026-11-11).
| job | queue (requested → ran) | nodes | walltime requested | elapsed | node-h | start → end (UTC, from .OU) | pool / done |
|---|---|---|---|---|---|---|---|
| 9074289 | prod → small | 256 | 0:45:00 | 00:38:37 | 164.8 | 11-11 07:40:23 → 08:18:54 | `tasks.jsonl`; state 6,144 done after the job |
| 9075362 | prod | 256 | 1:05:00 | 00:58:12 | 248.3 | 11-11 12:26:00 → 13:23:51 | `tasks_2019-24.jsonl`; 15,360 done (all pools) |
| 9075522 | prod | 256 | 1:05:00 | 00:57:41 | 246.1 | 11-11 13:40:50 → 14:38:03 | same pool; 24,120 done (all pools) |
| 9076613 | debug | 2 | 1:00:00 | 00:53:19 | 1.78 | 11-11 19:36:23 → 20:29:15 | `tasks_20260921.jsonl`; 24,180 done (all pools) |
- 9 task attempts failed with exit 1 after 2–29 s and succeeded on retry (`jobs/logs/failed_attempts.jsonl`: job, host, tile). The per-task
  log is overwritten by the retry, so the cause of those 9 is not recorded.
- Member logs (stdout+stderr of the task, start to end): `jobs/logs/S41u_plain_20250106_m00.log` and `S41u_sstanom_20250106_m00.log` (job
  9074289), with the task's `*.state.json` (host, tile, patches, argv) and `*.timing.json`. Upstream logs no line per step. In order (sstanom log): "Base model: latent=si, latent_cond=sst, sst_cond_mode=persistent, mc_dropout=True,
  temperature=1.0, feedback=base" (line 195), "Base model loaded from …rollout_ckpt_650.tar (epoch=2, iters=650)" (199), "Postprocess SI loaded …
  (iters=11400, 96 fields, max_lead_steps=10, blend=1.0)" (204), "rank 0: 1 init dates x 1 members [0]" (209), then nothing until
  "batch 0 done (1037s elapsed)" (213). Upstream's
  rank 0 inside a task is member m (`world_rank` set to m by the shard).
