"""UC-S2S (masak1112/S2S, branch uc-s2s_v3_multicond, config S2S_MULTICOND) on any device, incl. Aurora XPU.

The upstream repo (paths.ucs2s_repo(), i.e. ext/S2S) is never edited. Device handling is done
here by monkeypatches, applied before any upstream module is imported.

Modes
  random : build the S2S_MULTICOND model exactly as inference_postprocess_si.py does
           (VAE + Pangu det encoder/decoder from Trainer.get_model, StochasticInterpolant,
           optionally PostprocessSI) with RANDOM weights and synthetic inputs (no checkpoints,
           no h5 data), then roll out --steps autoregressive steps with the upstream step code
           (PostprocessInference._base_step / _apply_correction) and time every step.
  real   : run upstream inference_postprocess_si.py (its __main__) via runpy with the patches
           applied; everything after "--" is passed to it unchanged.

Examples (compute node, after `module load frameworks/2026.1.0` and the ucs2s-xpu venv):
  # one tile
  python -m s2storm.models.ucs2s_xpu --mode random --steps 45 --with_pp --out $S2STORM_DATA/runs/ucs2s/random_1tile.json
  # two ranks on one node, one tile each (rank r -> tile r)
  mpiexec -n 2 --ppn 2 python -m s2storm.models.ucs2s_xpu --mode random --ranks --members 2 --steps 45 --with_pp \
      --out $S2STORM_DATA/runs/ucs2s/random_2ranks.json
  # later, real checkpoints + h5 data (cwd for upstream is v2.0/HPC_scripts unless --workdir)
  mpiexec -n 12 --ppn 12 python -m s2storm.models.ucs2s_xpu --mode real --ranks -- \
      --yaml_config=../config/exp16_midway_sstanom.yaml --config=S2S_MULTICOND --run_num=aurora_test \
      --base_ckpt=... --pp_ckpt=... --init_date_set fuxi_s2s --years 2019 --max_dates 1 \
      --num_members 12 --inference_steps 45 --blend 1.0 --feedback base --base_latent si --sst_cond_mode evolving
Login node (CPU, build + count params only):
  python -m s2storm.models.ucs2s_xpu --mode random --device cpu --steps 0 --with_pp --out /tmp/ucs2s_build.json

Hard CUDA / NCCL uses found on the code path we exercise (grep cuda|nccl|GPUtil|nvidia|transformer_engine
in ext/S2S/v2.0) and what is done about each:
  train.py:47-48    torch.backends.cuda/cudnn.allow_tf32 = True            no-op off CUDA, left alone
  train.py:49       torch.set_float32_matmul_precision('high')             left alone (may allow TF32 on XPU)
  train.py:50       torch.cuda.empty_cache() at import                     -> torch.cuda.* redirect (below)
  train.py:53       dist.init_process_group(backend='nccl') at import, unconditional
                                                                           -> init_process_group patch: our backend,
                                                                              no-op if already initialised
  train.py:177      Trainer.device = torch.cuda.current_device() if cuda else 'cpu'
                    (on XPU this silently puts the whole model on the CPU) -> Trainer.device becomes a class
                                                                              property returning our device
  train.py:401-408  DistributedDataParallel(device_ids=[local_rank], output_device=[local_rank])
                                                                           -> DDP subclass: device_ids = our tile
                                                                              index (None on CPU)
  train.py:812      autocast(device_type="cuda") (training only)           -> torch.amp.autocast / torch.autocast /
                                                                              torch.cuda.amp.autocast remapped
  train.py:1278-9   torch.load(map_location='cuda:N') (restore_checkpoint)  -> torch.load patch
  train.py:1339-74  torch.cuda.device_count/get_device_name/memory_*/set_device (train __main__, not run)
  inference_postprocess_si.py:76       dist.init_process_group('nccl') at import  -> as train.py:53
  inference_postprocess_si.py:187,200,220  torch.load(map_location=f'cuda:{local_rank}') -> our device
  inference_postprocess_si.py:567      torch.cuda.manual_seed_all(seed)   -> torch.xpu.manual_seed_all
  inference_postprocess_si.py:835      torch.cuda.device_count() as WORLD_SIZE default -> xpu count (WORLD_SIZE
                                                                              is set from PMI/PALS anyway)
  inference_postprocess_si.py:842      torch.cuda.set_device(local_rank)  -> xpu.set_device(our tile)
  inference_postprocess_si.py:843      torch.backends.cudnn.benchmark     no-op off CUDA, left alone
  networks/stochastic_interpolant.py:533  torch.cuda.manual_seed_all (only when seed is passed) -> xpu
  networks/pangu.py:114-118, networks/pangu_vae.py:114-117  `import transformer_engine` + `from torch.cuda
                    import amp` when params.use_transformer_engine (False in S2S_MULTICOND)
                                                                           -> YParams patch forces it False when
                                                                              transformer_engine is not importable
  networks/pangu.py:1118,1126  F.scaled_dot_product_attention, 6-D q/k/v + float attn_mask
                    (portable op; whichever XPU SDPA kernel torch picks -- a likely first failure point)
  utils/data_loader_multifiles.py:103,837  pin_memory=torch.cuda.is_available() -> False on XPU (perf only)
  utils/utils.py:43-54  torch.cuda.memory_* in log_gpu_memory (decorator is commented out) -> redirected anyway
  No .cuda() calls, GPUtil, or nvidia-smi in the python inference path (nvidia-smi only in HPC_scripts/*.sh).
  Tensor.cuda / Module.cuda are redirected to our device anyway. Inference runs in fp32 without autocast
  (inference_postprocess_si.py has no autocast), so random mode does the same.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

from s2storm import device as dev
from s2storm import paths

CONFIG_FILE = 'v2.0/config/exp16_midway_sstanom.yaml'
CONFIG_NAME = 'S2S_MULTICOND'

# Current device, read by the patched Trainer.device property (switched by the CPU fallback).
_STATE = {'device': None}


# ---------------------------------------------------------------------------
# Ranks (PALS / PMI -> torch.distributed env)
# ---------------------------------------------------------------------------
def setup_rank_env(use_ranks: bool) -> tuple[int, int, int]:
    """Map PALS/PMI variables to RANK / WORLD_SIZE / LOCAL_RANK (single process unless use_ranks)."""
    env = os.environ
    if use_ranks:
        rank = env.get('RANK') or env.get('PMI_RANK') or env.get('PALS_RANKID') or '0'
        size = env.get('WORLD_SIZE') or env.get('PMI_SIZE') or env.get('PALS_SIZE')
        if size is None:   # Aurora PALS mpiexec sets PALS_RANKID but no size variable (S15 job 8997118)
            from mpi4py import MPI
            size = str(MPI.COMM_WORLD.Get_size())
            rank = str(MPI.COMM_WORLD.Get_rank())
        local = (env.get('LOCAL_RANK') or env.get('PALS_LOCAL_RANKID') or env.get('PMI_LOCAL_RANK')
                 or '0')
    else:
        rank, size, local = '0', '1', '0'
        # one-process group: take a free port, so 12 independent processes per node (task farm) don't
        # collide on MASTER_PORT (EADDRINUSE in S15 job 8997118)
        import socket as _socket
        with _socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            env['MASTER_ADDR'], env['MASTER_PORT'] = '127.0.0.1', str(sock.getsockname()[1])
    env['RANK'], env['WORLD_SIZE'], env['LOCAL_RANK'] = rank, size, local
    env.setdefault('MASTER_ADDR', 'localhost')
    env.setdefault('MASTER_PORT', '29500')
    return int(rank), int(size), int(local)


def dist_backend(device) -> str:
    """xccl on XPU if this torch has it, else oneCCL ('ccl'), else gloo; nccl on CUDA; gloo on CPU."""
    import torch.distributed as dist

    name = dev.backend(device)
    if name == 'xccl' and not getattr(dist, 'is_xccl_available', lambda: False)():
        try:
            import oneccl_bindings_for_pytorch  # noqa: F401  (registers 'ccl')
            name = 'ccl'
        except ImportError:
            name = 'gloo'
    return name


# ---------------------------------------------------------------------------
# Monkeypatches (before any upstream import)
# ---------------------------------------------------------------------------
def apply_patches(device) -> list[str]:
    """Make the CUDA-only upstream code run on `device`. Returns the names of the patches applied."""
    import torch
    import torch.distributed as dist

    applied = []
    _STATE['device'] = device
    backend = dist_backend(device)

    # torch.distributed.init_process_group('nccl') -> our backend; idempotent.
    orig_init = dist.init_process_group
    if not getattr(orig_init, '_ucs2s', False):
        def init_process_group(backend_arg=None, *args, **kwargs):
            if dist.is_initialized():
                return None
            kwargs.pop('backend', None)
            return orig_init(backend, *args, **kwargs)
        init_process_group._ucs2s = True
        dist.init_process_group = init_process_group
        applied.append(f'dist.init_process_group -> {backend}')

    if device.type == 'cuda':
        return applied

    # torch.load(map_location='cuda:N' / torch.device('cuda') / {..: 'cuda:N'}) -> our device.
    orig_load = torch.load
    if not getattr(orig_load, '_ucs2s', False):
        def is_cuda(loc):
            return (isinstance(loc, str) and loc.startswith('cuda')) or \
                   (isinstance(loc, torch.device) and loc.type == 'cuda')
        def load(f, map_location=None, *args, **kwargs):
            if is_cuda(map_location):
                map_location = _STATE['device']
            elif isinstance(map_location, dict):
                map_location = {k: (str(_STATE['device']) if is_cuda(v) else v) for k, v in map_location.items()}
            return orig_load(f, map_location, *args, **kwargs)
        load._ucs2s = True
        torch.load = load
        applied.append('torch.load map_location cuda -> device')

    # torch.cuda.* runtime calls -> torch.xpu.* (no-ops on CPU). is_available() is NOT patched.
    xpu = torch.xpu if device.type == 'xpu' else None
    idx = device.index or 0
    cuda = torch.cuda
    cuda.set_device = lambda d=None: xpu.set_device(idx) if xpu else None
    cuda.current_device = lambda: idx
    cuda.device_count = (lambda: xpu.device_count()) if xpu else (lambda: 1)
    cuda.manual_seed_all = (lambda s: xpu.manual_seed_all(s)) if xpu else (lambda s: None)
    cuda.manual_seed = (lambda s: xpu.manual_seed(s)) if xpu else (lambda s: None)
    cuda.synchronize = lambda d=None: dev.synchronize(_STATE['device'])
    cuda.empty_cache = (lambda: xpu.empty_cache()) if xpu else (lambda: None)
    cuda.max_memory_allocated = (lambda d=None: xpu.max_memory_allocated(idx)) if xpu else (lambda d=None: 0)
    cuda.memory_allocated = (lambda d=None: xpu.memory_allocated(idx)) if xpu else (lambda d=None: 0)
    cuda.memory_reserved = (lambda d=None: xpu.memory_reserved(idx)) if xpu else (lambda d=None: 0)
    cuda.reset_peak_memory_stats = (lambda d=None: xpu.reset_peak_memory_stats(idx)) if xpu else (lambda d=None: None)
    cuda.get_device_name = (lambda d=None: xpu.get_device_name(idx)) if xpu else (lambda d=None: 'cpu')
    applied.append(f'torch.cuda.{{set_device,current_device,device_count,manual_seed[_all],synchronize,'
                   f'empty_cache,max_memory_allocated,memory_*,get_device_name}} -> {device.type}')

    # .cuda() on tensors / modules -> our device.
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False, **kw: \
        self.to(_STATE['device'], non_blocking=non_blocking)
    torch.nn.Module.cuda = lambda self, device=None: self.to(_STATE['device'])
    applied.append('Tensor.cuda / Module.cuda -> device')

    # autocast('cuda') -> autocast(device.type)
    orig_autocast = torch.amp.autocast
    if not getattr(orig_autocast, '_ucs2s', False):
        class Autocast(orig_autocast):
            _ucs2s = True

            def __init__(self, device_type='cuda', *args, **kwargs):
                if device_type == 'cuda':
                    device_type = _STATE['device'].type
                super().__init__(device_type, *args, **kwargs)
        torch.amp.autocast = Autocast
        torch.autocast = Autocast
        torch.cuda.amp.autocast = lambda enabled=True, dtype=torch.float16, cache_enabled=True: \
            Autocast(_STATE['device'].type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
        applied.append(f'torch.amp.autocast / torch.autocast / torch.cuda.amp.autocast cuda -> {device.type}')

    # DistributedDataParallel(device_ids=[local_rank]) -> our tile (None on CPU).
    from torch.nn.parallel import DistributedDataParallel
    if not getattr(DistributedDataParallel, '_ucs2s', False):
        class DDP(DistributedDataParallel):
            _ucs2s = True

            def __init__(self, module, device_ids=None, output_device=None, *args, **kwargs):
                d = _STATE['device']
                if d.type == 'cpu':
                    device_ids, output_device = None, None
                else:
                    device_ids, output_device = [d.index or 0], d.index or 0
                super().__init__(module, device_ids, output_device, *args, **kwargs)
        torch.nn.parallel.DistributedDataParallel = DDP
        applied.append('DistributedDataParallel device_ids -> device index')
    return applied


def patch_upstream(applied: list[str]) -> None:
    """Patches on upstream classes (import after sys.path is set; these imports start nothing heavy)."""
    from utils.YParams import YParams

    try:
        import transformer_engine  # noqa: F401
        have_te = True
    except ImportError:
        have_te = False
    if not have_te and not getattr(YParams.__init__, '_ucs2s', False):
        orig = YParams.__init__

        def init(self, *args, **kwargs):
            orig(self, *args, **kwargs)
            if self.params.get('use_transformer_engine'):
                self['use_transformer_engine'] = False
        init._ucs2s = True
        YParams.__init__ = init
        applied.append('YParams: use_transformer_engine forced False (transformer_engine not installed)')


def patch_trainer(applied: list[str]) -> None:
    """Trainer.device -> our device (upstream uses torch.cuda.current_device() or 'cpu')."""
    import train

    train.Trainer.device = property(lambda self: _STATE['device'], lambda self, value: None)
    applied.append("train.Trainer.device -> device (was torch.cuda.current_device() or 'cpu')")


def add_repo_to_path() -> Path:
    v2 = paths.ucs2s_repo() / 'v2.0'
    if not v2.is_dir():
        raise FileNotFoundError(f'UC-S2S repo not found: {v2} (set UCS2S_REPO)')
    if str(v2) not in sys.path:
        sys.path.insert(0, str(v2))
    return v2


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------
def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def rank_file(out: Path, rank: int, world: int) -> Path:
    return out if world == 1 else out.with_name(f'{out.stem}.rank{rank}{out.suffix}')


def record_error(state: dict, phase: str, exc: BaseException, repo: Path | None = None) -> dict:
    """Append {phase, op, error, traceback} to state['errors']; op = deepest frame inside the upstream repo."""
    frames = traceback.extract_tb(exc.__traceback__)
    up = [f for f in frames if repo is not None and str(repo) in f.filename]
    last = (up or frames or [None])[-1]
    op = f'{last.filename}:{last.lineno} {last.name}: {last.line}' if last else ''
    err = {'phase': phase, 'op': op, 'error': f'{type(exc).__name__}: {exc}',
           'traceback': traceback.format_exc()}
    state.setdefault('errors', []).append(err)
    print(f'[ucs2s] ERROR in {phase}: {err["error"]}\n  at {op}', flush=True)
    return err


# ---------------------------------------------------------------------------
# Random mode: synthetic dataset + model build
# ---------------------------------------------------------------------------
class _Clim:
    """Stands in for the [366, H, W] climatology memmap: same map for every day, NaN outside domain."""

    def __init__(self, field):
        self.field = field

    def __getitem__(self, doy):
        return self.field


def synthetic_dataset(params, seed: int = 0):
    """A GetDataset instance without files: shapes, variable lists and statistics only.

    The upstream methods (_make_multi_cond, _doy_index, *_inv_transform) are used as they are;
    only _get_data (h5 read) is replaced by random physical fields.
    """
    import numpy as np
    import torch
    from utils.cond_anom_vars import REGISTRY, cond_vars
    from utils.data_loader_multifiles import GetDataset

    H, W = params.horizontal_resolution
    rng = np.random.default_rng(seed)
    ds = GetDataset.__new__(GetDataset)
    ds.params = params
    ds.land_variables = list(params.land_variables)
    ds.ocean_variables = list(params.ocean_variables)
    ds.surface_variables = list(params.surface_variables) + ds.land_variables + ds.ocean_variables
    ds.upper_air_variables = list(params.upper_air_variables)
    ds.diagnostic_variables = list(params.diagnostic_variables)
    ds.constant_boundary_variables = list(params.constant_boundary_variables)
    ds.varying_boundary_variables = list(params.varying_boundary_variables)
    ds.levels = list(params.levels)
    ds.mask_fill = params.mask_fill
    ds.calendar = getattr(params, 'calendar', 'standard')
    ds.datetime_class = ds.datetime_class_from_calendar(ds.calendar)
    ds.timedelta_hours = params.timedelta_hours

    # Synthetic land mask (smooth continents), replaces lsm from the constant-boundary h5 file.
    lat = np.linspace(-89.5, 89.5, H)[:, None]
    lon = np.linspace(0, 359, W)[None, :]
    land = (np.sin(np.deg2rad(2 * lon)) * np.cos(np.deg2rad(lat)) + 0.3 * np.sin(np.deg2rad(3 * lat)) > 0.25)
    land = land.astype(np.float32)
    ds.land_mask = torch.from_numpy(land)[None]                      # [1, H, W] as _load_constant_boundary_data
    cb = np.stack([land, rng.standard_normal((H, W)).astype(np.float32)])[:len(ds.constant_boundary_variables)]
    cb = (cb - cb.mean(axis=(1, 2), keepdims=True)) / (cb.std(axis=(1, 2), keepdims=True) + 1e-6)
    ds.constant_boundary_data = torch.from_numpy(cb.astype(np.float32))   # [n_const, H, W], standardised

    ns, nu, nl, nd = len(ds.surface_variables), len(ds.upper_air_variables), len(ds.levels), len(ds.diagnostic_variables)
    ds.surface_mean, ds.surface_std = torch.zeros(ns), torch.ones(ns)
    ds.upper_air_mean, ds.upper_air_std = torch.zeros(nu, nl), torch.ones(nu, nl)
    ds.diagnostic_mean, ds.diagnostic_std = torch.zeros(nd), torch.ones(nd)

    # Multi-variable anomaly conditioning (replaces cond_clim_<v>.npy / cond_scalar_<v>.npy).
    ds.sst_cond_anom = bool(getattr(params, 'sst_cond_anom', False))
    ds.sst_cond_size = tuple(getattr(params, 'sst_cond_size', (23, 45)))
    ds.cond_vars = cond_vars(params)
    ds.cond_clim, ds.cond_sigma = {}, {}
    for v in ds.cond_vars or []:
        key, masked = REGISTRY[v]
        clim = np.zeros((H, W), np.float32)
        if masked:   # NaN outside the variable's domain: land for sst/sic, ocean for soil water
            clim[(land > 0) if key != 'volumetric_soil_water_layer_1' else (land == 0)] = np.nan
        ds.cond_clim[v], ds.cond_sigma[v] = _Clim(clim), 1.0

    def get_data(start_time, out=False, variable_list=None):
        n = len(variable_list) if variable_list is not None else 1
        return rng.standard_normal((n, H, W)).astype(np.float32)
    ds._get_data = get_data
    ds.dates_all = None
    return ds


def load_params(config_file: Path, config_name: str, steps: int, rank: int, local_rank: int, world: int,
                overrides: list[str] | None = None):
    """YParams + the edits inference_postprocess_si.py's __main__ makes (+ --set key=value overrides)."""
    import yaml
    from utils.YParams import YParams

    params = YParams(str(config_file), config_name)
    for item in overrides or []:
        key, value = item.split('=', 1)
        params[key] = yaml.safe_load(value)
    params['config_filepath'] = str(config_file)
    params['has_diagnostic'] = bool(hasattr(params, 'diagnostic_variables') and len(params.diagnostic_variables) > 0)
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1
    params['log_to_wandb'] = False
    params['inference_steps'] = steps
    params['world_size'] = world
    params['local_rank'] = local_rank
    params['run_num'] = 'random'
    params['resuming'] = False
    params['run_iter'] = 1
    params['experiment_dir'] = ''
    return params


def make_random_inference(params, world_rank: int, with_pp: bool, args):
    """PostprocessInference with its data/checkpoint hooks replaced; everything else is upstream."""
    import torch
    import inference_postprocess_si as ips
    from networks.postprocess_si import PostprocessSI

    class RandomInference(ips.PostprocessInference):
        def get_dataset(self):
            # Trainer.get_dataset without files: same constant_boundary_data broadcast to batch_size.
            ds = synthetic_dataset(self.params, seed=args.seed)
            self.train_datasets = [ds]
            self.valid_dataset = ds
            self.constant_boundary_data = (ds.constant_boundary_data.unsqueeze(0)
                                           * torch.ones(self.params.batch_size, 1, 1, 1)).to(self.device)
            self.climatology = None

        def create_dirs(self, run_uuid):
            return None, None, None

        def _load_base_checkpoint(self):
            # random weights: only what the upstream loader does besides load_state_dict
            self.diff_model.freeze_encoder()

        def _load_pp_checkpoint(self, pp_ckpt):
            p = self.params
            self.pp_model = PostprocessSI(
                n_fields=self.n_fields,
                base_channels=int(getattr(p, 'pp_base_channels', 64)),
                c_mults=tuple(getattr(p, 'pp_c_mults', (1, 2, 4))),
                scale=tuple(getattr(p, 'pp_scale', (2, 2))),
                gamma_type=str(getattr(p, 'pp_gamma_type', 'brownian')),
                dropout=float(getattr(p, 'pp_dropout', 0.1)),
            ).to(self.device)
            self.pp_model.eval()
            for q in self.pp_model.parameters():
                q.requires_grad = False
            self.max_lead_steps = int(args.pp_max_lead_steps)   # upstream: ckpt.get('max_lead_steps', 10)

    mc = None if args.mc_dropout is None else args.mc_dropout == 'on'
    return RandomInference(params, world_rank, pp_ckpt=None, num_members=args.members, blend=args.blend,
                           feedback=args.feedback, apply_correction=with_pp, base_latent=args.base_latent,
                           mc_dropout=mc, latent_cond='sst', sst_cond_mode=args.sst_cond_mode)


def count_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def call_counter(module, counts: dict, key: str):
    """Forward pre-hook counting calls (verifies SI steps per model step) and recording input shapes."""
    def hook(mod, inputs):
        counts[key] = counts.get(key, 0) + 1
        if f'{key}_shapes' not in counts:
            counts[f'{key}_shapes'] = [list(t.shape) for t in inputs if hasattr(t, 'shape')]
    return module.register_forward_pre_hook(hook)


def move_to(inf, device, tensors: dict) -> dict:
    """CPU fallback: put models and the rollout state on `device`."""
    _STATE['device'] = device
    inf.diff_model.to(device)
    if inf.pp_model is not None:
        inf.pp_model.to(device)
    inf.constant_boundary_data = inf.constant_boundary_data.to(device)
    if getattr(inf, 'sst_cond_batch', None) is not None:
        inf.sst_cond_batch = inf.sst_cond_batch.to(device)
    inf._evo = None
    return {k: (v.to(device) if v is not None else None) for k, v in tensors.items()}


def run_random(args, device, rank: int, local_rank: int, world: int, state: dict, out_file: Path, repo: Path) -> None:
    import numpy as np
    import torch
    from torch.utils.data import default_collate

    def save():
        peak = dev.max_memory_gb(_STATE['device'])
        state['peak_mem_gb'] = round(peak, 3) if peak == peak else None   # NaN on CPU
        state['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
        write_json(out_file, state)

    # --- build -----------------------------------------------------------------
    state['phase'] = 'build'
    t0 = time.time()
    try:
        params = load_params(repo / CONFIG_FILE if not args.config_file else Path(args.config_file),
                             args.config, args.steps, rank, device.index or 0, world, args.set)
        try:
            inf = make_random_inference(params, rank, args.with_pp, args)
        except Exception as exc:   # DDP wrap is the usual suspect: retry with a plain holder
            record_error(state, 'build (with DDP)', exc, repo)
            import train

            class Holder(torch.nn.Module):
                def __init__(self, module, *a, **kw):
                    super().__init__()
                    self.module = module
            train.DistributedDataParallel = Holder
            state['ddp'] = 'replaced by plain holder after error'
            inf = make_random_inference(params, rank, args.with_pp, args)
        dev.synchronize(device)
    except Exception as exc:
        record_error(state, 'build', exc, repo)
        state['status'] = 'failed'
        save()
        return
    ds = inf.valid_dataset
    state['build_s'] = round(time.time() - t0, 2)
    state['ddp'] = state.get('ddp', type(inf.model_det).__name__)
    state['params'] = {
        'vae': count_params(inf.diff_model.encoder), 'det_pangu': count_params(inf.diff_model.model_det),
        'si_unet': count_params(inf.diff_model.unet),
        'pp_si': count_params(inf.pp_model) if inf.pp_model is not None else 0,
    }
    state['config_used'] = {
        'surface_variables (input)': list(ds.surface_variables),
        'upper_air_variables': list(ds.upper_air_variables), 'levels': list(ds.levels),
        'diagnostic_variables': list(ds.diagnostic_variables),
        'constant_boundary_variables': list(ds.constant_boundary_variables),
        'varying_boundary_variables': list(ds.varying_boundary_variables),
        'cond_anom_vars': list(ds.cond_vars or []), 'sst_cond_size': list(ds.sst_cond_size),
        'patch_size (set by PanguModel)': list(params.patch_size),
        'window_size (set by PanguModel)': list(params.window_size),
        'base_temperature': inf.base_temperature, 'mc_dropout': inf.mc_dropout,
        'base_latent': inf.base_latent, 'sst_cond_mode': inf.sst_cond_mode, 'feedback': inf.feedback,
        'si_sde_steps': 'StochasticInterpolant.sample default T=15 -> 14 Euler-Maruyama steps',
        'pp_n_fields': inf.n_fields, 'pp_sample_steps': int(getattr(params, 'pp_sample_steps', 20)),
        'pp_max_lead_steps': getattr(inf, 'max_lead_steps', None), 'blend': inf.blend,
        'dtype': 'float32, no autocast (as upstream inference)',
    }
    save()
    print(f'[ucs2s] rank {rank}: built in {state["build_s"]} s on {device}; params {state["params"]}', flush=True)

    # --- synthetic initial condition, as predict() gets it from default_collate([ds[i]]) -----
    state['phase'] = 'inputs'
    try:
        H, W = params.horizontal_resolution
        g = torch.Generator().manual_seed(args.seed)
        steps = int(args.steps)
        n_lead = int(params.forecast_lead_times[-1])
        start = datetime.datetime.strptime(args.init_date, '%Y-%m-%d')
        cond = ds._make_multi_cond(start) if ds.cond_vars else ds._make_sst_cond(None, start)
        sample = (torch.randn(len(ds.surface_variables), H, W, generator=g),
                  torch.randn(len(ds.upper_air_variables), len(ds.levels), H, W, generator=g),
                  torch.randn(n_lead + 1, len(ds.varying_boundary_variables), H, W, generator=g),
                  torch.tensor([start.year, start.month, start.day, start.hour]), cond)
        data = default_collate([sample])
        from utils.sst_cond import pop_sst_cond
        data, inf.sst_cond_batch = pop_sst_cond(data, params, device)
        in_sfc0, in_ua0, vbc, times = data
        in_sfc0, in_ua0, vbc = (x.to(device, dtype=torch.float32) for x in (in_sfc0, in_ua0, vbc))
        start_times = [ds.datetime_class(times[i, 0].item(), times[i, 1].item(), times[i, 2].item(),
                                         hour=times[i, 3].item()) for i in range(times.shape[0])]
        inf._start_times = start_times
        inf._start_times_tensor = torch.tensor([[d.year, d.month, d.day, d.hour] for d in start_times])
        B = in_sfc0.shape[0]
        cb = inf.constant_boundary_data[:B]
        state['shapes'] = {'surface_in': list(in_sfc0.shape), 'upper_air_in': list(in_ua0.shape),
                           'constant_boundary': list(cb.shape), 'varying_boundary': list(vbc.shape),
                           'sst_cond': list(inf.sst_cond_batch.shape) if inf.sst_cond_batch is not None else None}
        if steps > n_lead:
            state.setdefault('notes', []).append(f'steps clipped to forecast_lead_times[-1]={n_lead}')
            steps = n_lead
    except Exception as exc:
        record_error(state, 'inputs', exc, repo)
        state['status'] = 'failed'
        save()
        return

    counts = {}
    call_counter(inf.diff_model.unet, counts, 'si_unet')
    if inf.pp_model is not None:
        call_counter(inf.pp_model.unet, counts, 'pp_unet')

    # --- rollout (same member / step logic as PostprocessInference.predict) ----------------
    state['phase'] = 'rollout'
    nonneg = {'total_precipitation_24hr', 'total_precipitation_6hr', 'mean_top_net_long_wave_radiation_flux'}
    my_members = list(range(rank, args.members, world))
    state['members'] = {}
    batch_idx = 0
    fallback_left = None
    with torch.no_grad():
        for ens_id in my_members:
            rec = {'s_per_step': [], 'base_s': [], 'pp_s': [], 'host_s': [], 'si_unet_calls': [],
                   'pp_unet_calls': [], 'finite': [], 'sfc_std': [], 'device': []}
            state['members'][str(ens_id)] = rec
            seed = 100_000 * batch_idx + ens_id
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)   # patched -> xpu
            if args.keep_outputs:
                out_sfc = np.zeros((B, steps + 1, *in_sfc0.shape[1:]), np.float32)
                out_ua = np.zeros((B, steps + 1, *in_ua0.shape[1:]), np.float32)
                out_diag = np.zeros((B, steps + 1, inf.num_diagnostic_vars, H, W), np.float32)
            cur = {'s': in_sfc0, 'ua': in_ua0, 'diag': None, 'vbc': vbc, 'cb': cb}
            step = 0
            while step < steps:
                d = _STATE['device']
                phase = 'sync'
                try:
                    counts['si_unet'] = counts['pp_unet'] = 0
                    dev.synchronize(d)
                    t_a = time.perf_counter()
                    phase = 'base_step'
                    p_sfc, p_ua, p_diag = inf._base_step(cur['s'], cur['ua'], cur['cb'], cur['vbc'][:, step],
                                                         step=step, diag_prev=cur['diag'])
                    dev.synchronize(d)
                    t_b = time.perf_counter()
                    phase = 'postprocess_si'
                    c_sfc, c_ua, c_diag = inf._apply_correction(p_sfc, p_ua, p_diag, step)
                    dev.synchronize(d)
                    t_c = time.perf_counter()
                    phase = 'host_copy'
                    s_np = ds.surface_inv_transform(c_sfc.cpu()).numpy()
                    u_np = ds.upper_air_inv_transform(c_ua.cpu()).numpy()
                    d_np = ds.diagnostic_inv_transform(c_diag.cpu()).numpy()
                    for vi, vname in enumerate(ds.diagnostic_variables):
                        if vname in nonneg:
                            d_np[:, vi] = np.clip(d_np[:, vi], 0.0, None)
                    if args.keep_outputs:
                        out_sfc[:, step + 1], out_ua[:, step + 1], out_diag[:, step + 1] = s_np, u_np, d_np
                    t_d = time.perf_counter()
                except Exception as exc:
                    record_error(state, f'{phase} step {step} member {ens_id} on {d}', exc, repo)
                    if args.cpu_fallback and d.type != 'cpu':
                        state.setdefault('notes', []).append(
                            f'cpu_fallback: {phase} failed on {d} at step {step}; continuing on CPU for '
                            f'{args.cpu_fallback_steps} step(s)')
                        cur = move_to(inf, torch.device('cpu'), cur)
                        fallback_left = int(args.cpu_fallback_steps)
                        save()
                        continue
                    state['status'] = 'failed'
                    save()
                    return
                rec['base_s'].append(round(t_b - t_a, 4))
                rec['pp_s'].append(round(t_c - t_b, 4))
                rec['host_s'].append(round(t_d - t_c, 4))
                rec['s_per_step'].append(round(t_d - t_a, 4))
                rec['si_unet_calls'].append(counts.get('si_unet', 0))
                rec['pp_unet_calls'].append(counts.get('pp_unet', 0))
                rec['finite'].append(bool(np.isfinite(s_np).all() and np.isfinite(u_np).all()))
                rec['sfc_std'].append(float(np.nanstd(s_np)))
                rec['device'].append(str(d))
                if inf.feedback == 'corrected':
                    cur.update(s=c_sfc, ua=c_ua, diag=c_diag)
                else:
                    cur.update(s=p_sfc, ua=p_ua, diag=p_diag)
                step += 1
                state['steps_done'] = state.get('steps_done', 0) + 1
                state['unet_input_shapes'] = {k: v for k, v in counts.items() if k.endswith('_shapes')}
                save()
                print(f'[ucs2s] rank {rank} member {ens_id} step {step}/{steps}: {rec["s_per_step"][-1]:.3f} s '
                      f'(base {rec["base_s"][-1]:.3f}, pp {rec["pp_s"][-1]:.3f}, host {rec["host_s"][-1]:.3f}) '
                      f'peak {state["peak_mem_gb"]} GB', flush=True)
                if fallback_left is not None:
                    fallback_left -= 1
                    if fallback_left <= 0:
                        state['status'] = 'stopped after cpu_fallback steps'
                        save()
                        return
            s = rec['s_per_step'][1:] or rec['s_per_step']
            rec['mean_s_per_step_excl_first'] = round(sum(s) / len(s), 4) if s else None
    state['status'] = 'ok'
    state['phase'] = 'done'
    save()


# ---------------------------------------------------------------------------
# Real mode: upstream __main__ with patches
# ---------------------------------------------------------------------------
def run_real(args, upstream_args: list[str], state: dict, out_file: Path | None, repo: Path) -> None:
    import runpy

    script = repo / 'v2.0' / 'inference_postprocess_si.py'
    workdir = Path(args.workdir) if args.workdir else repo / 'v2.0' / 'HPC_scripts'
    os.chdir(workdir)   # upstream config paths (exp_dir: results, ../config) are relative to HPC_scripts
    sys.argv = [str(script)] + upstream_args
    state.update(phase='real', script=str(script), workdir=str(workdir), argv=upstream_args)
    if out_file:
        write_json(out_file, state)
    t0 = time.time()
    try:
        runpy.run_path(str(script), run_name='__main__')
        state['status'] = 'ok'
    except BaseException as exc:   # SystemExit from argparse too
        record_error(state, 'real', exc, repo)
        state['status'] = 'failed'
    state['wall_s'] = round(time.time() - t0, 1)
    peak = dev.max_memory_gb(_STATE['device'])
    state['peak_mem_gb'] = round(peak, 3) if peak == peak else None
    if out_file:
        write_json(out_file, state)


def run_dsi(launcher_args: list[str], state: dict, out_file: Path | None) -> None:
    """DSI's launcher (s2storm.models.ucs2s) on XPU. Upstream shards members as
    range(world_rank, num_members, world_size) using module globals of inference_postprocess_si;
    with S2STORM_MEMBER_SHARD='r/R' a single process (world 1) runs shard r of R, so a task farm can
    spread one (config, init) over R tiles. Seeds stay 100000*batch + member id (paired across configs)."""
    from s2storm.models import ucs2s as launcher

    shard = os.environ.get('S2STORM_MEMBER_SHARD')
    if shard:
        r, n = (int(v) for v in shard.split('/'))
        make = launcher._make_runner_class

        def make_sharded(base):
            ips = sys.modules[base.__module__]
            ips.world_rank, ips.world_size = r, n
            return make(base)

        launcher._make_runner_class = make_sharded
        state['member_shard'] = shard
    state.update(phase='dsi', argv=launcher_args)
    if out_file:
        write_json(out_file, state)
    t0 = time.time()
    try:
        launcher.main(launcher_args)
        state['status'] = 'ok'
    except BaseException as exc:
        record_error(state, 'dsi', exc, paths.ucs2s_repo())
        state['status'] = 'failed'
    state['wall_s'] = round(time.time() - t0, 1)
    peak = dev.max_memory_gb(_STATE['device'])
    state['peak_mem_gb'] = round(peak, 3) if peak == peak else None
    if out_file:
        write_json(out_file, state)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mode', choices=('random', 'real', 'dsi'), default='random',
                    help="dsi: run s2storm.models.ucs2s.main (DSI launcher: relaxation, bigrun output) with our XPU patches; "
                         "args after -- go to it. Env S2STORM_MEMBER_SHARD='r/R' runs members r, r+R, ... (task farm)")
    ap.add_argument('--ranks', action='store_true',
                    help='multi-process: take RANK/WORLD_SIZE/LOCAL_RANK from PMI_*/PALS_* (mpiexec)')
    ap.add_argument('--device', default=None, help="force a device, e.g. 'cpu' (default: xpu tile = local rank)")
    ap.add_argument('--out', default=None, help='JSON state/result file (rank r>0 writes <stem>.rank<r>.json)')
    ap.add_argument('--config_file', default=None, help=f'default <ucs2s_repo>/{CONFIG_FILE}')
    ap.add_argument('--config', default=CONFIG_NAME)
    ap.add_argument('--steps', type=int, default=45, help='rollout steps (0 = build only)')
    ap.add_argument('--members', type=int, default=None, help='total members, sharded over ranks (default world size)')
    ap.add_argument('--with_pp', action='store_true', help='also build/run the postprocess SI')
    ap.add_argument('--blend', type=float, default=1.0)
    ap.add_argument('--feedback', choices=('base', 'corrected'), default='base')
    ap.add_argument('--base_latent', choices=('si', 'deterministic'), default='si')
    ap.add_argument('--mc_dropout', choices=('on', 'off'), default=None, help='default: config (on)')
    ap.add_argument('--sst_cond_mode', choices=('persistent', 'evolving'), default='evolving',
                    help='README_inference run uses evolving')
    ap.add_argument('--pp_max_lead_steps', type=int, default=10,
                    help='lead normalisation of the pp SI (stored in the real ckpt; upstream default 10)')
    ap.add_argument('--init_date', default='2020-01-02')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep_outputs', type=int, default=1,
                    help='1: fill host arrays [steps+1, ...] like upstream (about 1 GB per member)')
    ap.add_argument('--cpu_fallback', action='store_true',
                    help='if a step fails on XPU: record it, move to CPU, run --cpu_fallback_steps more, stop. '
                         'Also sets PYTORCH_ENABLE_XPU_FALLBACK=1 (per-op CPU fallback for missing XPU kernels)')
    ap.add_argument('--cpu_fallback_steps', type=int, default=1)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='random mode: override a config key (YAML value), e.g. --set horizontal_resolution=[24,40] '
                         'for a tiny CPU shape check')
    ap.add_argument('--workdir', default=None, help='real mode: cwd for upstream (default v2.0/HPC_scripts)')
    args, upstream_args = ap.parse_known_args()
    if upstream_args and upstream_args[0] == '--':
        upstream_args = upstream_args[1:]

    if args.cpu_fallback:
        os.environ.setdefault('PYTORCH_ENABLE_XPU_FALLBACK', '1')   # must be set before torch is imported
    rank, world, local_rank = setup_rank_env(args.ranks)
    if args.members is None:
        args.members = world

    import torch
    import torch.distributed as dist

    device = torch.device(args.device) if args.device else dev.get_device(local_rank)
    if device.type == 'xpu' and device.index is not None:
        torch.xpu.set_device(device.index)
    _STATE['device'] = device

    out_file = None
    if args.out:
        out_file = rank_file(Path(args.out).resolve(), rank, world)
    elif args.mode == 'random':
        tag = os.environ.get('PBS_JOBID', str(os.getpid())).split('.')[0]
        out_file = rank_file(paths.runs() / 'ucs2s' / f'random_{tag}.json', rank, world)

    state = {'mode': args.mode, 'rank': rank, 'world_size': world, 'local_rank': local_rank,
             'device': str(device), 'host': socket.gethostname(), 'python': platform.python_version(),
             'torch': torch.__version__, 'with_pp': args.with_pp, 'steps_requested': args.steps,
             'steps_done': 0, 'members_total': args.members, 'errors': [], 'status': 'running',
             'started': datetime.datetime.now().isoformat(timespec='seconds'),
             'pbs_jobid': os.environ.get('PBS_JOBID'), 'ze_affinity_mask': os.environ.get('ZE_AFFINITY_MASK')}
    if device.type == 'xpu':
        state['device_name'] = torch.xpu.get_device_name(device.index or 0)

    repo = paths.ucs2s_repo()
    try:
        add_repo_to_path()
        state['patches'] = apply_patches(device)
        state['backend'] = dist_backend(device)
        state['phase'] = 'init_process_group'
        if out_file:
            write_json(out_file, state)   # early record: a hang in the collective setup is visible
        dist.init_process_group(init_method='env://')   # patched: our backend
        state['phase'] = 'all_reduce'
        if out_file:
            write_json(out_file, state)
        if world > 1 or args.ranks:
            t = torch.ones(1, device=device if state['backend'] != 'gloo' else 'cpu')
            dist.all_reduce(t)
            dev.synchronize(device)
            state['all_reduce'] = float(t.item())
            if state['all_reduce'] != float(world):
                raise RuntimeError(f'all_reduce sanity check: got {state["all_reduce"]}, expected {world}')
            dist.barrier()
        patch_upstream(state['patches'])
        patch_trainer(state['patches'])   # imports train.py (also runs its init_process_group: no-op now)
    except Exception as exc:
        record_error(state, 'setup', exc, repo)
        state['status'] = 'failed'
        if out_file:
            write_json(out_file, state)
        return 1
    if out_file:
        write_json(out_file, state)

    if args.mode == 'real':
        run_real(args, upstream_args, state, out_file, repo)
    elif args.mode == 'dsi':
        run_dsi(upstream_args, state, out_file)
    elif args.steps >= 0:
        run_random(args, device, rank, local_rank, world, state, out_file, repo)

    # rank 0: summary over the per-rank files
    try:
        if dist.is_initialized() and world > 1:
            dist.barrier()
        if rank == 0 and world > 1 and args.out:
            out = Path(args.out).resolve()
            ranks = []
            for r in range(world):
                f = rank_file(out, r, world)
                ranks.append(json.loads(f.read_text()) if f.exists() else {'rank': r, 'status': 'missing'})
            per_step = [s for st in ranks for m in st.get('members', {}).values()
                        for s in (m['s_per_step'][1:] or m['s_per_step'])]
            summary = {'mode': args.mode, 'world_size': world, 'with_pp': args.with_pp,
                       'status': 'ok' if all(st.get('status') == 'ok' for st in ranks) else 'failed',
                       'mean_s_per_step_excl_first': round(sum(per_step) / len(per_step), 4) if per_step else None,
                       'peak_mem_gb_max': max((st.get('peak_mem_gb') or float('nan')) for st in ranks),
                       'errors': [e for st in ranks for e in st.get('errors', [])],
                       'rank_files': [str(rank_file(out, r, world)) for r in range(world)],
                       'ranks': [{k: st.get(k) for k in ('rank', 'device', 'status', 'steps_done', 'peak_mem_gb',
                                                          'build_s', 'backend', 'all_reduce')} for st in ranks]}
            write_json(out, summary)
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        print(f'[ucs2s] summary/teardown failed: {exc}', flush=True)
    return 0 if state.get('status') == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
