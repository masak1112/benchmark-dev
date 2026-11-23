"""UC-S2S (masak1112/S2S @ uc-s2s_v3_multicond) launcher for s2storm.

The upstream clone is never edited. Everything we need is done here:
  * paths (data, checkpoints, conditioning climatologies, output dir) are set on the params object;
  * init dates come from ``--init_dates`` (monkeypatch of
    ``utils.data_loader_multifiles.get_initialization_dates``);
  * the process group is initialised by us with the right backend (upstream calls
    ``dist.init_process_group('nccl')`` at import time; we make that a no-op);
  * optional relaxation hook (s2storm.relax) applied after every daily base step, before the
    state is fed back and before the evolving conditioning is rebuilt;
  * per-rank timing / memory JSON, optional deterministic kernels.

Run under torchrun, e.g.
  torchrun --nproc_per_node=2 -m s2storm.models.ucs2s --run_name smoke --init_dates 2020-06-01 \
      --num_members 2 -- --inference_steps 45 --blend 1.0 --feedback base --base_latent si \
      --sst_cond_mode evolving
Arguments after ``--`` go to the upstream ``inference_postprocess_si.py`` parser unchanged.
"""
import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_CKPT = "/net/spaces/monsoon/bing/checkpoints_UC-S2S_v4_multicondi"
DEFAULT_H5 = "/net/spaces/monsoon/S2S/h5data"
DEFAULT_YAML = "v2.0/config/exp16_dsi_multicond.yaml"


def _our_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_root", default=None, help="default $S2STORM_DATA/runs/ucs2s")
    p.add_argument("--init_dates", nargs="+", required=False, default=None,
                   help="YYYY-MM-DD (00 UTC). Replaces the upstream Mon/Thu Jan-Apr calendar.")
    p.add_argument("--repo", default=None, help="UC-S2S clone (default $UCS2S_REPO or $S2STORM_DATA/ext/S2S)")
    p.add_argument("--yaml", default=DEFAULT_YAML, help="relative to --repo, or absolute")
    p.add_argument("--section", default="S2S_MULTICOND")
    p.add_argument("--ckpt_dir", default=os.environ.get("UCS2S_CKPT", DEFAULT_CKPT))
    p.add_argument("--base_ckpt_rel", default="loaded/3_base_full_state_rollout_finetuned_multicond/rollout_ckpt_1050.tar")
    p.add_argument("--pp_ckpt_rel", default="loaded/4_postprocess_si/pp_si_ckpt_12800.tar")
    p.add_argument("--data_dir", default=os.environ.get("UCS2S_H5", DEFAULT_H5))
    p.add_argument("--climatology_file", default=os.environ.get("UCS2S_CLIMATOLOGY"),
                   help="absolute path replacing params.climatology_file (upstream Trainer.get_dataset only opens it; "
                        "inference never reads it). Aurora: placeholder, the bundle has no 1979-2018_mean_climatology.nc")
    p.add_argument("--stats_dir", default=None, help="mean/std .nc dir (default = data_dir)")
    p.add_argument("--cond_anom_dir", default=None, help="default <ckpt_dir>/config/cond_anom")
    p.add_argument("--relax", default=None, help="YAML relaxation config (s2storm.relax; v2 = factorial)")
    p.add_argument("--relax_probe", nargs="*", default=None,
                   help="v2 YAMLs checked at the hook level on this (unmodified) rollout (E0, s2storm.relax.factorial.Probe)")
    p.add_argument("--relax_probe_out", default=None, help="JSON summary of the probe")
    p.add_argument("--deterministic", action="store_true",
                   help="cudnn.deterministic, no benchmark, no TF32 (for bitwise tests)")
    p.add_argument("--timing_json", default=None, help="per-rank timing file prefix")
    p.add_argument("--save_mode", choices=["upstream", "bigrun", "pilot", "trackb", "hindcast"], default="upstream",
                   help="bigrun: write <stem>_crop.nc (CONUS + tropics strip, selected vars) + global z500/OLR; "
                        "members in --full_members also get the upstream full file")
    p.add_argument("--full_members", type=int, nargs="*", default=[0])
    p.add_argument("--hc_grid", choices=["native", "1p5"], default="native",
                   help="hindcast mode: native 1 deg or conservative WB2 1.5 deg (s2storm.models.ucs2s_hindcast)")
    p.add_argument("--hc_complevel", type=int, default=4)
    p.add_argument("--hc_keepbits", type=int, default=None, help="hindcast mode: mantissa bits kept (lossy); default lossless")
    p.add_argument("--check_ckpts", action="store_true",
                   help="only build the model and try strict loading of every candidate checkpoint")
    return p


def _split_argv(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _setup_dist():
    """Initialise torch.distributed ourselves and neutralise upstream's import-time call."""
    import torch
    import torch.distributed as dist
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from s2storm.device import backend, get_device

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dev = get_device()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend(dev), init_method="env://")
    _orig = dist.init_process_group

    def _noop_if_initialised(*a, **k):
        if dist.is_initialized():
            return None
        return _orig(*a, **k)

    dist.init_process_group = _noop_if_initialised
    return dev


def _patch_calendar(init_dates):
    """Make GetDataset.dates_all == our dates. Upstream loops over the hard-coded years 2019-2024 calling
    get_initialization_dates(year) and concatenates; we return ALL requested dates on the first call of the loop
    (year 2019) and nothing for the other years, so any init year with h5 files works (e.g. 2025-2026, S16/S17)."""
    import pandas as pd
    import utils.data_loader_multifiles as dlm

    want = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in init_dates))
    bad = [d for d in want if d.hour != 0]
    if bad:
        raise ValueError(f"init dates must be 00 UTC: {bad}")

    def get_initialization_dates(year, start_month=1, end_month=4):
        return want if year == 2019 else want[:0]

    dlm.get_initialization_dates = get_initialization_dates
    return want


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ours, upstream = _split_argv(argv)
    a = _our_parser().parse_args(ours)

    repo = Path(a.repo or os.environ.get("UCS2S_REPO") or
                Path(os.environ.get("S2STORM_DATA", "/net/spaces/monsoon/jhayron/s2storm")) / "ext/S2S")
    v2 = repo / "v2.0"
    sys.path.insert(0, str(v2))
    os.chdir(v2 / "HPC_scripts")   # upstream relative paths assume this cwd

    dev = _setup_dist()
    import torch
    import torch.distributed as dist

    if a.init_dates:
        want = _patch_calendar(a.init_dates)

    import inference_postprocess_si as ips   # noqa: E402  (upstream; dist already initialised)
    from utils import logging_utils
    from utils.YParams import YParams

    # --- upstream argument parser, reused verbatim for the remaining flags -----------------
    up = _upstream_args(ips, upstream)

    yaml_path = Path(a.yaml) if os.path.isabs(a.yaml) else repo / a.yaml
    params = YParams(str(yaml_path), a.section)
    params["config_filepath"] = str(yaml_path)
    params["has_diagnostic"] = bool(getattr(params, "diagnostic_variables", None))
    if not hasattr(params, "num_ensemble_members"):
        params["num_ensemble_members"] = 1
    params["log_to_wandb"] = False

    ck = Path(a.ckpt_dir)
    params["checkpoint_path_vae_c1"] = str(ck / "loaded/1_vae_c1/ckpt.tar")
    params["checkpoint_path_det"] = str(ck / "loaded/2_pangu_deterministic/ckpt.tar")
    # stage-1 latent SI of the model in --ckpt_dir (multicond: latent_si_multicond_stage1; SST-anomaly: latent_si_sstanom_stage1).
    # Only read by upstream if checkpoint_path_finetuned is missing; one folder per checkpoint bundle.
    fb = sorted((ck / "fallback_not_loaded").glob("*/last_ckpt.tar"))
    params["checkpoint_path_diff"] = str(fb[0] if len(fb) == 1 else ck / "fallback_not_loaded/latent_si_multicond_stage1/last_ckpt.tar")
    if getattr(params, "sst_cond_anom", False) and not getattr(params, "cond_anom_vars", None):
        # S2S_SSTANOM: daily SST climatology (standardized units) for the conditioning map, shipped in <ckpt_dir>/config
        params["sst_clim_path"] = str(ck / "config/clim_00z_sst_std.npy")
    params["checkpoint_path_finetuned"] = str(ck / a.base_ckpt_rel)
    params["cond_anom_dir"] = a.cond_anom_dir or str(ck / "config/cond_anom")
    params["data_dir"] = a.data_dir
    if a.climatology_file:
        params["climatology_file"] = str(Path(a.climatology_file).resolve())
    if a.stats_dir:   # GetDataset joins data_dir + stats file names; absolute names win in os.path.join
        for k in ("surface_mean", "surface_std", "surface_ff_std", "upper_air_mean", "upper_air_std",
                  "upper_air_ff_std", "boundary_mean", "boundary_std", "diagnostic_mean", "diagnostic_std"):
            if k in params:
                params[k] = str(Path(a.stats_dir) / os.path.basename(params[k]))
    params["inference_steps"] = up.inference_steps or params.params.get("inference_steps", 45)
    params["init_date_set"] = None
    out_root = Path(a.out_root or Path(os.environ.get("S2STORM_DATA", "/net/spaces/monsoon/jhayron/s2storm")) / "runs/ucs2s")
    params["exp_dir"] = str(out_root)

    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    params["world_size"] = world_size
    params["local_rank"] = local_rank
    params["run_num"] = a.run_name
    params["resuming"] = False
    params["run_iter"] = 1

    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    exp_dir = out_root / a.section / a.run_name
    if world_rank == 0:
        (exp_dir / "predictions").mkdir(parents=True, exist_ok=True)
        logging_utils.log_to_file(logger_name=None, log_filename=str(exp_dir / "inference_postprocess_si.log"))
        logging_utils.log_versions()
        params.log()
    dist.barrier()
    params["experiment_dir"] = str(exp_dir)

    if a.check_ckpts:
        from s2storm.models.ucs2s_ckpt import check_all
        check_all(ips, params, ck, exp_dir, world_rank)
        dist.barrier()
        return

    relax = None
    if a.relax:
        from s2storm import config as s2cfg
        relax = s2cfg.load(a.relax)

    blend = up.blend if up.blend is not None else float(getattr(params, "pp_blend", 1.0))
    Runner = _make_runner_class(ips.PostprocessInference)
    runner = Runner(
        params, world_rank,
        pp_ckpt=str(ck / a.pp_ckpt_rel),
        num_members=up.num_members, blend=blend, feedback=up.feedback,
        apply_correction=not up.no_correction, years=up.years, max_dates=up.max_dates,
        save_vars=up.save_vars, overwrite=up.overwrite, base_latent=up.base_latent,
        mc_dropout=(None if up.mc_dropout is None else up.mc_dropout == "on"),
        latent_cond=up.latent_cond, sst_cond_mode=up.sst_cond_mode,
        relax_cfg=relax, device=dev, save_mode=a.save_mode, full_members=a.full_members,
        hc_opts=dict(grid=a.hc_grid, complevel=a.hc_complevel, keepbits=a.hc_keepbits,
                     attrs=dict(base_ckpt=a.base_ckpt_rel, pp_ckpt=a.pp_ckpt_rel, code=os.environ.get("S2STORM_CODE", ""))),
    )
    if a.relax_probe:
        from s2storm import config as s2cfg
        from s2storm.relax.factorial import Probe
        cfgs = [(Path(f).stem, s2cfg.load(f)) for f in a.relax_probe]
        runner.probe = Probe(cfgs, runner.valid_dataset, runner.params, runner.device)
    if a.init_dates and world_rank == 0:
        got = [str(d) for d in runner.valid_dataset.dates_all]
        import logging
        logging.info(f"s2storm: init dates requested {[str(d.date()) for d in want]} -> dataset {got}")
    t0 = time.time()
    runner.predict()
    wall = time.time() - t0
    if a.relax_probe and a.relax_probe_out:
        summ = runner.probe.summary()
        Path(a.relax_probe_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.relax_probe_out).write_text(json.dumps(summ, indent=1, default=str))
        import logging
        logging.info(f"s2storm: relax probe all_ok={summ['all_ok']} n={summ['n_configs']}")
    if a.timing_json or True:
        tj = Path(a.timing_json) if a.timing_json else exp_dir / "timing"
        tj.mkdir(parents=True, exist_ok=True)
        rec = runner.timing_record()
        rec.update(dict(rank=world_rank, world_size=world_size, wall_predict_s=wall,
                        host=os.uname().nodename,
                        gpu=(torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else str(dev)),
                        argv=argv, pp_ckpt=str(ck / a.pp_ckpt_rel), base_ckpt=str(ck / a.base_ckpt_rel)))
        (tj / f"rank{world_rank}.json").write_text(json.dumps(rec, indent=1, default=str))
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def _upstream_args(ips, upstream_argv):
    """Parse the upstream flags with an exact copy of the upstream parser definition."""
    import ast
    import inspect
    src = inspect.getsource(ips)
    tree = ast.parse(src)
    main_if = [n for n in tree.body if isinstance(n, ast.If)][-1]
    lines = []
    for node in main_if.body:
        seg = ast.get_source_segment(src, node)
        if "parse_args" in seg:
            break
        lines.append(seg)
    ns = {"argparse": argparse}
    exec("\n".join(lines), ns)
    return ns["parser"].parse_args(upstream_argv)


def _make_runner_class(Base):
    import numpy as np
    import torch

    class S2StormInference(Base):
        def __init__(self, *args, relax_cfg=None, device=None, save_mode="upstream", full_members=(0,), hc_opts=None, **kw):
            self.save_mode = save_mode
            self.hc_opts = hc_opts or {}
            self.full_members = set(full_members or [])
            self._t = dict(base_s=0.0, base_n=0, pp_s=0.0, pp_n=0, relax_s=0.0, relax_n=0, save_s=0.0, save_n=0)
            self._relax_cfg = relax_cfg
            super().__init__(*args, **kw)
            self.relaxer = None
            self._olr_flip = self._install_olr_fix() if save_mode == "hindcast" else None
            self.probe = None
            if relax_cfg:
                from s2storm.relax.factorial import make_relaxer
                self.relaxer = make_relaxer(relax_cfg, self.valid_dataset, self.params, self.device)

        def _sync(self):
            # on Aurora torch.cuda.is_available is patched to False: sync the XPU too, else timings are enqueue times
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()

        def _base_step(self, s_cur, ua_cur, cb, vbc_step, step=None, diag_prev=None):
            self._sync(); t = time.time()
            p_sfc, p_ua, p_diag = super()._base_step(s_cur, ua_cur, cb, vbc_step, step=step, diag_prev=diag_prev)
            self._sync(); self._t["base_s"] += time.time() - t; self._t["base_n"] += 1
            if self.relaxer is not None and step is not None:
                t = time.time()
                p_sfc, p_ua = self.relaxer(p_sfc, p_ua, self._start_times, step)
                self._sync(); self._t["relax_s"] += time.time() - t; self._t["relax_n"] += 1
            if self.probe is not None and step is not None:
                t = time.time()
                self.probe(p_sfc, p_ua, self._start_times, step)
                self._t["probe_s"] = self._t.get("probe_s", 0.0) + time.time() - t
            return p_sfc, p_ua, p_diag

        def _apply_correction(self, sfc, ua, diag, step):
            if self.pp_model is None:
                return sfc, ua, diag
            self._sync(); t = time.time()
            out = super()._apply_correction(sfc, ua, diag, step)
            self._sync(); self._t["pp_s"] += time.time() - t; self._t["pp_n"] += 1
            # size of the pp correction (normalised units, RMS over all surface / upper-air values): proves pp is active
            self._t.setdefault("pp_rms_sfc", []).append(float(((out[0] - sfc) ** 2).mean().sqrt()))
            self._t.setdefault("pp_rms_ua", []).append(float(((out[1] - ua) ** 2).mean().sqrt()))
            return out

        def _load_base_checkpoint(self):
            super()._load_base_checkpoint()
            rep = os.environ.get("S2STORM_STRICT_REPORT")
            if rep:
                self._strict_report(rep)

        def _strict_report(self, rep):
            """upstream loads the finetuned base state with strict=False (after strict loads of the VAE and the det model):
            record what that load matched; pass = no model key uncovered by the four checkpoints, no unexpected key, no shape
            mismatch, and every loaded tensor equal to the checkpoint after loading."""
            from collections import OrderedDict as _OD
            p = self.params
            base = getattr(p, "checkpoint_path_finetuned", None) or getattr(p, "checkpoint_path_diff", None)
            ck = torch.load(base, map_location="cpu", weights_only=False)
            st = ck["model_state"]
            if any(k.startswith("module.") for k in st):
                st = _OD((k[7:], v) for k, v in st.items())
            ms = self.diff_model.state_dict()
            missing = sorted(set(ms) - set(st))
            unexpected = sorted(set(st) - set(ms))
            shape = sorted(k for k in set(ms) & set(st) if tuple(ms[k].shape) != tuple(st[k].shape))
            neq = sorted(k for k in set(ms) & set(st) if k not in shape and
                         not torch.equal(ms[k].detach().cpu(), st[k].detach().cpu().to(ms[k].dtype)))
            covered = ("encoder.", "model_det.")     # strict loads of checkpoint_path_vae_c1 / checkpoint_path_det
            uncovered = [k for k in missing if not k.startswith(covered)]
            r = dict(base_ckpt=base, epoch=ck.get("epoch"), iters=ck.get("iters"), n_ckpt=len(st), n_model=len(ms),
                     n_missing=len(missing), n_missing_uncovered=len(uncovered), missing_uncovered=uncovered[:50],
                     missing_prefixes=sorted({k.split(".")[0] for k in missing}), n_unexpected=len(unexpected),
                     unexpected=unexpected[:50], shape_mismatch=shape[:50], n_not_equal_after_load=len(neq),
                     not_equal=neq[:50], vae=p.checkpoint_path_vae_c1, det=p.checkpoint_path_det,
                     sst_cond_anom=bool(getattr(p, "sst_cond_anom", False)), sst_cond_mode=getattr(self, "sst_cond_mode", None))
            r["pass"] = not (uncovered or unexpected or shape or neq)
            with open(rep, "w") as f:
                json.dump(r, f, indent=1)

        def _prediction_path(self, start_time, ens_id):
            p = super()._prediction_path(start_time, ens_id)
            if self.save_mode in ("bigrun", "pilot"):
                return p.replace(".nc", "_crop.nc")
            if self.save_mode == "trackb":
                return p.replace(".nc", "_light.nc")
            if self.save_mode == "hindcast":
                return p.replace(".nc", "_h25.nc")
            return p

        def _install_olr_fix(self):
            """Upstream predict() clips the saved diagnostics in ``nonneg_vars`` at 0, and that set includes
            mean_top_net_long_wave_radiation_flux, which is negative (ERA5 net TOA LW), so every saved OLR is 0. The clip only
            touches the saved copy; the trajectory uses the unclipped tensor. Without editing upstream: the dataset's
            diagnostic_inv_transform returns -OLR (positive, survives the clip), and save_prediction flips it back."""
            dv = list(getattr(self.valid_dataset, "diagnostic_variables", None) or [])
            name = "mean_top_net_long_wave_radiation_flux"
            if name not in dv:
                return None
            i = dv.index(name)
            orig = self.valid_dataset.diagnostic_inv_transform

            def inv(x, _o=orig, _i=i):
                y = _o(x).clone()
                y[:, _i] = -y[:, _i]
                return y
            self.valid_dataset.diagnostic_inv_transform = inv
            return i

        def save_prediction(self, surface_prediction, upper_air_prediction, start_times,
                            diagnostic_prediction=None, ens_id=None):
            t = time.time()
            if self._olr_flip is not None and diagnostic_prediction is not None:
                diagnostic_prediction = np.array(diagnostic_prediction, copy=True)
                diagnostic_prediction[:, :, self._olr_flip] *= -1.0      # [B, T, D, H, W]; lead 0 stays 0 (no diagnostic at the IC)
            if self.save_mode == "upstream" or ens_id in self.full_members:
                super().save_prediction(surface_prediction, upper_air_prediction, start_times,
                                        diagnostic_prediction=diagnostic_prediction, ens_id=ens_id)
            if self.save_mode == "hindcast":
                # full upstream file for --full_members (written by super() above under its own name), then the 25-field file
                # last: its existence = member done
                from s2storm.models.ucs2s_hindcast import save_hindcast
                for b, st in enumerate(start_times):
                    if st.strftime("%H") != "00":
                        continue
                    dg = None if diagnostic_prediction is None else diagnostic_prediction[b]
                    save_hindcast(self, surface_prediction[b], upper_air_prediction[b], dg, st,
                                  self._prediction_path(st, ens_id), attrs=dict(self.hc_opts.get("attrs", {}), member=str(ens_id)),
                                  **{k: v for k, v in self.hc_opts.items() if k != "attrs"})
            if self.save_mode in ("bigrun", "pilot", "trackb"):
                from s2storm.models.ucs2s_save import save_crops
                from s2storm.models.ucs2s_boxes import save_boxes, save_light
                for b, st in enumerate(start_times):
                    if st.strftime("%H") != "00":
                        continue
                    dg = None if diagnostic_prediction is None else diagnostic_prediction[b]
                    final = self._prediction_path(st, ens_id)     # written last: its existence = member done
                    if self.save_mode in ("pilot", "trackb"):
                        save_boxes(self, surface_prediction[b], upper_air_prediction[b], dg, st,
                                   final.replace("_crop.nc", "_boxes.nc").replace("_light.nc", "_boxes.nc"))
                    if self.save_mode == "trackb":
                        save_light(self, surface_prediction[b], upper_air_prediction[b], dg, st, final)
                    else:
                        save_crops(self, surface_prediction[b], upper_air_prediction[b], dg, st, final)
            self._t["save_s"] += time.time() - t; self._t["save_n"] += 1

        def timing_record(self):
            r = dict(self._t)
            if torch.cuda.is_available():
                r["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                r["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            r["s_per_base_step"] = r["base_s"] / max(r["base_n"], 1)
            r["s_per_pp_step"] = r["pp_s"] / max(r["pp_n"], 1)
            r["relax"] = self._relax_cfg
            return r

    return S2StormInference


if __name__ == "__main__":
    main()
