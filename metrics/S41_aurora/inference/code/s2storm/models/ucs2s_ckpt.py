"""Strict state-dict checks of every UC-S2S checkpoint candidate against every config section.

Called by ``python -m s2storm.models.ucs2s --check_ckpts``. Writes ``ckpt_check.json`` in the run dir.
"""
import copy
import json
import logging
import traceback
from collections import OrderedDict

SECTIONS = ["S2S_MULTICOND", "S2S_SSTANOM", "S2S_SSTCOND", "S2S"]
BASE_CANDIDATES = [
    "loaded/3_base_full_state_rollout_finetuned_multicond/rollout_ckpt_1050.tar",
    "fallback_not_loaded/latent_si_multicond_stage1/last_ckpt.tar",
]
PP_CANDIDATES = [
    "loaded/4_postprocess_si/pp_si_ckpt_11400.tar",
    "loaded/4_postprocess_si/pp_si_ckpt_12800.tar",
]


def _state(path, torch):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["model_state"]
    if any(k.startswith("module.") for k in st):
        st = OrderedDict((k[7:], v) for k, v in st.items())
    meta = {k: (v if isinstance(v, (int, float, str)) else type(v).__name__)
            for k, v in ck.items() if k != "model_state" and not hasattr(v, "state_dict")}
    if "field_names" in ck:
        meta["field_names_n"] = len(ck["field_names"])
    return st, meta


def _compare(module, st):
    own = module.state_dict()
    missing = [k for k in own if k not in st]
    unexpected = [k for k in st if k not in own]
    shape = [(k, list(own[k].shape), list(st[k].shape)) for k in own if k in st and own[k].shape != st[k].shape]
    strict_ok = not (missing or unexpected or shape)
    if strict_ok:
        module.load_state_dict(st, strict=True)
    return dict(strict_ok=strict_ok, n_model=len(own), n_ckpt=len(st), n_missing=len(missing),
                n_unexpected=len(unexpected), n_shape_mismatch=len(shape),
                missing=missing[:40], unexpected=unexpected[:40], shape_mismatch=shape[:40])


def check_all(ips, params0, ck, exp_dir, world_rank):
    import torch
    from utils.YParams import YParams

    if world_rank != 0:
        return
    out = {}
    for sec in SECTIONS:
        rec = out.setdefault(sec, {})
        try:
            p = YParams(params0.config_filepath, sec)
            for k, v in params0.params.items():   # keep our path/run overrides, section-specific architecture
                if k.startswith(("checkpoint_path", "cond_anom_dir", "data_dir", "exp_dir", "experiment_dir",
                                 "run_num", "world_size", "local_rank", "resuming", "run_iter",
                                 "log_to_wandb", "has_diagnostic", "inference_steps", "init_date_set",
                                 "config_filepath", "num_ensemble_members", "climatology_file")):
                    p[k] = v
            p["has_diagnostic"] = bool(getattr(p, "diagnostic_variables", None))
            if not getattr(p, "cond_anom_vars", None) and getattr(p, "sst_cond_anom", False):
                rec["note"] = "single-SST anomaly section; sst_clim_path = " + str(getattr(p, "sst_clim_path", None))
            # build without loading the finetuned base: point it at a non-existent file
            p["checkpoint_path_finetuned"] = "/nonexistent"
            p["checkpoint_path_diff"] = "/nonexistent"
            runner = ips.PostprocessInference(p, 0, pp_ckpt=None, num_members=1, apply_correction=False)
            fresh = copy.deepcopy(runner.diff_model.state_dict())
            for cand in BASE_CANDIDATES:
                runner.diff_model.load_state_dict(fresh, strict=True)
                st, meta = _state(str(ck / cand), torch)
                r = _compare(runner.diff_model, st)
                # a stage-1 checkpoint may hold only the UNet part
                if not r["strict_ok"] and hasattr(runner.diff_model, "unet"):
                    sub = OrderedDict((k[5:], v) for k, v in st.items() if k.startswith("unet."))
                    if sub:
                        r["unet_only"] = _compare(runner.diff_model.unet, sub)
                r["meta"] = meta
                rec[cand] = r
            runner.diff_model.load_state_dict(fresh, strict=True)
            for cand in PP_CANDIDATES:
                try:
                    runner.pp_model = None
                    runner._load_pp_checkpoint(str(ck / cand))
                    _, meta = _state(str(ck / cand), torch)
                    rec[cand] = dict(strict_ok=True, n_fields=runner.n_fields,
                                     max_lead_steps=runner.max_lead_steps, meta=meta)
                except Exception as e:  # noqa: BLE001
                    rec[cand] = dict(strict_ok=False, error=repr(e)[:2000])
            del runner
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            rec["build_error"] = repr(e)[:2000]
            rec["traceback"] = traceback.format_exc()[-4000:]
        logging.info(f"ckpt check {sec}: " + json.dumps({k: (v.get('strict_ok') if isinstance(v, dict) else v)
                                                         for k, v in rec.items()}, default=str))
    (exp_dir / "ckpt_check.json").write_text(json.dumps(out, indent=1, default=str))
