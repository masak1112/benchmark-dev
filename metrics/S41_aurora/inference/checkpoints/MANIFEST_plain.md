# Checkpoints used by HPC_scripts/nvidia_brev_infer_postprocess_si.sh  (backed up 2026-09-23)

Run: `../inference_postprocess_si.py --yaml_config=../config/exp16_nvidia_v3.yaml --config=S2S --pp_ckpt=...`
Loader: inference_postprocess_si.py `_load_base_checkpoint()` and `_load_pp_checkpoint()`.

## loaded/  (all four are read at start-up, in this order)
| # | folder | original path | config key / flag | what it holds |
|---|--------|---------------|-------------------|---------------|
| 1 | 1_vae_c1 | /data/bing/S2S/checkpoints/c1/ckpt.tar | checkpoint_path_vae_c1 | VAE encoder for the SST component (component 8) -> latent-SI conditioning z |
| 2 | 2_pangu_deterministic | /data/bing/S2S/checkpoints/determinstic/ckpt.tar | checkpoint_path_det | deterministic Pangu encoder/decoder (model_det) |
| 3 | 3_base_full_state_rollout_finetuned | .../results/S2S/rollout_finetune_finetune_rollout_crps/training_checkpoints/rollout_ckpt_1000.tar | checkpoint_path_finetuned | full 571-entry base state: encoder + rollout-finetuned model_det + latent-SI UNet. Loaded LAST with strict=False, so its model_det overrides #2 |
| 4 | 4_postprocess_si | .../results/S2S/postprocess_si_pp_si_v2/training_checkpoints/pp_si_ckpt_10800.tar | --pp_ckpt | output-space postprocess SI (iters=10800, 96 fields, max_lead_steps=10) |

Log evidence (results/S2S/pp_si_infer_v2 logs, 2026-08-22 and 2026-08-28):
  "Base model loaded from .../rollout_ckpt_1000.tar (epoch=12, iters=1000)"
  "Postprocess SI loaded from results/S2S/postprocess_si_pp_si_v2/training_checkpoints/pp_si_ckpt_10800.tar (iters=10800, ...)"

## fallback_not_loaded/
latent_si_c1_full_state/last_ckpt.tar = config `checkpoint_path_diff`. Only used if
`checkpoint_path_finetuned` is missing; it is the checkpoint the rollout finetune (#3) started from.

## config/
exp16_nvidia_v3.yaml and the run script, as of the backup date.

Note: the latent-SI cross-attention (SST conditioning) gates in #3 and in the fallback are ~0,
so the SST conditioning is inactive in this run. See nvidia_brev_infer_postprocess_si_nosstcond.sh.
