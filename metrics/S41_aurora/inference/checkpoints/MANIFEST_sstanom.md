# Checkpoints used by HPC_scripts/midway_infer_sstanom_2019.sh  (S2S_SSTANOM, backed up 2026-10-23)

Model: latent SI conditioned on the PERSISTENT SST ANOMALY (no SST VAE in the conditioning path),
rollout-CRPS-finetuned decoder, output-space postprocess SI.
Run: `../inference_postprocess_si.py --yaml_config=../config/exp16_midway_sstanom.yaml --config=S2S_SSTANOM --base_ckpt=... --pp_ckpt=...`
Loader: inference_postprocess_si.py `_load_base_checkpoint()` and `_load_pp_checkpoint()`.
Code: git commit d0356ba (branch sst-forecast-vae) of github.com/masak1112/S2S.

## loaded/  (all four are read at start-up, in this order)
| # | folder | original path (under S2S/v2.0/HPC_scripts/ unless absolute) | config key / flag | what it holds |
|---|--------|---------------|-------------------|---------------|
| 1 | 1_vae_c1 | checkpoints_UC-S2S_V2/loaded/1_vae_c1/ckpt.tar | checkpoint_path_vae_c1 | SST VAE encoder weights. Still loaded (frozen, strict) by the loaders but NOT used for conditioning in this model. Identical to V2. |
| 2 | 2_pangu_deterministic | checkpoints_UC-S2S_V2/loaded/2_pangu_deterministic/ckpt.tar | checkpoint_path_det | deterministic Pangu encoder/decoder (model_det). Identical to V2. |
| 3 | 3_base_full_state_rollout_finetuned_sstanom | results/S2S_SSTANOM/rollout_finetune_rollout_sstanom/training_checkpoints/rollout_ckpt_650.tar | checkpoint_path_finetuned / --base_ckpt | full state: encoder + rollout-finetuned model_det + stage-1 latent-SI UNet (frozen). iters=650, epoch 2. Decoder started from V2 rollout_ckpt_1000; rollout length ramped 1->2 steps (interval 500). Loaded LAST, overrides #2 |
| 4 | 4_postprocess_si | results/S2S_SSTANOM/postprocess_si_pp_si_sstanom/training_checkpoints/pp_si_ckpt_11400.tar | --pp_ckpt | output-space postprocess SI, 96 fields, max_lead_steps=10, iters=11400; warm-started from V2 pp_si_ckpt_10800 and trained against base #3 with rollout_steps=1 |

## fallback_not_loaded/
latent_si_sstanom_stage1/last_ckpt.tar = results/S2S_SSTANOM/si_c1_sstanom/training_checkpoints/last_ckpt.tar
(= diff_ckpt_2001, stage-1 latent SI on the persistent SST anomaly, warm-started from V2
latent_si_c1_full_state with the conditioning branch re-initialised). It is the UNet inside #3
and the `checkpoint_path_diff` of the config; the inference loader only reads it if
`checkpoint_path_finetuned` is missing.

## config/
exp16_midway_sstanom.yaml (block S2S_SSTANOM), the Midway training pipeline and inference scripts,
and clim_00z_sst_std.npy (366x180x360 daily SST climatology in standardized units, NaN=land;
`sst_clim_path` in the config, needed to build the conditioning map at inference).
Config paths point at the original locations; to run from this folder, repoint
checkpoint_path_* / --base_ckpt / --pp_ckpt to the files above.

## Notes
- Validation of the postprocess SI (iter 11500, 4 ICs x 4 members): SSR rises to ~0.6-1.2 but CRPS
  only improves for z500 3-5d and t2m 5d; best blend is lead dependent (~0.65 at 1d, 1.0-1.5 at 3-5d).
- SST conditioning in the latent SI is weak: cond_rel_effect ~0.012 and falling; the bottleneck
  (cross_attn_mid) branch received exactly zero gradient.
