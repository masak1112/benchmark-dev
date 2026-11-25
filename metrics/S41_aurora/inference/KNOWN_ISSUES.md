# Known issues (facts with evidence; anything not shown by a file, log or test is marked [inferred])

Line numbers refer to upstream `masak1112/S2S` at **db18364** (`v2.0/…`) unless the path starts with `code/` (this package, s2storm 0b3e213).
Numbers are from the S41 evaluation report (research-os `projects/s2storm/reports/2026-11-11_aurora_bing-evaluation.md`, sections named
below) and the fix-test summary (`runs/bing/eval/v1.5/fixtest/fixtest_summary.md` on Aurora).

## 1. +1.1 K SST jump from lead 0 to lead 1, both models
- **Plain model, production members (fix test "old"), 12 starts × 10 members:** global-ocean area-mean SST day 1 − day 0 = **+1.152 K**
  [95 % CI +1.146, +1.157]; ERA5 +0.006 K. Tropics +1.543 K (ERA5 +0.001 K). Every start +1.14 … +1.17 K (global) and +1.52 … +1.56 K (tropics).
  The SST stays high: global mean 293.03 K at days 1–2, 292.65 K at day 14, vs ERA5 291.89 → 291.95 K.
- **Both checkpoints (report §"v2.3 W3 SST initial offset", ocean only):** +1.08 K global, +1.44 K tropics (20°S–20°N) at lead 1 in the
  SST-anomaly and the plain model (starts 2025-01-06 and 2024-07-01). It then decays: +0.9 K (day 7), +0.65–0.75 (day 14), +0.45–0.6 (day 21),
  +0.3–0.47 (day 28).
- Lead 0 is the input. For start 2025-01-06 member 0, the h5 input SST and the saved lead-0 SST are identical on 1° ocean points (max |Δ| 0).
  The input regridded to 1.5° vs ERA5 1.5° differs by −0.0003 K.
- Our inputs are +0.29 K (2019, 2021) to +0.53 K (2024–25) warmer in the global ocean mean than the model's own 1979–2018 SST
  climatology (`clim_00z_sst_std.npy`) (report §v2.3).
- The fix-test maps (`FIX2_sst_day1m0_maps.png`) show the jump nearly uniform over the open ocean: +1.5 K in the tropics, falling to ≈ 0
  poleward of 50°. t2m has no such jump (global day 1 − day 0 +0.08 K vs ERA5 +0.02 K, fix-test summary).
- Not tested: base model alone (`--no_correction`) vs base + postprocess SI at step 0; normalised-space values at step 0.

### Where step 0 gets its SST (code path, no interpretation)
1. **h5 field.** `<data_dir>/<YYYY>_<NNNN>.h5`, group `input`, dataset `sea_surface_temperature` (180 × 360 float32, NaN on land, 21,972 cells).
   It is the point sample of the 00 UTC ARCO 0.25° value at each 1° cell centre (`code/s2storm/data/era5_h5.py:42-43, 56-62, 91-94`).
   Upstream reads it in `GetDataset._get_data` (`utils/data_loader_multifiles.py:549-561`) → `get_data_given_path` (`:75-83`). The variable
   order is upper-air × levels, then `surface_variables` (`:352-360`).
2. **Channel and land fill.** The surface channels are `surface_variables` + `land_variables` + `ocean_variables` (`:221-239`), so SST is
   channel 8 (t2m, u10, v10, msl, sp, swvl1, stl1, skt, sst). `_reshape_and_mask_variables` (`:362-371`) calls `_fill_mask` (`:396-408`)
   with `mask_fill['sea_surface_temperature'] = 270.` (config `mask_fill` in `S2S` / `S2S_SSTANOM`; the default at `:204-208` is the same).
   Land points therefore enter as 270 K.
3. **Normalisation.** `surface_transform` (`:488-489`): `(x − surface_mean) / surface_std`, applied at `:758` (inference builds the dataset with
   `validate=True`, `train.py:307-311`, which goes through the branch at `:725-759`). Mean and std come from `<data_dir>/pangu_s2s_1979-2018_surface_mean.nc`
   and `<data_dir>/pangu_s2s_1979-2018_surface_std.nc` (config keys `surface_mean`, `surface_std`), read by `load_mean_std` (`:288-289`,
   `:461-486`). sha256 in `data/normalisation_sha256.txt`.
4. **Into the model.** `predict` (`inference_postprocess_si.py:522-532`): `default_collate([valid_dataset[batch_idx]])`, then `pop_sst_cond`
   (`:525`, `utils/sst_cond.py:10-19`), then `in_sfc0` (normalised, float32).
5. **SST conditioning of the latent SI.**
   - SST-anomaly model (`sst_cond_anom: True`): `GetDataset.__getitem__` (`:569-578`) appends `_make_sst_cond(out[0], start_time)`
     (`:622-643`). That is the normalised IC SST channel minus `sst_clim[doy]` (`clim_00z_sst_std.npy`, loaded at `:251`; path set by
     `code/s2storm/models/ucs2s.py:162-164`), × ocean mask / `sst_anom_scale` (0.0583), masked-bilinear to `sst_cond_size` [23, 45], plus an
     ocean-fraction channel. With `--sst_cond_mode persistent` the same map is used at every step (`inference_postprocess_si.py:447-448`).
     The base model then uses it directly (`networks/diffusion.py:780-781`, VAE bypassed).
   - Plain model: `sst_cond` is None, so `ConditionalDiffusionModel._encode_vae` (`networks/diffusion.py:771-790`) runs
     `encoder._select_surface_component` (`networks/vae.py:228-238`, `component: 8` = SST of the concatenated input
     `[surface, constant boundary, varying boundary]`, `networks/stochastic_interpolant.py:537`), then `encoder.encode` (`networks/vae.py:250`).
     `vae_cond_mode` is absent in `S2S`, so the default `'sample'` applies (`networks/diffusion.py:707`) and the reparameterised sample z is
     used (`:786-790`). Bing's MANIFEST for the plain bundle says the cross-attention gates of this conditioning are ≈ 0.
6. **Step 0 base model.** `_base_step(in_sfc0, in_ua0, cb, vbc[:, 0], step=0)` (`inference_postprocess_si.py:596-597`) →
   `diff_model.prediction` (`:449-461`; `networks/stochastic_interpolant.py:507-625`). The varying boundary at step 0 is the IC-time
   `toa_incident_solar_radiation` (`utils/data_loader_multifiles.py:711-723`). MC dropout is on (`mc_dropout=True` in the job log; `:117-120`).
7. **Step 0 postprocess SI.** `_apply_correction(p_sfc, p_ua, p_diag, step=0)` (`:600-601`, `:349-363`): `lead_frac = (step + 1) / max_lead_steps`
   = 1/10, `corrected = member + blend · residual` (blend 1.0). With `--feedback base`, step 1's input is the uncorrected `p_sfc` (`:617-620`).
8. **Written.** Lead 0: `out_sfc[:, 0] = surface_inv_transform(in_sfc0)` (`:577-578`), i.e. the h5 SST with 270 K on land. Lead 1:
   `out_sfc[:, 1] = surface_inv_transform(c_sfc)` (`:605-606`), the corrected field. Our `save_prediction` override (`code/s2storm/models/ucs2s.py:398-433`)
   writes `sst = sfc[:, index('sea_surface_temperature')]` in K on the native grid, leads 0–45 (`code/s2storm/models/ucs2s_hindcast.py:103-117`),
   into `*_h25.nc`. Member 0's full upstream file comes from `super().save_prediction` (`inference_postprocess_si.py:651-`).

## 2. Member seeds repeat across starts (our launch)
- Every task runs one start (`--init_dates <date>`), so upstream's `batch_idx` = 0 and seed = `100000·0 + m` = m
  (`inference_postprocess_si.py:565-567`; README "How a member id becomes a seed"). Confirmed by a bitwise-reproducible control rerun
  (job 9077592). Bing's scripts run a year per job (`--years`), so her `batch_idx` differs per start.
- Effect (report §"Seed repetition"): each member carries a start-independent offset pattern (odd vs even starts r = 0.86–1.00 for SST,
  t2m, precipitation, u850). It is 49 % (SST-anomaly) / 41 % (plain) of the weeks-3–4 SST ensemble variance, 7 / 6 % for t2m,
  10 / 5 % for precipitation.
- Reseed test (per-start seed offset, s2storm 0685220; 89 starts × 10 members, job 9080400): offsets vanish (r −0.04 … +0.05). Spread/error
  is unchanged (±3 %). Paired CRPSS/ACC changes are noise for t2m, tp24, z500, olr, u850 and Niño-3.4. SST CRPSS is robustly lower with
  per-start seeds: global −0.017 (fixed clim) / −0.021 (trend).
- Not related to the day-1 SST jump: the jump is the same at every start and in the reseeded runs [inferred from the per-start values in
  §1, the reseed runs were not checked at day 1].

## 3. OLR saved as 0 upstream; fixed in our wrapper
- `predict` clips the saved diagnostics in `nonneg_vars` at 0 (`inference_postprocess_si.py:505-506, 612-614`). That set includes
  `mean_top_net_long_wave_radiation_flux`, which is negative (ERA5 net TOA LW), so every saved OLR value is 0. The rollout itself uses the
  unclipped tensor.
- Our fix (s2storm d7cede4, `code/s2storm/models/ucs2s.py:379-396, 401-403`): `diagnostic_inv_transform` returns −OLR (positive, survives
  the clip), and `save_prediction` flips it back. Lead 0 stays 0 (no diagnostic at the IC).
- Test 9074225: OLR at day 1 −361 … −43 W m⁻², global mean −227 (day 45 −225). Every other field is bitwise identical to the unfixed run
  (`runs/bing/debug/olrfix_check.json`). Bing's own output files have OLR = 0 for the same reason.

## 4. Lead-0 SST offset was our verification, not the model
- The saved fields contain 270 K on land (upstream `_fill_mask`, §1 step 2), and the model keeps it (lead 10 land SST 270.02 ± 0.17 K).
  Our 1° → 1.5° conservative remap mixed that fill into coastal cells. Saved lead 0 → 1.5° vs ERA5: −0.36 K (1st percentile −8.6 K); with
  land masked, −0.0003 K. Fixed in the evaluation (`bing-eval` 4b9a4ea): SST NaN on land and swvl1/stl1 NaN over sea before the remap,
  rescore v1.5 (report §v2.3).

## 5. Fix test of Bing's 7ceed2a (boundary off-by-one): day 0/1 bitwise identical, jump unchanged
- Code: `runs/bing/ext/S2S_fix_7ceed2a` = db18364 + cherry-pick of 7ceed2a = **6982d54**. The change: `boundary_times` = start +
  `range(1, max_lead_time + 1)` days instead of `range(max_lead_time)` (`utils/data_loader_multifiles.py:716`). The boundary list
  `[t0] + boundary_times` was `[t0, t0, t1, …]` and becomes `[t0, t1, …]`. Step s uses `vbc[:, s]`, so step 0 gets `t0` in both versions.
- Run: plain model, 12 starts × members 0–9, same launcher (0b3e213), same inputs and seeds (jobs 9097167 smoke, 9097212, 9097295).
- Result (`fixtest_summary.md`): **days 0 and 1 bitwise identical** in all starts for sst, t2m, olr, z500 and tp24; first difference at lead 2.
  The day-1 − day-0 SST is +1.152 K (new) vs +1.152 K (old) globally and +1.543 vs +1.543 K in the tropics. Weekly score changes ≤ 0.2 % RMSE.

## 6. Start 2026-09-21 had two bad input days at the end
- `2026_1228.h5` (2026-11-04) was built from ARCO on 2026-11-11 at 18:4x UTC while ARCO still had a gap: the file was all zeros. In
  `2026_1232.h5` (11-05) tp24 was incomplete. Both are now quarantined in `aurora_bundle_v5/quarantine_arco_gap/`. The 2026-09-21 runs
  (job 9076613, both models) read them as the varying boundary and validation targets of the last days. Days 44–45 can be affected; scores
  use leads ≤ 42 [inferred]. Rerun planned together with 2026-09-28, which is not run.

## 7. Small differences from upstream defaults worth knowing
- `checkpoint_path_diff` pointed at a non-existent file in production (the `_c16` checkpoint copies have no `fallback_not_loaded/`). Upstream reads
  it only when `checkpoint_path_finetuned` is missing (`inference_postprocess_si.py:196-212`), and the strict-load report shows the
  finetuned file was loaded (571/571 keys).
- Bundle `exp16_nvidia_v3.yaml` (`S2S`) has `train_year_start: 1984`; upstream db18364 has 1979 (`configs/resolved_section_diff.txt`).
- `--mc_dropout` was not passed, so MC dropout is on in the base model (`mc_dropout=True` in the job logs).
- 9 task attempts failed (exit 1 within 2–29 s) and succeeded on retry. Their logs were overwritten (`hpc/jobs/logs/failed_attempts.jsonl`).
