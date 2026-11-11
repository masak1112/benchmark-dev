"""
Field-level lead-time bias correction for MJO.

For each model, bias is estimated on u850_norm/u200_norm/OLR_norm
(leave-one-year-out, ensemble mean vs ERA5 normalized obs).
Corrected per-member fields are re-projected using the EOFs already stored
in each RMM output file — no external solver needed.

Output: rmm_output_fbcorr/ alongside existing rmm_output/.

Models: UC-S2S V2, UC-S2S SSTanom, ECMWF
"""

import numpy as np
import xarray as xr
import glob, os, re
import pandas as pd

YEARS   = {'2019', '2020', '2021'}
MAXLEAD = 46
OBS_CREATED = '/glade/work/bgong/MJO_results/obs/chi200_1p0deg/MJO_obs_created.nc'

MODELS = {
    'uc_s2s_v2':      '/glade/campaign/univ/uchi0018/bing/mjo/ours_si_v3/chi200_1p0deg',
    'uc_s2s_sstanom': '/glade/campaign/univ/uchi0018/bing/mjo/uc_s2s_sstanom/chi200_1p0deg',
    'ecmwf_s2s':      '/glade/campaign/univ/uchi0018/bing/mjo/ecmwf_s2s/chi200_1p0deg',
}

FC_FIELDS  = ['u850_norm', 'u200_norm', 'OLR_norm']
OBS_FIELDS = ['u850_norm', 'u200_norm', 'olr_norm']

print("Loading ERA5 normalized obs ...")
obs_ds    = xr.open_dataset(OBS_CREATED)
obs_times = pd.DatetimeIndex(obs_ds.time.values)
obs_idx   = {d.strftime('%Y%m%d'): i for i, d in enumerate(obs_times)}

def _date(fp):
    return re.search(r'\d{8}', os.path.basename(fp)).group()

def get_scales(ds_ref):
    """
    Determine sign conventions by comparing reprojection of stored fields
    against stored RMM values. Returns scale1, scale2 (+1 or -1).
    """
    e1_u850 = ds_ref['eof1_u850'].values
    e2_u850 = ds_ref['eof2_u850'].values
    e1_u200 = ds_ref['eof1_u200'].values
    e2_u200 = ds_ref['eof2_u200'].values
    e1_olr  = ds_ref['eof1_olr'].values
    e2_olr  = ds_ref['eof2_olr'].values
    eig1    = ds_ref['eig_vals'].values[0]
    eig2    = ds_ref['eig_vals'].values[1]

    u850 = ds_ref['u850_norm'].values[1, 0, :]  # lead 1, member 0
    u200 = ds_ref['u200_norm'].values[1, 0, :]
    olr  = ds_ref['OLR_norm'].values[1, 0, :]

    pc1 = (olr*e1_olr + u850*e1_u850 + u200*e1_u200).sum() / np.sqrt(eig1)
    pc2 = (olr*e2_olr + u850*e2_u850 + u200*e2_u200).sum() / np.sqrt(eig2)

    rmm1_stored = ds_ref['RMM1'].values[1, 0]
    rmm2_stored = ds_ref['RMM2'].values[1, 0]

    scale1 = 1 if np.sign(pc1) == np.sign(rmm1_stored) else -1
    scale2 = 1 if np.sign(pc2) == np.sign(rmm2_stored) else -1
    return scale1, scale2

def reproject(u850, u200, olr, ds_ref, scale1, scale2):
    """
    Project corrected fields onto EOFs stored in ds_ref.
    Fields shape: (leads, nmem, lon)
    Returns rmm1, rmm2: (leads, nmem)
    """
    e1_u850 = ds_ref['eof1_u850'].values
    e2_u850 = ds_ref['eof2_u850'].values
    e1_u200 = ds_ref['eof1_u200'].values
    e2_u200 = ds_ref['eof2_u200'].values
    e1_olr  = ds_ref['eof1_olr'].values
    e2_olr  = ds_ref['eof2_olr'].values
    eig1    = ds_ref['eig_vals'].values[0]
    eig2    = ds_ref['eig_vals'].values[1]

    pc1 = (olr*e1_olr + u850*e1_u850 + u200*e1_u200).sum(axis=-1) / np.sqrt(eig1)
    pc2 = (olr*e2_olr + u850*e2_u850 + u200*e2_u200).sum(axis=-1) / np.sqrt(eig2)
    return pc1 * scale1, pc2 * scale2

def load_emean_fields(rmm_dir):
    """
    Load ensemble-mean normalized fields for bias estimation.
    Returns:
      inits      : list of YYYYMMDD  (n,)
      fc_emean   : dict fv -> (n, MAXLEAD, 360)
      lead_dates : (n, MAXLEAD) np.datetime64
      all_files  : list of paths
    """
    files = sorted(glob.glob(os.path.join(rmm_dir, 'MJO_chi200_Forecast_Init_????????.nc')))
    files = [f for f in files if _date(f)[:4] in YEARS]
    print(f"  {len(files)} RMM files for years {sorted(YEARS)}")

    inits, all_files, lead_dates_list = [], [], []
    fc_emean = {fv: [] for fv in FC_FIELDS}

    for fp in files:
        dt    = _date(fp)
        ds    = xr.open_dataset(fp)
        times = ds.time.values
        n_t   = len(times)
        n_lon = ds.sizes['longitude']

        for fv in FC_FIELDS:
            em = ds[fv].values.mean(axis=1)   # ensemble mean (time, lon)
            if n_t < MAXLEAD:
                pad = np.full((MAXLEAD - n_t, n_lon), np.nan)
                em  = np.concatenate([em, pad], axis=0)
            else:
                em = em[:MAXLEAD]
            fc_emean[fv].append(em)

        if n_t < MAXLEAD:
            extra = np.array([np.datetime64('NaT')] * (MAXLEAD - n_t))
            times_padded = np.concatenate([times, extra])
        else:
            times_padded = times[:MAXLEAD]
        lead_dates_list.append(times_padded)
        inits.append(dt)
        all_files.append(fp)
        ds.close()

    fc_emean   = {fv: np.stack(fc_emean[fv], axis=0) for fv in FC_FIELDS}  # (n, MAXLEAD, 360)
    lead_dates = np.stack(lead_dates_list, axis=0)                           # (n, MAXLEAD)
    return inits, fc_emean, lead_dates, all_files

def get_era5_norm(obs_fvar, lead_dates):
    obs_data = obs_ds[obs_fvar].values   # (n_time_obs, 360)
    n, nl    = lead_dates.shape
    n_lon    = obs_data.shape[1]
    out      = np.full((n, nl, n_lon), np.nan)
    for i in range(n):
        for l in range(nl):
            dt = lead_dates[i, l]
            if dt is None or str(dt) == 'NaT':
                continue
            dstr = pd.Timestamp(dt).strftime('%Y%m%d')
            idx  = obs_idx.get(dstr)
            if idx is not None:
                out[i, l] = obs_data[idx]
    return out

def compute_loyo_bias(fc_emean_arr, era5, inits):
    years = np.array([d[:4] for d in inits])
    error = fc_emean_arr - era5
    bias  = np.full_like(error, np.nan)
    for i in range(len(inits)):
        other = years != years[i]
        if other.sum() == 0: continue
        bias[i] = np.nanmean(error[other], axis=0)
    return bias

def process_model(model_name, base_dir):
    rmm_dir = os.path.join(base_dir, 'rmm_output')
    out_dir = os.path.join(base_dir, 'rmm_output_fbcorr')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")

    print("  Loading ensemble-mean fields for bias estimation ...")
    inits, fc_emean, lead_dates, all_files = load_emean_fields(rmm_dir)
    print(f"  n_dates = {len(inits)}")

    bias_per_field = {}
    for fv, ov in zip(FC_FIELDS, OBS_FIELDS):
        print(f"  ERA5 match + LOYO bias for {fv} ...")
        era5 = get_era5_norm(ov, lead_dates)
        bias = compute_loyo_bias(fc_emean[fv], era5, inits)
        bias_per_field[fv] = bias
        print(f"    mean |bias| = {np.nanmean(np.abs(bias)):.4f}")

    print(f"  Writing corrected RMM files to {out_dir} ...")
    for i, (fp, dt) in enumerate(zip(all_files, inits)):
        ds    = xr.open_dataset(fp)
        n_t   = ds.sizes['time']
        leads = min(n_t, MAXLEAD)

        # subtract bias from each member
        u850_c = ds['u850_norm'].values[:leads].copy()  # (leads, nmem, lon)
        u200_c = ds['u200_norm'].values[:leads].copy()
        olr_c  = ds['OLR_norm'].values[:leads].copy()

        u850_c -= bias_per_field['u850_norm'][i, :leads, :][:, np.newaxis, :]
        u200_c -= bias_per_field['u200_norm'][i, :leads, :][:, np.newaxis, :]
        olr_c  -= bias_per_field['OLR_norm'][i, :leads, :][:, np.newaxis, :]

        # re-project using EOFs stored in this file
        scale1, scale2 = get_scales(ds)
        rmm1_new, rmm2_new = reproject(u850_c, u200_c, olr_c, ds, scale1, scale2)

        ds_new = ds.copy(deep=True)
        ds_new['RMM1'].values[:leads]       = rmm1_new
        ds_new['RMM2'].values[:leads]       = rmm2_new
        ds_new['RMM1_emean'].values[:leads] = rmm1_new.mean(axis=-1)
        ds_new['RMM2_emean'].values[:leads] = rmm2_new.mean(axis=-1)
        ds_new['u850_norm'].values[:leads]  = u850_c
        ds_new['u200_norm'].values[:leads]  = u200_c
        ds_new['OLR_norm'].values[:leads]   = olr_c
        ds_new.attrs['bias_corrected'] = 'leave-one-year-out field-level bias correction'

        ds_new.to_netcdf(os.path.join(out_dir, os.path.basename(fp)))
        ds.close(); ds_new.close()

        if (i+1) % 50 == 0 or i == len(inits)-1:
            print(f"    {i+1}/{len(inits)} done")

# ── Main ────────────────────────────────────────────────────────────────────
for model_name, base_dir in MODELS.items():
    process_model(model_name, base_dir)

print("\nAll done.")
