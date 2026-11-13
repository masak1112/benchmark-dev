"""
Field-level lead-time bias correction for FuXi and UC-S2S (S41u_plain) from Jhayron's archive.

Uses OLR-based RMM files (rmm_output/) produced by the pipeline.
LOYO bias estimated on u850_norm / u200_norm / OLR_norm vs ERA5 obs.
Obs coverage: 2019-2022 (OLR obs ends 2022-12-31).
Corrected files written to rmm_output_fbcorr/.

FuXi:  has_day0=True  (time[0] = init date)
S2S:   has_day0=False (time[0] = init+1)

Usage:
  python field_bias_correct_jhayron.py
"""

import numpy as np
import xarray as xr
import glob, os, re
import pandas as pd

YEARS_BC = {'2019', '2020', '2021', '2022'}   # years with obs coverage for bias estimation
MAXLEAD  = 46
OBS_FILE = '/glade/work/bgong/MJO_results/obs/olr_1p0deg/MJO_obs_created.nc'

MODELS = {
    'fuxi_jhayron': {
        'base_dir':  '/glade/campaign/univ/uchi0018/bing/mjo/fuxi_jhayron/olr_1p0deg',
        'has_day0':  True,
    },
    's2s_jhayron': {
        'base_dir':  '/glade/campaign/univ/uchi0018/bing/mjo/s2s_jhayron/olr_1p0deg',
        'has_day0':  False,
    },
}

FC_FIELDS  = ['u850_norm', 'u200_norm', 'OLR_norm']
OBS_FIELDS = ['u850_norm', 'u200_norm', 'olr_norm']

print("Loading ERA5 OLR obs ...")
obs_ds    = xr.open_dataset(OBS_FILE)
obs_times = pd.DatetimeIndex(obs_ds.time.values)
obs_idx   = {d.strftime('%Y%m%d'): i for i, d in enumerate(obs_times)}
print(f"  Obs covers {obs_times[0].date()} to {obs_times[-1].date()}")

def _date(fp):
    return re.search(r'\d{8}', os.path.basename(fp)).group()

def get_scales(ds_ref, has_day0):
    """Auto-detect EOF sign conventions by comparing reprojection vs stored RMM."""
    e1_u850 = ds_ref['eof1_u850'].values
    e2_u850 = ds_ref['eof2_u850'].values
    e1_u200 = ds_ref['eof1_u200'].values
    e2_u200 = ds_ref['eof2_u200'].values
    e1_olr  = ds_ref['eof1_olr'].values
    e2_olr  = ds_ref['eof2_olr'].values
    eig1    = ds_ref['eig_vals'].values[0]
    eig2    = ds_ref['eig_vals'].values[1]
    lead_idx = 1 if has_day0 else 0   # first valid lead
    u850 = ds_ref['u850_norm'].values[lead_idx, 0, :]
    u200 = ds_ref['u200_norm'].values[lead_idx, 0, :]
    olr  = ds_ref['OLR_norm'].values[lead_idx, 0, :]
    pc1 = (olr*e1_olr + u850*e1_u850 + u200*e1_u200).sum() / np.sqrt(eig1)
    pc2 = (olr*e2_olr + u850*e2_u850 + u200*e2_u200).sum() / np.sqrt(eig2)
    rmm1 = ds_ref['RMM1'].values[lead_idx, 0]
    rmm2 = ds_ref['RMM2'].values[lead_idx, 0]
    return (1 if np.sign(pc1)==np.sign(rmm1) else -1,
            1 if np.sign(pc2)==np.sign(rmm2) else -1)

def reproject(u850, u200, olr, ds_ref, scale1, scale2):
    """Fields shape: (leads, nmem, lon). Returns rmm1, rmm2: (leads, nmem)."""
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
    return pc1*scale1, pc2*scale2

def load_emean_fields(rmm_dir, has_day0):
    """Load ensemble-mean normalized fields for LOYO bias estimation."""
    files = sorted(glob.glob(os.path.join(rmm_dir, 'MJO_olr_Forecast_Init_????????.nc')))
    # Use ALL years for bias estimation (not just YEARS_BC) — LOYO will only use years with obs
    print(f"  {len(files)} RMM files found")

    inits, all_files, lead_dates_list = [], [], []
    fc_emean = {fv: [] for fv in FC_FIELDS}

    for fp in files:
        dt = _date(fp)
        ds = xr.open_dataset(fp)
        times  = ds.time.values
        n_t    = len(times)
        n_lon  = ds.sizes['longitude']

        # Pad lead array to MAXLEAD, inserting NaN at index 0 if has_day0=False
        lead_offset = 0 if has_day0 else 1   # actual data starts at lead_offset in MAXLEAD array

        for fv in FC_FIELDS:
            em = ds[fv].values.mean(axis=1)   # (time, lon) ensemble mean
            em_full = np.full((MAXLEAD, n_lon), np.nan)
            end = min(n_t, MAXLEAD - lead_offset)
            em_full[lead_offset:lead_offset+end] = em[:end]
            fc_emean[fv].append(em_full)

        # Lead dates aligned to MAXLEAD
        dates_full = np.array([np.datetime64('NaT', 'ns')] * MAXLEAD, dtype='datetime64[ns]')
        for l in range(min(n_t, MAXLEAD - lead_offset)):
            dates_full[lead_offset + l] = np.datetime64(times[l], 'ns')
        lead_dates_list.append(dates_full)
        inits.append(dt)
        all_files.append(fp)
        ds.close()

    fc_emean   = {fv: np.stack(fc_emean[fv])   for fv in FC_FIELDS}  # (n, MAXLEAD, lon)
    lead_dates = np.stack(lead_dates_list)                              # (n, MAXLEAD)
    return inits, fc_emean, lead_dates, all_files

def get_era5_norm(obs_fvar, lead_dates):
    obs_data = obs_ds[obs_fvar].values
    n, nl = lead_dates.shape
    n_lon = obs_data.shape[1]
    out   = np.full((n, nl, n_lon), np.nan)
    for i in range(n):
        for l in range(nl):
            dt = lead_dates[i, l]
            if np.isnat(np.datetime64(dt, 'ns')): continue
            dstr = pd.Timestamp(dt).strftime('%Y%m%d')
            idx  = obs_idx.get(dstr)
            if idx is not None:
                out[i, l] = obs_data[idx]
    return out

def compute_loyo_bias(fc_arr, era5, inits):
    """LOYO bias: for each init, mean error from all OTHER years that have obs."""
    years = np.array([d[:4] for d in inits])
    error = fc_arr - era5   # NaN where obs missing (2023-2024)
    bias  = np.full_like(error, np.nan)
    for i in range(len(inits)):
        yr = years[i]
        # other years that are in YEARS_BC (have obs)
        other = (years != yr) & np.array([y in YEARS_BC for y in years])
        if other.sum() == 0: continue
        bias[i] = np.nanmean(error[other], axis=0)
    return bias

def process_model(model_name, cfg):
    base_dir  = cfg['base_dir']
    has_day0  = cfg['has_day0']
    rmm_dir   = os.path.join(base_dir, 'rmm_output')
    out_dir   = os.path.join(base_dir, 'rmm_output_fbcorr')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  (has_day0={has_day0})")
    print(f"  LOYO bias years: {sorted(YEARS_BC)}")

    inits, fc_emean, lead_dates, all_files = load_emean_fields(rmm_dir, has_day0)
    print(f"  n_dates = {len(inits)}")

    bias_per_field = {}
    for fv, ov in zip(FC_FIELDS, OBS_FIELDS):
        print(f"  ERA5 match + LOYO bias for {fv} ...")
        era5 = get_era5_norm(ov, lead_dates)
        bias = compute_loyo_bias(fc_emean[fv], era5, inits)
        bias_per_field[fv] = bias
        print(f"    mean |bias| (non-NaN) = {np.nanmean(np.abs(bias)):.4f}")

    lead_offset = 0 if has_day0 else 1

    print(f"  Writing corrected RMM files to {out_dir} ...")
    for i, (fp, dt) in enumerate(zip(all_files, inits)):
        ds    = xr.open_dataset(fp)
        n_t   = ds.sizes['time']
        leads = min(n_t, MAXLEAD - lead_offset)

        u850_c = ds['u850_norm'].values[:leads].copy()  # (leads, nmem, lon)
        u200_c = ds['u200_norm'].values[:leads].copy()
        olr_c  = ds['OLR_norm'].values[:leads].copy()

        b_u850 = bias_per_field['u850_norm'][i, lead_offset:lead_offset+leads, :]
        b_u200 = bias_per_field['u200_norm'][i, lead_offset:lead_offset+leads, :]
        b_olr  = bias_per_field['OLR_norm'][i, lead_offset:lead_offset+leads, :]

        # Only apply bias where it's not NaN (i.e. where obs was available for estimation)
        u850_c -= np.where(np.isnan(b_u850), 0, b_u850)[:, np.newaxis, :]
        u200_c -= np.where(np.isnan(b_u200), 0, b_u200)[:, np.newaxis, :]
        olr_c  -= np.where(np.isnan(b_olr),  0, b_olr )[:, np.newaxis, :]

        scale1, scale2 = get_scales(ds, has_day0)
        rmm1_new, rmm2_new = reproject(u850_c, u200_c, olr_c, ds, scale1, scale2)

        ds_new = ds.copy(deep=True)
        ds_new['RMM1'].values[:leads]       = rmm1_new
        ds_new['RMM2'].values[:leads]       = rmm2_new
        ds_new['RMM1_emean'].values[:leads] = rmm1_new.mean(axis=-1)
        ds_new['RMM2_emean'].values[:leads] = rmm2_new.mean(axis=-1)
        ds_new['u850_norm'].values[:leads]  = u850_c
        ds_new['u200_norm'].values[:leads]  = u200_c
        ds_new['OLR_norm'].values[:leads]   = olr_c
        ds_new.attrs['bias_corrected'] = f'LOYO field-level BC using years {sorted(YEARS_BC)}'

        ds_new.to_netcdf(os.path.join(out_dir, os.path.basename(fp)))
        ds.close(); ds_new.close()

        if (i+1) % 50 == 0 or i == len(inits)-1:
            print(f"    {i+1}/{len(inits)} done")

for model_name, cfg in MODELS.items():
    process_model(model_name, cfg)

print("\nAll done.")
