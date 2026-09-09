"""
Compute regional ACC/RMSE metrics for deterministic PlaSim forecasts.
Supports global, tropics, NH, and SH regions.
"""

import numpy as np
import netCDF4
import h5py
import os
import pickle
from tqdm import tqdm

# ==================== Configuration ====================
results_path = './results/'
data_root_path = '/glade/campaign/univ/uchi0018/weidong/PLASIM/sim52/h5/plev_data/'
pred_root_path = '/glade/campaign/univ/uchi0014/weidong/FB/results/PlaSim/ShortTerm/'

intervel = 4
dt = int(intervel * 6)  # timestep in hours
train_start, train_end = 7, 47
model_type = 'FC_BC'  # FC_BC or FB
model_name = f'PlaSim_{model_type}'
fc_root_path = f'{pred_root_path}/{dt}/'
bc_root_path = f'{pred_root_path}/{dt}/'

inference_start = 106
inference_end = 109
num_forecast_steps = 15

print(model_name, train_start, train_end)

# Variable definitions
upper_air_vars = ['ta', 'ua', 'va', 'hus', 'zg']
surface_vars = ['tas', 'pl', 'ts', 'mrso', 'sst']

levs_pa = [100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0,
           40000.0, 30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 5000.0]

var_name_map = {
    'tas': '2m_temperature', 'pl': 'surface_pressure', 'ts': 'skin_temperature',
    'mrso': 'soil_moisture', 'sst': 'sea_surface_temperature',
    'ta': 'temperature', 'ua': 'u_wind', 'va': 'v_wind',
    'hus': 'specific_humidity', 'zg': 'geopotential'
}

# ==================== Regional Configuration ====================
REGION_CONFIG = {
    'global':  {'lat_min': -90, 'lat_max': 90,  'suffix': ''},
    'tropics': {'lat_min': -30, 'lat_max': 30,  'suffix': '_tropics'},
    'NH':      {'lat_min':  30, 'lat_max': 90,  'suffix': '_NH'},
    'SH':      {'lat_min': -90, 'lat_max': -30, 'suffix': '_SH'},
}

# PlaSim grid: 64 lat x 128 lon
lat_plasim = np.linspace(90 - 180/(2*64), -90 + 180/(2*64), 64)

# ==================== Utility Functions ====================
def gen_weights(lat):
    """Generate area weights based on latitude"""
    lon = np.arange(0, 360, 360/128)
    weights = np.cos(lat * np.pi / 180)
    W = np.tile(weights, (len(lon), 1)).T
    return W

def gen_regional_mask(lat, lat_min, lat_max):
    """Generate regional mask based on latitude bounds"""
    lon = np.arange(0, 360, 360/128)
    mask = (lat >= lat_min) & (lat <= lat_max)
    M = np.tile(mask, (len(lon), 1)).T
    return M

# Global weight matrix
weight = gen_weights(lat_plasim)

def weighted_average(X, W, mask=None):
    """Compute weighted average, optionally with regional mask"""
    if W.shape != X.shape:
        W = np.broadcast_to(W, X.shape)
    if mask is not None:
        if mask.shape != X.shape:
            mask = np.broadcast_to(mask, X.shape)
        indices = ~np.isnan(X) & mask
    else:
        indices = ~np.isnan(X)
    return np.average(X[indices], weights=W[indices])

def metric_rmse(F, A, W, mask=None):
    """Compute weighted RMSE"""
    return np.sqrt(weighted_average(np.square(F - A), W, mask))

def metric_acc(F, A, W, mask=None):
    """Compute weighted ACC (Anomaly Correlation Coefficient)"""
    F, A = np.squeeze(F), np.squeeze(A)
    a = weighted_average(F * A, W, mask)
    b = np.sqrt(weighted_average(A * A, W, mask))
    c = np.sqrt(weighted_average(F * F, W, mask))
    return a / b / c

# ==================== Data Loading ====================
def load_climatology(var_name, level=None):
    """Load climatology data"""
    with netCDF4.Dataset(f'{data_root_path}/climatology.nc') as nc:
        if var_name in upper_air_vars and level is not None:
            lev_idx = np.where(nc.variables['plev'][:] == level)[0][0]
            return nc[var_name][:, lev_idx, :, :]
        return nc[var_name][:, :, :]

def get_climatology_for_date(var_clim_all, date_str):
    """Get climatology for specific date"""
    step = int(date_str.split('_')[1])
    return var_clim_all[(step // 4) % 365]

def load_GT(date_str, var_name, level):
    """Load ground truth data"""
    with h5py.File(f"{data_root_path}/{date_str}.h5", "r") as h5file:
        if var_name in upper_air_vars:
            key = f"input/{var_name}_{level}"
        else:
            key = f"input/{var_name}"
        return h5file[key][:].data

def load_prediction(initial_date, direction, var_name, level):
    """Load prediction data for deterministic model"""
    year, step = initial_date.split('_')
    nc_var_name = var_name_map[var_name]
    
    # Construct prediction filename based on model type
    if model_type == 'FB':
        pred_file = f'{fc_root_path}/pred_FB_TrainF{train_start}to{train_end}_{direction}_{dt}_date{year}_{step}.nc'
    elif model_type == 'FC_BC':
        model_prefix = 'FC' if direction == 'forward' else 'BC'
        pred_file = f'{fc_root_path}/pred_{model_prefix}_TrainF{train_start}to{train_end}_{direction}_{dt}_date{year}_{step}.nc'
    
    if not os.path.exists(pred_file):
        return None, None
    
    try:
        with netCDF4.Dataset(pred_file) as nc:
            forecast_dates = [str(d) for d in nc.variables['forecast_date'][:]]
            
            if var_name in upper_air_vars and level is not None:
                plev_pred = nc.variables['level'][:]
                lev_idx = np.where(plev_pred == level)[0][0]
                data = nc[nc_var_name][:, lev_idx, :, :]
            else:
                data = nc[nc_var_name][:, :, :]
            
            return data, forecast_dates
    except Exception as e:
        print(f"\nError loading {pred_file}: {e}")
        return None, None

# ==================== Main Computation ====================
def compute_regional_metrics(var_name, level=None, regions=None):
    """
    Compute regional metrics for deterministic PlaSim forecasts.
    
    Parameters:
        var_name: Variable name (e.g., 'ta', 'zg', 'tas')
        level: Pressure level in Pa (e.g., 50000.0) or None for surface vars
        regions: List of regions to compute ['global', 'tropics', 'NH', 'SH']
    """
    if regions is None:
        regions = ['tropics', 'NH', 'SH']
    
    # Load climatology once
    var_clim_all = load_climatology(var_name, level)
    
    # Generate date list
    date_list = [f"{year}_{day*4:04d}" 
                 for year in range(inference_start, inference_end + 1) 
                 for day in range(0, 365, 5)]
    
    # Prepare regional masks
    regional_masks = {
        region: gen_regional_mask(lat_plasim, 
                                  REGION_CONFIG[region]['lat_min'],
                                  REGION_CONFIG[region]['lat_max'])
        for region in regions
    }
    
    print(f"\n{'='*60}")
    print(f"Processing {var_name}" + (f" @ {level/100:.0f}hPa" if level else " (surface)"))
    print(f"{'='*60}")
    print(f"Total cases: {len(date_list)}")
    print(f"Regions: {regions}")
    
    # Initialize metrics dictionary for all regions
    all_metrics = {}
    for region in regions:
        all_metrics[region] = {
            'FC': {'ACC': {}, 'RMSE': {}, 'direction': 'forward'},
            'BC': {'ACC': {}, 'RMSE': {}, 'direction': 'backward'},
            'config': REGION_CONFIG[region]
        }
    
    # ==================== Process Both Directions ====================
    for model_key, direction, desc in [('FC', 'forward', 'Forward'), 
                                        ('BC', 'backward', 'Backward')]:
        print(f"\n=== Computing {desc} Direction Metrics ===")
        
        for initial_date in tqdm(date_list, desc=desc):
            pred_data, forecast_dates = load_prediction(
                initial_date, direction, var_name, level
            )
            if pred_data is None:
                continue
            
            # Initialize per-region lists for this date
            region_acc = {r: [] for r in regions}
            region_rmse = {r: [] for r in regions}
            
            # Process each forecast step
            for i, target_date in enumerate(forecast_dates[:num_forecast_steps]):
                try:
                    var_GT = load_GT(target_date, var_name, level)
                    var_clim = get_climatology_for_date(var_clim_all, target_date)
                    GT_anom = var_GT - var_clim
                    pred_anom = pred_data[i] - var_clim
                    
                    # Compute metrics for each region
                    for region in regions:
                        mask = regional_masks[region]
                        region_acc[region].append(
                            metric_acc(pred_anom, GT_anom, weight, mask)
                        )
                        region_rmse[region].append(
                            metric_rmse(pred_anom, GT_anom, weight, mask)
                        )
                except Exception as e:
                    continue
            
            # Store metrics for each region
            for region in regions:
                if region_acc[region]:
                    all_metrics[region][model_key]['ACC'][initial_date] = region_acc[region]
                    all_metrics[region][model_key]['RMSE'][initial_date] = region_rmse[region]
    
    # ==================== Save Results ====================
    os.makedirs(results_path, exist_ok=True)
    save_name = f"{int(level/100)}" if level is not None else 'surface'
    
    # Save combined file with all regions
    output_file = f"{results_path}/{var_name}_{save_name}_{model_name}_Train{train_start}to{train_end}_regional_metrics.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(all_metrics, f)
    print(f"\nSaved combined: {output_file}")
    
    # Save individual region files
    for region in regions:
        suffix = REGION_CONFIG[region]['suffix']
        region_file = f"{results_path}/{var_name}_{save_name}_{model_name}_Train{train_start}to{train_end}{suffix}_metrics.pkl"
        with open(region_file, "wb") as f:
            pickle.dump(all_metrics[region], f)
        print(f"Saved: {region_file}")
    
    # ==================== Print Summary ====================
    print_summary(all_metrics, regions)
    
    return all_metrics


def print_summary(all_metrics, regions):
    """Print summary statistics for all regions"""
    print("\n" + "="*70)
    print("=== Summary Statistics (Deterministic Model) ===")
    print("="*70)
    
    for region in regions:
        print(f"\n>>> Region: {region.upper()} <<<")
        print("-"*50)
        
        for direction_name, model_key in [('FORWARD', 'FC'), ('BACKWARD', 'BC')]:
            metrics = all_metrics[region][model_key]
            if metrics['ACC']:
                acc_all = np.concatenate(list(metrics['ACC'].values()))
                rmse_all = np.concatenate(list(metrics['RMSE'].values()))
                
                print(f"  {direction_name}:")
                print(f"    Cases: {len(metrics['ACC'])}")
                print(f"    ACC:  mean={acc_all.mean():.4f}, std={acc_all.std():.4f}, "
                      f"min={acc_all.min():.4f}, max={acc_all.max():.4f}")
                print(f"    RMSE: mean={rmse_all.mean():.4f}, std={rmse_all.std():.4f}, "
                      f"min={rmse_all.min():.4f}, max={rmse_all.max():.4f}")


# ==================== Entry Point ====================
if __name__ == "__main__":
    os.makedirs(results_path, exist_ok=True)
    
    # Define regions to compute
    # regions_to_compute = ['tropics', 'NH', 'SH']
    regions_to_compute = ['global', 'tropics', 'NH', 'SH']
    
    # Surface variables
    # compute_regional_metrics('tas', regions=regions_to_compute)
    
    # Upper air variables
    # compute_regional_metrics('ta', level=50000.0, regions=regions_to_compute)
    compute_regional_metrics('zg', level=50000.0, regions=regions_to_compute)
    compute_regional_metrics('ua', level=85000.0, regions=regions_to_compute)
    compute_regional_metrics('ua', level=25000.0, regions=regions_to_compute)