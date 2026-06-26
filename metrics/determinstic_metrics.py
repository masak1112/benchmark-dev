import numpy as np
from datetime import datetime, timedelta, date
import os
import netCDF4
import h5py
from tqdm import tqdm
import pickle
from numpy.lib.stride_tricks import as_strided

# ==================== Configuration ====================
results_path = '/glade/work/bgong'
data_root_path = '/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/'
fc_root_path = '/glade/campaign/univ/uchi0014/weidong/FB/results/ERA5/S2S_8001/'
bc_root_path = '/glade/derecho/scratch/yqsun/FB/s2s-retrainE-BC/results/S2S/2003/predictions/'

'''
3 seeds version:
FC: 
/glade/campaign/univ/uchi0014/weidong/FB/results/ERA5/S2S_8001/    ~ 45 seed1
FC_len, BC_len = 45, 20

/glade/derecho/scratch/yqsun/FB/s2s-retrainD/results/S2S/1003/predictions/   seed2
/glade/campaign/univ/uchi0014/yqsun/s2s-retrainD/results/S2S/1002/predictions/   seed3   ~~ replace


BC: 
/glade/derecho/scratch/yqsun/FB/s2s-retrainE-BC/results/S2S/2003/predictions/   seed1
/glade/derecho/scratch/yqsun/FB/s2s-retrainE-BC/results/S2S/2004/predictions/   seed2
/glade/derecho/scratch/yqsun/FB/s2s-retrainE-BC/results/S2S/2005/predictions/   seed3   ~~ replace
'''

FC_len, BC_len = 45, 20
model_name = 'FC-BC_seed1'
print(model_name)
is_FC_s2s = True
FC_idx, BC_idx = '8001', '2003'

# ==================== Regional Configuration ====================
# Options: 'global', 'tropics', 'NH', 'SH'
# tropics: 30S-30N
# NH: 30N-90N
# SH: 90S-30S
REGION = 'tropics'  # Change this to 'global', 'tropics', 'NH', or 'SH'

REGION_CONFIG = {
    'global': {'lat_min': -90, 'lat_max': 90, 'suffix': ''},
    'tropics': {'lat_min': -30, 'lat_max': 30, 'suffix': '_tropics'},
    'NH': {'lat_min': 30, 'lat_max': 90, 'suffix': '_NH'},
    'SH': {'lat_min': -90, 'lat_max': -30, 'suffix': '_SH'},
}

upper_air_vars = ['u_component_of_wind', 'v_component_of_wind',
                  'temperature', 'specific_humidity', 'geopotential']
surface_vars = ['2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind',
                'mean_sea_level_pressure','surface_pressure',
                'sea_surface_temperature',
                'skin_temperature','soil_temperature_level_1','volumetric_soil_water_layer_1']

levs = [5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
if is_FC_s2s:
    plevs_fc = [5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    plevs_bc = [5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
else:
    plevs_fc = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    plevs_bc = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# ==================== Utility Functions ====================
def gen_weights(lat):
    """Generate cosine latitude weights"""
    lon = np.arange(0, 360, 1)
    weights = np.cos(lat*np.pi/180)
    W = np.tile(weights, (len(lon), 1)).T
    return W

def gen_regional_mask(lat, lat_min, lat_max):
    """Generate regional mask based on latitude bounds"""
    lon = np.arange(0, 360, 1)
    mask = (lat >= lat_min) & (lat <= lat_max)
    M = np.tile(mask, (len(lon), 1)).T
    return M

# Full latitude array (assuming 1-degree resolution from 89.5 to -89.5)
lat_array = np.arange(89.5, -89.5-0.5, -1)
weight = gen_weights(lat_array)

# Generate regional mask
region_config = REGION_CONFIG[REGION]
regional_mask = gen_regional_mask(lat_array, region_config['lat_min'], region_config['lat_max'])
print(f"Region: {REGION} (lat: {region_config['lat_min']} to {region_config['lat_max']})")

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
    z = np.average(X[indices], weights=W[indices])
    return z

def metric_rmse(F, A, W, mask=None):
    X = np.square(F - A)
    return np.sqrt(weighted_average(X, W, mask))

def metric_acc(F, A, W, mask=None):
    F = np.squeeze(F)
    A = np.squeeze(A)
    a = weighted_average(F*A, W, mask)
    b = np.sqrt(weighted_average(A*A, W, mask))
    c = np.sqrt(weighted_average(F*F, W, mask))
    return a/b/c

def to_file_format(dt: datetime) -> str:
    year = dt.year
    year_start = datetime(year, 1, 1, 0, 0, 0)
    delta = dt - year_start
    step = delta.days * 4 + delta.seconds // 21600
    return f"{year}_{step:04d}"

def current_date_to_file_format(initial_date, lead_time):
    start_date = datetime.strptime(initial_date, "%Y-%m-%d")
    new_date = start_date + timedelta(days=lead_time)
    date_str = to_file_format(new_date)
    return date_str

# ==================== Data Loading Functions ====================
def load_GT(date_str, var_name, level):
    h5file = h5py.File(f"{data_root_path}/h5data/{date_str}.h5", "r")
    if var_name in upper_air_vars:
        lev_str = f"{level:.1f}"
        var_full_name = f"input/{var_name}_{lev_str}"
    else:
        var_full_name = f"input/{var_name}"
    data = h5file[var_full_name][:].data[::-1]
    h5file.close()
    return data

# ==================== Main Computation ====================
def compute_metrics(var_name, level, region=REGION):
    """
    Compute metrics for 2020 full year at 00Z
    var_name: variable name (e.g., '2m_temperature', 'geopotential')
    level: 'surface' or pressure level (e.g., 500, 850)
    region: 'global', 'tropics', 'NH', or 'SH'
    """
    
    # Get regional configuration
    region_cfg = REGION_CONFIG[region]
    regional_mask = gen_regional_mask(lat_array, region_cfg['lat_min'], region_cfg['lat_max'])
    region_suffix = region_cfg['suffix']
    
    print(f"\n{'='*60}")
    print(f"Computing metrics for region: {region}")
    print(f"Latitude range: {region_cfg['lat_min']} to {region_cfg['lat_max']}")
    print(f"{'='*60}")
    
    # Load climatology
    climatology = netCDF4.Dataset(f'{data_root_path}1979-2018_mean_climatology.nc')
    if var_name in upper_air_vars:
        var_clim_all = climatology[var_name][:, levs.index(level), ::-1]
    else:
        var_clim_all = climatology[var_name][:, ::-1]
    
    def load_clim(date_str):
        var_clim = var_clim_all[int(int(date_str[5:])/4)]
        return var_clim
    
    # Get all dates in 2020 at 00Z
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    current_date = start_date
    selected_date_list = []
    while current_date <= end_date:
        selected_date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    print(f"Processing {var_name} at level {level}")
    print(f"Total cases: {len(selected_date_list)}")
    print(f"Date range: {selected_date_list[0]} to {selected_date_list[-1]}")
    
    # ==================== Metrics Dictionary ====================
    metric_dict = {
        'FC': {'ACC': {}, 'RMSE': {}, 'direction': 'forward'},
        'BC': {'ACC': {}, 'RMSE': {}, 'direction': 'backward'},
        'region': region,
        'lat_min': region_cfg['lat_min'],
        'lat_max': region_cfg['lat_max'],
    }
    
    # ==================== Forward Direction ====================
    print("\n=== Computing Forward Direction Metrics ===")
    
    for initial_date in tqdm(selected_date_list, desc="Forward"):
        acc_fc_list, rmse_fc_list = [], []
        
        date_str1 = initial_date.replace('-', '') + '00'
        
        try:
            nc_file_fc = netCDF4.Dataset(f'{fc_root_path}pangu_plasim_{FC_idx}_24h_{FC_len}step_{date_str1}.nc')
            
            if var_name in upper_air_vars:
                FC_var = nc_file_fc[var_name][:].data[:, plevs_fc.index(level), ::-1]
            else:
                FC_var = nc_file_fc[var_name][:].data[:, ::-1]
            
            for i in range(15):
                date_str = current_date_to_file_format(initial_date, i+1)
                var_GT_data = load_GT(date_str, var_name, level)
                var_clim_data = load_clim(date_str)
                var_GT_anom = var_GT_data - var_clim_data
                
                fc_anom = FC_var[i+1] - var_clim_data
                rmse_fc_list.append(metric_rmse(fc_anom, var_GT_anom, weight, regional_mask))
                acc_fc_list.append(metric_acc(fc_anom, var_GT_anom, weight, regional_mask))
            
            metric_dict['FC']['ACC'][initial_date] = acc_fc_list
            metric_dict['FC']['RMSE'][initial_date] = rmse_fc_list
                
        except Exception as e:
            print(f"\nError processing forward {initial_date}: {e}")
            continue
    
    # ==================== Backward Direction ====================
    print("\n=== Computing Backward Direction Metrics ===")
    for initial_date in tqdm(selected_date_list, desc="Backward"):
        acc_bc_list, rmse_bc_list = [], []
        
        try:
            date_str1 = initial_date.replace('-', '') + '00'
            nc_file_bc = netCDF4.Dataset(f'{bc_root_path}pangu_plasim_{BC_idx}_-24h_{BC_len}step_{date_str1}.nc')
            
            if var_name in upper_air_vars:
                BC_var = nc_file_bc[var_name][:].data[:, plevs_bc.index(level), ::-1]
            else:
                BC_var = nc_file_bc[var_name][:].data[:, ::-1]
            
            for i in range(15):
                date_str = current_date_to_file_format(initial_date, -(i+1))
                var_GT_data = load_GT(date_str, var_name, level)
                var_clim_data = load_clim(date_str)
                var_GT_anom = var_GT_data - var_clim_data
                
                bc_anom = BC_var[i+1] - var_clim_data
                rmse_bc_list.append(metric_rmse(bc_anom, var_GT_anom, weight, regional_mask))
                acc_bc_list.append(metric_acc(bc_anom, var_GT_anom, weight, regional_mask))
                
            metric_dict['BC']['ACC'][initial_date] = acc_bc_list
            metric_dict['BC']['RMSE'][initial_date] = rmse_bc_list
            
        except Exception as e:
            print(f"\nError processing backward {initial_date}: {e}")
            continue
    
    # ==================== Save Combined Results ====================
    os.makedirs("results", exist_ok=True)
    save_name = 'surface' if var_name in surface_vars else level
    
    # Include region in filename
    output_file = f"results/{var_name}_{save_name}_{model_name}{region_suffix}_metrics.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(metric_dict, f)
    print(f"\nSaved: {output_file}")
    
    # ==================== Print Summary Statistics ====================
    print("\n" + "="*60)
    print(f"=== Summary Statistics ({region}) ===")
    print("="*60)
    
    for direction, models in [('FORWARD', ['FC']), ('BACKWARD', ['BC'])]:
        print(f"\n{'='*60}")
        print(f"{direction} Direction:")
        print(f"{'='*60}")
        
        for model in models:
            if metric_dict[model]['ACC']:
                acc_values = []
                rmse_values = []
                for date in metric_dict[model]['ACC'].keys():
                    acc_values.extend(metric_dict[model]['ACC'][date])
                    rmse_values.extend(metric_dict[model]['RMSE'][date])
                
                acc_array = np.array(acc_values)
                rmse_array = np.array(rmse_values)
                
                print(f"\n  {model}:")
                print(f"    Cases: {len(metric_dict[model]['ACC'])}")
                print(f"    ACC:  mean={acc_array.mean():.4f}, std={acc_array.std():.4f}, "
                      f"min={acc_array.min():.4f}, max={acc_array.max():.4f}")
                print(f"    RMSE: mean={rmse_array.mean():.4f}, std={rmse_array.std():.4f}, "
                      f"min={rmse_array.min():.4f}, max={rmse_array.max():.4f}")
    
    print("\n" + "="*60)
    return metric_dict

if __name__ == "__main__":
    # Compute for different regions
    # for region in ['tropics', 'NH', 'SH']:
    for region in ['SH', ]:
        print(f"\n{'#'*60}")
        print(f"# Processing region: {region}")
        print(f"{'#'*60}")
        
        compute_metrics('geopotential', 500, region=region)
        compute_metrics('u_component_of_wind', 850, region=region)
        compute_metrics('u_component_of_wind', 250, region=region)