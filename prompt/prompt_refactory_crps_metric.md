# Task
Now working on the folder /glade/work/bgong/benchmark-dev/metrics/CRPS. We need to refactory the code to make it more adapt to various input data source and data format

## Input Data
Forecast data location:
/glade/derecho/scratch/bgong/amip_s2s_ensembles

## Reference Code
Base implementation (Jupyter notebook):
/glade/work/bgong/benchmark-dev/metrics/CRPS/fast_crps.ipynb

Convert this notebook to a standalone Python script (.py).

## Environment
Use the following conda environment for all dependencies:
/glade/work/bgong/conda-envs

## Job Submission
Prepare a PBS job submission script to run the Python script on Derecho.
Use the following script as a template for job configuration (queue, walltime, CPU/memory resources, environment activation, and module loading):
/glade/work/bgong/download_fuxi_s2s.sh

## Output
Save directory:
/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/amip_results_2019_2022

File naming convention:
amip_2019_2022_crps_ssr_{variable_name}.nc

Output format reference (match this exactly):
/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/gencast_results_2019_2024/gencast_2019_2024_crps_ssr_2m_temperature.nc



