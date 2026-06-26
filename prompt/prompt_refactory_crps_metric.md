# Task
Refactor the CRPS/SSR evaluation codebase to support multiple input data sources and formats.

Working directory:
/glade/work/bgong/benchmark-dev/metrics/CRPS

## Version Control
Before making any changes, create a git branch or backup of the current working directory to preserve the original code.

## Input Data

Three example input data sources:

| # | Path | Format |
|---|------|--------|
| 1 | /glade/derecho/scratch/bgong/amip_s2s_ensembles | Format A |
| 2 | /glade/derecho/scratch/bgong/amip_s2s_train_scratch | Format A (same as #1) |
| 3 | /glade/derecho/scratch/bgong/ours_si | Format B (different structure) |

Ground truth (ERA5) data location (same for all sources):
/glade/derecho/scratch/bgong/era5

## Code Refactoring Rules

1. **Configuration file**: Store all input data paths and settings in a separate configuration file. Do not hardcode paths or settings in the main script (e.g., amip_crps_ssr.py).

2. **Modular data loading**: Implement data loading and preprocessing as separate, pluggable modules — one per data format. This ensures that adding a new data source with a different format only requires adding a new loader module, with no changes to the main script.

3. **Main script stays clean**: The main script (e.g., amip_crps_ssr.py) should remain format-agnostic and unchanged when onboarding new data sources.

## Environment
Use the following conda environment for all dependencies:
/glade/work/bgong/conda-envs

## Job Submission
Prepare a PBS job submission script to run the Python script on Derecho.
Use the following script as a template for job configuration (queue, walltime, CPU/memory resources, environment activation, and module loading):
/glade/work/bgong/benchmark-dev/metrics/CRPS/submit_amip_crps_ssr.sh

## Output

Parent output directory:
/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/

Child directory: use the model name derived from the input data directory name.

Example: input from /glade/derecho/scratch/bgong/ours_si → output saved to:
/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/ours_si/

File naming convention:
{model_name}_{start_year}_{end_year}_crps_ssr_{variable_name}.nc

Example:
ours_si_2019_2022_crps_ssr_2m_temperature.nc

Output format reference (match this exactly):
/glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr/gencast_results_2019_2024/gencast_2019_2024_crps_ssr_2m_temperature.nc





-----------------------------------
# Task
Enhance the CRPS codebase for better organisation, cleanliness, and documentation.

Working directory:
/glade/work/bgong/benchmark-dev/metrics/CRPS


## Codebase Structure
Organise the codebase into the following directory structure:

CRPS/

├── loaders/          # Data loading and preprocessing modules (one per data format)

├── configuration/    # Configuration files (input paths, settings)

├── job_scripts/      # PBS job submission scripts for Derecho

└── README.md         # Project documentation (see below)


## Documentation
Create a `README.md` in the root of the `CRPS/` directory covering:

1. **Overview** — what the codebase does and which variables/metrics are supported
2. **Quick start** — how to configure and run the evaluation
3. **Job submission** — how to submit a job on Derecho using the scripts in `job_scripts/`
4. **Adding a new data source** — step-by-step guide for adding a new loader module in `loaders/` without modifying the main script

## Code Cleanup
- Remove all unused scripts, notebooks, and dead code
- Keep only files that are actively used by the current pipeline
- Leave a comment or note if a file is intentionally kept for reference