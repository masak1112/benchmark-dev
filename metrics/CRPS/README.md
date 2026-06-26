# CRPS / SSR Evaluation

Compute **Continuous Ranked Probability Score (CRPS)** and **Spread–Skill Ratio (SSR)** for S2S ensemble forecasts, verified against ERA5 reanalysis.

## Directory layout

```
CRPS/
├── crps_ssr_main.py        # Format-agnostic evaluation driver
├── configuration/
│   └── config.yaml         # All paths, settings, and variable mappings
├── job_scripts/
│   └── submit_crps_ssr.sh  # PBS submission script for Derecho
└── loaders/
    ├── __init__.py          # Loader registry (format key → module)
    ├── format_a.py          # Loader for amip_s2s_ensembles style data
    └── format_b.py          # Loader for pangu_plasim / ours_si style data
```

---

## Overview

### Metrics

| Metric | Description |
|--------|-------------|
| **CRPS** | Mean over all init dates, lead times, lat, lon. Lower is better. |
| **SSR** | √(mean spread² / mean error²). Target ≈ 1 (perfectly spread ensemble). |

Output files contain `crps` and `ssr` as 1-D arrays indexed by `prediction_timedelta` (integer hours).

### Supported variables (configured per model)

| Config key | ERA5 variable | Notes |
|---|---|---|
| `2m_temperature` | `2m_temperature` | Surface temperature |
| `geopotential_500` | `geopotential` | 500 hPa level |
| `sea_surface_temperature` | `sea_surface_temperature` | |
| `total_precipitation_24hr` | `total_precipitation_24hr` | Unit conversion applied |

Variables are defined per model in `configuration/config.yaml` — different models can expose different variable sets.

---

## Quick start

### 1. Configure

Edit `configuration/config.yaml`.  The key sections are:

```yaml
era5:
  dir: /glade/derecho/scratch/bgong/era5   # ERA5 root
  file_pattern: "{variable}/{year}_180x360.nc"

output:
  parent_dir: /glade/work/bgong/gencast_neuralGCM_crps_ssr/gencast_neuralGCM_crps_ssr

compute:
  max_members: 25   # cap ensemble size; null = use all

models:
  my_model:
    input_dir: /path/to/forecasts
    format: format_a          # or format_b
    years: [2019, 2020, 2021, 2022]
    variables:
      2m_temperature:
        model_var: 2m_temperature
        era5_var: 2m_temperature
```

### 2. Run interactively

```bash
module load conda
conda activate /glade/work/bgong/conda-envs/my_env

CRPS=/glade/work/bgong/benchmark-dev/metrics/CRPS

# All models, all variables
python $CRPS/crps_ssr_main.py

# Single model
python $CRPS/crps_ssr_main.py --model ours_si

# Single model, single variable
python $CRPS/crps_ssr_main.py --model ours_si --var 2m_temperature

# Custom config file
python $CRPS/crps_ssr_main.py --config /path/to/my_config.yaml
```

### 3. Output location

Results are written to:

```
{output.parent_dir}/{model_name}/{model_name}_{start_year}_{end_year}_crps_ssr_{variable}.nc
```

Example:
```
.../gencast_neuralGCM_crps_ssr/ours_si/ours_si_2020_2020_crps_ssr_2m_temperature.nc
```

Existing output files are skipped automatically.

---

## Job submission on Derecho

The PBS script `job_scripts/submit_crps_ssr.sh` reads its settings from
`configuration/config.yaml`.  That file controls which models run, which
variables are evaluated, which years are included, and where output is written.
The PBS script itself only sets cluster resources and passes optional filters.

### Typical workflow

**1. Edit the configuration** to select what you want to run:

```yaml
# configuration/config.yaml

compute:
  max_members: 25     # reduce to speed up a test run

models:
  ours_si:
    years: [2020]     # narrow the date range for a quick test
    ...
```

**2. Submit the job**, optionally filtering to a single model or variable:

```bash
cd /glade/work/bgong/benchmark-dev/metrics/CRPS

# All models and variables defined in config.yaml
qsub job_scripts/submit_crps_ssr.sh

# One model only (all its variables from config.yaml)
qsub -v MODEL=ours_si job_scripts/submit_crps_ssr.sh

# One model, one variable
qsub -v MODEL=ours_si,VAR=2m_temperature job_scripts/submit_crps_ssr.sh

# Override the job name
qsub -N crps_ours_si -v MODEL=ours_si job_scripts/submit_crps_ssr.sh
```

**3. Use a different config file** (e.g. for a test run or a new experiment):

```bash
# Pass a custom config via the CONFIG variable
qsub -v CONFIG=/path/to/test_config.yaml job_scripts/submit_crps_ssr.sh

# Combine with a model filter
qsub -v CONFIG=/path/to/test_config.yaml,MODEL=ours_si job_scripts/submit_crps_ssr.sh
```

> The `CONFIG` variable in the PBS script defaults to
> `configuration/config.yaml`.  Passing `-v CONFIG=...` overrides it without
> editing the script.

### Default PBS settings

Edit `job_scripts/submit_crps_ssr.sh` to change these:

| Setting | Value |
|---------|-------|
| Queue | `develop` |
| Walltime | 5h 40m |
| CPUs | 4 |
| Memory | 100 GB |
| Account | `UCHI0018` |

---

## Adding a new data source

No changes to `crps_ssr_main.py` are needed. Follow these three steps:

### Step 1 — Add a loader module

Create `loaders/format_c.py` (name it after the format, not the model).  The module must expose a single function with this signature:

```python
def load_forecast(
    input_dir: str,
    years: list[int],
    var_name: str,
    var_cfg: dict,
    max_members: int | None,
) -> xr.Dataset:
    ...
```

**Contract:** return an `xr.Dataset` containing the variable named `var_cfg['era5_var']` with dimensions `(time, number, prediction_timedelta, lat, lon)` where:

- `time` — init dates, `datetime64[ns]`
- `number` — integer ensemble member IDs
- `prediction_timedelta` — lead times, `timedelta64[ns]` from init time
- `lat` / `lon` — matching ERA5 spatial coordinates (rename if needed)

See `loaders/format_a.py` (amip_s2s style) and `loaders/format_b.py` (pangu_plasim/ours_si style) as reference implementations.

### Step 2 — Register the loader

In `loaders/__init__.py`, import your module and add it to `LOADERS`:

```python
from . import format_a, format_b, format_c   # add format_c

LOADERS = {
    "format_a": format_a,
    "format_b": format_b,
    "format_c": format_c,                     # add this line
}
```

### Step 3 — Add the model to config

Add an entry under `models:` in `configuration/config.yaml`:

```yaml
models:
  my_new_model:
    input_dir: /glade/derecho/scratch/bgong/my_new_model
    format: format_c          # key registered in Step 2
    years: [2020, 2021]
    variables:
      2m_temperature:
        model_var: 2m_temperature   # variable name inside the model files
        era5_var: 2m_temperature    # ERA5 variable to verify against
      geopotential_500:
        model_var: geopotential
        era5_var: geopotential
        level: 500.0
        level_dim: plev             # pressure-level coordinate name in model files
```

That's it — run `python crps_ssr_main.py --model my_new_model` and results appear under `{output.parent_dir}/my_new_model/`.

---

## Data format reference

### Format A — amip_s2s_ensembles / amip_s2s_train_scratch

```
{input_dir}/{YYYYMMDD}/ensemble_{YYYYMMDD}_seed{N}.nc
```

- Dims: `(member, time, lat, lon[, plev])`
- `time`: 46 valid-time steps; `member`: 4 per seed file
- Multiple seed files per init date are concatenated along `number`

### Format B — ours_si (pangu_plasim)

```
{input_dir}/{prefix}_{YYYYMMDDHH}_ens_{N}.nc
```

- Dims: `(time, latitude, longitude[, level])`
- One ensemble member per file; `latitude`/`longitude` renamed to `lat`/`lon`
- Init date and member index parsed from the filename

### ERA5

```
{era5.dir}/{variable}/{year}_180x360.nc
```

Grid: 180 × 360 (1° resolution).  Files available for 2018–2024.
