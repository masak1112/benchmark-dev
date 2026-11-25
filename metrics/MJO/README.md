# MJO Modular Evaluation Pipeline

A configurable, multi-dataset MJO forecast evaluation pipeline built on top of [MJOcast](https://github.com/WillyChap/MJOcast).

## Overview

This pipeline evaluates MJO forecast skill using RMM (Real-time Multivariate MJO) indices derived from configurable input variables:

| Variable | Description |
|---|---|
| `olr` | Outgoing Longwave Radiation (standard RMM) |
| `chi200` | Velocity potential at 200 hPa (OLR proxy for models without OLR output) |
| `chi250` | Velocity potential at 250 hPa |

**Skill metrics produced:**
- Bivariate anomaly correlation coefficient (ACC) vs lead time
- Bivariate RMSE vs lead time
- RMM phase space trajectories

**Supported resolutions:** 1.0° and 1.5° (obs regridded automatically if needed)

---

## Directory Structure

```
MJO_modular/
├── config/
│   └── pipeline_config.yaml    ← Edit this to configure the run
├── loaders/
│   ├── __init__.py             ← Loader registry
│   ├── fuxi_s2s.py             ← FuXi S2S loader (7z archives)
│   └── amip_s2s.py             ← AMIP S2S loader (seed NC files)
├── src/
│   ├── config.py               ← Config loader and path derivation
│   ├── step1_build_obs.py      ← Build ERA5 obs files
│   ├── step2_prepare_forecasts.py  ← Prepare forecast NC files
│   ├── step3_process_eofs.py   ← Compute observed EOFs
│   ├── step4_run_forecasts.py  ← MJOcast RMM projection
│   ├── step5_acc_plot.py       ← Bivariate ACC skill curve
│   └── step5_phase_space.py    ← RMM phase space plots
├── run_scripts/
│   ├── run_pipeline.sh         ← Full pipeline runner (sequential)
│   ├── run_step2_array.pbs     ← PBS array job for step 2
│   └── run_step4_step5.pbs     ← PBS job for step 4+5
└── README.md
```

---

## Quick Start

### 1. Configure the pipeline

Edit `config/pipeline_config.yaml`:

```yaml
variable: chi200     # olr | chi200 | chi250
resolution: 1.0      # 1.0 | 1.5 degrees

datasets:
  fuxi_s2s:
    enabled: true
    src_dir: /glade/derecho/scratch/bgong/fuxi_s2s
    out_dir: /glade/work/bgong/MJO_results/fuxi_s2s
  amip_s2s:
    enabled: true
    src_dir: /glade/derecho/scratch/bgong/amip_s2s_train_scratch
    out_dir: /glade/work/bgong/MJO_results/amip_s2s_train_scratch

plots:
  phase_space_months:
    - "202005"   # May 2020
    - "202007"   # July 2020
```

### 2. Run the full pipeline

```bash
cd /glade/work/bgong/MJO_modular
bash run_scripts/run_pipeline.sh
```

Or, to run each step independently:

```bash
PYTHON=/glade/work/bgong/mjocast_env/bin/python

$PYTHON src/step1_build_obs.py
$PYTHON src/step2_prepare_forecasts.py
$PYTHON src/step3_process_eofs.py
$PYTHON src/step4_run_forecasts.py
$PYTHON src/step5_acc_plot.py
$PYTHON src/step5_phase_space.py
```

### 3. Inspect outputs

Results are written to subdirectories of each dataset's `out_dir`, organised by `{variable}_{resolution}deg/`:

```
/glade/work/bgong/MJO_results/
├── obs/chi200_1p0deg/          ← ERA5 obs files (shared)
├── fuxi_s2s/chi200_1p0deg/
│   ├── forecast_files/         ← Prepared forecast NC files
│   ├── rmm_output/             ← MJOcast RMM output
│   └── plots/                  ← Per-dataset plots
├── amip_s2s_train_scratch/chi200_1p0deg/
│   ├── forecast_files/
│   ├── rmm_output/
│   └── plots/
└── plots/
    ├── bivariate_acc_comparison.png   ← All datasets on one ACC figure
    ├── phase_space_202005.png         ← May 2020 phase space
    └── phase_space_202007.png         ← July 2020 phase space
```

---

## Adding a New Dataset

To add a new dataset (e.g. `my_model`) without modifying any existing scripts:

1. **Create a loader module** `loaders/my_model.py` that implements two functions:

   ```python
   def list_dates(src_dir: str) -> list[str]:
       """Return sorted list of YYYYMMDD strings with available data."""
       ...

   def process_date(date_str, src_dir, out_dir, variable, resolution,
                    max_members=None, scratch_dir="/tmp") -> str:
       """
       Build and return a MJOcast-compatible forecast NC file.
       Output must have variables: rlut, ua_850, ua_200
       Dimensions: (ensemble, time, lat, lon)
       """
       ...
   ```

2. **Add an entry** in `config/pipeline_config.yaml`:

   ```yaml
   datasets:
     my_model:
       enabled: true
       loader: my_model          # matches loaders/my_model.py
       src_dir: /path/to/model/data
       out_dir: /glade/work/bgong/MJO_results/my_model
   ```

3. Re-run step 2 and onwards — the new dataset is automatically included in all skill plots.

**Required output format for a loader:**

| Variable | Meaning | Units |
|---|---|---|
| `rlut` | Primary variable (OLR or chi) occupying the OLR slot in MJOcast | W m⁻² or m² s⁻¹ |
| `ua_850` | Zonal wind at 850 hPa | m s⁻¹ |
| `ua_200` | Zonal wind at 200 hPa (or 250 hPa proxy) | m s⁻¹ |

Dimensions must be `(ensemble, time, lat, lon)`.

---

## Job Submission on Derecho

### Option A: Full sequential pipeline (interactive / small tests)

```bash
cd /glade/work/bgong/MJO_modular
bash run_scripts/run_pipeline.sh
```

### Option B: Parallel step 2 (large datasets)

Step 2 (preparing forecast files) is the most compute-intensive step and can be
parallelised across init dates using the PBS array job.

```bash
cd /glade/work/bgong/MJO_modular

# 1. Generate a list of dates to process
ls /glade/derecho/scratch/bgong/fuxi_s2s/*.7z \
    | xargs -I{} basename {} .7z | sort \
    > run_scripts/all_dates.txt

# 2. Submit the array job (chunk of 8 dates per task)
NDATES=$(wc -l < run_scripts/all_dates.txt)
CHUNK=8
NJOBS=$(( (NDATES + CHUNK - 1) / CHUNK ))

STEP2_ID=$(qsub -J 0-$((NJOBS-1)) run_scripts/run_step2_array.pbs)
echo "Step 2 array job: ${STEP2_ID}"

# 3. Submit step 4+5 with dependency on step 2
qsub -W depend=afterany:${STEP2_ID%.*}[] run_scripts/run_step4_step5.pbs
```

Monitor with:
```bash
qstat -u $USER
```

### Customising PBS resources

Edit the `#PBS` headers at the top of each PBS script.  Key settings:
- `#PBS -A UCHI0018` — project account code
- `#PBS -q main` — queue (`main` or `develop`)
- `#PBS -l walltime=06:00:00` — wall-clock limit

---

## Reference Pipeline

The original (non-modular) chi250/chi200 evaluation pipeline is at:
```
/glade/work/bgong/MJO/chi250_eval/
```
Scripts-only backup (preserved before modularisation):
```
/glade/work/bgong/MJO/chi250_eval_scripts_backup/
```

The modular pipeline reuses the same underlying MJOcast EOF and forecast
processing logic; only the data loading and path management have been
generalised.
