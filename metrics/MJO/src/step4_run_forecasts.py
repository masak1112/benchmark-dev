"""
src/step4_run_forecasts.py — Run MJOcast RMM projection for all enabled datasets.

For each enabled dataset this script:
  1. Re-derives the EOF solver from the obs file built in step 3
  2. Identifies forecast files not yet projected (no matching RMM output)
  3. Uses a temp directory with symlinks to avoid re-processing existing dates
  4. Writes RMM1, RMM2 (per member + ensemble mean) and observed RMM to
       <rmm_dir>/MJO_{variable}_Forecast_Init_YYYYMMDD.nc

Usage:
  python src/step4_run_forecasts.py [--config CFG] [--dataset NAME] [--date YYYYMMDD]

  --dataset : restrict to one dataset
  --date    : process only this init date (may repeat)
"""

import argparse
import glob
import os
import re
import shutil
import sys
import random
import string
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PipelineConfig


def run_forecasts(cfg: PipelineConfig,
                  dataset_name: str | None = None,
                  dates: list[str] | None = None) -> None:
    sys.path.insert(0, cfg.mjocast_base_dir)
    from MJOcast.utils.ProcessOBS      import MJOobsProcessor
    from MJOcast.utils.ProcessForecasts import MJOforecaster

    obs_nc = os.path.join(cfg.obs_dir, "MJO_obs_created.nc")
    if not os.path.exists(obs_nc):
        raise FileNotFoundError(
            f"{obs_nc} not found.  Run step3_process_eofs.py first.")

    target_datasets = {dataset_name: cfg.datasets[dataset_name]} \
        if dataset_name else cfg.datasets

    for ds_name, ds_cfg in target_datasets.items():
        print(f"\n{'='*60}")
        print(f"Step 4 — dataset: {ds_name}")

        fc_dir  = cfg.forecast_dir(ds_name)
        rmm_dir = cfg.rmm_dir(ds_name)
        os.makedirs(rmm_dir, exist_ok=True)
        os.makedirs(cfg.ds_plot_dir(ds_name), exist_ok=True)

        # identify forecast files without RMM output
        pattern   = os.path.join(fc_dir, f"{cfg.variable}_forecast_????????.nc")
        fc_files  = sorted(glob.glob(pattern))
        if not fc_files:
            print(f"  No forecast files found in {fc_dir} — run step2 first.")
            continue

        todo = []
        for fp in fc_files:
            m = re.search(r"(\d{8})\.nc$", os.path.basename(fp))
            if not m:
                continue
            d = m.group(1)
            if dates and d not in dates:
                continue
            rmm_fp = os.path.join(rmm_dir,
                                   f"MJO_{cfg.variable}_Forecast_Init_{d}.nc")
            if not os.path.exists(rmm_fp):
                todo.append((d, fp))

        if not todo:
            print("  All RMM files already exist — nothing to do.")
            continue
        print(f"  Processing {len(todo)} new init dates ...")

        # use temp dir with symlinks so MJOcast only sees the dates we want
        os.makedirs(ds_cfg.scratch_dir, exist_ok=True)
        # letters-only name: MJOcast takes the init date from the first date-like digit
        # run in the FULL path, so digits in a mkdtemp() suffix (e.g. "...0682...")
        # would become the init label and make every output file collide.
        while True:
            tmp_dir = os.path.join(ds_cfg.scratch_dir, "mjocast_step4_" +
                                   "".join(random.choices(string.ascii_lowercase, k=10)))
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                break
        try:
            for d, fp in todo:
                dst = os.path.join(tmp_dir, os.path.basename(fp))
                os.symlink(fp, dst)

            yaml_path = cfg.generate_mjocast_yaml(
                ds_name, forecast_data_loc=tmp_dir, tmp_dir=tmp_dir
            )

            print("  Re-deriving EOF solver from obs ...")
            # scaling_dict bypasses the OLR-orientation check (needed for chi200/chi250)
            scaling_dict = {"loc1": 0, "loc2": 1, "scale1": 1, "scale2": -1}
            obs_proc  = MJOobsProcessor(yaml_path, scaling_dict=scaling_dict)
            _, _, _, MJO_fobs, eof_dict = obs_proc.make_observed_MJO()

            print("  Running forecast projection ...")
            forecaster = MJOforecaster(yaml_path, eof_dict, MJO_fobs)
            forecaster.create_forecasts(num_files=None)

            print(f"  RMM output: {rmm_dir}")
        finally:
            shutil.rmtree(tmp_dir)

    print("\nStep 4 complete.")


def _parse_args():
    p = argparse.ArgumentParser(description="Run MJOcast RMM projection.")
    p.add_argument("--config",  default="config/pipeline_config.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--date",    action="append", dest="dates", metavar="YYYYMMDD")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = PipelineConfig(args.config)
    run_forecasts(cfg, dataset_name=args.dataset, dates=args.dates)
