"""
src/step3_process_eofs.py — Compute observed MJO EOFs with MJOcast.

Reads the filtered obs file from step1 (ERA5_{variable}_anom_filt120.nc),
computes the multivariate EOF of [chi/olr, U850, U250] over 15S–15N, and
saves the observed RMM indices and EOF structures to:

  <obs_dir>/MJO_obs_created.nc

This file is the observational reference for step4 (EOF projection and
120-day filter normalisation).

Usage:
  python src/step3_process_eofs.py [--config CFG]
"""

import argparse
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PipelineConfig


def process_eofs(cfg: PipelineConfig) -> None:
    sys.path.insert(0, cfg.mjocast_base_dir)
    from MJOcast.utils.ProcessOBS import MJOobsProcessor

    # For step3 we need any dataset's YAML — we use a temp dir since step3 only
    # needs obs paths and mjocast base_dir (not dataset-specific paths).
    # Pick the first enabled dataset to satisfy generate_mjocast_yaml's signature.
    ds_name = next(iter(cfg.datasets))
    tmp_dir = tempfile.mkdtemp(prefix="mjocast_step3_")
    try:
        yaml_path = cfg.generate_mjocast_yaml(ds_name, tmp_dir=tmp_dir)

        print(f"Initialising MJOobsProcessor  (variable={cfg.variable}) ...")
        obs_proc = MJOobsProcessor(yaml_path)

        print("Computing observed EOFs ...")
        OBS_DS, eof_list, pcs, MJO_fobs, eof_dict = obs_proc.make_observed_MJO()

        print("\nVariance fraction of first 5 EOFs:")
        for i, vf in enumerate(obs_proc.varfrac[:5]):
            print(f"  EOF{i+1}: {vf*100:.1f}%")

        obs_nc = os.path.join(cfg.obs_dir, "MJO_obs_created.nc")
        print(f"\nObserved MJO file saved to: {obs_nc}")
        print("\nStep 3 complete.")
    finally:
        shutil.rmtree(tmp_dir)


def _parse_args():
    p = argparse.ArgumentParser(description="Compute observed MJO EOFs with MJOcast.")
    p.add_argument("--config", default="config/pipeline_config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = PipelineConfig(args.config)
    process_eofs(cfg)
