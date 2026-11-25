"""
src/step2_prepare_forecasts.py — Prepare forecast files for all enabled datasets.

For each dataset enabled in pipeline_config.yaml, this script:
  1. Loads the appropriate loader module from loaders/
  2. Builds standardised forecast NC files (rlut, ua_850, ua_200) per init date
  3. Skips dates whose output file already exists

Usage:
  python src/step2_prepare_forecasts.py [--config CFG] [--dataset NAME] [--date YYYYMMDD]

  --dataset  : process only this dataset (default: all enabled datasets)
  --date     : process only this init date (default: all available dates)
  --all      : process all dates (default when --date not given)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PipelineConfig
import loaders


def prepare_forecasts(cfg: PipelineConfig,
                      dataset_name: str | None = None,
                      dates: list[str] | None = None) -> None:
    """
    Run step 2 for all (or a subset of) enabled datasets and dates.

    Parameters
    ----------
    cfg          : loaded PipelineConfig
    dataset_name : if given, restrict to this dataset
    dates        : if given, process only these YYYYMMDD strings
    """
    target_datasets = {}
    if dataset_name:
        if dataset_name not in cfg.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in config. "
                             f"Available: {list(cfg.datasets.keys())}")
        target_datasets[dataset_name] = cfg.datasets[dataset_name]
    else:
        target_datasets = cfg.datasets

    for ds_name, ds_cfg in target_datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  (loader: {ds_cfg.loader})")
        print(f"  src: {ds_cfg.src_dir}")
        print(f"  out: {cfg.forecast_dir(ds_name)}")

        loader = loaders.get_loader(ds_cfg.loader)
        out_dir = cfg.forecast_dir(ds_name)
        os.makedirs(out_dir, exist_ok=True)

        if dates:
            date_list = dates
        else:
            date_list = loader.list_dates(ds_cfg.src_dir)
            print(f"  Found {len(date_list)} available dates.")

        for d in date_list:
            try:
                resolution = ds_cfg.resolution if ds_cfg.resolution is not None else cfg.resolution
                loader.process_date(
                    date_str    = d,
                    src_dir     = ds_cfg.src_dir,
                    out_dir     = out_dir,
                    variable    = cfg.variable,
                    resolution  = resolution,
                    max_members = ds_cfg.max_members,
                    scratch_dir = ds_cfg.scratch_dir,
                )
            except Exception as e:
                print(f"  ERROR processing {d}: {e}")

    print("\nStep 2 complete.")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Prepare forecast files for all enabled datasets.")
    p.add_argument("--config",  default="config/pipeline_config.yaml")
    p.add_argument("--dataset", default=None,
                   help="Process only this dataset (must match a key in config)")
    p.add_argument("--date",    action="append", dest="dates", metavar="YYYYMMDD",
                   help="Process only this init date (may repeat)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = PipelineConfig(args.config)
    prepare_forecasts(cfg, dataset_name=args.dataset, dates=args.dates)
