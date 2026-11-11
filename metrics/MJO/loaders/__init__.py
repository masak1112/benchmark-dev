"""
Loader registry — maps loader names (from pipeline_config.yaml) to modules.

To add a new dataset:
  1. Create loaders/<your_loader>.py implementing list_dates() and process_date()
  2. No changes to any other file are required.
"""

import importlib


def get_loader(name: str):
    """Return the loader module for the given name (e.g. 'fuxi_s2s')."""
    try:
        return importlib.import_module(f"loaders.{name}")
    except ModuleNotFoundError:
        raise ValueError(
            f"No loader module found for '{name}'. "
            f"Create loaders/{name}.py with list_dates() and process_date()."
        )
