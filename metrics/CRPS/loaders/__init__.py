"""
Loader registry.

Each loader module must expose:
    load_forecast(input_dir, years, var_name, var_cfg, max_members) -> xr.Dataset

To add a new format, create loaders/format_X.py and add its key here.
"""

from . import format_a, format_b

LOADERS = {
    "format_a": format_a,
    "format_b": format_b,
}


def get_loader(format_key: str):
    if format_key not in LOADERS:
        raise ValueError(
            f"Unknown format '{format_key}'. Available: {list(LOADERS)}"
        )
    return LOADERS[format_key]
