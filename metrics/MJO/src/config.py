"""
config.py — load pipeline_config.yaml and derive all paths used by the pipeline.

Usage
-----
from src.config import PipelineConfig
cfg = PipelineConfig("/glade/work/bgong/MJO_modular/config/pipeline_config.yaml")

Key attributes
--------------
cfg.variable          : "olr" | "chi200" | "chi250"
cfg.resolution        : 1.0 | 1.5
cfg.run_key           : e.g. "chi200_1p0deg"
cfg.obs_dir           : /glade/work/bgong/MJO_results/obs/chi200_1p0deg/
cfg.datasets          : dict of DatasetConfig (one per enabled dataset)
cfg.generate_mjocast_yaml(ds_name, forecast_dir, out_dir) -> path to tmp YAML
"""

import os
import re
import yaml
import tempfile
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    name: str
    loader: str
    src_dir: str
    out_dir: str
    max_members: Optional[int]
    scratch_dir: Optional[str]
    resolution: Optional[float]  # overrides global resolution if set

    @property
    def forecast_dir(self, run_key: str = "") -> str:
        return ""  # computed below via PipelineConfig.forecast_dir()


class PipelineConfig:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self.config_path = os.path.abspath(config_path)
        self._raw = raw

        self.variable: str = raw["variable"]
        self.resolution: float = float(raw["resolution"])
        self.regrid_forecast: bool = raw.get("regrid_forecast", False)

        obs = raw["obs"]
        self.era5_vp200_dir: str = obs["era5_vp200_dir"]
        self.era5_h5_dir: str = obs["era5_h5_dir"]
        self.era5_mjocast_obs_dir: str = obs["era5_mjocast_obs_dir"]
        self.base_obs_dir: str = obs["base_obs_dir"]
        self.climo_start_year: int = int(obs["climo_start_year"])
        self.climo_end_year: int = int(obs["climo_end_year"])
        self.anomaly_start_year: int = int(obs["anomaly_start_year"])
        self.lat_south: float = float(obs["lat_south"])
        self.lat_north: float = float(obs["lat_north"])

        mc = raw["mjocast"]
        self.mjocast_base_dir: str = mc["base_dir"]
        self.mjocast_user: str = mc["user"]

        self.datasets: dict[str, DatasetConfig] = {}
        for ds_name, ds_raw in raw.get("datasets", {}).items():
            if not ds_raw.get("enabled", True):
                continue
            max_mem = ds_raw.get("max_members")
            scratch = ds_raw.get("scratch_dir", "/tmp")
            ds_res  = ds_raw.get("resolution")
            self.datasets[ds_name] = DatasetConfig(
                name=ds_name,
                loader=ds_raw["loader"],
                src_dir=ds_raw["src_dir"],
                out_dir=ds_raw["out_dir"],
                max_members=int(max_mem) if max_mem is not None else None,
                scratch_dir=scratch,
                resolution=float(ds_res) if ds_res is not None else None,
            )

        plots = raw.get("plots", {})
        self.phase_space_months: list[str] = plots.get("phase_space_months", [])
        self.acc_plot_filename: str = plots.get("acc_plot_filename",
                                                  "bivariate_acc_comparison.png")
        self.plot_dir: str = plots.get("plot_dir",
                                        "/glade/work/bgong/MJO_results/plots")

        self.conda_env: str = raw.get("conda_env", "")
        self.pbs: dict = raw.get("pbs", {})

    # ── derived keys ──────────────────────────────────────────────

    @property
    def run_key(self) -> str:
        """Short string encoding variable + resolution, used in subdirectory names."""
        res_str = f"{self.resolution:.1f}".replace(".", "p") + "deg"
        return f"{self.variable}_{res_str}"

    @property
    def obs_dir(self) -> str:
        return os.path.join(self.base_obs_dir, self.run_key)

    # ── per-dataset paths ─────────────────────────────────────────

    def _ds_run_key(self, ds_name: str) -> str:
        """run_key using per-dataset resolution override if set."""
        ds = self.datasets[ds_name]
        res = ds.resolution if ds.resolution is not None else self.resolution
        res_str = f"{res:.1f}".replace(".", "p") + "deg"
        return f"{self.variable}_{res_str}"

    def forecast_dir(self, ds_name: str) -> str:
        ds = self.datasets[ds_name]
        return os.path.join(ds.out_dir, self._ds_run_key(ds_name), "forecast_files")

    def rmm_dir(self, ds_name: str) -> str:
        ds = self.datasets[ds_name]
        return os.path.join(ds.out_dir, self._ds_run_key(ds_name), "rmm_output")

    def ds_plot_dir(self, ds_name: str) -> str:
        ds = self.datasets[ds_name]
        return os.path.join(ds.out_dir, self._ds_run_key(ds_name), "plots")

    def ensure_dirs(self, ds_name: str) -> None:
        for d in [self.obs_dir, self.forecast_dir(ds_name),
                  self.rmm_dir(ds_name), self.ds_plot_dir(ds_name),
                  self.plot_dir]:
            os.makedirs(d, exist_ok=True)

    # ── MJOcast variable-specific names ──────────────────────────

    @property
    def usr_named_obs(self) -> str:
        if self.variable == "olr":
            return "ERA5_Meridional_Mean_Anomaly_Filtered120.nc"
        return f"ERA5_{self.variable}_anom_filt120.nc"

    @property
    def forecast_name_str(self) -> str:
        return f"{self.variable}_forecast_*.nc"

    @property
    def output_files_string(self) -> str:
        return f"/MJO_{self.variable}_Forecast_Init"

    # ── MJOcast YAML generation ───────────────────────────────────

    def generate_mjocast_yaml(self, ds_name: str,
                               forecast_data_loc: Optional[str] = None,
                               tmp_dir: Optional[str] = None) -> str:
        """
        Write a MJOcast-compatible settings YAML for the given dataset and return
        the path.  If forecast_data_loc is provided, use that directory (symlink
        trick for single-date runs); otherwise use self.forecast_dir(ds_name).

        The YAML is written into tmp_dir (or a system tempdir) and must be cleaned
        up by the caller.
        """
        fc_loc = forecast_data_loc or self.forecast_dir(ds_name)
        out_dir = self.rmm_dir(ds_name)
        plot_dir = self.ds_plot_dir(ds_name)

        # trailing slash required by MJOcast's path concatenation
        obs_data_loc = self.obs_dir.rstrip("/") + "/"
        output_files_loc = out_dir.rstrip("/") + "/"

        yml = {
            "user": self.mjocast_user,
            "base_dir": self.mjocast_base_dir,
            "user_defined_info": {
                "use_era5": False,
                "usr_named_obs": self.usr_named_obs,
                "obs_data_loc": obs_data_loc,
                "forecast_data_loc": fc_loc,
                "forecast_data_name_str": self.forecast_name_str,
                "forecast_olr_name": "rlut",
                "forecast_u200_name": "ua_200",
                "forecast_u850_name": "ua_850",
                "forecast_ensemble_dimension_name": "ensemble",
                "output_plot_loc": plot_dir,
                "output_files_loc": output_files_loc,
                "output_files_string": self.output_files_string,
                "use_forecast_climo": False,
                "use_observed_climo": False,
                "regenerate_climo": False,
                "use_dask_for_climo": False,
            },
        }

        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="mjocast_")
        os.makedirs(tmp_dir, exist_ok=True)
        yaml_path = os.path.join(tmp_dir, f"settings_{ds_name}.yaml")
        with open(yaml_path, "w") as fh:
            yaml.dump(yml, fh, default_flow_style=False)
        return yaml_path
