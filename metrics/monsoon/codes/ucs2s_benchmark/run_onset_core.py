#!/usr/bin/env python
"""
Run the legacy onset benchmark ("core") on any registered model with flexible
initialisation dates.

The three core scripts are imported UNCHANGED:
  mae_far_mr_probabilistic_models.py   (MAE / FAR / MR maps, CMZ averages)
  onset_reliability_diagram_cmz.py     (reliability diagram, CMZ)
  binned_skill_score_cmz.py            (fair BSS / RPSS / AUC vs IMD climatology, CMZ)
Only their forecast loader `get_forecast_probabilistic_twice_weekly` is
replaced.  The legacy loader keeps Mon/Thu dates of a fixed calendar year
(date_filter_year=2024), which does not match the new UC-S2S init schedule.
The replacement:
  * keeps every init between 1 May and 31 Jul ("native"), or only the dates
    shared by all models in config.COMPARISON_SET ("common");
  * applies the per-model lead alignment (config.MODELS[...]['day_offset']);
  * uses the first MEM_NUM members, converts to mm/day and loads into memory.
Thresholds, onset definition, MOK filter, windows, tolerances, bins and the
climatological reference are exactly those of the core scripts.

Steps (as in complete_onset_benchmarking_pipeline_for_ncar.sh):
  1 MAE/FAR/MR days 1-15 (tol 3)   2 MAE/FAR/MR days 16-30 (tol 5)
  3 reliability 15 d               4 reliability 30 d
  5 skill scores 15 d              6 skill scores 30 d
  7 skill scores in WEEKLY bins (days 1-7, 8-14, 15-21, 22-28 + "after day 28"),
    computed with the core's own functions (only the bins differ from step 5/6)

Usage:
  python run_onset_core.py --model ucs2s_plain --init-mode native --step all
  python run_onset_core.py --model ngcm --init-mode common --step 5
"""
import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import config as C
import common as U

CORE = {"mae": C.CORE_MAE, "reliability": C.CORE_RELIABILITY, "skill": C.CORE_SKILL,
        "skill_weekly": C.CORE_SKILL}
STEPS = {
    1: ("mae", dict(tol=3, vw=1, fd=15, mfd=15, tag="1_15_day")),
    2: ("mae", dict(tol=5, vw=16, fd=30, mfd=30, tag="16_30_day")),
    3: ("reliability", dict(mfd=15)),
    4: ("reliability", dict(mfd=30)),
    5: ("skill", dict(mfd=15)),
    6: ("skill", dict(mfd=30)),
    7: ("skill_weekly", dict(mfd=C.ONSET_WEEK_BINS[-1][1], bins=C.ONSET_WEEK_BINS)),
}


def import_core(kind):
    spec = importlib.util.spec_from_file_location(f"core_{kind}", CORE[kind])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def init_dates_by_year(model_key, mode, years):
    out = {}
    for y in years:
        if mode == "native":
            d = U.model_init_dates(model_key, y, C.ONSET_INIT_MONTHS)
        elif mode == "common":
            d = U.common_init_dates(C.COMPARISON_SET, y, C.ONSET_INIT_MONTHS)
        else:
            raise ValueError(mode)
        if len(d):
            out[y] = d
    return out


def make_loader(model_key, dates):
    def loader(yr, model_forecast_dir=None, mem_num=C.MEM_NUM, date_filter_year=None, file_pattern=None):
        da = U.load_model_tp(model_key, int(yr), mem_num=mem_num, init_dates=dates[int(yr)])
        print(f"[ucs2s loader] {model_key} {yr}: {da.sizes['init_time']} inits, "
              f"{da.sizes['member']} members, steps {int(da.step[0])}..{int(da.step[-1])}", flush=True)
        return da, da.init_time.values
    return loader


def build_argv(kind, p, model_key, name, years, out_dir):
    m = C.MODELS[model_key]
    common = ["--model_forecast_dir", str(m["tp_dir"]), "--imd_folder", str(C.IMD_DIR),
              "--thres_file", str(C.THRESH_FILE), "--mem_num", str(C.MEM_NUM),
              "--file_pattern", "{}.nc", "--date_filter_year", "2024"]
    yrs = ["--years"] + [str(y) for y in years]
    if kind == "mae":
        return yrs + common + [
            "--shpfile_path", str(C.SHAPEFILE), "--tolerance_days", str(p["tol"]),
            "--verification_window", str(p["vw"]), "--forecast_days", str(p["fd"]),
            "--max_forecast_day", str(p["mfd"]), "--mok",
            "--output_file", str(out_dir / f"results_{name}_{p['tag']}_MOK.nc"),
            "--plot_dir", str(out_dir)]
    if kind == "reliability":
        return yrs + common + ["--max_forecast_day", str(p["mfd"]), "--save_path", str(out_dir), "--mok"]
    if kind == "skill":
        return yrs + common + ["--max_forecast_day", str(p["mfd"]), "--model_name", name,
                               "--save_dir", str(out_dir), "--mok"]
    raise ValueError(kind)


def run_skill_custom_bins(mod, model_key, name, years, out_dir, day_bins, max_day, tag="weekly"):
    """Same sequence as binned_skill_score_cmz.main(), with caller-defined day bins."""
    import xarray as xr
    m = C.MODELS[model_key]
    kw = dict(date_filter_year=2024, file_pattern="{}.nc")
    print(f"custom bins {day_bins}, max day {max_day}")
    fo = mod.multi_year_forecast_obs_pairs(years, str(m["tp_dir"]), str(C.IMD_DIR), str(C.THRESH_FILE),
                                           C.MEM_NUM, max_day, day_bins, mok=True, **kw)
    brier_f, rps_f, auc_f = mod.calculate_brier_score(fo), mod.calculate_rps(fo), mod.calculate_auc(fo)
    thresh = xr.open_dataset(C.THRESH_FILE)["MWmean"]
    clim = mod.compute_climatological_onset_dataset(str(C.IMD_DIR), thresh, years=None, mok=True)
    co = mod.multi_year_climatological_forecast_obs_pairs(clim, years, day_bins, C.MEM_NUM, str(m["tp_dir"]),
                                                          max_forecast_day=max_day, mok=True, **kw)
    brier_c, rps_c, auc_c = (mod.calculate_brier_score_climatology(co), mod.calculate_rps(co),
                             mod.calculate_auc_climatology(co))
    skill = mod.calculate_skill_scores(brier_f, rps_f, brier_c, rps_c)
    mod.save_results(auc_f, brier_f, skill, rps_f, auc_c, brier_c, f"{name}_{tag}", max_day, save_dir=str(out_dir))
    mod.create_heatmap(skill, auc_f, auc_c, brier_f, brier_c, f"{name}_{tag}", max_day, save_dir=str(out_dir))


def run_step(model_key, mode, step, years, dates, out_dir):
    kind, p = STEPS[step]
    name = f"{model_key}_{mode}"
    mod = import_core(kind)
    mod.get_forecast_probabilistic_twice_weekly = make_loader(model_key, dates)
    argv = build_argv(kind, p, model_key, name, years, out_dir) if kind != "skill_weekly" else []
    log = out_dir / "logs" / f"step{step}_{kind}_{p['mfd']}d.log"
    U.ensure_dir(log.parent)
    t0 = time.time()
    print(f"step {step} ({kind}, {p}) -> log {log}", flush=True)
    old_argv = sys.argv
    sys.argv = [str(CORE[kind])] + argv
    try:
        with open(log, "w") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            if kind == "skill_weekly":
                run_skill_custom_bins(mod, model_key, name, years, out_dir, p["bins"], p["mfd"])
            else:
                print("argv:", " ".join(sys.argv))
                mod.main()
    finally:
        sys.argv = old_argv
    print(f"step {step} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--init-mode", default="native", choices=["native", "common"])
    ap.add_argument("--step", default="all", help="1-6 or 'all'")
    ap.add_argument("--years", nargs="+", type=int, default=C.EVAL_YEARS)
    a = ap.parse_args()

    dates = init_dates_by_year(a.model, a.init_mode, a.years)
    years = sorted(dates)
    skipped = sorted(set(a.years) - set(years))
    out_dir = U.ensure_dir(C.OUT_ROOT / "onset" / a.init_mode / a.model)
    info = dict(model=a.model, label=C.MODELS[a.model]["label"], init_mode=a.init_mode,
                years=years, years_without_inits=skipped, mem_num=C.MEM_NUM,
                day_offset=C.MODELS[a.model]["day_offset"],
                init_shift_days=C.MODELS[a.model].get("init_shift_days", 0),
                n_inits={str(y): len(d) for y, d in dates.items()},
                init_dates={str(y): [x.strftime("%Y-%m-%d") for x in d] for y, d in dates.items()})
    with open(out_dir / "run_info.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"{a.model} [{a.init_mode}] years {years} (no inits: {skipped}); "
          f"inits/yr {info['n_inits']}", flush=True)

    steps = list(STEPS) if a.step == "all" else [int(s) for s in a.step.split(",")]
    for s in steps:
        run_step(a.model, a.init_mode, s, years, dates, out_dir)


if __name__ == "__main__":
    main()
