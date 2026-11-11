#!/usr/bin/env python3
"""
Format-agnostic rank histogram (Talagrand diagram) driver.

Reuses the same config, loaders and masking conventions as crps_ssr_main.py,
but writes rank counts instead of CRPS/SSR. Existing CRPS outputs are untouched.

For each (init date, lead, grid point) the observation's rank among the N
ensemble members is counted, giving N+1 bins. Ties are broken randomly
(Hamill 2001), which matters for variables with many identical values
(e.g. precipitation zeros). Counts are accumulated per lead time.

Usage:
    python rank_histogram_main.py --model MODEL [--var VAR] [--config config.yaml]

Examples:
    python rank_histogram_main.py --model ours_si_postprocess_v2_0903 --var geopotential_500
    python rank_histogram_main.py --model amip_s2s_train_scratch --var geopotential_500
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import binary_dilation

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))
from loaders import get_loader
from crps_ssr_main import load_era5, _detect_member_dim


# ── Rank histogram core ───────────────────────────────────────────────────────

def _ranks_with_random_ties(obs: np.ndarray, ens: np.ndarray, rng) -> np.ndarray:
    """
    Rank of `obs` among ensemble members, with random tie-breaking.

    Parameters
    ----------
    obs : array, shape (...)
    ens : array, shape (n_mem, ...)   -- member dim FIRST

    Returns
    -------
    ranks : float array, shape (...); integer ranks in [0, n_mem], NaN where
            obs or any member is NaN.

    Method (Hamill 2001): the rank is the number of members strictly below the
    observation, plus a uniform draw over the block of members exactly equal to
    it. If `k` members tie with the observation, the observation may occupy any
    of the k+1 slots within that block with equal probability, so the offset is
    drawn from DiscreteUniform{0..k} -- NOT from Binomial(k, 0.5), which would
    concentrate the offset near k/2 and carve spurious structure into the
    histogram for zero-heavy or discretised fields such as precipitation.
    """
    n_mem = ens.shape[0]

    valid = np.isfinite(obs) & np.all(np.isfinite(ens), axis=0)

    below = np.sum(ens < obs, axis=0)
    ties  = np.sum(ens == obs, axis=0)

    ranks = below.astype(np.float64)
    has_tie = ties > 0
    if np.any(has_tie):
        # place obs uniformly at random within the tied block (k+1 slots)
        ranks[has_tie] += rng.integers(0, ties[has_tie] + 1)

    ranks[~valid] = np.nan
    del below, ties, valid
    return ranks


def compute_rank_histogram(
    forecast: xr.Dataset,
    analysis: xr.Dataset,
    var: str,
    *,
    time_dim: str = "time",
    lead_dim: str = "prediction_timedelta",
    member_dim: str = "number",
    seed: int = 0,
    weekly: bool = False,
    region: str = "all",
    bias_correct: bool = False,
    quiet: bool = False,
) -> xr.Dataset:
    """
    Accumulate rank counts per lead time over all init dates and grid points.

    Returns a Dataset with:
        rank_counts (prediction_timedelta, rank)  -- int64 counts, N+1 bins
        n_valid     (prediction_timedelta)        -- total counted samples
    """
    if var not in forecast or var not in analysis:
        raise KeyError(f"'{var}' must exist in both forecast and analysis.")

    f    = forecast[var]
    mdim = _detect_member_dim(f, member_dim)
    n_mem = f.sizes[mdim]
    n_bins = n_mem + 1

    valid_time = forecast[time_dim] + forecast[lead_dim]
    f = f.assign_coords(valid_time=valid_time)
    t = analysis[var].sel({time_dim: f["valid_time"]})

    # Align mismatched-but-equivalent grids, as in evaluate_crps_fast.
    for coord in ("lat", "lon"):
        if coord in f.coords and coord in t.coords:
            t = t.assign_coords({coord: f.coords[coord]})

    # Same static open-ocean / land mask as the CRPS driver, so that rank
    # histograms and CRPS are computed over identical spatial support.
    land_2d = np.all(np.isnan(t.values), axis=tuple(
        i for i, d in enumerate(t.dims) if d not in ("lat", "lon")
    ))
    if land_2d.any():
        exclude_2d = binary_dilation(land_2d, iterations=1)
        keep_2d = xr.DataArray(~exclude_2d, dims=["lat", "lon"], coords={
            "lat": t["lat"], "lon": t["lon"],
        })
        f = f.where(keep_2d)
        t = t.where(keep_2d)
        if not quiet: print(f"  Spatial mask: {int((~exclude_2d).sum())}/{exclude_2d.size} "
              f"grid points kept ({100*(~exclude_2d).sum()/exclude_2d.size:.1f}%).")

    if var == "sea_surface_temperature":
        # Same spurious-point screen as the CRPS driver: drop sea ice (ERA5
        # below the freezing point of sea water) and coastal fill values (any
        # member at/below it, where the model land-sea mask disagrees with
        # ERA5). Without this, ice-edge and coastal points dominate the SST
        # rank histogram -- they give a ~4.5 K rms error with ~0 K mean bias,
        # which looks like a model deficiency but is a masking artifact.
        SST_THRESH = 271.15
        era5_ok = t >= SST_THRESH
        ens_ok = f.min(dim=mdim) >= SST_THRESH
        keep_sst = era5_ok & ens_ok
        f = f.where(keep_sst)
        t = t.where(keep_sst)
        frac = float(keep_sst.mean())
        if not quiet: print(f"  SST mask (>{SST_THRESH} K, ERA5 and all members): "
              f"{100*frac:.1f}% of points kept.")

    corr_by_year = None
    if bias_correct:
        # Lead-dependent model climatology correction. For each (lead, lat, lon)
        # we estimate the systematic error as the mean of (ensemble mean - ERA5)
        # over init dates, then subtract it from every member.
        #
        # LEAVE-ONE-YEAR-OUT cross-validation: the correction applied to an init
        # date in year Y is estimated from all OTHER years only. Estimating and
        # applying on the same dates would absorb part of the random error too
        # and make the histogram look better than the correction really is.
        #
        # Memory: accumulate per-year sums one year at a time and keep only the
        # small (n_year, lead, lat, lon) correction table. Materialising the
        # full (time, lead, lat, lon) correction instead peaks above 100 GB on
        # the 305-date UC-S2S record and gets the job OOM-killed.
        yrs_all = f[time_dim].dt.year.values
        uniq = np.unique(yrs_all)
        if len(uniq) < 2:
            raise RuntimeError("--bias-correct needs >=2 years for leave-one-year-out.")
        s_y, c_y = {}, {}
        for y in uniq:
            idx = np.flatnonzero(yrs_all == y)
            e_y = (f.isel({time_dim: idx}).mean(dim=mdim) - t.isel({time_dim: idx}))
            s_y[y] = e_y.sum(dim=time_dim, skipna=True).compute() \
                     if hasattr(e_y, "compute") else e_y.sum(dim=time_dim, skipna=True)
            c_y[y] = e_y.notnull().sum(dim=time_dim).compute() \
                     if hasattr(e_y, "compute") else e_y.notnull().sum(dim=time_dim)
            del e_y
        sum_all = sum(s_y.values())
        cnt_all = sum(c_y.values())
        corr_by_year = {}
        for y in uniq:
            c_out = cnt_all - c_y[y]
            corr_by_year[y] = ((sum_all - s_y[y]) / c_out.where(c_out > 0)).values
        mabs = np.nanmean([np.nanmean(np.abs(v)) for v in corr_by_year.values()])
        if not quiet:
            print(f"  Bias correction: leave-one-year-out lead-dependent model "
                  f"climatology over years {list(uniq)}; "
                  f"mean |correction| = {mabs:.3f}", flush=True)

    if region != "all":
        # Restrict to land or ocean. The NaN-based mask above only fires for
        # fields that are undefined over land (e.g. SST); t2m is valid
        # everywhere, so an explicit land-sea mask is needed to subset it.
        lsm_path = ("/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/newdata/"
                    "land_sea_mask_180x360.nc")
        lsm = xr.open_dataset(lsm_path)["land_sea_mask"].squeeze()
        lsm = lsm.assign_coords(lat=t["lat"], lon=t["lon"])
        keep = (lsm > 0.5) if region == "land" else (lsm <= 0.5)
        f = f.where(keep)
        t = t.where(keep)
        if not quiet: print(f"  Region mask '{region}': {int(keep.sum())}/{keep.size} grid "
              f"points kept ({100*float(keep.mean()):.1f}%).")

    if weekly:
        # Standard S2S weekly aggregation: average BOTH the forecast members and
        # the verifying analysis over each 7-day window (days 1-7, 8-14, ...),
        # then rank the weekly means. Averaging before ranking is what makes this
        # a weekly-mean skill measure; pooling daily ranks afterwards would not.
        lead_h = f[lead_dim].values / np.timedelta64(1, "h")
        lead_d = np.rint(lead_h / 24).astype(int)
        groups, labels = [], []
        for w in range(1, 7):                      # weeks 1..6 -> days 1-7 ... 36-42
            sel = np.flatnonzero((lead_d >= 7 * (w - 1) + 1) & (lead_d <= 7 * w))
            if len(sel) == 0:
                continue
            if len(sel) < 7:
                print(f"  WARNING: week {w} has only {len(sel)}/7 daily steps; "
                      f"its mean is over fewer days.")
            groups.append(sel)
            labels.append(w)
        if not groups:
            raise RuntimeError("No complete weekly windows in the lead axis.")
        f = xr.concat([f.isel({lead_dim: g}).mean(lead_dim, skipna=False)
                       for g in groups], dim=lead_dim)
        t = xr.concat([t.isel({lead_dim: g}).mean(lead_dim, skipna=False)
                       for g in groups], dim=lead_dim)
        # Label each window by its week number (stored in the lead coord).
        f = f.assign_coords({lead_dim: np.array(labels)})
        t = t.assign_coords({lead_dim: np.array(labels)})
        if not quiet: print(f"  Weekly means: {len(labels)} windows (weeks {labels[0]}-{labels[-1]}), "
              f"{len(groups[0])} days each.")

    leads = f[lead_dim].values
    init_times = f[time_dim].values
    rng = np.random.default_rng(seed)

    counts = np.zeros((len(leads), n_bins), dtype=np.int64)

    # Loop over init dates to bound peak memory: each iteration holds only
    # (n_mem, n_lead, lat, lon) in RAM rather than the full 5-D cube.
    for i_t, init in enumerate(init_times):
        try:
            f_i = f.isel({time_dim: i_t}).transpose(mdim, lead_dim, "lat", "lon").values
            t_i = t.isel({time_dim: i_t}).transpose(lead_dim, "lat", "lon").values
        except Exception as exc:
            # A truncated/corrupt source file only fails here, when the lazy
            # read is forced. Drop that init date rather than lose the run.
            print(f"  SKIP init {str(init)[:10]}: read failed ({type(exc).__name__}: "
                  f"{str(exc)[:80]})", flush=True)
            continue

        if corr_by_year is not None:
            # subtract this year's leave-one-year-out correction from every member
            y = int(str(np.datetime_as_string(init, unit="Y")))
            f_i = f_i - corr_by_year[y][None, ...]

        ranks = _ranks_with_random_ties(t_i, f_i, rng)  # (n_lead, lat, lon)

        for i_l in range(len(leads)):
            r = ranks[i_l]
            r = r[np.isfinite(r)]
            if r.size:
                counts[i_l] += np.bincount(r.astype(np.int64), minlength=n_bins)

        del f_i, t_i, ranks
        if not quiet and ((i_t + 1) % 10 == 0 or i_t == len(init_times) - 1):
            print(f"  ... {i_t+1}/{len(init_times)} init dates processed "
                  f"({counts.sum():,} samples so far)", flush=True)

    if weekly:
        lead_hours = np.asarray(leads).astype(np.int64)   # week numbers 1..6
    else:
        lead_hours = (leads / np.timedelta64(1, "h")).astype(np.int64)

    return xr.Dataset(
        {
            "rank_counts": xr.DataArray(counts, dims=[lead_dim, "rank"]),
            "n_valid":     xr.DataArray(counts.sum(axis=1), dims=[lead_dim]),
        },
        coords={lead_dim: lead_hours, "rank": np.arange(n_bins)},
        attrs={"n_members": n_mem, "tie_breaking": "random (Hamill 2001)",
               "seed": seed},
    )


def _reliability_diagnostics(counts: np.ndarray) -> tuple[float, float, float]:
    """
    Return (ri, delta, bias) for one lead's rank counts.

    ri    : reliability index, sum(|observed_freq - 1/k|). This is the headline
            number for COMPARING models: it is a sample-size-independent effect
            size, reading as the fraction of probability mass sitting in the
            wrong bin. 0 = perfectly flat; ~0.5 = severely miscalibrated.
    delta : reduced chi-square against uniformity. This is a SIGNIFICANCE test,
            not an effect size -- it grows linearly with sample count, so with
            O(10^7) samples per lead even a trivial departure yields delta in the
            thousands. Reported for completeness; do NOT compare it across models
            with different numbers of init dates. Use `ri` for that.
    bias  : mean rank normalised to [-1, 1]; 0 = unbiased, >0 = forecasts
            too low (obs falls high in the ensemble), <0 = forecasts too high.
    """
    n = counts.sum()
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = len(counts)
    expected = n / k
    chi2 = ((counts - expected) ** 2 / expected).sum()
    delta = chi2 / (k - 1)

    freq = counts / n
    ri = float(np.abs(freq - 1.0 / k).sum())

    mean_rank = (np.arange(k) * counts).sum() / n
    bias = (mean_rank - (k - 1) / 2) / ((k - 1) / 2)
    return ri, float(delta), float(bias)


# ── Per-variable evaluation ───────────────────────────────────────────────────

def evaluate_variable(
    model_name: str,
    var_name: str,
    var_cfg: dict,
    model_cfg: dict,
    era5_cfg: dict,
    output_parent: str,
    max_members: int | None,
    seed: int,
    overwrite: bool,
    keep_dates: set[str] | None = None,
    date_tag: str = "",
    only_years: set[int] | None = None,
    weekly: bool = False,
    region: str = "all",
    bias_correct: bool = False,
) -> None:
    years     = model_cfg["years"]
    input_dir = model_cfg["input_dir"]
    fmt_key   = model_cfg["format"]

    if only_years:
        years = [y for y in years if y in only_years]
        if not years:
            raise ValueError(
                f"--years {sorted(only_years)} leaves nothing for {model_name} "
                f"(config years: {model_cfg['years']}).")

    year_tag = f"{min(years)}_{max(years)}" if len(years) > 1 else str(years[0])
    out_dir  = os.path.join(output_parent, model_name)
    suffix   = f"_{date_tag}" if date_tag else ""
    if weekly:
        suffix = "_weekly" + suffix
    if region != "all":
        suffix = f"_{region}" + suffix
    if bias_correct:
        suffix = "_bc" + suffix
    out_path = os.path.join(
        out_dir, f"{model_name}_{year_tag}_rank_hist_{var_name}{suffix}.nc")

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {out_path} already exists (use --overwrite to redo).")
        return

    era5_var = var_cfg["era5_var"]
    print(f"\n{'='*65}")
    print(f"  Model    : {model_name}")
    print(f"  Variable : {var_name}")
    print(f"  Model var: {var_cfg['model_var']}  x{var_cfg.get('model_scale', 1.0)}")
    print(f"  ERA5 var : {era5_var}  x{var_cfg.get('era5_scale', 1.0)}")
    if var_cfg.get("level"):
        print(f"  Level    : {var_cfg['level']}")
    print(f"{'='*65}")

    loader = get_loader(fmt_key)

    # The climatological baseline's "members" are calendar years, and slicing
    # them positionally would keep only the EARLIEST years (1979-1998 for a
    # max_members of 20), biasing the distribution to a colder era rather than
    # sampling the full climate record. Use every year instead.
    eff_max_members = None if fmt_key == "format_era5_clim" else max_members
    if eff_max_members != max_members:
        print(f"  (using all climatology years; ignoring max_members={max_members})")

    # ── streaming path (large datasets) ──────────────────────────────────────
    # Loaders that accept an `on_date` callback hand us one init date at a time
    # so the full (dates x members x leads x lat x lon) cube is never held in
    # memory. FuXi needs this: its cube peaks near 277 GB (2.67 GB/date x 104),
    # which exceeded every queue limit available.
    import inspect
    streaming = "on_date" in inspect.signature(loader.load_forecast).parameters
    if streaming and bias_correct:
        # The leave-one-year-out correction needs all init dates at once.
        raise NotImplementedError(
            f"--bias-correct is not supported on the streaming loader "
            f"'{fmt_key}' (it sees one init date at a time).")
    if streaming:
        print("Loading ERA5 analysis ...")
        analysis = load_era5(era5_cfg, years, var_cfg)

        acc = {"counts": None, "bins": None, "leads": None, "n": 0}

        def _on_date(ds_one):
            if keep_dates is not None:
                d = np.datetime_as_string(ds_one["time"].values, unit="D")
                if np.char.replace(d, "-", "")[0] not in keep_dates:
                    return
            res = compute_rank_histogram(
                ds_one, analysis, var=era5_var, time_dim="time",
                lead_dim="prediction_timedelta", member_dim="number",
                seed=seed, weekly=weekly, region=region, quiet=True)
            c = res["rank_counts"].values
            if acc["counts"] is None:
                acc["counts"] = c.astype(np.int64)
                acc["leads"] = res["prediction_timedelta"].values
                acc["bins"] = res["rank"].values
            else:
                acc["counts"] += c
            acc["n"] += 1
            if acc["n"] % 10 == 0:
                print(f"  ... {acc['n']} init dates counted "
                      f"({acc['counts'].sum():,} samples)", flush=True)

        loader.load_forecast(input_dir, years, var_name, var_cfg,
                             eff_max_members, on_date=_on_date)
        if acc["counts"] is None:
            raise RuntimeError(f"No init dates counted for {model_name}/{var_name}.")
        res = xr.Dataset(
            {"rank_counts": (("prediction_timedelta", "rank"), acc["counts"]),
             "n_valid": (("prediction_timedelta",), acc["counts"].sum(axis=1))},
            coords={"prediction_timedelta": acc["leads"], "rank": acc["bins"]},
            attrs={"n_members": len(acc["bins"]) - 1,
                   "tie_breaking": "random (Hamill 2001)", "seed": seed,
                   "n_init_dates": acc["n"]})
        print(f"  streamed {acc['n']} init dates")
        _finish(res, out_dir, out_path, weekly)
        return

    print("Loading forecast ...")
    forecast = loader.load_forecast(input_dir, years, var_name, var_cfg,
                                    eff_max_members)

    # Optional common-date subsetting. Applied after loading so it is
    # loader-agnostic (format_a keys off date dirs, format_b off filenames).
    if keep_dates is not None:
        init_str = np.datetime_as_string(forecast["time"].values, unit="D")
        init_str = np.char.replace(init_str, "-", "")
        mask = np.isin(init_str, sorted(keep_dates))
        n_before = forecast.sizes["time"]
        forecast = forecast.isel(time=np.flatnonzero(mask))
        n_after = forecast.sizes["time"]
        print(f"  Date filter: kept {n_after}/{n_before} init dates "
              f"({len(keep_dates)} requested).")
        if n_after == 0:
            raise RuntimeError(
                f"No init dates left after filtering for {model_name}/{var_name}.")
        missing = sorted(set(keep_dates) - set(init_str[mask]))
        if missing:
            print(f"  NOTE: {len(missing)} requested dates absent from this model, "
                  f"e.g. {missing[:5]}")

    forecast = forecast.chunk({
        "time": 1, "number": -1, "prediction_timedelta": -1, "lat": -1, "lon": -1
    })

    print("Loading ERA5 analysis ...")
    analysis = load_era5(era5_cfg, years, var_cfg)

    print("Computing rank histogram ...")
    res = compute_rank_histogram(
        forecast, analysis,
        var=era5_var,
        time_dim="time",
        lead_dim="prediction_timedelta",
        member_dim="number",
        seed=seed,
        weekly=weekly,
        region=region,
        bias_correct=bias_correct,
    )

    _finish(res, out_dir, out_path, weekly)


def _finish(res, out_dir, out_path, weekly):
    """Attach RI/delta/bias diagnostics, print the table, and write the file."""
    counts = res["rank_counts"].values
    ris, deltas, biases = [], [], []
    lead_label = "Week" if weekly else "Lead (h)"
    print(f"\n  {lead_label:>8}  {'N samples':>14}  {'RI':>7}  {'delta':>12}  {'bias':>8}")
    for i, lead in enumerate(res["prediction_timedelta"].values):
        r, d, b = _reliability_diagnostics(counts[i])
        ris.append(r); deltas.append(d); biases.append(b)
        print(f"  {int(lead):>8}  {counts[i].sum():>14,}  {r:>7.4f}  {d:>12.1f}  {b:>8.3f}")

    res["ri"]    = xr.DataArray(np.array(ris, dtype=np.float32),
                                dims=["prediction_timedelta"])
    res["delta"] = xr.DataArray(np.array(deltas, dtype=np.float32),
                                dims=["prediction_timedelta"])
    res["bias"]  = xr.DataArray(np.array(biases, dtype=np.float32),
                                dims=["prediction_timedelta"])
    res["ri"].attrs["long_name"]    = "reliability index, sum|freq - 1/(N+1)| (0 = flat)"
    res["delta"].attrs["long_name"] = "reduced chi-square (sample-size dependent; not comparable across models)"
    res["bias"].attrs["long_name"]  = "normalised mean-rank bias in [-1, 1]"

    os.makedirs(out_dir, exist_ok=True)
    res.to_netcdf(out_path)
    print(f"\nSaved -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute rank histograms for S2S ensemble forecasts.")
    parser.add_argument("--config",
                        default=str(Path(__file__).parent / "configuration" / "config.yaml"),
                        help="Path to YAML config file.")
    parser.add_argument("--model", default=None,
                        help="Run only this model (key in config models:).")
    parser.add_argument("--var", default=None,
                        help="Run only this variable (key in model variables:).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for tie-breaking (default 0).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if the output file exists.")
    parser.add_argument("--dates-file", default=None,
                        help="File of YYYYMMDD init dates (one per line) to restrict "
                             "the evaluation to, e.g. the common dates across models.")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Restrict to these years (subset of the model's config "
                             "years). Output filename uses the restricted range, so "
                             "per-year runs do not overwrite each other.")
    parser.add_argument("--region", choices=["all", "land", "ocean"], default="all",
                        help="Restrict the verification domain. Output gets a "
                             "_land/_ocean tag so it never overwrites the global run.")
    parser.add_argument("--bias-correct", action="store_true",
                        help="Remove a lead-dependent, grid-point-wise model "
                             "climatology bias from every member before "
                             "ranking. Estimated leave-one-year-out so the "
                             "correction is never fitted on the date it is "
                             "applied to. Output gets a _bc tag.")
    parser.add_argument("--weekly", action="store_true",
                        help="Average forecast and analysis over 7-day windows "
                             "(weeks 1-6) before ranking, the standard S2S "
                             "weekly-mean convention. Output gets a _weekly tag.")
    parser.add_argument("--date-tag", default="",
                        help="Suffix added to output filenames when --dates-file is "
                             "used, e.g. 'common104', so filtered runs do not "
                             "overwrite the full-period outputs.")
    args = parser.parse_args()

    keep_dates = None
    if args.dates_file:
        with open(args.dates_file) as fh:
            keep_dates = {ln.strip() for ln in fh if ln.strip()}
        print(f"Restricting to {len(keep_dates)} init dates from {args.dates_file}")
        if not args.date_tag:
            print("WARNING: --dates-file without --date-tag will overwrite "
                  "full-period outputs; consider passing --date-tag.")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    era5_cfg      = cfg["era5"]
    output_parent = cfg["output"]["parent_dir"]
    max_members   = cfg["compute"].get("max_members")
    models_cfg    = cfg["models"]

    if args.model:
        if args.model not in models_cfg:
            print(f"ERROR: model '{args.model}' not in config. Available: {list(models_cfg)}")
            sys.exit(1)
        models_cfg = {args.model: models_cfg[args.model]}

    for model_name, model_cfg in models_cfg.items():
        variables = model_cfg["variables"]
        if args.var:
            if args.var not in variables:
                print(f"ERROR: variable '{args.var}' not in model '{model_name}'. "
                      f"Available: {list(variables)}")
                sys.exit(1)
            variables = {args.var: variables[args.var]}

        for var_name, var_cfg in variables.items():
            evaluate_variable(
                model_name, var_name, var_cfg,
                model_cfg, era5_cfg, output_parent, max_members,
                args.seed, args.overwrite,
                keep_dates=keep_dates, date_tag=args.date_tag,
                only_years=set(args.years) if args.years else None,
                weekly=args.weekly,
                region=args.region,
                bias_correct=args.bias_correct,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
