"""W1 extremes (plan v2 §2): Brier skill score of weekly / biweekly 10th- and 90th-percentile events (exploratory).

* Thresholds q10, q90 per start, window and grid point = linear quantiles of the package's 30-member same-calendar-window ERA5
  reference ensembles (fixed 1991-2020 and trend-adjusted; ``Climatology.ensembles``, the same ensembles as the tercile edges).
* Events: below q10 (``lo``), above q90 (``hi``); strict inequalities as the tercile code. A point is not scored for an event where
  the threshold is degenerate (q10 equal to the smallest reference member, q90 equal to the largest; e.g. dry-season precipitation).
* Forecast probability = fraction of the drift-corrected members (members minus the package's LOYO drift ``drift_bias.npy``,
  precipitation clipped at 0) beyond the threshold.
* Brier score BS = (p - o)^2; reference = the constant climatological probability (0.1), BS_c = (0.1 - o)^2.
* BSS_D = 1 - mean BS / (mean BS_c + D), D = p_c (1 - p_c) / M (Weigel et al. 2007 ensemble-size correction, the Brier analogue
  of the package's debiased RPSS), so that an M-member ensemble drawn from the climatological distribution scores 0.
"""
from __future__ import annotations

import numpy as np

from s41eval.bing_eval import core as K

QS = (0.1, 0.9)
QUINT = (0.2, 0.4, 0.6, 0.8)                         # quintile categories (AI Weather Quest style, weekly means) [inferred setup]
P_CLIM = 0.1
EVENTS = ('lo', 'hi')


def quantile_edges(ref_ens, qs=QS):
    """ref_ens [n, ...] -> (edges [len(qs), ...], valid [len(qs), ...]); linear quantiles; valid = edge strictly inside
    (min, max) of the members for the lower tail / upper tail."""
    s = np.sort(np.asarray(ref_ens, np.float32), axis=0)
    n = s.shape[0]
    e, v = [], []
    for q in qs:
        h = (n - 1) * q
        j = int(np.floor(h))
        f = h - j
        x = s[j] if j + 1 >= n else s[j] + f * (s[j + 1] - s[j])
        e.append(x)
        v.append((x > s[0]) if q < 0.5 else (x < s[-1]))
    return np.stack(e).astype(np.float32), np.stack(v)


def quantiles(ref_ens, qs=QUINT):
    """ref_ens [n, ...] -> [len(qs), ...] linear quantiles (numpy 'linear')."""
    s = np.sort(np.asarray(ref_ens, np.float32), axis=0)
    n = s.shape[0]
    out = []
    for q in qs:
        h = (n - 1) * q
        j = int(np.floor(h))
        f = h - j
        out.append(s[j] if j + 1 >= n else s[j] + f * (s[j + 1] - s[j]))
    return np.stack(out).astype(np.float32)


def rps_k(ens, obs, edges):
    """ranked probability score with K = len(edges) + 1 categories (cumulative, not normalised): forecast F_k = fraction of
    members < edge_k, observed O_k = 1[obs < edge_k]; climatological F_k = k / K. -> (rps, rps_clim) [...], NaN where obs/edges NaN."""
    e = np.asarray(ens)
    o = np.asarray(obs)
    K_ = edges.shape[0] + 1
    r = np.zeros(o.shape, np.float64)
    rc = np.zeros(o.shape, np.float64)
    for k in range(K_ - 1):
        Fk = (e < edges[k][None]).mean(0)
        Ok = (o < edges[k]).astype(np.float64)
        r += (Fk - Ok) ** 2
        rc += ((k + 1) / K_ - Ok) ** 2
    ok = np.isfinite(o) & np.isfinite(edges).all(0)
    return np.where(ok, r, np.nan), np.where(ok, rc, np.nan)


def rpss_D(M, K_=5):
    """Weigel et al. (2007) ensemble-size correction (K^2 - 1) / (6 K M)."""
    return (K_ * K_ - 1) / (6.0 * K_ * M)


def event_probs(ens, edges):
    """ens [m, ...], edges [2, ...] -> [2, ...] fraction of members below edges[0] / above edges[1]."""
    e = np.asarray(ens)
    return np.stack([(e < edges[0][None]).mean(0), (e > edges[1][None]).mean(0)])


def obs_events(obs, edges):
    o = np.asarray(obs)
    return np.stack([(o < edges[0]).astype(np.float64), (o > edges[1]).astype(np.float64)])


def brier_maps(ens, obs, edges, valid, p_clim=P_CLIM):
    """-> bs_f, bs_c, ev [2, ...] with NaN where obs / edge is missing or the edge is degenerate."""
    p = event_probs(ens, edges)
    o = obs_events(obs, edges)
    ok = np.isfinite(np.asarray(obs))[None] & np.isfinite(edges) & valid
    bs_f = np.where(ok, (p - o) ** 2, np.nan)
    bs_c = np.where(ok, (p_clim - o) ** 2, np.nan)
    ev = np.where(ok, o, np.nan)
    return bs_f, bs_c, ev


def debias_D(M, p_clim=P_CLIM):
    return p_clim * (1 - p_clim) / M


def bss_d(bs_f, bs_c, M, p_clim=P_CLIM):
    """pooled debiased BSS from per-start (region-mean) Brier scores."""
    bs_f, bs_c = np.asarray(bs_f, np.float64), np.asarray(bs_c, np.float64)
    ok = np.isfinite(bs_f) & np.isfinite(bs_c)
    if not ok.any():
        return np.nan
    return 1 - bs_f[ok].mean() / (bs_c[ok].mean() + debias_D(M, p_clim))


def score_system(F_dc, O, E_edges, E_valid, regions=None, grid='g15', Q_edges=None):
    """per-start region means. F_dc: callable s -> members [M, W, y, x] (drift corrected) or None; O [S, W, y, x];
    E_edges / E_valid: dict ref -> [S, 2, W, y, x]. -> dict ref -> arrays [S, W, R, 2] of bs_f, bs_c, ev (region means)."""
    regions = regions or K.RK
    mk = K.masks(grid)
    S, W = O.shape[:2]
    out = {r: {k: np.full((S, W, len(regions), 2), np.nan) for k in ('bs_f', 'bs_c', 'ev')} for r in E_edges}
    if Q_edges is not None:
        for r in Q_edges:
            out[r]['rps5'] = np.full((S, W, len(regions), 2), np.nan)          # [..., 0] forecast, [..., 1] climatology
    for s in range(S):
        X = F_dc(s)
        if X is None:
            continue
        for r in E_edges:
            ed = np.moveaxis(np.asarray(E_edges[r][s]), 0, 0)               # [2, W, y, x]
            va = np.asarray(E_valid[r][s]).astype(bool)
            bf, bc, ev = brier_maps(X, O[s], ed, va)                         # [2, W, y, x]
            for j, rn in enumerate(regions):
                if rn not in mk:
                    continue
                for k, x in (('bs_f', bf), ('bs_c', bc), ('ev', ev)):
                    out[r][k][s, :, j, :] = np.moveaxis(K.amean(x, mk[rn], grid), 0, -1)
            if Q_edges is not None:
                rp, rc = rps_k(X, O[s], np.asarray(Q_edges[r][s]))
                for j, rn in enumerate(regions):
                    if rn in mk:
                        out[r]['rps5'][s, :, j, 0] = K.amean(rp, mk[rn], grid)
                        out[r]['rps5'][s, :, j, 1] = K.amean(rc, mk[rn], grid)
    return out
