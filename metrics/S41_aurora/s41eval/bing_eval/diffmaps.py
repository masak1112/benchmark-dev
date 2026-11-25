"""W2 difference maps (plan v2 §2; exploratory): per grid point, paired A - B of CRPSS (each system against its own reference
CRPS of the same reference type) and of centred ACC (anomalies vs each reference climatology), on the common starts, with a
moving-block bootstrap over starts in date order (block 4, 1,000 draws) and the Wilks (2016) / Benjamini-Hochberg false-discovery-rate
control at alpha_FDR = 0.1 over the points of the map.

p-values: two-sided normal approximation from the bootstrap standard error, p = 2 Phi(-|d| / se_boot) (continuous p-values;
percentile p-values with 1,000 draws are too coarse for FDR over ~29,000 points). Points where a system is NaN in any start are
left NaN. Also returned: per-point sums (mean CRPS, mean reference CRPS) so land / ocean region CRPSS can be formed afterwards.
"""
from __future__ import annotations

import math

import numpy as np

from s41eval.bing_eval import stats as ST


def fdr_wilks(p, alpha=0.1):
    """Benjamini-Hochberg / Wilks (2016) on the finite p-values: -> boolean significance [same shape] and the p threshold."""
    p = np.asarray(p, np.float64)
    ok = np.isfinite(p)
    v = np.sort(p[ok])
    n = v.size
    if n == 0:
        return np.zeros(p.shape, bool), 0.0
    k = np.flatnonzero(v <= alpha * np.arange(1, n + 1) / n)
    if not k.size:
        return np.zeros(p.shape, bool), float('nan')
    thr = float(v[k[-1]])
    return ok & (p <= thr), thr


def pvals(d0, draws):
    """two-sided p from the bootstrap SE (draws [B, P])."""
    se = np.nanstd(draws, 0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = d0 / se
    from scipy.special import erfc
    return np.where(np.isfinite(z), erfc(np.abs(z) / math.sqrt(2)), np.nan)


def crpss_diff_map(ca, cra, cb, crb, block=4, n_draws=1000, seed=0, batch=200):
    """ca, cra, cb, crb [S, P] per-start per-point CRPS (A, ref of A, B, ref of B). -> dict d [P], p [P], crpss_a, crpss_b."""
    S = ca.shape[0]
    ok = np.isfinite(ca).all(0) & np.isfinite(cb).all(0) & np.isfinite(cra).all(0) & np.isfinite(crb).all(0)
    X = [np.where(ok, x, 0.0).astype(np.float64) for x in (ca, cra, cb, crb)]

    def st(c):
        sa, sra, sb, srb = (c @ x for x in X)
        with np.errstate(invalid='ignore', divide='ignore'):
            return (1 - sa / sra), (1 - sb / srb)
    a0, b0 = st(np.ones((1, S)))
    idx = ST.mbb_indices(S, block, n_draws, seed)
    dr = []
    for i in range(0, n_draws, batch):
        c = ST.counts_from_idx(idx[i:i + batch], S)
        a, b = st(c)
        dr.append(a - b)
    dr = np.concatenate(dr)
    d0 = (a0 - b0)[0]
    p = pvals(d0, dr)
    nan = np.where(ok, 1.0, np.nan)
    return dict(d=d0 * nan, p=p * nan, a=a0[0] * nan, b=b0[0] * nan)


def acc_diff_map(fa, oa, fb, ob, block=4, n_draws=1000, seed=0, batch=100):
    """fa, oa, fb, ob [S, P] anomalies; per-point centred correlation difference with block bootstrap."""
    S = fa.shape[0]
    ok = np.isfinite(fa).all(0) & np.isfinite(fb).all(0) & np.isfinite(oa).all(0) & np.isfinite(ob).all(0)
    A, OA, B, OB = (np.where(ok, x, 0.0).astype(np.float64) for x in (fa, oa, fb, ob))
    pre = (A, OA, A * A, OA * OA, A * OA, B, OB, B * B, OB * OB, B * OB)

    def st(c):
        ra = ST.acc_from_sums(c, *pre[:5])
        rb = ST.acc_from_sums(c, *pre[5:])
        return ra, rb
    a0, b0 = st(np.ones((1, S)))
    idx = ST.mbb_indices(S, block, n_draws, seed)
    dr = []
    for i in range(0, n_draws, batch):
        c = ST.counts_from_idx(idx[i:i + batch], S)
        a, b = st(c)
        dr.append(a - b)
    dr = np.concatenate(dr)
    d0 = (a0 - b0)[0]
    p = pvals(d0, dr)
    nan = np.where(ok, 1.0, np.nan)
    return dict(d=d0 * nan, p=p * nan, a=a0[0] * nan, b=b0[0] * nan)
