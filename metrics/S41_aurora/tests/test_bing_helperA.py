"""Synthetic CPU tests of helper A's additions (scorecard over windows / systems, BSS of 10th/90th-percentile events, FDR maps).
Run: ``PYTHONPATH=envs/bing_pylib:<worktree> python -m pytest -q tests/test_bing_helperA.py``."""
from __future__ import annotations

import numpy as np

from s41eval.bing_eval import diffmaps as DM
from s41eval.bing_eval import extremes as EX
from s41eval.bing_eval import scorecard as SC
from s41eval.bing_eval import stats as ST


def test_quantile_edges_match_numpy_and_flag_degenerate():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(30, 4, 5)).astype(np.float32)
    e, v = EX.quantile_edges(E)
    np.testing.assert_allclose(e, np.quantile(E, [0.1, 0.9], axis=0), rtol=1e-5, atol=1e-6)
    assert v.all()
    Z = np.zeros((30, 3))
    Z[-2:] = 1.0                                  # dry: q10 = 0 = minimum -> lower tail degenerate
    e, v = EX.quantile_edges(Z)
    assert not v[0].any() and e[0].max() == 0


def test_bss_climatological_ensemble_is_zero_and_perfect_is_positive():
    rng = np.random.default_rng(1)
    n, M, P_ = 4000, 10, 50
    obs = rng.normal(size=(n, P_))
    edges = np.stack([np.full(P_, np.quantile(rng.normal(size=10 ** 6), 0.1)), np.full(P_, np.quantile(rng.normal(size=10 ** 6), 0.9))])
    valid = np.ones((2, P_), bool)
    bf_c, bc_c, bf_p = [], [], []
    for i in range(n):
        ens = rng.normal(size=(M, P_))                                   # same distribution as obs, no information
        bf, bc, _ = EX.brier_maps(ens, obs[i], edges, valid)
        bf_c.append(np.nanmean(bf, 1))
        bc_c.append(np.nanmean(bc, 1))
        ens_p = obs[i][None] + 0.1 * rng.normal(size=(M, P_))            # informative
        bp, _, _ = EX.brier_maps(ens_p, obs[i], edges, valid)
        bf_p.append(np.nanmean(bp, 1))
    bf_c, bc_c, bf_p = map(np.array, (bf_c, bc_c, bf_p))
    for k in range(2):
        assert abs(EX.bss_d(bf_c[:, k], bc_c[:, k], M)) < 0.02          # debiased: ~0 for a climatological M-member ensemble
        assert 1 - bf_c[:, k].mean() / bc_c[:, k].mean() < -0.05          # without the correction it is negative (~ -1/M)
        assert EX.bss_d(bf_p[:, k], bc_c[:, k], M) > 0.6


def test_paired_acc_diff2_equals_headline_version_for_shared_verification():
    rng = np.random.default_rng(2)
    S = 40
    o = rng.normal(size=(S, 121, 240)).astype(np.float32)
    fa = (o + rng.normal(size=o.shape)).astype(np.float32)
    fb = (o + 1.5 * rng.normal(size=o.shape)).astype(np.float32)
    mk = np.zeros((121, 240))
    mk[50:60, 10:30] = 1
    r1 = ST.paired_acc_diff(fa, fb, o, mk, n_draws=50, seed=3)
    r2 = SC.paired_acc_diff2(fa, o, fb, o, mk, n_draws=50, seed=3)
    assert r1['diff'] == r2['diff'] and r1['ci'] == r2['ci']


def test_fdr_wilks():
    p = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590,
                  1.000, np.nan])
    sig, thr = DM.fdr_wilks(p, 0.05)                                  # Benjamini & Hochberg (1995) example: 4 rejections at q = 0.05
    assert sig.sum() == 4 and thr == 0.0095
    sig, thr = DM.fdr_wilks(np.array([0.5, 0.9]), 0.1)
    assert sig.sum() == 0


def test_diff_maps_zero_for_identical_systems_and_detect_a_real_difference():
    rng = np.random.default_rng(4)
    S, P_ = 60, 200
    cr = rng.gamma(2, 1, size=(S, P_))
    ca = cr * 0.8
    m = DM.crpss_diff_map(ca, cr, ca.copy(), cr, n_draws=100)
    assert np.allclose(m['d'], 0) and np.allclose(m['a'], 0.2)
    cb = cr * np.where(np.arange(P_) < 100, 0.9, 0.8) * rng.uniform(0.95, 1.05, size=(S, P_))
    m = DM.crpss_diff_map(ca, cr, cb, cr, n_draws=200)
    sig, _ = DM.fdr_wilks(m['p'], 0.1)
    assert sig[:100].mean() > 0.9 and sig[100:].mean() < 0.2
    o = rng.normal(size=(S, P_))
    f = o + rng.normal(size=(S, P_))
    m = DM.acc_diff_map(f, o, f, o, n_draws=50)
    assert np.allclose(m['d'], 0)


def test_quintile_rps_climatological_ensemble_debiased_zero():
    rng = np.random.default_rng(5)
    n, M, P_ = 3000, 10, 40
    big = rng.normal(size=10 ** 6)
    edges = np.stack([np.full(P_, np.quantile(big, q)) for q in EX.QUINT])
    r, rc = [], []
    for i in range(n):
        a, b = EX.rps_k(rng.normal(size=(M, P_)), rng.normal(size=P_), edges)
        r.append(a.mean())
        rc.append(b.mean())
    rpss = 1 - np.mean(r) / (np.mean(rc) + EX.rpss_D(M, 5))
    assert abs(rpss) < 0.02
    e3 = np.stack([np.full(P_, np.quantile(big, q)) for q in (1 / 3, 2 / 3)])
    from s41eval.bing_eval import core as K
    ens, obs = rng.normal(size=(M, P_)), rng.normal(size=P_)
    a, _ = EX.rps_k(ens, obs, e3)
    plo, phi = K.tercile_probs(ens, e3)
    np.testing.assert_allclose(a, K.rps3(plo, phi, obs, e3))          # K = 3 equals the package's tercile RPS (no ties)
