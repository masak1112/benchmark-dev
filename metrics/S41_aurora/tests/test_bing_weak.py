"""synthetic CPU tests of the helper-B diagnostics (s41eval.bing_weak)."""
from datetime import date, timedelta

import numpy as np
import pytest

from s41eval import bing_weak as B
from s41eval.bing_eval import core as K


def test_zonal_spectrum_parseval_and_peak():
    lat = np.linspace(-90, 90, 121)
    lon = np.arange(240) * 1.5
    x = 3.0 + 2.0 * np.cos(np.deg2rad(5 * lon))[None, :] * np.ones((121, 1))
    P = B.zonal_spectrum(x, lat, (30, 60))
    assert np.argmax(P) == 5
    assert P[5] == pytest.approx(2.0 ** 2 / 2, rel=1e-9)           # variance of 2 cos = 2
    rng = np.random.default_rng(0)
    y = rng.normal(size=(121, 240))
    P = B.zonal_spectrum(y, lat, (-90, 90))
    w = np.cos(np.deg2rad(lat))
    assert P.sum() == pytest.approx((y.var(-1) * w).sum() / w.sum(), rel=1e-9)


def test_hist_quantile_and_frac():
    rng = np.random.default_rng(1)
    x = rng.exponential(5.0, size=(200, 121, 240)).astype(np.float32)
    h = B.wavg_hist(x, np.ones((121, 240)))
    q = B.hist_quantile(h, 0.99)
    assert q == pytest.approx(np.quantile(x, 0.99), rel=0.03)
    f = B.hist_frac(h, B.HEDGES[0], B.HEDGES[np.argmin(np.abs(B.HEDGES - 1.0))])
    assert f == pytest.approx((x < B.HEDGES[np.argmin(np.abs(B.HEDGES - 1.0))]).mean(), abs=1e-6)


def test_rank_flatness_and_reliability():
    p, d, u = B.rank_flatness(np.ones(31))
    assert d == pytest.approx(0) and u == pytest.approx(1)
    p, d, u = B.rank_flatness(np.r_[10, np.ones(29), 10])
    assert u > 5 and d > 0
    # perfectly reliable counts: observed frequency = forecast probability
    pb = np.linspace(0, 1, 11)
    c = np.stack([np.ones(11), pb, pb], 1)
    pm, of, wf, rel, res, obar = B.reliability_curve(c)
    assert rel == pytest.approx(0) and np.allclose(pm, of)


def _synthetic(S=60, M=30, seed=3, drift=0.0, ny=121, nx=240, january=False):
    rng = np.random.default_rng(seed)
    starts = ([date(2019 + i // 8, 1, 1 + 3 * (i % 8)) for i in range(S)] if january else
              [date(2019 + i // 10, 1 + (i % 10), 7) for i in range(S)])
    sig = rng.normal(size=(S, 1, ny, nx)).astype(np.float32)           # predictable signal
    O = sig + rng.normal(size=(S, 1, ny, nx)).astype(np.float32)
    F = sig[:, None] + rng.normal(size=(S, M, 1, ny, nx)).astype(np.float32) + drift
    clim = np.zeros((S, 1, ny, nx), np.float32)
    # reference: 30-member climatological ensemble N(0, 2)
    E = rng.normal(scale=np.sqrt(2), size=(30, S, 1, ny, nx)).astype(np.float32)
    CC = np.stack([K.fair_crps(E[:, s], O[s]) for s in range(S)]).astype(np.float32)
    return starts, F, O, {'fixed': CC}, {'fixed': clim}


def test_subset_scores_drift_removed_and_members():
    mk = {'global': np.ones((121, 240))}
    st, F, O, CC, C = _synthetic(drift=0.0)
    a = B.subset_scores(F, O, CC, C, st, np.arange(30), mk)
    st, F2, O2, CC2, C2 = _synthetic(drift=5.0)                       # same random draws + a constant drift
    b = B.subset_scores(F2, O2, CC2, C2, st, np.arange(30), mk)
    assert a[('crpss', 'fixed', 'global', 0)] == pytest.approx(b[('crpss', 'fixed', 'global', 0)], abs=1e-4)
    assert a[('acc', 'fixed', 'global', 0)] == pytest.approx(b[('acc', 'fixed', 'global', 0)], abs=1e-4)
    # theory for this toy (LOYO drift from 5 training starts adds (1 + 1/M) / 5 of noise variance):
    # ensemble-mean ACC with M members = 1 / sqrt(2 (1 + 1/M + (1 + 1/M) / 5)); fair CRPS nearly independent of M
    acc5 = np.mean([B.subset_scores(F, O, CC, C, st, np.arange(k, k + 5), mk)[('acc', 'fixed', 'global', 0)] for k in (0, 5, 10)])
    assert a[('acc', 'fixed', 'global', 0)] == pytest.approx(1 / np.sqrt(2 * (1 + 1 / 30 + (1 + 1 / 30) / 5)), abs=0.02)
    assert acc5 == pytest.approx(1 / np.sqrt(2 * (1 + 1 / 5 + (1 + 1 / 5) / 5)), abs=0.02)
    c5 = B.subset_scores(F, O, CC, C, st, np.arange(5), mk)[('crpss', 'fixed', 'global', 0)]
    assert c5 == pytest.approx(a[('crpss', 'fixed', 'global', 0)], abs=0.02)


def test_windows_of_shape():
    x = np.arange(46, dtype=np.float32)[:, None, None, None, None] * np.ones((46, 2, 3, 4, 5), np.float32)
    w = B.windows_of(x)
    assert w.shape == (4, 2, 3, 4, 5)
    assert w[0, 0, 0, 0, 0] == pytest.approx(4.0)                      # w1 = mean of days 1..7
    assert w[2, 0, 0, 0, 0] == pytest.approx(21.5)                     # w34 = days 15..28


def test_h5_path():
    assert B.h5_path(date(2019, 1, 7)).name == '2019_0024.h5'
    assert B.h5_path(date(2020, 12, 31)).name == '2020_1460.h5'


def test_member_offsets_detects_persistent_member_bias():
    rng = np.random.default_rng(5)
    S, M = 80, 10
    m = np.ones((121, 240))
    F = rng.normal(size=(S, M, 121, 240)).astype(np.float32)
    r, frac, b = B.member_offsets(F, m)
    assert abs(r.mean()) < 0.05 and frac == pytest.approx(1 / S, rel=0.2)
    off = rng.normal(scale=0.5, size=(1, M, 121, 240)).astype(np.float32)         # fixed per-member pattern
    r2, frac2, _ = B.member_offsets(F + off, m)
    assert r2.mean() > 0.8 and frac2 > 0.2


def test_permember_correction_removes_fixed_member_offsets():
    mk = {'global': np.ones((121, 240))}
    st, F, O, CC, C = _synthetic(S=64, M=10, january=True, ny=61, nx=120)   # 56 LOYO training starts per target
    mk = {'global': np.ones((61, 120))}
    a = B.permember_scores(F, O, CC, st, mk, grid='g30')
    assert a[('dc_member', 'fixed', 'global', 0)] == pytest.approx(a[('dc_ens', 'fixed', 'global', 0)], abs=0.03)
    off = np.random.default_rng(9).normal(scale=1.0, size=(1, 10, 1, 61, 120)).astype(np.float32)
    b = B.permember_scores(F + off, O, CC, st, mk, grid='g30')
    # the per-member correction removes fixed member offsets exactly; the ensemble correction keeps them as spread
    assert b[('dc_member', 'fixed', 'global', 0)] == pytest.approx(a[('dc_member', 'fixed', 'global', 0)], abs=1e-5)
    assert abs(b[('dc_ens', 'fixed', 'global', 0)] - a[('dc_ens', 'fixed', 'global', 0)]) > 1e-3
