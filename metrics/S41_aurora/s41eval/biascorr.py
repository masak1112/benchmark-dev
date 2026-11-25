"""S40 symmetric bias correction (2026-11-04 DSI brief task 5; identical code path for UC-S2S and AIFS).

Three levels, applied to daily forecast fields already on the common 1.5-degree grid:
  raw     x
  mean    x - b,                        b = m_mod - m_obs
  mspread (x - m_mod) * s_obs / s_mod + m_obs
where, per grid point and lead day,
  m_mod, s_mod = mean and standard deviation over all members of all TRAINING starts,
  m_obs, s_obs = mean and standard deviation of the verifying observation over the same training starts,
TRAINING starts for a start s = starts of the OTHER year(s) (leave-one-year-out) whose start day-of-year lies within +-30 days of s
(circular). The four statistics are then smoothed over leads with a centred 7-day running mean (truncated at the ends). Starts
without any training start (e.g. Oct - Dec, present only in 2025) get NaN for the corrected levels (= raw scores only).
The mean+spread level rescales the model anomaly about its own (training) climatology by the observed/model standard-deviation
ratio, i.e. the classical mean-and-variance adjustment; it changes both the ensemble spread and the amplitude of the ensemble mean.

Exactness (tested by `selftest`): if x = k * o + c (any k > 0, constant c) the mean level returns x - c exactly and the mean+spread
level returns o exactly, for any training set, because every statistic is linear in o and the smoothing is linear.

Array layout is given by an axis string, e.g. 'SMLYX' (start, member, lead, lat, lon) or 'SLMYX'; obs layout 'SLYX'.

    python -m s41eval.biascorr selftest
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np

WINDOW_D = 30
LEAD_SMOOTH = 7


def _to_std(x, layout, want):
    return np.transpose(x, [layout.index(c) for c in want])


def _from_std(x, layout, have):
    return np.transpose(x, [have.index(c) for c in layout])


def _smooth_leads(a, n=LEAD_SMOOTH):
    """a [L, ...] -> centred running mean over axis 0, truncated window at the ends (NaN-aware)."""
    h = n // 2
    out = np.empty_like(a)
    for l in range(a.shape[0]):
        out[l] = np.nanmean(a[max(0, l - h):l + h + 1], axis=0)
    return out


def _doy_dist(a: date, b: date):
    d = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(d, 365 - d)


def training_sets(starts, window=WINDOW_D):
    """starts: list of date -> list (per start) of indices of other-year starts within +-window days (circular day of year)."""
    return [[j for j, t in enumerate(starts) if t.year != s.year and _doy_dist(s, t) <= window] for s in starts]


def correct(fc, obs, starts, fc_layout='SMLYX', obs_layout='SLYX', window=WINDOW_D, lead_smooth=LEAD_SMOOTH):
    """fc [.. S, M, L, Y, X ..] forecasts, obs [.. S, L, Y, X ..] verifying observations (same valid days).
    Returns dict(raw=fc, mean=..., mspread=..., ntrain=[S]) in the input layout (float32)."""
    F = _to_std(np.asarray(fc, np.float64), fc_layout, 'SMLYX')
    O = _to_std(np.asarray(obs, np.float64), obs_layout, 'SLYX')
    S = F.shape[0]
    assert O.shape[0] == S and O.shape[1:] == F.shape[2:], (F.shape, O.shape)
    tr = training_sets(list(starts), window)
    mean_c = np.full_like(F, np.nan)
    msp_c = np.full_like(F, np.nan)
    for s in range(S):
        idx = tr[s]
        if not idx:
            continue
        Ft = F[idx]                                         # [n, M, L, Y, X]
        Ot = O[idx]                                         # [n, L, Y, X]
        m_mod = Ft.mean(axis=(0, 1))
        s_mod = Ft.reshape(-1, *Ft.shape[2:]).std(axis=0)
        m_obs = Ot.mean(axis=0)
        s_obs = Ot.std(axis=0)
        if lead_smooth and lead_smooth > 1:
            m_mod, s_mod, m_obs, s_obs = (_smooth_leads(a, lead_smooth) for a in (m_mod, s_mod, m_obs, s_obs))
        mean_c[s] = F[s] - (m_mod - m_obs)[None]
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(s_mod > 0, s_obs / s_mod, np.nan)
        msp_c[s] = (F[s] - m_mod[None]) * r[None] + m_obs[None]
    back = lambda a: _from_std(a, fc_layout, 'SMLYX').astype(np.float32)
    return dict(raw=np.asarray(fc, np.float32), mean=back(mean_c), mspread=back(msp_c), ntrain=[len(t) for t in tr])


def selftest():
    rng = np.random.default_rng(0)
    starts = [date(2025, 5, 1), date(2025, 5, 15), date(2025, 6, 2), date(2025, 11, 3),
              date(2026, 5, 4), date(2026, 5, 18), date(2026, 6, 1)]
    S, M, L, Y, X = len(starts), 4, 12, 5, 6
    obs = rng.normal(280, 3, (S, L, Y, X))
    ok = True
    # constant bias c on a forecast that equals the observation: the mean level must recover c exactly and return obs;
    # x = k o + c: the mean+spread level must return obs exactly; both for two array layouts and both latitude orders.
    for name, lay, olay in (('UC-S2S layout SMLYX, lat ascending', 'SMLYX', 'SLYX'),
                            ('AIFS layout SLMYX, lat descending', 'SLMYX', 'SLYX')):
        o = obs if 'ascending' in name else obs[..., ::-1, :].copy()
        c, k = 1.75, 1.6
        back = lambda a: np.transpose(a, [lay.index(ch) for ch in 'SMLYX'])
        # x = k o + c (members identical): mean+spread level must return o exactly
        fk = k * o[:, None] + c + np.zeros((1, M, 1, 1, 1))
        rk = correct(np.transpose(fk, ['SMLYX'.index(ch) for ch in lay]), o, starts, lay, olay)
        has = np.array(rk['ntrain']) > 0
        msp_err = float(np.nanmax(np.abs(back(rk['mspread'])[has] - o[has][:, None])))
        # constant-bias exactness: x = o + c (k = 1, no scatter)
        f1 = o[:, None] + c + np.zeros((1, M, 1, 1, 1))
        r1 = correct(np.transpose(f1, ['SMLYX'.index(ch) for ch in lay]), o, starts, lay, olay)
        b_err = float(np.nanmax(np.abs((back(r1['raw']) - back(r1['mean']))[has].astype(np.float64) - c)))
        m_err = float(np.nanmax(np.abs(back(r1['mean'])[has] - o[has][:, None])))
        nan_ok = bool(np.isnan(back(r1['mean'])[~has]).all() and np.isnan(back(r1['mspread'])[~has]).all())
        tol = 1e-3                                                 # float32 output of ~280 K fields (ulp 3e-5)
        p = b_err < tol and m_err < tol and msp_err < tol and nan_ok
        ok &= p
        print(f'{name}: constant bias c={c}: recovered bias max err {b_err:.2e}, corrected = obs max err {m_err:.2e}; '
              f'x = {k} o + {c}: mean+spread = obs max err {msp_err:.2e}; starts without training (Nov) -> NaN: {nan_ok}; '
              f'ntrain {rk["ntrain"]} -> {"PASS" if p else "FAIL"}', flush=True)
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        raise SystemExit(0 if selftest() else 1)
