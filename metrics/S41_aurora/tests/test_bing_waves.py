"""Synthetic tests of the Wheeler-Kiladis spectrum and filter (bing_waves): prescribed waves must land in their band."""
import numpy as np

from s41eval import bing_waves as W

LON = np.deg2rad(np.arange(240) * 1.5)
LAT = W.LATEQ


def _wave(k, freq_cpd, T=45, lat_shape='sym', amp=1.0, seed=0):
    t = np.arange(T)[:, None, None]
    lam = LON[None, None, :]
    y = LAT[None, :, None]
    env = np.exp(-(y / 8.0) ** 2) * (1.0 if lat_shape == 'sym' else y / 8.0)
    x = amp * env * np.cos(k * lam - 2 * np.pi * freq_cpd * t)
    rng = np.random.default_rng(seed)
    return x + 0.05 * rng.standard_normal(x.shape)


def _kelvin_freq(k, h=25.0):
    return float(W.disp_freq(k, h, 'kelvin'))


def test_kelvin_lands_in_kelvin_band():
    k = 6
    f = _kelvin_freq(k)                       # ~ 0.17 cpd for h = 25 m
    T = 45
    f = round(f * T) / T                      # on a Fourier frequency
    sym, anti = W.wk_power(_wave(k, f, T), taper=0.1)
    freq, ks = W.wk_axes(T)
    j, i = np.unravel_index(np.argmax(sym), sym.shape)
    assert ks[i] == k and abs(freq[j] - f) < 1e-9
    assert W.band_mask(freq, ks, 'kelvin')[j, i]
    pk = W.band_power(sym, freq, ks, 'kelvin')
    assert pk > 0.8 * sym[1:].sum()
    assert W.band_power(sym, freq, ks, 'er') < 0.02 * pk
    assert anti.sum() < 0.05 * sym.sum()


def test_westward_er_and_mrg():
    T = 45
    # ER n=1, k = -4, h = 25 m
    f = round(float(W.disp_freq(-4, 25.0, 'er')) * T) / T
    sym, anti = W.wk_power(_wave(-4, f, T), taper=0.1)
    freq, ks = W.wk_axes(T)
    j, i = np.unravel_index(np.argmax(sym), sym.shape)
    assert ks[i] == -4 and W.band_mask(freq, ks, 'er')[j, i]
    # MRG: antisymmetric structure, k = -4, h = 25 m
    f = round(float(W.disp_freq(-4, 25.0, 'mrg')) * T) / T
    sym, anti = W.wk_power(_wave(-4, f, T, lat_shape='anti'), taper=0.1)
    j, i = np.unravel_index(np.argmax(anti), anti.shape)
    assert ks[i] == -4 and W.band_mask(freq, ks, 'mrg')[j, i]
    assert sym.sum() < 0.05 * anti.sum()


def test_filter_recovers_kelvin_and_rejects_er():
    T = 180
    fk = round(_kelvin_freq(6) * T) / T
    fe = round(float(W.disp_freq(-4, 25.0, 'er')) * T) / T
    a = _wave(6, fk, T, seed=1)
    b = _wave(-4, fe, T, seed=2)
    y = W.wk_filter(a + b, 'kelvin')
    clean = a - (a - _wave(6, fk, T, amp=1.0, seed=1)) - 0.0
    r = np.corrcoef(y[20:-20].ravel(), _wave(6, fk, T, seed=9)[20:-20].ravel())[0, 1]
    assert r > 0.95
    ye = W.wk_filter(a + b, 'er')
    r2 = np.corrcoef(ye[20:-20].ravel(), _wave(-4, fe, T, seed=9)[20:-20].ravel())[0, 1]
    assert r2 > 0.95
    assert np.abs(W.wk_filter(a, 'er')).max() < 0.1


def test_mjo_band_and_dispersion_values():
    # Kelvin h = 25 m: c = 15.7 m/s -> k = 1 period ~ 29.5 d
    assert abs(1 / _kelvin_freq(1) - 29.5) < 0.5
    T = 45
    f = 1 / 45
    sym, _ = W.wk_power(_wave(2, f, T), taper=0.1)
    freq, ks = W.wk_axes(T)
    assert W.band_power(sym, freq, ks, 'mjo') > 0.8 * sym[1:].sum()
