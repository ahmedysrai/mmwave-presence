import numpy as np
import pytest

from mmpresence.doppler import (
    doppler_axis_hz,
    doppler_fft,
    low_band_excess,
    low_band_score,
    velocity_axis_mps,
)
from mmpresence.fmcw import range_resolution_m
from mmpresence.rangeproc import range_fft, remove_static
from mmpresence.sim import ChirpConfig, if_frames


def test_slow_time_tone_lands_in_its_doppler_bin():
    n = 128
    f_hz = doppler_axis_hz(n, 0.05)
    k = np.flatnonzero(np.isclose(f_hz, 1.25))[0]  # 8 bins above zero
    x = 0.4 * np.exp(2j * np.pi * 1.25 * np.arange(n) * 0.05)
    X = doppler_fft(x[:, None])[:, 0]
    assert np.argmax(np.abs(X)) == k
    assert abs(X[k]) == pytest.approx(0.4, rel=1e-9)


def test_target_moving_away_has_positive_velocity():
    # 10 mm/s over 128 frames at 20 Hz moves 64 mm, a bit over one range
    # bin, so the target smears over two bins; take the stronger one.
    cfg = ChirpConfig()
    n = 128
    t = np.arange(n) * cfg.frame_period_s
    r = 50 * range_resolution_m(cfg.bandwidth_hz) + 0.010 * t
    X = doppler_fft(
        range_fft(if_frames(cfg, r[None, :], 1.0, 0.0, np.random.default_rng(0)))
    )
    k_dop, _ = np.unravel_index(np.argmax(np.abs(X)), X.shape)
    v = velocity_axis_mps(cfg, n)
    assert v[k_dop] == pytest.approx(0.010, abs=v[1] - v[0])


def test_static_removal_empties_zero_doppler():
    cfg = ChirpConfig()
    wall = if_frames(cfg, np.full((1, 64), 0.3), 5.0, 0.0, np.random.default_rng(0))
    X = doppler_fft(remove_static(range_fft(wall)))
    assert np.abs(X).max() < 1e-12


def test_unambiguous_span_is_plus_minus_half_the_frame_rate():
    cfg = ChirpConfig()
    v = velocity_axis_mps(cfg, 400)
    assert v[0] == pytest.approx(-0.0244, abs=1e-4)
    assert v[200] == 0.0


def test_low_band_excess_keeps_a_slow_tone():
    # By Parseval a windowed tone puts a^2 n sum(w^2) / sum(w)^2 over all
    # Doppler bins, about 1.5 a^2 for Hann, and here all of it is below 4 Hz.
    n = 400
    f_hz = doppler_axis_hz(n, 0.05)
    x = 0.4 * np.exp(2j * np.pi * 0.25 * np.arange(n) * 0.05)
    p = np.abs(doppler_fft(x[:, None])) ** 2
    w = np.hanning(n)
    expected = 0.16 * n * (w**2).sum() / w.sum() ** 2
    assert low_band_excess(p, f_hz)[0] == pytest.approx(expected, rel=1e-6)


def test_low_band_excess_counts_fast_tones_against_on_both_sides():
    n = 400
    f_hz = doppler_axis_hz(n, 0.05)
    t = np.arange(n) * 0.05
    for fd in (6.0, -6.0):
        x = np.exp(2j * np.pi * fd * t)
        p = np.abs(doppler_fft(x[:, None])) ** 2
        assert low_band_excess(p, f_hz)[0] < 0


def test_low_band_excess_of_white_noise_is_near_zero():
    rng = np.random.default_rng(0)
    n, n_bins = 400, 2000
    x = rng.normal(size=(n, n_bins)) + 1j * rng.normal(size=(n, n_bins))
    p = np.abs(doppler_fft(x)) ** 2
    e = low_band_excess(p, doppler_axis_hz(n, 0.05))
    low_power = p[np.abs(doppler_axis_hz(n, 0.05)) < 4.0].sum(axis=0)
    # mean over bins goes to zero; each bin scatters around it
    assert abs(e.mean()) < 0.02 * low_power.mean()


def test_low_band_score_does_not_change_with_power_scale():
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=(n, 50)) + 1j * rng.normal(size=(n, 50))
    p = np.abs(doppler_fft(x)) ** 2
    f_hz = doppler_axis_hz(n, 0.05)
    np.testing.assert_allclose(
        low_band_score(100.0 * p, f_hz), low_band_score(p, f_hz), rtol=1e-9, atol=1e-12
    )


def test_low_band_score_of_white_noise_spreads_by_about_1_4():
    # Hann bins k and k+1 have power correlation 4/9, k and k+2 1/36, so a
    # sum over many bins has 1 + 8/9 + 1/18 = 1.94 times the independent
    # variance: std about 1.39 rather than 1.
    rng = np.random.default_rng(1)
    n, n_bins = 400, 4000
    x = rng.normal(size=(n, n_bins)) + 1j * rng.normal(size=(n, n_bins))
    z = low_band_score(np.abs(doppler_fft(x)) ** 2, doppler_axis_hz(n, 0.05))
    assert abs(z.mean()) < 0.1
    assert z.std() == pytest.approx(1.39, abs=0.08)


def test_low_band_score_picks_out_a_slow_tone_in_noise():
    rng = np.random.default_rng(2)
    n = 400
    t = np.arange(n) * 0.05
    noise = rng.normal(size=n) + 1j * rng.normal(size=n)
    x = np.exp(2j * np.pi * 0.25 * t) + noise
    f_hz = doppler_axis_hz(n, 0.05)
    assert low_band_score(np.abs(doppler_fft(x[:, None])) ** 2, f_hz)[0] > 5
