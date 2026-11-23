import numpy as np
import pytest

from mmpresence.doppler import (
    doppler_axis_hz,
    doppler_fft,
    low_band_excess,
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
