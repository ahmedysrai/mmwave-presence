import numpy as np
import pytest

from mmpresence.fmcw import range_resolution_m
from mmpresence.rangeproc import motion_energy, range_axis_m, range_fft
from mmpresence.sim import ChirpConfig, breathing_range_m, if_frames


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_on_bin_tone_has_its_amplitude():
    cfg = ChirpConfig()
    r = 50 * range_resolution_m(cfg.bandwidth_hz)
    x = if_frames(cfg, [[r]], 0.3, 0.0, np.random.default_rng(0))
    X = range_fft(x)
    assert np.argmax(np.abs(X[0])) == 50
    assert abs(X[0, 50]) == pytest.approx(0.3, rel=1e-9)
    assert range_axis_m(cfg)[50] == pytest.approx(r)


def test_static_target_leaves_no_motion_energy():
    cfg = ChirpConfig()
    x = if_frames(cfg, np.full((1, 100), 2.3), 1.0, 0.0, np.random.default_rng(0))
    assert motion_energy(range_fft(x), cfg) < 1e-20


def test_breathing_target_shows_motion_energy_at_its_range(rng):
    cfg = ChirpConfig()
    t = np.arange(400) * cfg.frame_period_s
    r = breathing_range_m(t, 2.5, 0.25, 0.008)
    X = range_fft(if_frames(cfg, r[None, :], 1.0, 0.0, rng))
    p_moving = np.mean(np.abs(X - X.mean(axis=0)) ** 2, axis=0)
    bin_m = range_resolution_m(cfg.bandwidth_hz)
    assert abs(range_axis_m(cfg)[np.argmax(p_moving)] - 2.5) < bin_m


def test_moving_fraction_follows_bessel_j0():
    # Sinusoidal phase beta sin(wt) leaves J0(beta)^2 of the power static.
    # The range bin's phase follows the chirp's mean frequency, not fc:
    # 60 GHz + 0.5 * 255 samples * 12 MHz/sample = 61.53 GHz. For 1 mm
    # peak-to-peak, beta = 4 pi 0.5 mm * 61.53 GHz / c = 1.2896 rad and
    # J0(beta) = 0.6255 (with fc alone it would be 0.6421, which fails).
    cfg = ChirpConfig()
    t = np.arange(400) * cfg.frame_period_s  # 5 whole breaths at 0.25 Hz
    r = breathing_range_m(t, 50 * range_resolution_m(cfg.bandwidth_hz), 0.25, 1e-3)
    X = range_fft(if_frames(cfg, r[None, :], 1.0, 0.0, np.random.default_rng(0)))
    fraction = motion_energy(X, cfg) / np.mean(np.abs(X[:, 50]) ** 2)
    assert fraction == pytest.approx(1 - 0.6255**2, abs=2e-4)
