import numpy as np
import pytest

from mmpresence.fmcw import range_resolution_m
from mmpresence.sim import ChirpConfig, breathing_range_m, if_frames


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_default_config_fits_iwr6843_band():
    cfg = ChirpConfig()
    assert 60e9 <= cfg.fc_hz and cfg.fc_hz + cfg.bandwidth_hz <= 64e9
    assert cfg.max_range_m == pytest.approx(12.49, abs=0.01)
    assert range_resolution_m(cfg.bandwidth_hz) == pytest.approx(0.0488, abs=1e-4)


def test_static_target_beat_lands_in_its_range_bin(rng):
    cfg = ChirpConfig()
    bin_m = range_resolution_m(cfg.bandwidth_hz)
    r = 40 * bin_m  # exactly on bin 40, so no leakage into neighbours
    x = if_frames(cfg, [[r]], 1.0, 0.0, rng)
    spec = np.abs(np.fft.fft(x[0]))
    assert np.argmax(spec) == 40
    assert spec[40] == pytest.approx(cfg.n_samples)


def test_chirp_to_chirp_phase_follows_displacement(rng):
    # 0.5 mm of motion is 4 pi 0.5e-3 / 5 mm = 1.26 rad at 60 GHz
    cfg = ChirpConfig()
    r = np.array([[2.0, 2.0005]])
    x = if_frames(cfg, r, 1.0, 0.0, rng)
    dphi = np.angle(x[1, 0] / x[0, 0])
    assert dphi == pytest.approx(4 * np.pi * 0.5e-3 / cfg.wavelength_m, rel=1e-6)


def test_noise_power_matches_noise_std(rng):
    cfg = ChirpConfig()
    x = if_frames(cfg, np.full((1, 200), 3.0), 0.0, 0.2, rng)
    # 51200 complex samples: std error of the mean power is about 0.5 %
    assert np.mean(np.abs(x) ** 2) == pytest.approx(0.04, rel=0.03)


def test_breathing_range_swings_by_depth():
    t = np.arange(0, 8, 0.05)  # two whole breaths, so the sine averages out
    r = breathing_range_m(t, 2.0, 0.25, 0.008)
    assert r.max() - r.min() == pytest.approx(0.008, rel=1e-3)
    assert r.mean() == pytest.approx(2.0, abs=1e-6)
