import numpy as np
import pytest

from mmpresence.fmcw import range_resolution_m
from mmpresence.sim import (
    ChirpConfig,
    breathing_range_m,
    fan_echo,
    fan_ranges_m,
    if_frames,
)


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


def test_fan_tip_range_spans_near_and_far_side_of_hub():
    # one revolution at 120 rpm is 0.5 s; sample it finely
    t = np.linspace(0, 0.5, 2001)
    r = fan_ranges_m(t, 3.0, 1.4, 0.66, 120, n_blades=1, n_points=4)
    tip = r[-1]
    assert tip.min() == pytest.approx(np.hypot(3.0 - 0.66, 1.4), abs=1e-5)
    assert tip.max() == pytest.approx(np.hypot(3.0 + 0.66, 1.4), abs=1e-5)


def test_fan_looks_the_same_after_one_blade_spacing():
    # three blades: the set of scatterer ranges repeats every third of a turn
    rpm = 150
    dt = 60 / rpm / 3
    t = np.array([0.0137, 0.0137 + dt])
    r = fan_ranges_m(t, 2.5, 1.2, 0.6, rpm, rng=np.random.default_rng(3))
    np.testing.assert_allclose(np.sort(r[:, 0]), np.sort(r[:, 1]), atol=1e-12)
    assert not np.allclose(r[:, 0], r[:, 1])  # but each point has moved on


def test_fan_segment_flashes_across_line_of_sight():
    # far field, one blade, one segment, theta0 = 0, 60 rpm (theta = 2 pi t).
    # At t = 0 the blade points along the line of sight. It lies across it
    # when the blade is perpendicular to the hub-to-centre vector seen from
    # the radar: cos(theta) = -rho_c / D, a little past a quarter turn.
    lam, hub_m, radius = 5e-3, 2000.0, 0.66  # 2 L^2 / lam is 98 m
    rho_c = 0.625 * radius
    t_across = np.arccos(-rho_c / hub_m) / (2 * np.pi)
    _, g = fan_echo([0.0, t_across], hub_m, 0.0, radius, 60, lam, 1, n_segments=1)
    assert abs(g[0, 1]) == pytest.approx(1.0, abs=1e-3)
    assert abs(g[0, 0]) < 0.01


def test_fan_segment_mean_power_over_a_turn():
    # far field |g|^2 is sinc^2 around each of the two broadside crossings,
    # each worth lambda / (2 L) rad, so the mean over a turn is lambda / (2 pi L)
    lam, radius = 5e-3, 0.66
    t = np.linspace(0, 1, 20_000, endpoint=False)
    _, g = fan_echo(t, 200.0, 0.0, radius, 60, lam, n_blades=1, n_segments=2)
    seg_m = 0.75 * radius / 2
    np.testing.assert_allclose(
        np.mean(np.abs(g) ** 2, axis=1), lam / (2 * np.pi * seg_m), rtol=0.02
    )
