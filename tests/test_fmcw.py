import pytest

from mmpresence.fmcw import beat_to_range_m, range_resolution_m


def test_resolution_for_4ghz_sweep():
    # 60-64 GHz sweep gives about 3.75 cm
    assert range_resolution_m(4e9) == pytest.approx(0.0375, rel=1e-3)


def test_beat_to_range():
    # 100 MHz/us slope, target at 3 m -> beat of 2 R S / c
    slope = 100e6 / 1e-6
    beat = 2 * 3.0 * slope / 299_792_458.0
    assert beat_to_range_m(beat, slope) == pytest.approx(3.0)


def test_zero_bandwidth():
    with pytest.raises(ValueError):
        range_resolution_m(0)
