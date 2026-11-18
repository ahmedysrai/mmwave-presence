C_MPS = 299_792_458.0


def range_resolution_m(bandwidth_hz):
    """Range resolution of an FMCW chirp, c / (2 B)."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth must be positive")
    return C_MPS / (2.0 * bandwidth_hz)


def beat_to_range_m(beat_hz, slope_hz_per_s):
    """Target range from the beat frequency: R = c f_b / (2 S)."""
    return C_MPS * beat_hz / (2.0 * slope_hz_per_s)
