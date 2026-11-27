"""Doppler processing over slow time.

With one chirp per frame, slow time is the frame index and the Doppler FFT
runs across frames in each range bin. At the default 20 Hz frame rate the
unambiguous Doppler span is -10 to +10 Hz, or +-24 mm/s of range rate.
That covers chest motion; anything faster, such as a fan blade, aliases.
"""

import numpy as np

from mmpresence.fmcw import C_MPS


def doppler_fft(profiles):
    """Hann-windowed FFT over slow time, zero Doppler in the middle.

    profiles: complex (n_frames, n_bins) range profiles. Returns complex
    (n_frames, n_bins), Doppler along axis 0 in the order of doppler_axis_hz.
    Scaled by the window sum, so a slow-time tone of amplitude a centred on
    a Doppler bin gives |X| = a. Pass profiles after remove_static if the
    wall is not wanted in the zero-Doppler row.
    """
    w = np.hanning(profiles.shape[0])
    X = np.fft.fft(profiles * w[:, None], axis=0) / w.sum()
    return np.fft.fftshift(X, axes=0)


def doppler_axis_hz(n_frames, frame_period_s):
    return np.fft.fftshift(np.fft.fftfreq(n_frames, frame_period_s))


def velocity_axis_mps(cfg, n_frames):
    """Range rate for each Doppler bin, positive moving away from the radar.

    The sign follows if_frames, whose phase grows with range. The wavelength
    is the one at the chirp's mean frequency, which the slow-time phase
    follows (NOTES.md, 2026-11-19).
    """
    lam_m = C_MPS / (cfg.fc_hz + cfg.bandwidth_hz / 2)
    return doppler_axis_hz(n_frames, cfg.frame_period_s) * lam_m / 2


def low_band_excess(power, f_hz, cut_hz=4.0):
    """Power below cut_hz in Doppler, less the flat floor seen above it.

    power: (n_frames, n_bins) |X|^2 from doppler_fft, f_hz its Doppler axis.
    The bins at |f_d| >= cut_hz estimate a floor that is flat over the whole
    span (white noise, and a fan whose flashes alias everywhere, NOTES.md
    2026-11-22). That floor is taken off every bin below cut_hz, so noise
    and a fan come out near zero and breathing, which stays below about
    4 Hz, comes out positive. Returns (n_bins,).
    """
    low = np.abs(f_hz) < cut_hz
    floor = power[~low].mean(axis=0)
    return power[low].sum(axis=0) - low.sum() * floor


def low_band_score(power, f_hz, cut_hz=4.0, pool_bins=0):
    """low_band_excess divided by how much it scatters when there is no target.

    With only a flat floor in a range bin, each bin of power scatters with
    some spread s, which the bins at |f_d| >= cut_hz estimate. The low-band
    sum of n_low bins then has variance n_low s^2 and the floor estimate,
    scaled up to n_low bins, adds n_low^2 s^2 / n_high. The excess is
    divided by the square root of their sum. The score has no units, so a
    strong fan whose excess is a large number around zero scores the same
    as a weak one (NOTES.md 2026-11-24).

    The Hann window correlates neighbouring Doppler bins, which this
    ignores, so for white noise the score spreads a little wider than a
    unit normal. Thresholds come from the empty room, so that only matters
    when reading a score as a z value.

    With pool_bins = k > 0, s in each range bin is raised to the median s
    of the 2k + 1 range bins around it (edge bins repeated at the ends),
    never lowered. Taking the max over range otherwise picks bins whose s
    came out low by chance; a bin with a strong flat-spectrum return, like
    a fan, keeps its own larger s (NOTES.md 2026-11-27). Returns (n_bins,).
    """
    low = np.abs(f_hz) < cut_hz
    n_low, n_high = low.sum(), (~low).sum()
    s = power[~low].std(axis=0, ddof=1)
    if pool_bins:
        padded = np.pad(s, pool_bins, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * pool_bins + 1)
        s = np.maximum(s, np.median(windows, axis=-1))
    return low_band_excess(power, f_hz, cut_hz) / (
        s * np.sqrt(n_low * (1 + n_low / n_high))
    )
