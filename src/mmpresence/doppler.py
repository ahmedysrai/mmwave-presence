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
