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
