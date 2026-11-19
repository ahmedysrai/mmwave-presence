"""Range processing over a stack of chirps."""

import numpy as np

from mmpresence.fmcw import range_resolution_m


def range_fft(frames):
    """Hann-windowed range FFT of each chirp.

    frames: complex (n_frames, n_samples). Returns complex
    (n_frames, n_samples) range profiles. Complex sampling means every bin
    is a positive beat frequency, so all n_samples bins are kept. Scaled by
    the window sum, so a tone of amplitude a centred on a bin gives |X| = a.
    """
    w = np.hanning(frames.shape[-1])
    return np.fft.fft(frames * w, axis=-1) / w.sum()


def range_axis_m(cfg):
    return np.arange(cfg.n_samples) * range_resolution_m(cfg.bandwidth_hz)


def remove_static(profiles):
    """Subtract the mean over slow time from every range bin.

    Takes out the wall, furniture and anything else that does not move over
    the window. A sitting person is mostly static too: only the part of the
    chest return that changes with breathing survives.
    """
    return profiles - profiles.mean(axis=0, keepdims=True)


def motion_energy(profiles, cfg, r_min_m=0.5):
    """Largest slow-time power over range bins beyond r_min_m.

    The crude presence feature: mean |X|^2 over frames after static removal,
    maximised over range. Bins nearer than r_min_m are skipped to keep the
    wall and the antenna leakage out.
    """
    p = np.mean(np.abs(remove_static(profiles)) ** 2, axis=0)
    return p[range_axis_m(cfg) >= r_min_m].max()
