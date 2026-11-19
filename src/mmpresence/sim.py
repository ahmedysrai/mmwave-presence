"""IF signal of an FMCW radar looking at point targets.

Ideal dechirped model, one receive channel, complex baseband samples:

    s(t) = sum_k a_k exp(j (2 pi f_b,k t + 4 pi fc R_k / c))

with f_b,k = 2 S R_k / c. The range R_k of each target may change from one
chirp to the next (breathing, a fan blade), but is held constant inside a
chirp, which is fine for the mm/s speeds of chest motion over a ~50 us chirp.
Ignored: phase noise, the residual video phase term and range migration of
the beat frequency within a chirp.
"""

from dataclasses import dataclass

import numpy as np

from mmpresence.fmcw import C_MPS


@dataclass(frozen=True)
class ChirpConfig:
    """One chirp per frame, sampled as complex baseband.

    Defaults are an assumed configuration inside the TI IWR6843 limits
    (SWRS219F, device comparison table: 60 to 64 GHz RF, 10 MHz max IF,
    12.5 Msps max complex sampling). 256 samples at 5 Msps with a
    60 MHz/us slope sweeps 3.07 GHz, so range bins are 4.9 cm and the
    unambiguous range is 12.5 m (beat 5 MHz, under the 10 MHz IF limit).
    A 20 Hz frame rate is well above twice a 0.5 Hz breathing rate.
    """

    fc_hz: float = 60e9
    slope_hz_per_s: float = 60e12
    fs_hz: float = 5e6
    n_samples: int = 256
    frame_period_s: float = 0.05

    @property
    def bandwidth_hz(self):
        return self.slope_hz_per_s * self.n_samples / self.fs_hz

    @property
    def wavelength_m(self):
        return C_MPS / self.fc_hz

    @property
    def max_range_m(self):
        return self.fs_hz * C_MPS / (2 * self.slope_hz_per_s)


def two_way_amplitude(range_m, rcs_m2, wall_loss_db=0.0):
    """Relative echo amplitude: sqrt(RCS) / R^2, less the wall loss both ways.

    Unitless; only ratios between targets and against the noise level mean
    anything. wall_loss_db is the one-way power loss through the wall.
    """
    return np.sqrt(rcs_m2) / np.asarray(range_m) ** 2 * 10 ** (-wall_loss_db / 10)


def if_frames(cfg, ranges_m, amplitudes, noise_std, rng):
    """Simulate a stack of chirps.

    ranges_m and amplitudes: (n_targets, n_frames) or broadcastable to it,
    in metres and linear amplitude. Returns complex (n_frames, n_samples).
    noise_std is the std of the complex white noise per sample (split
    equally between I and Q).
    """
    r = np.atleast_2d(ranges_m)
    a = np.broadcast_to(np.atleast_2d(amplitudes), r.shape)
    t = np.arange(cfg.n_samples) / cfg.fs_hz
    f_b = 2 * cfg.slope_hz_per_s * r / C_MPS
    phase0 = 4 * np.pi * cfg.fc_hz * r / C_MPS
    arg = 2 * np.pi * f_b[..., None] * t + phase0[..., None]
    x = (a[..., None] * np.exp(1j * arg)).sum(axis=0)
    noise = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    return x + noise_std / np.sqrt(2) * noise


def breathing_range_m(t_s, range_m, rate_hz, depth_m, rng=None):
    """Chest range over time for a sinusoidal breathing pattern.

    depth_m is the peak-to-peak chest displacement. Real breathing is not a
    pure sine and drifts in rate; that is left for a later model.
    """
    phase = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    return range_m + depth_m / 2 * np.sin(2 * np.pi * rate_hz * t_s + phase)
