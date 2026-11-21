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


def _fan_point_ranges_m(t_s, hub_range_m, hub_height_m, rho_m, rpm, n_blades, theta0):
    """Ranges (n_blades, len(rho_m), len(t_s)) to points rho_m out along each blade."""
    w_rad_s = 2 * np.pi * rpm / 60
    blades = theta0 + 2 * np.pi * np.arange(n_blades) / n_blades
    theta = blades[:, None, None] + w_rad_s * np.asarray(t_s)
    rho = np.asarray(rho_m)[None, :, None]
    x = hub_range_m + rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.sqrt(x**2 + y**2 + hub_height_m**2)


def fan_ranges_m(
    t_s, hub_range_m, hub_height_m, radius_m, rpm, n_blades=3, n_points=4, rng=None
):
    """Range to points along the blades of a ceiling fan.

    The blades turn in a horizontal plane hub_height_m above the radar, with
    the hub hub_range_m away horizontally. Each blade is n_points scatterers
    spread from a quarter of radius_m out to the tip. Returns
    (n_blades * n_points, len(t_s)) ranges in metres, blade by blade.

    Ignored: blade pitch, the antenna's elevation pattern, and the fact that
    a flat blade reflects mostly when it faces the radar (fan_echo has
    that). The motor housing is static and belongs in the scene as its own
    target if wanted.
    """
    theta0 = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    rho = np.linspace(0.25, 1.0, n_points) * radius_m
    r = _fan_point_ranges_m(t_s, hub_range_m, hub_height_m, rho, rpm, n_blades, theta0)
    return r.reshape(n_blades * n_points, -1)


def fan_echo(
    t_s,
    hub_range_m,
    hub_height_m,
    radius_m,
    rpm,
    wavelength_m,
    n_blades=3,
    n_segments=4,
    rng=None,
):
    """Blade segments of a ceiling fan with an aspect-dependent return.

    Same geometry as fan_ranges_m, but each blade from a quarter of radius_m
    to the tip is cut into n_segments straight segments, and each segment
    is the coherent sum of points spaced wavelength_m / 8 along it. The sum
    is near 1 when the segment lies across the line of sight and falls off
    like a sinc as it turns away (a uniform line target), so the return
    comes in short flashes. The sum is exact in range, so the near-field
    blur of a segment a few metres away is included.

    Returns (ranges_m, gain): segment centre ranges in metres and complex
    gains (unit magnitude at broadside in the far field), both
    (n_blades * n_segments, len(t_s)). Scale gain by the segment's peak
    amplitude and use the centre range in if_frames.

    The gain is worked out at one wavelength; pass the one at the chirp's
    mean frequency, which is what the slow-time phase follows. Over a 3 GHz
    sweep the pattern narrows by 5 % from start to end, which is ignored, as
    are blade width, pitch and edge diffraction.
    """
    theta0 = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    r_in = 0.25 * radius_m
    seg_m = (radius_m - r_in) / n_segments
    n_sub = int(np.ceil(8 * seg_m / wavelength_m))
    u = (np.arange(n_sub) + 0.5) / n_sub - 0.5  # position along a segment, -1/2..1/2
    centres = r_in + (np.arange(n_segments) + 0.5) * seg_m
    rho = (centres[:, None] + u * seg_m).ravel()
    r_sub = _fan_point_ranges_m(
        t_s, hub_range_m, hub_height_m, rho, rpm, n_blades, theta0
    ).reshape(n_blades, n_segments, n_sub, -1)
    r_c = _fan_point_ranges_m(
        t_s, hub_range_m, hub_height_m, centres, rpm, n_blades, theta0
    )
    gain = np.exp(4j * np.pi * (r_sub - r_c[:, :, None]) / wavelength_m).mean(axis=2)
    return r_c.reshape(n_blades * n_segments, -1), gain.reshape(
        n_blades * n_segments, -1
    )
