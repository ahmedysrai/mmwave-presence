"""Where in Doppler each scene puts its motion energy.

Same scenes, seeds and wall as energy_threshold.py. For each 20 s window
(400 chirps at 20 Hz) the static part is removed, the Doppler FFT runs over
all 400 chirps, and the printed number is the share of the motion energy in
the range bin that has the most of it (beyond 0.5 m, the bin the energy
threshold looks at) that lies at |f_d| above each cut. Median and 5th to
95th percentile over the trials.

    python experiments/doppler_spread.py
    python experiments/doppler_spread.py --noise 1.0

A spectrum that is flat over the +-10 Hz span gives (10 - cut) / 10.
"""

import argparse

import numpy as np
from energy_threshold import (
    FAN_RCS_M2,
    TEST_SEED,
    WALL_M,
    WALL_RCS_M2,
    scene_targets,
)

from mmpresence.doppler import doppler_axis_hz, doppler_fft
from mmpresence.rangeproc import range_axis_m, range_fft, remove_static
from mmpresence.sim import ChirpConfig, if_frames, two_way_amplitude

CUTS_HZ = (2.0, 4.0, 6.0, 8.0)


def high_doppler_shares(cfg, kind, noise_std, rng, n_frames=400, r_min_m=0.5):
    t = np.arange(n_frames) * cfg.frame_period_s
    r, a = scene_targets(cfg, kind, t, rng, FAN_RCS_M2)
    ranges = np.vstack([np.full((1, n_frames), WALL_M), r])
    amps = np.vstack(
        [np.full((1, n_frames), two_way_amplitude(WALL_M, WALL_RCS_M2)), a]
    )
    x = if_frames(cfg, ranges, amps, noise_std, rng)
    p = np.abs(doppler_fft(remove_static(range_fft(x)))) ** 2
    p = p[:, range_axis_m(cfg) >= r_min_m]
    p = p[:, np.argmax(p.sum(axis=0))]
    f_hz = np.abs(doppler_axis_hz(n_frames, cfg.frame_period_s))
    return [p[f_hz > cut].sum() / p.sum() for cut in CUTS_HZ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--noise", type=float, default=0.0)
    args = ap.parse_args()
    cfg = ChirpConfig()
    kinds = [k for k in TEST_SEED if k != "empty" or args.noise > 0]
    print("scene  " + "".join(f"{f'|f_d| > {c:.0f} Hz':>20}" for c in CUTS_HZ))
    for kind in kinds:
        rng = np.random.default_rng(TEST_SEED[kind])
        s = np.array(
            [
                high_doppler_shares(cfg, kind, args.noise, rng)
                for _ in range(args.trials)
            ]
        )
        lo, med, hi = np.percentile(s, [5, 50, 95], axis=0)
        print(
            f"{kind:6s} "
            + "".join(f"{m:6.2f} [{a:.2f}, {b:.2f}]" for m, a, b in zip(med, lo, hi))
        )


if __name__ == "__main__":
    main()
