"""Low-band excess against the motion energy threshold, same protocol.

Same scenes, seeds and wall as energy_threshold.py. Each window gives two
features: the motion energy (the baseline, rangeproc.motion_energy) and the
low-band excess (doppler.low_band_excess, max over range bins beyond
0.5 m). Each feature gets its own threshold, the largest empty-room value
on the validation seed, and the counts over it on the test seeds are
printed side by side. A pet or fan over the threshold is a false alarm.

    python experiments/low_band.py
    python experiments/low_band.py --cut 3

Both features come from the same simulated window, so the energy column is
the energy_threshold.py table.
"""

import argparse

import numpy as np
from energy_threshold import FAN_RCS_M2, TEST_SEED, WALL_M, WALL_RCS_M2, scene_targets

from mmpresence.doppler import doppler_axis_hz, doppler_fft, low_band_excess
from mmpresence.rangeproc import motion_energy, range_axis_m, range_fft, remove_static
from mmpresence.sim import ChirpConfig, if_frames, two_way_amplitude


def features(cfg, kind, noise_std, rng, cut_hz, n_frames=400, r_min_m=0.5):
    t = np.arange(n_frames) * cfg.frame_period_s
    r, a = scene_targets(cfg, kind, t, rng, FAN_RCS_M2)
    ranges = np.vstack([np.full((1, n_frames), WALL_M), r])
    amps = np.vstack(
        [np.full((1, n_frames), two_way_amplitude(WALL_M, WALL_RCS_M2)), a]
    )
    profiles = range_fft(if_frames(cfg, ranges, amps, noise_std, rng))
    p = np.abs(doppler_fft(remove_static(profiles))) ** 2
    excess = low_band_excess(p, doppler_axis_hz(n_frames, cfg.frame_period_s), cut_hz)
    return motion_energy(profiles, cfg, r_min_m), excess[
        range_axis_m(cfg) >= r_min_m
    ].max()


def run(cfg, kind, noise_std, seed, n, cut_hz):
    rng = np.random.default_rng(seed)
    return np.array([features(cfg, kind, noise_std, rng, cut_hz) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.1, 1.0, 3.0, 10.0])
    ap.add_argument("--cut", type=float, default=4.0, help="Hz")
    args = ap.parse_args()
    cfg = ChirpConfig()
    n = args.trials
    head = "  ".join(f"{k:>7}" for k in TEST_SEED)
    print(f"noise_std  energy: {head}   low band: {head}")
    for noise_std in args.noise:
        thr = run(cfg, "empty", noise_std, 1, n, args.cut).max(axis=0)
        over = np.array(
            [
                (run(cfg, k, noise_std, seed, n, args.cut) > thr).sum(axis=0)
                for k, seed in TEST_SEED.items()
            ]
        )
        cols = ["  ".join(f"{c:3d}/{n:<3d}" for c in over[:, j]) for j in (0, 1)]
        print(f"{noise_std:9.2f}          {cols[0]}             {cols[1]}")


if __name__ == "__main__":
    main()
