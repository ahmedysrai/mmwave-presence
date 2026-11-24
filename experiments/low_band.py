"""Low-band excess against the motion energy threshold, same protocol.

Same scenes, seeds and wall as energy_threshold.py. Each window gives two
features: the motion energy (the baseline, rangeproc.motion_energy) and the
low-band excess (doppler.low_band_excess, max over range bins beyond
0.5 m). Each feature gets its own threshold, the largest empty-room value
on the validation seed, and the counts over it on the test seeds are
printed side by side. A pet or fan over the threshold is a false alarm.

    python experiments/low_band.py
    python experiments/low_band.py --cut 3
    python experiments/low_band.py --noise 0.1 --fan-detail

--fan-detail also prints, for each noise level, the fans over the
low-band threshold by hub range and by speed.

Both features come from the same simulated window, so the energy column is
the energy_threshold.py table.
"""

import argparse
from itertools import pairwise

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


def fan_draws(seed, n, noise_std, cut_hz, cfg):
    """Hub range and speed of each fan trial in run(cfg, "fan", ...).

    scene_targets draws the hub range and then the speed first in every fan
    trial, so a copy of the generator taken before each trial gives them.
    """
    rng = np.random.default_rng(seed)
    hub_rpm = []
    for _ in range(n):
        peek = np.random.default_rng()
        peek.bit_generator.state = rng.bit_generator.state
        hub_rpm.append((peek.uniform(2.0, 4.0), peek.uniform(80, 200)))
        features(cfg, "fan", noise_std, rng, cut_hz)
    return np.array(hub_rpm)


def print_fan_detail(hub_rpm, over):
    for name, col, edges in [
        ("hub m", 0, np.arange(2.0, 4.01, 0.5)),
        ("rpm", 1, np.arange(80, 201, 20)),
    ]:
        idx = np.digitize(hub_rpm[:, col], edges[1:-1])
        cells = [
            f"{lo:g}-{hi:g}: {over[idx == i].sum()}/{(idx == i).sum()}"
            for i, (lo, hi) in enumerate(pairwise(edges))
        ]
        print(f"    fans over, by {name}:  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.1, 1.0, 3.0, 10.0])
    ap.add_argument("--cut", type=float, default=4.0, help="Hz")
    ap.add_argument("--fan-detail", action="store_true")
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
        if args.fan_detail:
            seed = TEST_SEED["fan"]
            fan = run(cfg, "fan", noise_std, seed, n, args.cut)
            hub_rpm = fan_draws(seed, n, noise_std, args.cut, cfg)
            print_fan_detail(hub_rpm, fan[:, 1] > thr[1])


if __name__ == "__main__":
    main()
