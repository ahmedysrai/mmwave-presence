"""Low-band excess and score against the motion energy threshold.

Same scenes, seeds and wall as energy_threshold.py. Each window gives three
features: the motion energy (the baseline, rangeproc.motion_energy), the
low-band excess (doppler.low_band_excess) and the low-band score
(doppler.low_band_score), the last two as the max over range bins beyond
0.5 m. Each feature gets its own threshold from the empty-room values on
the validation seed, and the counts over it on the test seeds are printed
side by side. A pet or fan over the threshold is a false alarm.

By default the threshold is the largest of --trials empty windows. With
--empty 1000 --quantile 0.99 it is the 99th percentile of 1000 empty
windows instead, which a single heavy-tail window cannot move much; the
empty column then reads about 1 in 100 by design.

--pool K passes pool_bins=K to the score, which takes the floor spread
of each range bin from the K bins either side where its own is lower.
The default 0 gives the score of 2026-11-25.

    python experiments/low_band.py
    python experiments/low_band.py --cut 3
    python experiments/low_band.py --noise 0.1 --fan-detail
    python experiments/low_band.py --empty 1000 --quantile 0.99
    python experiments/low_band.py --empty 1000 --quantile 0.99 --pool 8

--fan-detail also prints, for each noise level, the fans over the
excess and score thresholds by hub range and by speed.

All features come from the same simulated window, so the energy column is
the energy_threshold.py table.
"""

import argparse
from itertools import pairwise

import numpy as np
from energy_threshold import FAN_RCS_M2, TEST_SEED, WALL_M, WALL_RCS_M2, scene_targets

from mmpresence.doppler import (
    doppler_axis_hz,
    doppler_fft,
    low_band_excess,
    low_band_score,
)
from mmpresence.rangeproc import motion_energy, range_axis_m, range_fft, remove_static
from mmpresence.sim import ChirpConfig, if_frames, two_way_amplitude


def features(cfg, kind, noise_std, rng, cut_hz, pool_bins=0, n_frames=400, r_min_m=0.5):
    t = np.arange(n_frames) * cfg.frame_period_s
    r, a = scene_targets(cfg, kind, t, rng, FAN_RCS_M2)
    ranges = np.vstack([np.full((1, n_frames), WALL_M), r])
    amps = np.vstack(
        [np.full((1, n_frames), two_way_amplitude(WALL_M, WALL_RCS_M2)), a]
    )
    profiles = range_fft(if_frames(cfg, ranges, amps, noise_std, rng))
    p = np.abs(doppler_fft(remove_static(profiles))) ** 2
    f_hz = doppler_axis_hz(n_frames, cfg.frame_period_s)
    far = range_axis_m(cfg) >= r_min_m
    return (
        motion_energy(profiles, cfg, r_min_m),
        low_band_excess(p, f_hz, cut_hz)[far].max(),
        low_band_score(p, f_hz, cut_hz, pool_bins)[far].max(),
    )


def run(cfg, kind, noise_std, seed, n, cut_hz, pool_bins=0):
    rng = np.random.default_rng(seed)
    return np.array(
        [features(cfg, kind, noise_std, rng, cut_hz, pool_bins) for _ in range(n)]
    )


def fan_draws(seed, n, noise_std, cut_hz, cfg, pool_bins=0):
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
        features(cfg, "fan", noise_std, rng, cut_hz, pool_bins)
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
    ap.add_argument(
        "--empty", type=int, help="empty validation windows (default --trials)"
    )
    ap.add_argument(
        "--quantile", type=float, default=1.0, help="of the empty windows, 1 = max"
    )
    ap.add_argument(
        "--pool", type=int, default=0, help="score: range bins each side for s"
    )
    args = ap.parse_args()
    cfg = ChirpConfig()
    n = args.trials
    head = "  ".join(f"{k:>7}" for k in TEST_SEED)
    print(f"noise_std  energy: {head}   excess: {head}   score: {head}")
    for noise_std in args.noise:
        empty = run(cfg, "empty", noise_std, 1, args.empty or n, args.cut, args.pool)
        thr = np.quantile(empty, args.quantile, axis=0)
        over = np.array(
            [
                (run(cfg, k, noise_std, seed, n, args.cut, args.pool) > thr).sum(axis=0)
                for k, seed in TEST_SEED.items()
            ]
        )
        cols = ["  ".join(f"{c:3d}/{n:<3d}" for c in over[:, j]) for j in (0, 1, 2)]
        print(f"{noise_std:9.2f}          " + "           ".join(cols))
        if args.fan_detail:
            seed = TEST_SEED["fan"]
            fan = run(cfg, "fan", noise_std, seed, n, args.cut, args.pool)
            hub_rpm = fan_draws(seed, n, noise_std, args.cut, cfg, args.pool)
            for j, name in [(1, "excess"), (2, "score")]:
                print(f"  {name}")
                print_fan_detail(hub_rpm, fan[:, j] > thr[j])


if __name__ == "__main__":
    main()
