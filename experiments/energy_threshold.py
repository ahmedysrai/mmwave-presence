"""Crude end-to-end check: motion energy threshold, person vs empty room.

Each trial is one 20 s window (400 chirps at 20 Hz) seen through a wall.
The threshold is the largest empty-room energy on the validation seed, and
detection and false alarms are counted on a separate test seed.

    python experiments/energy_threshold.py

Assumed scene, no measurement behind any of it: wall return at 0.3 m with
RCS 10 m^2, 3 dB one-way wall loss, person chest RCS 0.5 m^2 at 1.5 to
5 m, breathing 0.2 to 0.33 Hz with 4 to 10 mm peak-to-peak chest motion.
"""

import argparse

import numpy as np

from mmpresence.rangeproc import motion_energy, range_fft
from mmpresence.sim import ChirpConfig, breathing_range_m, if_frames, two_way_amplitude

WALL_M = 0.3
WALL_RCS_M2 = 10.0
WALL_LOSS_DB = 3.0
CHEST_RCS_M2 = 0.5


def trial(cfg, person, noise_std, rng, n_frames=400):
    t = np.arange(n_frames) * cfg.frame_period_s
    ranges = [np.full(n_frames, WALL_M)]
    amps = [two_way_amplitude(WALL_M, WALL_RCS_M2)]
    if person:
        r0 = rng.uniform(1.5, 5.0)
        ranges.append(
            breathing_range_m(
                t, r0, rng.uniform(0.2, 0.33), rng.uniform(4e-3, 10e-3), rng
            )
        )
        amps.append(two_way_amplitude(r0, CHEST_RCS_M2, WALL_LOSS_DB))
    x = if_frames(cfg, np.array(ranges), np.array(amps)[:, None], noise_std, rng)
    return motion_energy(range_fft(x), cfg)


def energies(cfg, person, noise_std, seed, n):
    rng = np.random.default_rng(seed)
    return np.array([trial(cfg, person, noise_std, rng) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.1, 1.0, 3.0, 10.0])
    args = ap.parse_args()
    cfg = ChirpConfig()
    print("noise_std  threshold  detected  false_alarms")
    for noise_std in args.noise:
        thr = energies(cfg, False, noise_std, 1, args.trials).max()
        hits = (energies(cfg, True, noise_std, 2, args.trials) > thr).sum()
        fas = (energies(cfg, False, noise_std, 3, args.trials) > thr).sum()
        print(
            f"{noise_std:9.2f}  {thr:9.2e}  {hits:4d}/{args.trials}"
            f"  {fas:4d}/{args.trials}"
        )


if __name__ == "__main__":
    main()
