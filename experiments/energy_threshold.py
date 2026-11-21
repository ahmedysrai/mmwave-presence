"""Crude end-to-end check: motion energy threshold on four scenes.

Each trial is one 20 s window (400 chirps at 20 Hz) seen through a wall:
an empty room, a breathing person, a breathing pet, or a ceiling fan. The
threshold is the largest empty-room energy on the validation seed. The
other scenes are counted against it on a separate test seed; a pet or a
fan over the threshold is a false alarm.

    python experiments/energy_threshold.py

Assumed scene, no measurement behind any of it: wall return at 0.3 m with
RCS 10 m^2, 3 dB one-way wall loss. Person: chest RCS 0.5 m^2 at 1.5 to
5 m, breathing 0.2 to 0.33 Hz with 4 to 10 mm peak-to-peak chest motion.
Pet (cat or small dog): RCS 0.05 m^2 at 1.5 to 5 m, breathing 0.33 to
0.67 Hz, 1 to 3 mm peak-to-peak. Fan: 3 blades of 0.66 m radius (a 52 in
fan) at 80 to 200 rpm, hub 2 to 4 m away and 1.4 m above the radar, each
blade cut into 4 segments that flash as they cross the line of sight
(sim.fan_echo). --fan-rcs is a segment's RCS averaged over a revolution,
0.02 m^2 by default; the peak is that times 2 pi L / lambda, about 160x.
--isotropic-fan puts back the 2026-11-20 model: 4 point scatterers per
blade that return the same at every angle, each with --fan-rcs.
"""

import argparse

import numpy as np

from mmpresence.fmcw import C_MPS
from mmpresence.rangeproc import motion_energy, range_fft
from mmpresence.sim import (
    ChirpConfig,
    breathing_range_m,
    fan_echo,
    fan_ranges_m,
    if_frames,
    two_way_amplitude,
)

WALL_M = 0.3
WALL_RCS_M2 = 10.0
WALL_LOSS_DB = 3.0
CHEST_RCS_M2 = 0.5
PET_RCS_M2 = 0.05
FAN_RCS_M2 = 0.02
# test seed per scene; person and empty keep the seeds of the 2026-11-19 run
TEST_SEED = {"empty": 3, "person": 2, "pet": 4, "fan": 5}


def scene_targets(cfg, kind, t, rng, fan_rcs_m2, isotropic_fan=False):
    """Ranges and amplitudes, (n_targets, n_frames), of the moving targets."""
    if kind == "empty":
        return np.empty((0, len(t))), np.empty((0, len(t)))
    if kind == "person":
        r0 = rng.uniform(1.5, 5.0)
        r = breathing_range_m(
            t, r0, rng.uniform(0.2, 0.33), rng.uniform(4e-3, 10e-3), rng
        )
        return r[None, :], two_way_amplitude(r[None, :], CHEST_RCS_M2, WALL_LOSS_DB)
    if kind == "pet":
        r0 = rng.uniform(1.5, 5.0)
        r = breathing_range_m(
            t, r0, rng.uniform(0.33, 0.67), rng.uniform(1e-3, 3e-3), rng
        )
        return r[None, :], two_way_amplitude(r[None, :], PET_RCS_M2, WALL_LOSS_DB)
    if kind == "fan":
        hub_m, rpm = rng.uniform(2.0, 4.0), rng.uniform(80, 200)
        if isotropic_fan:
            r = fan_ranges_m(t, hub_m, 1.4, 0.66, rpm, rng=rng)
            return r, two_way_amplitude(r, fan_rcs_m2, WALL_LOSS_DB)
        # chirp mean frequency, which the slow-time phase follows (NOTES 11-19)
        lam_m = C_MPS / (cfg.fc_hz + cfg.bandwidth_hz / 2)
        seg_m = 0.75 * 0.66 / 4
        peak_rcs_m2 = fan_rcs_m2 * 2 * np.pi * seg_m / lam_m
        r, gain = fan_echo(t, hub_m, 1.4, 0.66, rpm, lam_m, rng=rng)
        return r, two_way_amplitude(r, peak_rcs_m2, WALL_LOSS_DB) * gain
    raise ValueError(f"unknown scene {kind!r}")


def trial(cfg, kind, noise_std, rng, fan_rcs_m2, isotropic_fan, n_frames=400):
    t = np.arange(n_frames) * cfg.frame_period_s
    r, a = scene_targets(cfg, kind, t, rng, fan_rcs_m2, isotropic_fan)
    ranges = np.vstack([np.full((1, n_frames), WALL_M), r])
    amps = np.vstack(
        [np.full((1, n_frames), two_way_amplitude(WALL_M, WALL_RCS_M2)), a]
    )
    x = if_frames(cfg, ranges, amps, noise_std, rng)
    return motion_energy(range_fft(x), cfg)


def energies(cfg, kind, noise_std, seed, n, fan_rcs_m2=FAN_RCS_M2, iso=False):
    rng = np.random.default_rng(seed)
    return np.array(
        [trial(cfg, kind, noise_std, rng, fan_rcs_m2, iso) for _ in range(n)]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.1, 1.0, 3.0, 10.0])
    ap.add_argument(
        "--fan-rcs", type=float, default=FAN_RCS_M2, help="m^2 per blade scatterer"
    )
    ap.add_argument("--isotropic-fan", action="store_true")
    args = ap.parse_args()
    cfg = ChirpConfig()
    print("noise_std  threshold  " + "  ".join(f"{k:>7}" for k in TEST_SEED))
    for noise_std in args.noise:
        thr = energies(cfg, "empty", noise_std, 1, args.trials).max()
        over = [
            (
                energies(
                    cfg,
                    k,
                    noise_std,
                    seed,
                    args.trials,
                    args.fan_rcs,
                    args.isotropic_fan,
                )
                > thr
            ).sum()
            for k, seed in TEST_SEED.items()
        ]
        print(
            f"{noise_std:9.2f}  {thr:9.2e}  "
            + "  ".join(f"{n:3d}/{args.trials:<3d}" for n in over)
        )


if __name__ == "__main__":
    main()
