"""CA-CFAR over range on the motion energy profile, against a fixed threshold.

Same scenes, seeds and wall as energy_threshold.py. The profile is the mean
|X|^2 over the window's frames after static removal, per range bin beyond
0.5 m (what rangeproc.motion_energy takes the max of). Two detectors look
at it:

- energy: the max over range against a fixed threshold, the largest of
  --trials empty windows at --train-noise and kept for every noise level;
- CFAR: cfar.ca_cfar along range, a detection if any bin fires. The
  per-bin pfa is --window-pfa over the number of bins, and n_looks is the
  number of frames less one (remove_static takes out the mean).

The fixed threshold is right only at the noise level it was set at; CFAR
sets its own from the bins around each one, so it should hold its empty
room rate at every level.

    python experiments/cfar_range.py
    python experiments/cfar_range.py --train 8 --guard 4
"""

import argparse

import numpy as np
from energy_threshold import FAN_RCS_M2, TEST_SEED, WALL_M, WALL_RCS_M2, scene_targets

from mmpresence.cfar import ca_cfar
from mmpresence.rangeproc import range_axis_m, range_fft, remove_static
from mmpresence.sim import ChirpConfig, if_frames, two_way_amplitude


def profile(cfg, kind, noise_std, rng, n_frames=400, r_min_m=0.5):
    t = np.arange(n_frames) * cfg.frame_period_s
    r, a = scene_targets(cfg, kind, t, rng, FAN_RCS_M2)
    ranges = np.vstack([np.full((1, n_frames), WALL_M), r])
    amps = np.vstack(
        [np.full((1, n_frames), two_way_amplitude(WALL_M, WALL_RCS_M2)), a]
    )
    x = if_frames(cfg, ranges, amps, noise_std, rng)
    p = np.mean(np.abs(remove_static(range_fft(x))) ** 2, axis=0)
    return p[range_axis_m(cfg) >= r_min_m]


def profiles(cfg, kind, noise_std, seed, n):
    rng = np.random.default_rng(seed)
    return np.array([profile(cfg, kind, noise_std, rng) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--train-noise", type=float, default=1.0)
    ap.add_argument("--train", type=int, default=16, help="training bins per side")
    ap.add_argument("--guard", type=int, default=2, help="guard bins per side")
    ap.add_argument("--window-pfa", type=float, default=0.01)
    args = ap.parse_args()
    cfg = ChirpConfig()
    thr = profiles(cfg, "empty", args.train_noise, 1, args.trials).max(axis=1).max()
    print(f"energy threshold {thr:.3e} from noise {args.train_noise}")
    print("noise_std  detector  " + "  ".join(f"{k:>7}" for k in TEST_SEED))
    for noise_std in args.noise:
        rows = {"energy": [], "CFAR": []}
        for kind, seed in TEST_SEED.items():
            p = profiles(cfg, kind, noise_std, seed, args.trials)
            pfa = args.window_pfa / p.shape[1]
            cf = [ca_cfar(q, args.train, args.guard, pfa, n_looks=399).any() for q in p]
            rows["energy"].append((p.max(axis=1) > thr).sum())
            rows["CFAR"].append(int(np.sum(cf)))
        for name, over in rows.items():
            print(
                f"{noise_std:9.2f}  {name:8s}  "
                + "  ".join(f"{n:3d}/{args.trials:<3d}" for n in over)
            )


if __name__ == "__main__":
    main()
