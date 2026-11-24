"""Doppler spectrum of a ceiling fan at a few speeds, noise-free.

Checks the 2026-11-23 guess that the fans left over the low-band
threshold are blade-pass harmonics folding into |f_d| < 4 Hz. Each window
is the fan of energy_threshold.py alone (no wall, no noise), 400 chirps at
20 Hz. The printed table is the share of the Doppler power below 4 Hz in
the strongest range bin, mean over five hub ranges per speed; a flat
spectrum gives 8 / 20 = 0.40. The figure has that bin's spectrum for a
breathing person, a fan whose blade-pass rate divides the frame rate
(100 rpm, 5 Hz) and one whose rate does not (183 rpm, 9.15 Hz).

    python experiments/fan_spectrum.py              # about 15 s
    python experiments/fan_spectrum.py --no-plot
"""

import argparse
from pathlib import Path

import numpy as np
from energy_threshold import CHEST_RCS_M2, FAN_RCS_M2

from mmpresence.doppler import doppler_axis_hz, doppler_fft
from mmpresence.fmcw import C_MPS
from mmpresence.rangeproc import range_fft, remove_static
from mmpresence.sim import (
    ChirpConfig,
    breathing_range_m,
    fan_echo,
    if_frames,
    two_way_amplitude,
)

N_FRAMES = 400
FIG = Path(__file__).resolve().parent.parent / "figures" / "fan_spectrum.png"


def strongest_bin_spectrum(cfg, ranges_m, amps, rng):
    x = if_frames(cfg, ranges_m, amps, 0.0, rng)
    p = np.abs(doppler_fft(remove_static(range_fft(x)))) ** 2
    return p[:, p.sum(axis=0).argmax()]


def fan_spectrum(cfg, rpm, hub_m, rng):
    t = np.arange(N_FRAMES) * cfg.frame_period_s
    lam_m = C_MPS / (cfg.fc_hz + cfg.bandwidth_hz / 2)
    peak_rcs_m2 = FAN_RCS_M2 * 2 * np.pi * (0.75 * 0.66 / 4) / lam_m
    r, gain = fan_echo(t, hub_m, 1.4, 0.66, rpm, lam_m, rng=rng)
    return strongest_bin_spectrum(cfg, r, two_way_amplitude(r, peak_rcs_m2) * gain, rng)


def person_spectrum(cfg, rng):
    t = np.arange(N_FRAMES) * cfg.frame_period_s
    r = breathing_range_m(t, 3.0, 0.25, 6e-3, rng)[None, :]
    return strongest_bin_spectrum(cfg, r, two_way_amplitude(r, CHEST_RCS_M2), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=float, default=4.0, help="Hz")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    cfg = ChirpConfig()
    f_hz = doppler_axis_hz(N_FRAMES, cfg.frame_period_s)
    low = np.abs(f_hz) < args.cut

    print("   rpm  blade-pass Hz  low-band share")
    for rpm in range(80, 201, 5):
        shares = []
        for seed, hub_m in enumerate(np.linspace(2.0, 4.0, 5)):
            p = fan_spectrum(cfg, rpm, hub_m, np.random.default_rng(seed))
            shares.append(p[low].sum() / p.sum())
        print(f"{rpm:6d}  {3 * rpm / 60:13.2f}  {np.mean(shares):14.2f}")

    if args.no_plot:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    traces = [
        ("person, 0.25 Hz breathing", person_spectrum(cfg, rng)),
        ("fan, 100 rpm", fan_spectrum(cfg, 100, 3.0, rng)),
        ("fan, 183 rpm", fan_spectrum(cfg, 183, 3.0, rng)),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvspan(-args.cut, args.cut, color="0.9", label=f"|f_d| < {args.cut:g} Hz")
    for label, p in traces:
        ax.plot(f_hz, 10 * np.log10(p / p.max() + 1e-12), lw=0.8, label=label)
    ax.set_xlim(f_hz[0], -f_hz[0])
    ax.set_ylim(-60, 3)
    ax.set_xlabel("Doppler frequency (Hz)")
    ax.set_ylabel("power in strongest range bin (dB re peak)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=120)
    print(f"wrote {FIG.relative_to(FIG.parent.parent)}")


if __name__ == "__main__":
    main()
