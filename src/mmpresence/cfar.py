"""Cell-averaging CFAR along one axis.

Each cell under test is compared with alpha times the mean of the training
cells on both sides of it, skipping guard cells next to it so a target that
spreads over a few bins does not raise its own threshold. The noise level
never enters: it scales the cell and the training mean alike, so the false
alarm rate stays put when the noise floor moves.

alpha comes from the design false alarm probability per cell. If the cells
are independent and each is the mean of n_looks exponential powers (a
square-law detector over n_looks frames), the cell over the training mean
is F distributed with (2 n_looks, 2 n_looks n_train) degrees of freedom,
so alpha is its upper pfa point. For n_looks = 1 that is the textbook
n_train (pfa^(-1/n_train) - 1). Neighbouring range bins after a Hann
range FFT are not independent, which the guard cells only partly cover;
the measured rate is in NOTES.md (2026-11-28).
"""

import numpy as np
from scipy.stats import f as f_dist


def ca_cfar_threshold(power, n_train, n_guard, pfa, n_looks=1):
    """Threshold for each cell of a 1-D power profile.

    power: (n,) non-negative. n_train and n_guard are cells on each side.
    Near the ends the window is cut off and the cell uses the training
    cells it has (alpha follows their count); a cell with none on either
    side gets an infinite threshold. Returns (n,).
    """
    power = np.asarray(power, dtype=float)
    n = power.size
    csum = np.concatenate(([0.0], np.cumsum(power)))
    i = np.arange(n)
    # training cells [i - g - t, i - g - 1] and [i + g + 1, i + g + t], clipped
    lo_a, lo_b = np.clip(i - n_guard - n_train, 0, n), np.clip(i - n_guard, 0, n)
    hi_a, hi_b = (
        np.clip(i + n_guard + 1, 0, n),
        np.clip(i + n_guard + n_train + 1, 0, n),
    )
    count = (lo_b - lo_a) + (hi_b - hi_a)
    total = (csum[lo_b] - csum[lo_a]) + (csum[hi_b] - csum[hi_a])
    thr = np.full(n, np.inf)
    ok = count > 0
    alpha = f_dist.isf(pfa, 2 * n_looks, 2 * n_looks * count[ok])
    thr[ok] = alpha * total[ok] / count[ok]
    return thr


def ca_cfar(power, n_train, n_guard, pfa, n_looks=1):
    """Boolean detections, power over ca_cfar_threshold."""
    return np.asarray(power) > ca_cfar_threshold(power, n_train, n_guard, pfa, n_looks)
