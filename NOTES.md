# Notes

## 2026-11-19

First end-to-end path: IF simulator, Hann range FFT, mean removal over a
20 s window, and the largest remaining range-bin power as the presence
feature. Threshold is the largest empty-room value on a validation seed,
reported on a separate test seed, 100 windows per class.
(`experiments/energy_threshold.py`)

    noise_std  threshold  detected  false_alarms
         0.10   7.08e-05   100/100     0/100
         1.00   7.08e-03    51/100     0/100
         3.00   6.37e-02    14/100     0/100
        10.00   7.08e-01     3/100     0/100

The empty room is only a static wall plus white noise, so mean removal
takes the wall out exactly and the threshold is pure noise floor; it scales
with noise_std^2 as it should. At noise_std 1.0, binned by person range:
1.5-2.5 m 32/32, 2.5-3.5 m 19/32, 3.5-5 m 0/36. That is the R^-4 of the
two-way path doing what it should. The noise levels are arbitrary, so the
table only says where the cliff is relative to them. The real test of this
feature is a fan or a pet, which the scene doesn't have yet.

While writing the Bessel test for the moving fraction of a breathing echo,
the prediction with fc = 60 GHz came out at 0.588 against 0.609 from the
simulator. The slow-time phase of a range bin follows the chirp's mean
frequency, 61.53 GHz here, not its start frequency: over a 3 GHz sweep that
is 2.5 % more phase per mm. With 61.53 GHz the prediction matches to 1e-5.
Worth remembering for the breathing-rate and displacement estimate.
