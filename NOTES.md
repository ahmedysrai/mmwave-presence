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

## 2026-11-20

Added a pet and a ceiling fan to the scene. Same threshold rule as
yesterday, same seeds for the person and the empty room, 100 windows per
scene; the counts are windows over the empty-room threshold.
(`experiments/energy_threshold.py`)

    noise_std  threshold    empty   person      pet      fan
         0.10   7.08e-05    0/100  100/100   99/100  100/100
         1.00   7.08e-03    0/100   51/100   11/100    3/100
         3.00   6.37e-02    0/100   14/100    1/100    2/100
        10.00   7.08e-01    0/100    3/100    1/100    2/100

The pet is the expected problem. At noise 0.1 it is over the threshold
99/100, the same as the person. At 1.0 it falls to 11/100 only because it
is a weaker target (RCS 0.05 against 0.5 m^2, 1 to 3 mm of chest motion),
not because the feature tells it apart.

The fan came out much weaker than I expected: 3/100 at noise 1.0 with the
default 0.02 m^2 per scatterer. Static removal leaves all of its echo in
place (a blade tip moves about 10 m/s at 140 rpm, so its slow-time phase is
effectively random from one chirp to the next), but the blades sweep about
24 range bins for a hub 3 m out, and the max-over-bins feature only sees
one of them. Noise-free median energy over 50 windows: person 1.4e-3, pet
1.2e-4, fan 7.0e-5.

So the fan result depends almost entirely on the blade RCS, which is a
guess. At noise 1.0:

    --fan-rcs 0.02    3/100
    --fan-rcs 0.2    41/100
    --fan-rcs 2.0   100/100

A flat blade flashes when it faces the radar and returns much less at other
angles, and the model has nothing like that yet. Next: an aspect-dependent
blade return, then the Doppler FFT, where the fan's aliased, wideband
spectrum should stand apart from a breathing line near 0 Hz.
