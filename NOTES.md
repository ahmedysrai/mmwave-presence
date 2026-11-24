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

## 2026-11-21

The fan blades now flash. Each blade is 4 straight segments, and each segment is
the coherent sum of points lambda/8 apart along it (`sim.fan_echo`), so it
returns strongly only while it lies across the line of sight. In the far
field the mean of |gain|^2 over a turn is lambda / (2 pi L), 0.0063 for a
12.4 cm segment at the chirp's mean frequency; the test checks this to 2 %.
At 2 to 4 m the segment is well inside its far-field distance
(2 L^2 / lambda, about 6 m), and the flash is blunted and wider: with the
hub 2 to 4 m away and 1.4 m up, peak |gain|^2 is 0.67 to 0.89 and the mean
over a turn 0.0077 to 0.0066.

To change one thing at a time, `--fan-rcs` now means a segment's RCS
averaged over a turn and the peak is set to that times 2 pi L / lambda
(about 160x), so the fan sends back about the same power as yesterday's
isotropic points. Same seeds as before; `--isotropic-fan` gives yesterday's
model and reproduces its fan column exactly.

    noise_std  threshold    empty   person      pet      fan
         0.10   7.08e-05    0/100  100/100   99/100  100/100
         1.00   7.08e-03    0/100   51/100   11/100   17/100
         3.00   6.37e-02    0/100   14/100    1/100    2/100
        10.00   7.08e-01    0/100    3/100    1/100    2/100

At noise 1.0, fan false alarms against the RCS averaged over a turn:

    --fan-rcs   isotropic   flashing
       0.02       3/100      17/100
       0.2       41/100      98/100
       2.0      100/100     100/100

Same mean power, but the flashing fan beats the energy threshold far more
often. The reason is where the power lands in range. A segment flashes when
it lies across the line of sight, and so does the rest of its blade, so a
flash puts the whole blade into one or two range bins. Noise-free, 50
windows: the largest bin holds a median 51 % of the fan's motion energy
against 8 % for the isotropic points (more than half the peak in 2 bins
against 12), and the max-over-bins feature goes from 6.7e-5 to 4.8e-4. The
person is at 1.4e-3. A fan is a real false-alarm risk for this feature
after all, and it still rests on a guessed blade RCS. The Doppler FFT is
next. At 140 rpm a flash lasts about 2 ms, much less than the 50 ms
between chirps, so a chirp either lands in one or misses it, and the fan
should spread over every Doppler bin while breathing stays near 0 Hz.

## 2026-11-22

Doppler FFT over slow time (`doppler.doppler_fft`, Hann over all 400 chirps
of a window, so 0.05 Hz bins over +-10 Hz). `experiments/doppler_spread.py`
takes the range bin with the most motion energy, the one the energy
threshold fires on, and prints the share of its energy above a Doppler
cut. Same scenes and seeds as the energy threshold, 50 trials, median and
[5th, 95th] percentile. Noise-free:

    scene   |f_d| > 2 Hz        |f_d| > 4 Hz        |f_d| > 8 Hz
    person  0.32 [0.00, 0.61]   0.00 [0.00, 0.01]   0.00 [0.00, 0.00]
    pet     0.01 [0.00, 0.17]   0.00 [0.00, 0.00]   0.00 [0.00, 0.00]
    fan     0.80 [0.77, 0.84]   0.60 [0.57, 0.64]   0.20 [0.14, 0.24]

The fan comes out flat over the whole span: a flat spectrum gives
(10 - cut) / 10, which is 0.80, 0.60 and 0.20. That is what yesterday's
guess said. A flash lasts about 2 ms and the chirps are 50 ms apart, so
whether a chirp catches one is close to random from chirp to chirp, and
the tip Doppler (about 4 kHz at 140 rpm) aliases many times over.

Breathing stays under 4 Hz. Phase modulation by a sine of beta radians
spreads out to about beta times the breathing rate, with
beta = 2 pi d / lambda for peak-to-peak depth d. At the 10 mm, 0.33 Hz end
of the person range that is 4.3 Hz, which matches the 0.01 at the 95th
percentile. The pet, with 1 to 3 mm, stays under 2 Hz most of the time.

With noise the picture depends on the margin. At noise 0.1 the person is
still 0.03 [0.00, 0.13] above 4 Hz and the fan 0.61, but the pet goes to
0.20 [0.02, 0.50] because the weaker pets are near the noise. At noise 1.0
everything reads like the empty room (0.59 above 4 Hz; person 0.53
[0.16, 0.65]), since the noise is white too. So the high-Doppler share
alone cannot tell a fan from noise; it has to go with the energy.

One way to use this: in each range bin, the band above 4 Hz estimates the
flat floor (noise plus fan), so 4/6 of the high-band power estimates the
floor inside |f_d| < 4 Hz. Low-band power minus that is near zero for
noise and for the fan and positive for breathing. Not tried yet.

## 2026-11-23

Tried the low-band excess from yesterday (`doppler.low_band_excess`): in
each range bin, Doppler power at |f_d| < 4 Hz minus the number of those
bins times the mean power at |f_d| >= 4 Hz, then the max over range beyond
0.5 m. `experiments/low_band.py` runs it on the same windows as the energy
threshold, each feature with its own threshold (largest empty-room value
on the validation seed), 100 trials per scene:

    noise_std   energy: person  pet  fan    low band: person  pet  fan
    0.10                100     99   100              100     97   34
    1.00                 51     11    17               46      8    0
    3.00                 14      1     2                7      0    0
    10.00                 3      1     2                0      0    0

(Empty room 0/100 for both at every noise level.)
The energy column is the energy_threshold.py table, as it should be.

The fan false alarms drop from 17 to 0 at noise 1.0 and from 100 to 34 at
noise 0.1. It costs some person detections at the weaker end (51 to 46 at
noise 1.0, 14 to 7 at noise 3.0): the floor estimate from 120 high bins
has its own spread, and taking it off 160 low bins adds that spread to the
feature, so the empty-room threshold sits higher relative to a weak
breather. The pet goes through as before, which was expected, since it
breathes below 4 Hz too.

The 34 fans left at noise 0.1 are not spread evenly over speed:

    rpm       80-100  100-120  120-140  140-160  160-180  180-200
    fans          18       12       12       22       19       17
    over thr       9        1        2        5       10        7

Three blades flash at the blade-pass rate 3 rpm / 60, 4 to 10 Hz here.
My guess is that the flash train is periodic enough to have harmonics, and
those fold back into the low band: at 160 to 200 rpm the second harmonic
(16 to 20 Hz) aliases to -4 to 0 Hz, and at 80 to 100 rpm the fundamental
sits right at the 4 Hz cut. Not checked yet; looking at the spectrum of
one 180 rpm fan would settle it.

## 2026-11-24

Checked yesterday's guess that the fans left over the low-band threshold
are blade-pass harmonics folding into |f_d| < 4 Hz. It is wrong.

`experiments/fan_spectrum.py` runs the fan alone, noise-free, at 80 to
200 rpm in 5 rpm steps (five hub ranges each) and prints the share of the
strongest range bin's Doppler power below 4 Hz. A flat spectrum gives
0.40. Almost every speed gives 0.32 to 0.42. The exceptions are the speeds
whose blade-pass rate divides the 20 Hz frame rate exactly (80, 100, 160
and 200 rpm, i.e. 4, 5, 8 and 10 Hz): 0.08, 0.00, 0.08 and 0.00, with
0.24 at 120 rpm (6 Hz, 3 blade passes every 10 frames). Only there is the
spectrum a few clean lines, and they sit on multiples of the blade-pass
rate, outside the low band. At any other speed a 2 ms flash caught every
50 ms lands at a different point of the flash each chirp, and the
spectrum looks like noise (figures/fan_spectrum.png, 183 rpm against
100 rpm and a breathing person).

So the low-band excess of a fan is a random number around zero whose
spread grows with the fan's power. `low_band.py --fan-detail` splits the
34 fans over the threshold at noise 0.1:

    hub m     2-2.5  2.5-3  3-3.5  3.5-4
    over      18/28   6/22   4/25   6/25

    rpm       80-100  100-120  120-140  140-160  160-180  180-200
    over        9/18     1/12     2/12     5/22    10/19     7/17

The rpm row is yesterday's table. Hub range sorts them much better: two
thirds of the near fans (strongest return) cross, against a quarter or
less beyond 2.5 m. I now think the rpm pattern is mostly which hub ranges
happened to be drawn with which speeds, but I haven't checked that
directly. Taking the floor off is right on average; what is missing is a
test of whether the excess is large compared with how much the floor
estimate itself scatters. Next thing to try: divide the excess by the
standard error of the floor (from the spread of the bins above 4 Hz)
and threshold that.
