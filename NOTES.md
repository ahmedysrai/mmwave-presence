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

## 2026-11-25

Tried yesterday's idea: divide the low-band excess by how much it
scatters when the range bin holds only a flat floor
(`doppler.low_band_score`). The spread s of one Doppler bin's power comes
from the 241 bins at |f_d| >= 4 Hz, and the excess is divided by
s * sqrt(n_low * (1 + n_low / n_high)). For white noise that should be a
unit normal if the bins were independent; the Hann window correlates
neighbours (power correlation 4/9 one bin apart, 1/36 two apart), so the
std comes out at 1.43 over 4000 range bins, against 1.39 from those
correlations.

`experiments/low_band.py` now prints three columns, each with its own
empty-room threshold on the validation seed:

    noise   energy (person pet fan)   excess (person pet fan)   score (person pet fan)
     0.1        100   99  100              100   97   34              100   89    0
     1.0         51   11   17               46    8    0               41    3    0
     3.0         14    1    2                7    0    0                2    1    0
    10.0          3    1    2                0    0    0                0    1    0

out of 100 each, empty room 0/100 everywhere. The 34 fans at noise 0.1
are gone, at every hub range (`--fan-detail`: 0/28 at 2-2.5 m, where the
excess let 18/28 through). The pet at noise 0.1 drops from 97 to 89,
which I didn't expect; probably the weaker pets at the far end, whose
excess was large in absolute terms only because noise was low, but I
haven't split it by range.

The cost is at higher noise: 46 -> 41 people at 1.0, 7 -> 2 at 3.0. The
empty-room score does not change with noise level at all (threshold 8.52
at 0.1, 1.0 and 3.0, same seeds), because the room is the wall, which
static removal takes out, plus noise, and the score has no units. That is
the useful property: one fixed threshold holds the false alarm rate
whatever the noise, the way CFAR does. But 8.52 is high. The empty
windows have a median max-over-range score of 4.67 and the top five are
6.35, 6.38, 6.57, 7.54 and 8.52, so the threshold is set by one or two
windows out of 100, about 6 times the per-bin std. The median max over
245 range bins of a std-1.43 normal is about 4, so the tail is heavier
than normal; my guess is range bins where s comes out low by chance. The threshold would be steadier from a quantile of more empty
windows than from the max of 100.

## 2026-11-26

Yesterday's guess was that the score threshold (8.52) was high because
it was the max of only 100 empty windows. `low_band.py` now takes
`--empty N --quantile q`; the defaults (N = --trials, q = 1) give the
old max and reproduce the 2026-11-25 table exactly. With 1000 empty
windows on the same validation seed (the first 100 are yesterday's):

    empty-room score, max over range   noise 0.1   1.0    3.0
      max of 1000                        8.70      8.70   8.70
      99.9th pct                         8.52      8.52   8.52
      99th pct                           7.85      7.85   7.85
      median                             4.63      4.63   4.63

So 8.52 was not a fluke of a small sample: it is about the 1-in-1000
point, and even the 99th percentile sits at 7.85, 1.7 times the median.
The tail really is heavy. Same identical numbers at every noise level
as yesterday.

With every feature thresholded at its own 99th percentile of the 1000
empty windows (test-seed empty room should then read about 1/100):

    noise   energy (empty person pet fan)   excess (empty person pet fan)   score (empty person pet fan)
     0.1        0  100   99  100                 0  100   98   36                 1  100   93    0
     1.0        0   51   11   22                 0   49   10    0                 1   43    5    0
     3.0        0   14    1    5                 0    9    1    0                 1    3    1    0
    10.0        0    3    1    5                 0    0    1    0                 1    0    1    0

out of 100 each. The score gets back 2 people at noise 1.0 (41 -> 43) and
1 at 3.0 (2 -> 3) and still has no fans over it, but it stays behind the
raw excess (49, 9) and the energy (51, 14). Energy and excess hardly move
because their empty-room values are tight (energy q99 equals the max to
three figures at noise 1.0). The empty column for energy and excess still
reads 0/100 on the test seed, the score 1/100, as it should.

So the score's loss of people is not about how the threshold is
estimated. It comes from the shape of the empty-room score distribution.
The next thing to check is whether the high empty windows are the ones
where the floor spread s came out low, which would point at estimating s
over more bins (or over neighbouring range bins) rather than at the
threshold.

## 2026-11-27

Are the high empty-room scores the windows where the floor spread s
came out low? 300 empty windows on the validation seed at noise 1.0,
looking at the range bin that gives each window its max score, with s
divided by that window's median s over range:

    window                 score   s / median s   excess / median s
    top 1                   8.52       0.73             101
    top 2                   8.50       0.74             102
    top 3                   8.43       0.62              85
    top 10 (range)    6.82-8.52   0.62-0.89          83-119
    median window           4.67       0.82              62

Over all range bins of all windows s / median s has 1st and 5th
percentiles 0.77 and 0.83. So the bin that wins the max over range
already has a low s in a typical window (0.82), and the top windows have
s in about the lowest 1 to 2 percent, though their excess is high as well
(correlation of score and s at the max bin -0.32). Low s is part of it,
not all of it.

First try: replace s in each bin by the median s of the 2k + 1 bins
around it (noise after the range FFT is flat in range, so they should
agree). The empty tail shrinks (q99 7.85 -> 5.97 at k = 8) and people
come back, but at noise 0.1 fans go from 0 to 34-35 in 100 for k >= 4,
which is the raw excess again. Of course: a strong fan in its own bin
has a large s, and that is exactly what the score divides by. Pooling
throws that away.

So s is raised to the neighbourhood median but never lowered
(`low_band_score(..., pool_bins=k)`, `low_band.py --pool k`). Same
1000 empty windows and q99 thresholds as yesterday, score column only:

    k                     0      2      4      8     16
    empty median        4.63   4.13   4.01   3.96   3.94
    empty q99           7.85   6.57   6.13   5.97   5.79
    noise 0.1   fan        0      0      0      0      0
                pet       93     96     97     98     98
    noise 1.0   person    43     46     48     49     49
                pet        5      7      9      8     10
    noise 3.0   person     3      6      9      8      9

out of 100. k = 0 is yesterday's column exactly. With k = 8 the score
matches the excess at noise 1.0 (49) and is close at 3.0 (8 against 9),
with no fans over it at any noise level, where the excess lets 36
through at 0.1. Still behind the energy at 3.0 (14), but the energy lets
22 fans through at 1.0. The test-seed empty room reads 2/100 at k = 8
(1/100 at k = 0) against an expected 1 in 100.

The pet drop at noise 0.1 (97 -> 89/93) that I noted on 11-25 mostly goes
away too (98 at k = 8), so it was probably the same low-s effect rather
than anything about range. The pet is a breathing target, so the score
is right to fire on it; telling it apart from a person is the
classifier's job later.

k = 8 is a guess that sits on the flat part of the table; anything from 4
up gives about the same. Next is CFAR over range, which needs the same
kind of neighbourhood estimate, so this may get folded into it.
