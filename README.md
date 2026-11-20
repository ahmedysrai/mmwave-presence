# mmwave-presence

Presence sensing through a wall with a 60 GHz FMCW radar: detect people from
their breathing and tell them apart from pets and ceiling fans.

The project is simulation based. So far there is an IF signal simulator for
point targets, the range FFT, and a crude presence check (motion energy
after static clutter removal, against a fixed threshold) run on four scenes:
an empty room, a breathing person, a breathing pet and a ceiling fan. The
energy check passes a pet as a person whenever the pet is close enough to be
seen, which is the case the classifier has to handle.

```
pip install -e .[dev]
pytest
python experiments/energy_threshold.py    # about 40 s
```

Results so far are in NOTES.md.
