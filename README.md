# mmwave-presence

Presence sensing through a wall with a 60 GHz FMCW radar: detect people from
their breathing and tell them apart from pets and ceiling fans.

The project is simulation based. So far there is an IF signal simulator for
point targets, the range FFT, and a crude presence check (motion energy
after static clutter removal, against a fixed threshold) run on a breathing
person and an empty room. Pets and fans come next.

```
pip install -e .[dev]
pytest
python experiments/energy_threshold.py    # about 10 s
```

Results so far are in NOTES.md.
