# MicroscopeVuilder

A physics-based puzzle game about building a microscope. The player assembles
diascopic and episcopic light paths on a virtual optical bench; configurations are
verified against real optical invariants (conjugate planes, NA matching, tube-length
conventions, Nyquist sampling) and scored by the image the build actually produces.

Aimed at graduate students in the biomedical sciences: rigor over abstraction, real
manufacturer conventions, and the governing equation shown with your numbers in it.

See [`docs/PLAN.md`](docs/PLAN.md) for the full design and implementation plan.

## Status

Milestones M0-M4 complete: paraxial engine, CFI60 catalog, wavefront and diffraction
layers, partially coherent imaging, the bench and rule engine, rounds 1-2, and the Qt
workspace.

```sh
python -m microscopevuilder --list           # implemented rounds
python -m microscopevuilder --round 2        # grade the reference build headlessly
python -m microscopevuilder --ui --round 2   # open the workspace (needs the 'ui' extra)
```

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
