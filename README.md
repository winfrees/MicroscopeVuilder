# MicroscopeVuilder

A physics-based puzzle game about building a microscope. The player assembles
diascopic and episcopic light paths on a virtual optical bench; configurations are
verified against real optical invariants (conjugate planes, NA matching, tube-length
conventions, Nyquist sampling) and scored by the image the build actually produces.

Aimed at graduate students in the biomedical sciences: rigor over abstraction, real
manufacturer conventions, and the governing equation shown with your numbers in it.

See [`docs/PLAN.md`](docs/PLAN.md) for the full design and implementation plan.

## Status

Early. Milestone M1 (paraxial ABCD engine, stops and pupils) is in progress.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
