# MicroscopeVuilder

A physics-based puzzle game about building a microscope. The player assembles
diascopic and episcopic light paths on a virtual optical bench; configurations are
verified against real optical invariants (conjugate planes, NA matching, tube-length
conventions, Nyquist sampling) and scored by the image the build actually produces.

Aimed at graduate students in the biomedical sciences: rigor over abstraction, real
manufacturer conventions, and the governing equation shown with your numbers in it.

See [`docs/PLAN.md`](docs/PLAN.md) for the full design and implementation plan.

## Download

Self-contained executables for each platform are attached to every
[release](../../releases) — one file, no installer, no Python, no `pip`.

Every commit to `main` publishes a development build (a prerelease tagged
`build-N`); tagged `v*` versions publish a stable release. Both run the full test
suite and exercise real rounds against the frozen binary before publishing.
On macOS and Linux, `chmod +x MicroscopeVuilder-*` before the first run. The macOS
and Windows builds are not code-signed, so the first launch needs Right-click → Open
(macOS) or "More info" → "Run anyway" (Windows).

## Status

Milestones M0–M9 complete: paraxial engine, CFI60 catalog, wavefront and diffraction
layers, partially coherent imaging, the bench and rule engine, the Qt workspace, and
packaging. Rounds 1–5 and 9–12 are playable, plus a sandbox. Rounds 6–8 (colour, flat
field, camera port) and 13–15 (contrast techniques) are not yet implemented.

```sh
python -m microscopevuilder --list           # implemented rounds
python -m microscopevuilder --round 2        # grade the reference build headlessly
python -m microscopevuilder --ui --round 2   # open the workspace (needs the 'ui' extra)
```

### Building the executable

```sh
pip install -e ".[dev,ui]" "pyinstaller>=6.22"
pyinstaller packaging/microscopevuilder.spec --noconfirm
./dist/MicroscopeVuilder --round 12          # smoke-test the frozen build
```

`.github/workflows/release.yml` builds Linux, Windows and both macOS architectures.
It publishes a prerelease on every push to `main` and a stable release on a `v*`
tag; running it by hand builds the artifacts without publishing anything.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
