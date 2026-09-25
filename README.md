# MicroscopeVuilder

A physics-based puzzle game about building a microscope. You assemble diascopic and
episcopic light paths on a virtual optical bench; builds are checked against real
optical invariants — conjugate planes, NA matching, tube-length conventions, Nyquist
sampling — and scored by the image the build actually forms.

Aimed at graduate students in the biomedical sciences: rigor over abstraction, real
manufacturer conventions, and the governing equation shown with your numbers in it.

Fifteen rounds, from a single-lens loupe to phase contrast and DIC, plus a sandbox.
See [`docs/PLAN.md`](docs/PLAN.md) for the design.

## Download

Self-contained executables are attached to every [release](../../releases) — one file,
no installer, no Python, no `pip`.

Every commit to `main` publishes a development build (a prerelease tagged `build-N`);
a `v*` tag publishes a stable release. Both run the full test suite and exercise real
rounds against the frozen binary before publishing.

On macOS and Linux, `chmod +x MicroscopeVuilder-*` first. The macOS and Windows builds
are not code-signed, so the first launch needs Right-click → Open (macOS) or
"More info" → "Run anyway" (Windows SmartScreen).

## Running it

```sh
python -m microscopevuilder --list           # rounds, with your progress
python -m microscopevuilder --round 4        # grade a reference build headlessly
python -m microscopevuilder --ui --round 4   # open the workspace (needs the 'ui' extra)
python -m microscopevuilder --verify-catalog # what still needs a datasheet check
```

Useful flags: `--bench FILE` grades your own saved build, `--diff` compares a failing
build against a working one, `--strict` grades positions at the physical tolerance, and
`--no-measure` skips the rules that render an image.

## Placing components

The bench view zooms (wheel, `Ctrl` `+`/`-`, `Ctrl+0` to fit) and carries a ruler.
Dragging snaps to a selectable grid and to optical planes — image, pupil and focal —
with `Alt` to suspend snapping. The inspector takes a typed position, which is the exact
route. Units switch between millimetres and inches. `F1` explains the workspace.

Positions are graded within **5%** by default. The physical tolerance is usually far
tighter — on round 11 the depth of focus is 0.19 mm — and the scorecard names it on
every row, so the number you take away is the one a bench would hold you to. The toolbar
switches grading to that tolerance; `--strict` does the same headlessly.

## A note on the component catalog

The Nikon CFI60 specifications in `microscopevuilder/assets/components.toml` are
recorded from secondary knowledge and are **not** datasheet-verified; every entry is
marked `verified = false` and the app says so. The aberration budgets are pedagogical by
design — manufacturers do not publish Zernike budgets — and are anchored to quotable
figures such as the `f/2000` secondary spectrum of an achromat. Run `--verify-catalog`
for the checklist.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,ui]"
pytest -q
```

Everything outside `ui/` is importable and testable headless. The Qt tests run on the
offscreen platform, which CI uses too.

### Building the executable

```sh
pip install "pyinstaller>=6.22"
pyinstaller packaging/microscopevuilder.spec --noconfirm
./dist/MicroscopeVuilder --round 12   # smoke-test the frozen build
```

`.github/workflows/release.yml` builds Linux, Windows and both macOS architectures. It
publishes a prerelease on every push to `main` and a stable release on a `v*` tag;
running it by hand builds the artifacts without publishing.
