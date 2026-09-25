# MicroscopeVuilder — design and plan

A distributable Python puzzle game about *building* a working microscope. The player
assembles diascopic and episcopic illumination paths plus an imaging path on a virtual
optical bench, verified against real optical physics. Progression runs from a single
lens to an infinity-corrected stand with contrast techniques.

This document is the design and the current state. How each piece came to be built is
in the commit history; constraints discovered along the way live in comments and tests
beside the code they constrain.

---

## 1. Decisions

| # | Decision | Consequence |
|---|---|---|
| 1 | **Audience: graduate students in biomedical sciences** | Rigor over abstraction. Nikon CFI60 is the reference system (200 mm tube lens, 60 mm parfocal, `f = 200/M`); DIN 160 mm appears in the early finite-tube rounds and Olympus's 180 mm in round 11 as a cross-vendor trap. Every scorecard row shows the governing equation with the player's numbers in it. Tolerances come from what matters at a bench, not from difficulty tuning. |
| 2 | **Rigor ceiling: thin lens + wavefront aberrations** | No sequential ray trace, no glass prescriptions, no Snell at surfaces. Elements are ideal thin lenses carrying an assigned aberration budget. |
| 3 | **2D bench now, 3D later** | The bench is stored in 3D from the start — position and direction per element, folds are real rotations — and only the renderer is 2D. A 3D view is a second renderer, not a rewrite. |
| 4 | **Contrast techniques are rounds, gated on Köhler** | Phase, polarization and DIC are rounds 13–15, unlocked by round 4. A phase ring is unteachable without conjugate planes already understood, since it lives in the objective back focal plane. Requires complex amplitude throughout, not just intensity. |

## 2. Design pillars

1. **Physics is the puzzle.** No scripted answer slots. A build wins because the trace
   and the generated image satisfy the round's spec.
2. **Illumination is a first-class problem.** Köhler and critical, diascopic and
   episcopic. Most real misery in microscopy is illumination.
3. **Tolerances are derived, never chosen.** Focus tolerances come from the Rayleigh
   quarter-wave criterion (`dz ≤ λ/2NA²`) and are validated against the diffraction
   engine: a build exactly at tolerance measures Strehl 0.8. Pupil conjugates use a
   different criterion — defocusing a pupil blurs nothing, so the test is whether the
   lamp image still fits the stop (`δ ≤ r/NA`). Conventional numbers, like the 70–100%
   condenser NA ratio, say so rather than implying a derivation.
4. **Legible feedback.** Every failure names the violated invariant, the measured
   value, the target and a remedy.
5. **One new idea per round**, never invalidating what came before.
6. **Ships as a double-clickable executable**, not a `pip install` chore.

---

## 3. Optical model

### 3.1 Geometry — paraxial ABCD on a 3D bench

Every element carries a 3D position and direction, a clear aperture, and a 2×2 ABCD
matrix. Mirrors and dichroics rotate the propagation direction; the trace runs in
unfolded path coordinate `s` and the renderer projects.

An **episcopic path needed more than folding**. It is a second path that joins the
imaging path at the beamsplitter and runs *backwards* through the objective, which
serves as its own condenser. The bench therefore carries named arms with their own path
coordinate, joining the main axis at a junction and continuing forward or in reverse.
The objective appears in both systems at the correct distance along each path.

The paraxial pass runs every frame: system matrix between any two planes, image location
(where `B = 0`), magnification, stop and pupil identification, marginal and chief rays.
The chief ray passes through the **centre of the aperture stop** — not parallel to the
axis, which only coincides when the stop sits at the front focal plane.

### 3.2 Wavefront — the aberration budget

Rather than deriving aberrations from surfaces, each catalog objective carries a budget
scaled by the Seidel dependences. Terms are stored **in the units each is naturally
quoted in**, which keeps them checkable:

| Term | Stored as | Scales as |
|---|---|---|
| Spherical | µm RMS at the entry's NA | `NA⁴`, field-independent |
| Astigmatism | µm RMS at full field | `h² NA²` |
| Field curvature | longitudinal sag in µm | `h²`, converted via `NA²` |
| Secondary spectrum | fraction of focal length | converted via `NA²` |

The achromat figure is the textbook `f/2000`, which a student can look up. Both
longitudinal terms convert as `W020 = dz·NA²/2`, so the same physical focus error costs
a fast objective far more — which is why chromatic correction matters most on fast
lenses, and why a 4× suffers ten times the chromatic shift of a 40×.

**Defocus is computed, not assigned**: a wrong position becomes a real defocus
coefficient, referred to object space as `dz/M²`.

### 3.3 Image formation

- **Coherent transfer**: pupil `P = A·exp(i2πW/λ)`, amplitude PSF by FFT, incoherent
  PSF `|h|²`, OTF by autocorrelation.
- **Partial coherence** by **Abbe source integration, not Hopkins TCC**: each
  condenser-pupil point contributes one coherent image and intensities sum. Abbe costs
  `N_source` FFTs and `O(n²)` memory against the TCC's `O(n⁴)`, and is exact for a
  spatially incoherent Köhler source. The TCC only amortizes when many specimens pass
  through one fixed system — the opposite of this game.
  The cutoff is `(1+S)·NA/λ`, so `S = 1` reaches the full incoherent `2NA/λ` and
  `S → 0` gives half. Note `S > 1` is *not* "more incoherent": source points outside the
  objective pupil contribute no undiffracted background, so the build slides toward
  darkfield and never converges to the incoherent convolution.
- **Complex amplitude throughout**, so a pure phase object is genuinely invisible in
  brightfield and becomes visible only once the player fits a phase ring and a matched
  annulus. Polarization carries a Jones state; a birefringent specimen is stored as
  retardance and azimuth maps, so what it transmits depends on the polars actually
  fitted.
- **Contrast techniques are pupil and source engineering**, not rendering modes: an
  annular source and a complex pupil mask for phase contrast, a sheared field with bias
  retardation for DIC. A misaligned ring fails for the reason it would at a bench.
- **Photometry**: relative irradiance ∝ `NA²/M²`, reported against a 10×/0.25 reference
  build, with shot noise — so a build that is correct but too dim comes out grainy.

### 3.4 The rule checker

Rules are evaluated on the built bench and return pass/fail **plus** the measured value,
the target, the governing equation and a remedy. This is both the win condition and the
tutorial.

| Invariant | Check |
|---|---|
| Optical tube length | Back focal plane to intermediate image = design `L`, so `M = L/f`. Measured from the BFP, not the objective (that is `L + f`) |
| Köhler conjugates | lamp ≡ aperture diaphragm ≡ objective back focal plane |
| Field conjugates | field diaphragm ≡ specimen ≡ intermediate image ≡ sensor |
| Relaxed eye | Intermediate image at the eyepiece front focal plane |
| NA match | `NA_cond ≈ NA_obj` (0.7–1.0×), immersion consistency |
| Clear aperture | nothing but the stop clips the marginal or chief ray |
| Epi separation | dichroic edge inside the Stokes shift; bleedthrough below threshold |
| Infinity space | specimen at the objective front focus; collimated to the tube lens |
| Sampling | Nyquist at the sensor, with a warning band above it for empty magnification |
| Parfocality | every turret objective shares a shoulder height |

Rules run on **two clocks**. Geometry rules are microseconds and run on every drag;
measured rules render an image and run on **Run**. Rounds needing only geometry pay
nothing for the second pass.

---

## 4. Game structure

### Workspace

Side-elevation bench view with components as glyphs, draggable along the axis. Marginal
rays in one colour, chief rays in another, illumination in a third, each toggleable. A
**conjugate-plane ribbon** under the axis shows field and pupil planes as labelled
ticks, so Köhler becomes visibly "line up the ticks". An inspector shows the selected
element and takes a typed position. A parts budget makes the cheap correct solution beat
the brute-force one.

### The white card

A probe the player can drop anywhere on the axis, mirroring the habit that answers most
beginner questions at a bench: hold a card in the beam and look. It reports a sharp
image and its magnification, an evenly filled pupil disc, or a blur with the distance to
focus — implementing the classic diagnostic directly, since **the marginal ray crossing
the axis marks an image plane and the chief ray crossing marks a pupil plane**.

It is excluded from the trace, from stop-finding and from the rules, so holding one up
never changes what it measures.

### Placement

The derived tolerances turned out finer than the input resolution: on round 11 the depth
of focus is 0.19 mm across a 380 mm bench, so at fit-to-window one pixel of drag moved a
component 0.50 mm — two and a half times the tolerance. The physics was right and the
game was unplayable.

Fixed on both sides. **Input**: the view zooms, a ruler runs under the axis with tick
density following the zoom, dragging snaps to a selectable grid *and* to optical planes,
and the inspector takes a typed position. The default grid is 0.1 mm, because most round
targets land on a tenth and a 1 mm grid would step over them. **Grading**: positions are
accepted within the larger of the physical tolerance and 5% of the position. Relaxing
never tightens, and every row names the physical tolerance whatever the policy, so the
number a student takes away is the one a bench would hold them to.

### Testing loop

1. Choose a specimen: USAF target, Siemens star, Ronchi ruling, sine grating, beads,
   stained section, unstained cell, birefringent fibres, polished metal.
2. **Run** → the image this bench forms, through the resolved pupil.
3. **Scorecard**: measured contrast, field uniformity, SNR, Strehl, dominant aberration,
   irradiance and sampling, with the measured rules' verdict above them.
4. **Star rating**: one for solving it, one for no warnings, one for staying in budget.
5. **Diff** against a working build after a failure — what differs, deliberately not
   what to do about it.

### Rounds

| # | Round | Introduces | Win condition |
|---|---|---|---|
| 1 | Single lens loupe | Conjugates, `1/f = 1/s + 1/s'` | Focused 5× image of a ruling |
| 2 | Compound scope | Objective + eyepiece, optical tube length | 100× at the eyepoint, relaxed eye |
| 3 | Make it bright | Critical illumination, collected NA | Lamp on the specimen and enough light. Two collector positions focus; only the near one is bright enough, since irradiance goes as NA². Passes *with* a warning that the filament is visible |
| 4 | Köhler | Field and aperture diaphragms, four-plane conjugacy | Even field, filament gone, diaphragms both doing a job |
| 5 | Resolution | NA, Abbe limit, condenser matching | Resolve 0.50 µm |
| 6 | Colour | Secondary spectrum, achromat vs apochromat | The F-to-C band inside the depth of focus |
| 7 | Flat field | Field curvature, plan objectives | Corner MTF within spec of centre |
| 8 | Camera port | Nyquist sampling, empty magnification | Sampled, and not over-magnified |
| 9 | Episcopic | Vertical illuminator, branched axis, objective as its own condenser | Epi-Köhler: lamp on the BFP, field diaphragm on the specimen |
| 10 | Fluorescence | Dichroic, Stokes shift, bleedthrough | Right cube for the dye, bleedthrough below threshold |
| 11 | Infinity | Infinity space, tube lens | Same `M` at any separation, and a filter that shifts no focus |
| 12 | Full stand | Turret parfocality | All rules green across three objectives |
| 13 | Phase contrast | Annulus and ring conjugate to the BFP | Unstained cell visible |
| 14 | Polarization | Jones calculus, crossed polars | Birefringent structure on a dark background |
| 15 | DIC | Wollaston shear, bias retardation | Relief image along the shear axis |

Rounds 1–4 teach, 5–8 tighten tolerances, 9–12 restructure the bench, 13–15 are the
contrast techniques. A sandbox unlocks alongside round 5 and grades nothing.

---

## 5. Architecture

```
microscopevuilder/
  optics/    paraxial, wavefront, psf, coherence, contrast, jones, spectra
  bench/     bench model and arms, catalog, white-card probe
  rules/     base, geometry checks, measured checks, derived tolerances
  imaging/   build_optics, synthesis, metrics, specimens, techniques
  game/      rounds, scoring, progress
  ui/        workspace, scene, view, ruler, geometry, trace_model
  assets/    components.toml
tests/       golden optical cases, rule tests, round solvability, headless Qt
```

- Python 3.11+, `numpy` for everything numeric, `platformdirs` for save locations.
- **PySide6** for the UI: `QGraphicsScene` gives a real scene graph, and this game is
  substantially inspector panels. Qt also packages cleanly.
- The engine is **pure and UI-free**: `Bench → TraceResult → RuleReport → RenderedImage`.
  Everything outside `ui/` is importable and testable headless, and the Qt tests run on
  the offscreen platform in CI.
- Projection lives in `ui/geometry.py` with no Qt import, so the fold-aware drawing is
  unit-tested and a 3D renderer stays possible.
- Determinism: fixed seeds for noise, so scores are reproducible.

**Performance.** The paraxial and rule pass carries the live UI. A 384² Abbe integration
over ~200 source points takes ~0.3 s, so image synthesis runs on a worker thread behind
**Run**, never on a drag. The worker is parented to the window and waited on at close,
because destroying a running QThread aborts the process.

**Validation.** Golden tests against closed-form cases: the imaging equation, Airy first
zero, 84% encircled energy, the analytic diffraction-limited MTF, the `2NA/λ` cutoff,
quarter-wave defocus giving Strehl 0.8, and the `(1+S)NA/λ` partially coherent cutoff.

---

## 6. Distribution

- **PyInstaller one-file per platform** — an executable, not an installer. The audience
  is students on lab machines who may lack rights to run an installer or to
  `pip install`.
- CI matrix on GitHub Actions: Linux, Windows, macOS Intel and Apple silicon. Every
  commit to `main` publishes a development build as a prerelease tagged `build-N`; a
  `v*` tag publishes a stable release. Development builds are prereleases so they never
  displace the "Latest release" badge.
- Two constraints learned by building it: Linux must build on the **oldest** supported
  runner (glibc is forward but not backward compatible) and must install the xcb
  libraries at *build* time, or PyInstaller ships a Qt platform plugin that cannot load.
- Saves and progress go to the per-user config dir, never beside the executable. Bench
  files are plain JSON so players can share builds.
- No network dependency at runtime.

---

## 7. Status

Rounds 1–15 are playable, plus a sandbox. The engine, rule checker, workspace, testing
loop, scoring and packaging are built and covered by the test suite.

| Area | State |
|---|---|
| Optics engine | Paraxial ABCD, stops and pupils, folded and branched benches, wavefront budgets, PSF/OTF, partial coherence, Jones, DIC shear |
| Catalog | Nikon CFI60: plan achromats, plan apochromats, non-plan achromats, with per-field provenance |
| Rules | Geometry checks on the live clock, measured checks on Run, tolerances derived from the quarter-wave criterion |
| Rounds | 1–15 and a sandbox, each with a reference build that passes and wrong builds that fail for the stated reason |
| Workspace | Bench view with zoom and ruler, ray overlays, conjugate ribbon, white card, inspector, scorecard, measurements |
| Game shell | Parts budget, star rating, structural diff, atomic progress saves, nine specimens, prerequisite unlocking |
| Packaging | One-file executables for four platforms, published per commit to `main` |

### What remains

- **The catalog datasheet pass.** All 13 objective entries are `verified = false`: their
  specifications were written from secondary knowledge and have not been confirmed
  against Nikon's current datasheets. `--verify-catalog` prints the checklist, and the
  loader refuses any entry marked verified without a source and a date. Nothing in this
  repository should be read as claiming these numbers are datasheet-accurate.
- **The ribbon playtest.** `docs/PLAYTEST.md` is the script. It needs a person who has
  not seen the workspace, which is the one part that cannot be automated.
- **Code signing** for macOS and Windows, so first launch does not need Right-click →
  Open or "Run anyway". A cost decision, not a technical one.
- **A 3D renderer**, deferred by decision 3 and kept possible by the 3D-native bench.

### Risks

| Risk | State |
|---|---|
| Partial-coherence synthesis defensible and fast enough | **Closed.** Abbe source integration, ~0.3 s at 384². The incoherent fallback is unavailable anyway, since decision 4 needs complex amplitude |
| The ribbon makes Köhler legible *to a person* | **Open, and unanswerable from code.** Rows are captioned and ticks named, which is as far as this goes without a subject. The playtest script permits cutting the ribbon if the white card does its job better |
| Catalog numbers a student will check against their own bench | **Partly open.** Provenance tracking, UI disclaimer, checklist and a loader guard are built; the datasheet check itself is not done. The aberration budgets stay pedagogical permanently and by design |
| Frozen-build portability | **Closed by building locally**, not by trusting CI: PyInstaller ≥ 6.22 for numpy 2.4, xcb libraries at build time, oldest supported Linux runner |
| Unsigned binaries | **Open by choice**, stated in the release body and README |
| 3D view | **Deferred by design**, not blocked |
