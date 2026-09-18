# MicroscopeVuilder — Design & Implementation Plan

A distributable Python puzzle game about *building* a working microscope. The player
assembles diascopic (transmitted) and episcopic (reflected/epi) illumination paths plus
an imaging path on a virtual optical bench. Geometry is verified against real optical
physics; progression runs from a simple compound microscope to a modern
infinity-corrected stand.

---

## 0. Locked decisions (answers to §7)

| # | Decision | Consequence |
|---|---|---|
| 1 | **Audience: graduate students in biomedical sciences.** | Favor rigor. **Nikon CFI60 is the reference system** (200 mm tube lens, 60 mm parfocal, `f = 200/M`); DIN 160 mm appears only in the early finite-tube rounds, and the Olympus 180 mm tube length appears in round 11 as the cross-vendor trap. Real manufacturer conventions (RMS thread, real NA/WD/immersion values), real NA/immersion values, correct nomenclature throughout. Math is surfaced, not hidden: every scorecard shows the governing equation with the player's numbers substituted in. Tolerances are set from what actually matters at the bench (e.g. Nyquist, NA matching), not from difficulty tuning. |
| 2 | **Rigor ceiling: thin lens + wavefront aberrations.** | No sequential real-ray trace, no glass prescriptions, no Snell at surfaces. Elements are ideal thin lenses (or ideal 4f groups) carrying an *assigned* wavefront-aberration budget expressed in Zernike terms. Removes `optics/raytrace.py` and `materials.py`-as-glass-catalog from the plan; adds `optics/wavefront.py`. See §2.2. |
| 3 | **2D bench now, 3D later.** | The bench model is stored in 3D from day one (position + direction vector per element, folds are real rotations); only the *renderer* is 2D. No geometry is flattened into screen space. A 3D view becomes a second renderer against the same model, not a rewrite. |
| 4 | **Contrast techniques are rounds, after Köhler.** | Phase, DIC and polarization become rounds 13–15, gated on a completed Köhler build (round 4). This makes them pedagogically honest — phase contrast is *unteachable* without conjugate planes already understood, since the phase ring lives in the objective back focal plane. Requires the engine to carry complex amplitude, not just intensity (see §2.3). |

---

## 1. Design pillars

1. **Physics is the puzzle.** No scripted "right answer" slots. A configuration wins
   because the ray trace and the generated image actually satisfy the round's spec
   (magnification, NA, field of view, conjugate planes, uniformity).
2. **Two coupled path problems.** Illumination (Köhler/critical, diascopic and
   episcopic) is a first-class puzzle, not decoration. Most real misery in microscopy
   is illumination, and that is where the interesting constraints live.
3. **Legible feedback.** Every failure names the violated optical invariant
   ("field diaphragm is not conjugate to the specimen", "objective NA > condenser NA:
   resolution limited by illumination").
4. **Rounds add one new idea at a time**, and never invalidate what was learned.
5. **Ships as a double-clickable app**, not a `pip install` chore.

---

## 2. Optical model (the core engine)

Two layers, both required.

### 2.1 Geometric layer — paraxial ABCD on a 3D bench

- **Elements**: lamp, collector, field diaphragm, aperture diaphragm, condenser,
  specimen stage, objective, tube lens, mirrors/prisms, dichroic, filters, eyepiece,
  camera sensor, beamsplitter cube, and (rounds 13–15) phase ring, Wollaston prism,
  polarizer/analyzer.
- **Representation**: every element carries a 3D `position` and unit `direction` on the
  bench graph, a clear aperture (semi-diameter), and a 2x2 ABCD matrix. Ideal thin
  lenses and ideal groups only — per decision 2. Mirrors and 45 deg dichroics rotate the
  propagation direction, so an episcopic path is a genuinely folded bench; the trace
  runs in unfolded path-length coordinate `s`, and the 2D renderer projects. The 3D
  storage is what keeps decision 3 cheap.
- **Paraxial pass** runs every frame: system matrix between any two planes, image
  location (where `B = 0`), transverse and angular magnification, cardinal points,
  stop/pupil identification, and marginal/chief ray traces. This drives the whole UI.
- **Stops and pupils**: the aperture stop is found by tracing a probe ray from the axial
  object point and picking the element with the smallest ratio of clear aperture to ray
  height; the field stop likewise from the chief ray. Entrance/exit pupils are the stop
  imaged by the preceding/following subsystems. This is what makes the conjugate-plane
  ribbon (see §3) computable rather than authored.

### 2.2 Wavefront layer — where the optics get hard

Instead of deriving aberrations from surfaces, each element carries an **aberration
budget**: a Zernike coefficient vector over its own pupil, scaled with field height and
NA by the standard Seidel dependences, plus a chromatic term.

- A catalog of realistic components: an **achromat** 20x/0.5 carries residual spherical
  and secondary longitudinal color; an **apochromat** carries far less but costs more
  and has a shorter working distance; a **plan** objective carries little field curvature
  while a non-plan one carries a lot. These numbers are *chosen to be representative of
  real catalog parts* and documented with their source in `assets/components.yaml`.
- The player's build sums contributions into a **total pupil phase** `W(rho, theta, h, lambda)`.
- **Defocus and misalignment are computed, not assigned**: a wrong `z` becomes a real
  defocus coefficient; a tilted fold becomes coma/astigmatism.
- This is the whole reason the game teaches anything: the image degrades because the
  wavefront is wrong, and the player can *open the pupil view* and see which Zernike
  term is dominant and which element contributed it.

### 2.2b Image formation

- **Coherent transfer**: pupil function `P = A * exp(i*2*pi*W/lambda)`, amplitude PSF by FFT,
  incoherent PSF = `|h|^2`, OTF by autocorrelation of `P`.
- **Partial coherence**: condenser-to-objective NA ratio `S = NA_cond / NA_obj` sets the
  illumination pupil fill. Implemented by **Abbe source integration, not Hopkins TCC**:
  each condenser-pupil point contributes one coherent image and the intensities sum.
  Abbe costs `N_source` FFTs and `O(n^2)` memory against the TCC's `O(n^4)`, and the TCC
  only amortizes when many specimens pass through one fixed system — the opposite of
  this game. Abbe is also exact for a spatially incoherent Köhler source rather than an
  approximation of it.
  The verified result: the partially coherent cutoff is `(1 + S) NA / lambda`, so `S = 1`
  reaches the full incoherent `2 NA / lambda` and `S -> 0` gives half that. Two
  corrections to earlier assumptions, both now pinned by tests:
  (a) `S > 1` is *not* "more incoherent" — source points outside the objective pupil
  contribute no undiffracted background, so the build slides toward darkfield and the
  image never converges to the incoherent convolution;
  (b) source shifts are applied by `np.roll`, which wraps, so an oversized source
  silently folds high frequencies back into the passband. It now raises instead.
- **Complex amplitude throughout** (decision 4): the specimen is a complex transmittance
  `t = A * exp(i*phi)`, so a pure phase object is genuinely invisible in brightfield and
  becomes visible only once the player puts a phase ring in the objective back focal
  plane and a matched annulus in the condenser front focal plane. Polarization rides
  along as a Jones vector per ray bundle for rounds 14–15.
- **Diffraction limit** falls out: Abbe `d = lambda / (NA_obj + NA_cond)`, Airy radius
  `0.61 lambda / NA`. Both are golden-test anchors.
- **Photometry**: relative irradiance proportional to NA^2 / M^2, vignetting from clipped
  apertures, sensor/eye response with shot noise, so "correct but too dim to see" is a
  real failure mode.

### 2.3 Verified geometry — the rule checker

A declarative rule set evaluated on the built bench. Examples:

| Invariant | Check |
|---|---|
| Optical tube length | Back focal plane to intermediate image = design L, so `M = L/f`. Measured from the BFP, not the objective (that is `L + f`), and not the mechanical tube length |
| Köhler conjugates | lamp ≡ aperture diaphragm ≡ objective back focal plane ≡ eyepoint |
| Relaxed eye | Intermediate image at the eyepiece front focal plane, so output is collimated |
| Field conjugates | field diaphragm ≡ specimen ≡ intermediate image ≡ retina/sensor |
| NA match | `NA_cond ≈ NA_obj` (0.7–1.0×), immersion medium consistency |
| Aperture sufficiency | no element clips the marginal or chief ray unintentionally |
| Epi separation | excitation and emission bands separated by dichroic + filters |
| Infinity space | specimen at objective front focus; only collimated light between objective and tube lens |
| Sampling | Nyquist at the sensor: pixel ≤ (d·M)/2 |

Each rule returns pass/fail **plus** the measured value, the target, and a one-line
explanation. This is both the win condition and the tutorial.

---

## 3. Game structure

### Workspace
- Side-elevation **optical bench** view: axis, components as recognizable glyphs,
  drag along `z`, snap to rails, rotate folds. Aperture/field diaphragms are draggable
  irises with live diameter readout.
- **Ray overlay**: axial (marginal) rays in one color, field (chief) rays in another —
  the classic teaching diagram, live. Toggle illumination path / imaging path / both.
- **Conjugate-plane ribbon** under the axis: two rows (field set, aperture set) with
  planes drawn as ticks; the puzzle becomes visibly "line up the ticks".
- **Inspector** for the selected element (f, NA, clear aperture, coatings, glass).
- **Parts bin** with a per-round budget — components cost, so the cheap correct
  solution beats the brute-force one.

### Testing loop
1. Choose a **test object** from a library: USAF 1951 target, Siemens star, diatom,
   stained histology section, fluorescent beads, live-cell phase object, mirror-polished
   metal (for epi), Ronchi ruling.
2. **Run** → paraxial check, rule check, real-ray trace, image synthesis.
3. **Scorecard**: rendered image next to the target image, plus measured magnification,
   resolved line pairs, field flatness, evenness of illumination, chromatic error,
   throughput. Star rating from the metrics; failures link back to the offending element.
4. **Diff view** against the reference build for the round, available after one failure.

### Rounds

| # | Round | Introduces | Win condition |
|---|---|---|---|
| 1 | Single lens loupe | Conjugates, `1/f = 1/s + 1/s'`, real vs virtual | Focused 5× image of a ruling |
| 2 | Compound scope | Objective + eyepiece, 160 mm tube, intermediate image | 100× at the eyepoint, in focus |
| 3 | Make it bright | Critical illumination, lamp collector | Adequate irradiance, filament visible (the setup for round 4) |
| 4 | Köhler | Field + aperture diaphragms, four-plane conjugacy | Even field, filament gone, field stop imaged sharply |
| 5 | Resolution | NA, Abbe limit, condenser matching | Resolve group 7 of the USAF target |
| 6 | Color | Achromat vs apochromat, lateral color, filters | Chromatic error under tolerance |
| 7 | Flat field | Field curvature, plan objectives, sensor size | Corner MTF within spec |
| 8 | Camera port | Beamsplitter, C-mount, Nyquist sampling | Sampled, not empty-magnified |
| 9 | Episcopic | Vertical illuminator, epi-brightfield, folded axis | Reflected-light image of opaque metal |
| 10 | Fluorescence | Dichroic, excitation/emission, stray light | Signal/background above threshold |
| 11 | Infinity | Remove tube length, infinity space, tube lens | Same M, and inserting a filter in infinity space doesn't shift focus |
| 12 | Full stand | Combined dia + epi, turret parfocality | All rules green across three objectives |
| 13 | Phase contrast | Phase annulus + ring conjugate to the objective BFP, phase object | Unstained cell visible; halo artifact understood |
| 14 | Polarization | Jones calculus, crossed polars, birefringent specimen | Extinction achieved, birefringent structure resolved |
| 15 | DIC | Wollaston prisms, shear, bias retardation | Pseudo-relief image with correct shear axis |

Rounds 1–4 teach, 5–8 tighten tolerances, 9–12 restructure the bench, and 13–15 are the
contrast techniques — gated on round 4, because a phase ring is meaningless to a player
who does not yet own the objective back focal plane as a concept. A free **sandbox**
unlocks alongside round 5.

---

## 4. Technical architecture

```
microscopevuilder/
  optics/       paraxial.py, wavefront.py, pupil.py, psf.py, coherence.py, jones.py
  bench/        component library, Bench model, folding, serialization (JSON)
  rules/        invariant checks, tolerances, explanations
  imaging/      specimen textures, image synthesis, metrics (MTF, uniformity)
  game/         round definitions, scoring, progression, save files
  ui/           workspace canvas, inspector, scope view, scorecard
  assets/       specimen images, glyphs, fonts
tests/          golden optical cases, rule tests, round-solvability tests
```

**Stack**
- Python 3.11+, `numpy` for everything numeric, `scipy` optional for FFT/optimize.
- **UI: PySide6 (Qt)** — `QGraphicsScene` gives a real scene graph for the bench
  (drag, snap, z-order, hit testing) and `QImage` for the rendered scope view. pygame
  was considered and rejected: it has no widget layer, and this game is 60% inspector
  panels. Qt also packages cleanly.
- Bench model is 3D-native with a 2D renderer (decision 3); `ui/render2d.py` is kept
  behind a `BenchRenderer` protocol so `ui/render3d.py` can be added later.
- Engine is **pure and UI-free**: `Bench -> TraceResult -> RuleReport -> RenderedImage`.
  Everything in `optics/`, `rules/`, `imaging/` is importable and testable headless.
- Determinism: fixed seeds for noise so scores are reproducible.

**Performance**: paraxial + rule pass targets <5 ms (interactive, every drag). PSF and
partially coherent synthesis run on a worker thread with a progress indicator. Measured:
a 384² Abbe integration over ~200 source points takes ~0.3 s, so the testing loop is
responsive but the workspace must not attempt it on every drag — the paraxial pass and
rule report carry the live UI, and image synthesis is triggered by **Run**.

**Validation**: golden tests against textbook cases (thin-lens conjugates, known
Cooke-triplet spot sizes, Airy radius `0.61λ/NA`, Abbe limit) with numeric tolerances.
Where practical, compare a few benches against a published prescription.

---

## 5. Distribution

- Package with **PyInstaller** (one-folder for size, one-file option) per platform;
  CI matrix on GitHub Actions producing macOS `.app`/dmg, Windows `.exe`, Linux
  AppImage or tarball.
- Also publish to PyPI as `microscopevuilder` with a `microscopevuilder` console entry
  point for the `pip` crowd.
- Saves, progress and custom benches in a per-user config dir (`platformdirs`), never
  next to the executable. Bench files are plain JSON so players can share builds.
- No network dependency at runtime.

---

## 6. Milestones

| M | Deliverable | Gate |
|---|---|---|
| M0 | Repo scaffold, CI, test harness | `pytest` green on 3 OSes |
| M1 | `optics/` paraxial ABCD, stops/pupils, 3D folding | Golden tests pass |
| M2 | `optics/` wavefront + PSF/OTF + CFI60 catalog | **Done for the incoherent path**: Airy zero, 84% encircled energy, analytic MTF, quarter-wave Strehl all reproduce |
| M2b | Partial coherence (Abbe source integration) | **Done**: S->0 matches the coherent formula to 1e-12; the `(1+S) NA/lambda` cutoff holds at S = 0, 0.5, 1; a 0.2 rad phase object is invisible in brightfield |
| M3 | Bench model, rule engine, rounds 1–2, headless runner | **Done**: `python -m microscopevuilder --round N`; reference builds pass and wrong builds fail for the stated reason |
| M4 | Qt workspace: drag, ray overlay, inspector, conjugate ribbon, Run loop | **Done**: `python -m microscopevuilder --ui --round N`; headless Qt tests cover drag, retrace, scorecard and worker thread |
| M5 | Testing loop, scorecard, conjugate ribbon | Rounds 1–5 |
| M6 | Illumination depth: Köhler, NA matching | Rounds 3–5 tuned, tolerances validated |
| M7 | Epi + fluorescence, folded benches | Rounds 9–10 |
| M8 | Infinity correction, full stand, sandbox | Rounds 11–12 |
| M8b | Complex-amplitude contrast: phase, polarization, DIC | Rounds 13–15 |
| M9 | Packaging, installers, onboarding, art pass | Signed builds published |

**Highest-risk items, front-loaded**: (a) partial-coherence image synthesis (Hopkins TCC)
that is both physically defensible and fast enough — prototype in M2; the documented
fallback is an incoherent-PSF approximation, but note that decision 4 makes this harder
to give up, since phase contrast *requires* complex amplitude; (b) making conjugate
planes *visible* enough that Köhler is a puzzle rather than a guessing game — prototype
the ribbon UI in M4 and playtest before building rounds on it; (c) sourcing defensible
aberration budgets for the component catalog (§2.2) — these are the numbers a graduate
student will check against their own bench, so each gets a cited source.

---

## 7. Open questions — RESOLVED

All four answered; see §0 for the decisions and their consequences. New questions that
arise during M1–M2 get appended here rather than blocking work.
