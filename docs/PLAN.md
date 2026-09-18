# MicroscopeVuilder — Design & Implementation Plan

A distributable Python puzzle game about *building* a working microscope. The player
assembles diascopic (transmitted) and episcopic (reflected/epi) illumination paths plus
an imaging path on a virtual optical bench. Geometry is verified against real optical
physics; progression runs from a simple compound microscope to a modern
infinity-corrected stand.

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

### 2.1 Geometric layer — paraxial + real ray trace

- **Elements**: lamp, collector, field diaphragm, aperture diaphragm, condenser,
  specimen stage, objective, tube lens, mirrors/prisms, dichroic, filters, eyepiece,
  camera sensor, field stop, beamsplitter cube.
- **Representation**: each element is a surface list on a directed bench axis with
  position `z`, clear aperture (semi-diameter), and either a thin-lens focal length or
  a real surface set (radius, thickness, glass index, conic). Start thin-lens +
  ABCD matrices; keep a `Surface` abstraction so a real sequential trace can replace it
  without touching the game layer.
- **ABCD/paraxial pass** gives, cheaply and every frame: image positions, transverse
  and angular magnification, conjugate-plane sets, pupil positions, vignetting limits.
  This drives the live UI.
- **Real ray pass** (Snell at each surface, meridional + skew fans) runs on demand:
  spot diagrams, defocus, spherical/field curvature, chromatic spread using a 3-line
  (F, d, C) dispersion model per glass.
- **Folding**: mirrors and 45° dichroics transform the axis, so an episcopic path is
  modeled as a genuine folded bench rather than a special case. Geometry verification
  works in the unfolded coordinate.

### 2.2 Image-formation layer — what the player sees

- **Diffraction limit**: Abbe `d = λ / (NA_obj + NA_cond)`; incoherent PSF from the
  exit-pupil autocorrelation (OTF), FFT-based convolution of the specimen texture.
- **Aberration → wavefront**: the real-ray OPD at the exit pupil becomes a pupil phase,
  so a badly corrected build produces a genuinely blurred/colored image rather than a
  cosmetic filter. Defocus, coma, astigmatism and lateral color all fall out of this.
- **Illumination → image**: condenser NA sets partial coherence (illumination-pupil
  fill); the field diaphragm image sets the illuminated field edge; a non-conjugate
  lamp produces visible filament structure (the critical-vs-Köhler lesson, shown not
  told).
- **Photometry**: relative irradiance ∝ NA² / M², vignetting from clipped apertures,
  and a simple sensor/eye response with noise so "technically correct but dim" is a
  real failure mode.

### 2.3 Verified geometry — the rule checker

A declarative rule set evaluated on the built bench. Examples:

| Invariant | Check |
|---|---|
| Parfocality / tube length | Objective shoulder-to-image = 160 mm (DIN) or ∞ + tube lens |
| Köhler conjugates | lamp ≡ aperture diaphragm ≡ objective back focal plane ≡ eyepoint |
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
| 12 | Full stand | Combined dia + epi, turret parfocality, DIC/phase (stretch) | All rules green across three objectives |

Rounds 1–4 teach, 5–8 tighten tolerances, 9–12 restructure the bench. A free
**sandbox** unlocks alongside round 5.

---

## 4. Technical architecture

```
microscopevuilder/
  optics/       surfaces, abcd.py, raytrace.py, materials.py, pupil.py, psf.py
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
- Engine is **pure and UI-free**: `Bench -> TraceResult -> RuleReport -> RenderedImage`.
  Everything in `optics/`, `rules/`, `imaging/` is importable and testable headless.
- Determinism: fixed seeds for noise so scores are reproducible.

**Performance**: paraxial + rule pass targets <5 ms (interactive, every drag). Real-ray
and PSF synthesis run on a worker thread with a progress indicator; 512² FFT convolution
is well under a second.

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
| M1 | `optics/` paraxial + ABCD + real trace | Golden tests pass |
| M2 | `imaging/` PSF + convolution + metrics | Airy/Abbe reproduce analytically |
| M3 | Headless round 1–2 solvable via script | Solver test passes |
| M4 | Qt workspace: drag, ray overlay, inspector | Playable round 2 |
| M5 | Testing loop, scorecard, conjugate ribbon | Rounds 1–5 |
| M6 | Illumination depth: Köhler, NA matching | Rounds 3–5 tuned, tolerances validated |
| M7 | Epi + fluorescence, folded benches | Rounds 9–10 |
| M8 | Infinity correction, full stand, sandbox | Rounds 11–12 |
| M9 | Packaging, installers, onboarding, art pass | Signed builds published |

**Highest-risk items, front-loaded**: (a) partial-coherence image synthesis that is both
physically defensible and fast enough — prototype in M2 and be willing to fall back to
an incoherent-PSF approximation with a documented caveat; (b) making conjugate planes
*visible* enough that Köhler is a puzzle rather than a guessing game — prototype the
ribbon UI in M4 and playtest before building rounds on it.

---

## 7. Open questions for the user

1. **Audience**: teaching tool for microscopy students (favor rigor, real
   prescriptions, named manufacturers' conventions) or a puzzle game for general
   players (favor abstraction and tighter feedback loops)? This changes tolerance
   design and how much math is surfaced.
2. **Rigor ceiling**: is a thin-lens + wavefront-aberration model acceptable
   throughout, or should real glass prescriptions and full sequential tracing be the
   target? The former is ~3× less work and plays nearly the same.
3. **2D bench only**, or does the episcopic/fluorescence work need a 3D view?
4. **Contrast techniques** (phase, DIC, polarization) — in scope as rounds, or a later
   expansion?
