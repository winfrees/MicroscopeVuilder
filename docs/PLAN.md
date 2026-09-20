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

### Placement and grading tolerance

The derived tolerances of M6 turned out to be **finer than the input resolution**.
On round 11 the depth of focus is 0.19 mm across a 380 mm bench: fitted to a
760-pixel view that is 0.50 mm per pixel, so a single pixel of drag overshot the
tolerance by two and a half times. The physics was right and the game was
unplayable.

Fixed on both sides:

- **Input**: the view zooms (a pixel at 10x is 0.05 mm), a ruler runs under the
  axis with tick density following the zoom, dragging snaps to a selectable grid
  *and* to optical planes, and the inspector takes a typed position. The default
  grid is 0.1 mm — most round targets land on a tenth (17.6, 193.6, 172.6) and a
  1 mm grid would step straight over them.
- **Grading**: a `TolerancePolicy` grades positions at the larger of the physical
  tolerance and 5% of the position. Relaxing never tightens — near the origin 5% is
  a fraction of a millimetre. Every graded row names the physical tolerance
  whatever the policy, so a student is never left thinking 5% is what a bench
  accepts.

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

### The white card

A diagnostic surface the player can drop anywhere on the axis, mirroring the one
habit that answers most beginner questions at a real bench: hold a card in the beam
and look. It reports what would land there — a sharp image and its magnification, an
evenly filled pupil disc, or a blur with the distance to the nearest image plane.

The card implements the classic diagnostic directly: **the marginal ray crossing the
axis marks an image (field) plane; the chief ray crossing marks a pupil (aperture)
plane.** It is a *probe*, excluded from the trace, from stop-finding and from the
rules, so holding one up never changes the answer it is being used to measure. This
is how conjugate planes become tangible before round 4 asks the player to reason
about four of them at once.

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
| 3 | Make it bright | Critical illumination, collected NA, lamp collector | Lamp imaged on the specimen AND enough light gathered. Two collector positions focus correctly (the near and far conjugates); only the near one is bright enough, since irradiance goes as NA². Passes *with* a warning that the filament is visible |
| 4 | Köhler | Field + aperture diaphragms, four-plane conjugacy | Even field, filament gone, field stop imaged sharply |
| 5 | Resolution | NA, Abbe limit, condenser matching | Resolve group 7 of the USAF target |
| 6 | Color | Achromat vs apochromat, lateral color, filters | Chromatic error under tolerance |
| 7 | Flat field | Field curvature, plan objectives, sensor size | Corner MTF within spec |
| 8 | Camera port | Beamsplitter, C-mount, Nyquist sampling | Sampled, not empty-magnified |
| 9 | Episcopic | Vertical illuminator, epi-brightfield, branched axis, objective as its own condenser | Epi-Köhler: lamp on the objective back focal plane, field diaphragm on the specimen |
| 10 | Fluorescence | Dichroic, Stokes shift, excitation/emission bands, bleedthrough | Correct cube for the dye, and bleedthrough below threshold |
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

- Package with **PyInstaller one-file**, per platform — an executable, not an
  installer. The audience is students on lab machines who may not have rights to run
  an installer or to `pip install`, so the deliverable is a single downloadable file.
  CI matrix on GitHub Actions (Linux, Windows, macOS Intel and Apple silicon).
  **Every commit to `main` publishes a development build** as a prerelease tagged
  `build-N`, so the current state is always downloadable; a `v*` tag publishes a
  stable release. Development builds are prereleases specifically so they never
  displace the "Latest release" badge that points at a real version. Two constraints learned by building it:
  Linux builds must run on the **oldest** supported runner (glibc is forward but not
  backward compatible) and must install the xcb development libraries at *build*
  time, or PyInstaller cannot resolve the Qt platform plugin's dependencies and
  silently ships a binary that will not start.
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
| M5 | Illumination rounds, conjugacy rules, illumination ray path | **Done**: rounds 3–5 with the four-plane Köhler check, throughput, condenser NA matching |
| M6 | Derived tolerances | **Done**: every focus tolerance traces to the Rayleigh quarter-wave criterion and is validated against the PSF engine; pupil conjugates use a separate geometric criterion |
| M7 | Epi + fluorescence on **branched** benches | **Done**: rounds 9–10; the bench gained arms, since an epi path is a second path, not a folded one |
| M8 | Infinity correction, parfocal turret, sandbox | **Done**: rounds 11–12; a glass-plate element makes the infinity-space argument demonstrable, and the bench gained enabled/disabled elements so a turret traces one objective |
| M8b | Complex-amplitude contrast: phase, polarization, DIC | **Done**: rounds 13–15. Contrast techniques are expressed as pupil and source engineering (`optics/jones.py`, annular sources, phase-ring pupil masks, Wollaston shear), so a misaligned ring fails for the reason it would on a bench |
| M9 | Packaging: one-file executables, published to Releases | **Done**: PyInstaller one-file per platform, built and smoke-tested in CI, attached to a GitHub Release on `v*` tags. Unsigned — noted in the release body |
| M10 | Close the testing loop on the real bench | **Done**: `imaging/build_optics.py` resolves a bench into its optical state, `imaging/synthesis.py` renders through it, `imaging/metrics.py` measures the result. An achromat and an apochromat now render differently, and the catalog's chromatic term is anchored to the textbook `f/2000` |
| M11 | Rounds 6–8: colour, flat field, camera port | **Done**: graded by a second *measured* pass that renders, alongside the live geometry pass. Non-plan objectives added to the catalog so round 7 has something to contrast against |
| M12 | Game shell: parts budget, scoring, progress, specimen library | **Done**: budgets enforced, three-star rating, structural diff after a failure, atomic progress saves in the user config dir, nine specimens, and rounds that unlock on their prerequisite |
| M13 | Verification and polish | **Partly done**: verification *tooling* built (`--verify-catalog`, and a loader that refuses `verified = true` without a citation), onboarding and ribbon labelling done, playtest script written. The datasheet check itself and the playtest are human work that remains — see below |

**Recommended order: M10 → M11 → M8b → M12 → M13.** M10 unblocks M11 (see below) and
supplies the metrics M12's scoring depends on. M8b sits after M11 because phase contrast
and DIC are the most demanding consumers of the synthesis path M10 builds.

### M10 — closing the testing loop (done)

`imaging/build_optics.py` resolves a bench into the optical state synthesis needs:
NA and magnification from the trace, the aberration budget from the objective's
catalog entry scaled to the field height and aperture in use, **defocus computed**
from where the image actually lands versus where the detector is, and photometry.
`imaging/synthesis.py` renders through that resolved pupil; `imaging/metrics.py`
measures the result. The workspace's Run button now renders the player's own bench
and reports what was measured off the picture.

Three things the work changed about the design:

- **The aberration model moved to quotable units.** Storing field curvature and
  secondary spectrum as wavefront coefficients hid the fact that both scale as `NA²`
  when converted from a focus error. They are now stored as *distances*: field
  curvature as a longitudinal sag in microns, secondary spectrum as a fraction of
  focal length. The achromat figure is the textbook `f/2000`, which a student can
  look up — far better provenance than "illustrative". It also makes the physics
  right: a 4× (f = 50 mm) suffers ten times the chromatic focus shift of a 40×.
- **Plan sags were an order of magnitude too large.** "Plan" means the sag sits
  *inside* the depth of focus, which at NA 0.75 is only 0.49 µm — so sub-micron, not
  "a few microns" as first written.
- **An objective that declares no grade is treated as ideal**, not matched to the
  nearest catalog entry by aperture. Matching on NA alone silently handed an
  uncharacterised objective an achromat's secondary spectrum, and it would then have
  imaged badly in blue for reasons nothing on the bench explained.

Measurement note: the SNR estimator is band-limited rather than region-based. The
optics band-limit the image, so power beyond `2 NA / λ` is noise. A dark-region
estimator reported *infinite* SNR on a photon-starved image (dark pixels quantize to
zero, so their spread is zero), and an adjacent-pixel-difference estimator counted a
grating's own modulation as noise, reporting SNR 6 on an image with 4500 photons per
pixel.

### M11 — measured rounds (done)

Rounds 6–8 introduced a second grading pass. `Round.evaluate` stays on the live
clock (ray geometry, microseconds, runs on every drag); `Round.measure` renders and
therefore runs on **Run**, the same two-clock split the workspace already used for
image synthesis. Rounds that need only geometry leave `measure` unset and pay
nothing. Headless, `python -m microscopevuilder --round 7` runs both, and
`--no-measure` grades on geometry alone.

Thresholds stay derived. The colour criterion is "the F-to-C band lands inside the
depth of focus", which is what colour correction *means* rather than a tolerance
someone picked; flatness is a measured corner-to-centre MTF ratio; sampling is
Nyquist at the sensor, with a warning band above it for empty magnification.

Two measurement lessons:

- **A flatness ratio measured near the diffraction cutoff is noise.** At a 1.0 µm
  period against a 0.85 µm cutoff, a *flat* plan objective read 0.57 and a badly
  curved one read 0.91 — the ratio inverts, because both numbers are already near
  zero. The test period moved to mid-band, and the rule now refuses to answer when
  centre modulation is below 5% rather than answering wrongly.
- **Round 7 needed non-plan objectives**, which the catalog did not have; every
  entry was a plan design. Three `achromat`-grade entries were added with Petzval
  sags of 20–30 µm, 7–31× the depth of focus against the plan designs' 0–3×.

### M8b — contrast techniques (done)

All three are pupil and polarization engineering rather than rendering modes, so
they are driven by components the player places. `imaging/techniques.py` translates
whatever is on the bench into a source, a pupil mask and a field transform.

- **Phase contrast** is an annular source plus a complex pupil mask over the
  matching ring. Both of the ring's jobs are necessary and tested as such: the
  quarter-wave shift alone, and the attenuation alone, each produce less than half
  the contrast of the two together. A ring with no annulus does nothing and says so.
- **Polarization** carries a Jones state. A birefringent specimen is stored as
  retardance and azimuth maps, *not* as transmittance, so what it transmits is
  decided by the polarizer and analyzer actually fitted — rotate the analyzer and
  the background lifts, exactly as Malus predicts. With no polars fitted it is
  completely invisible, which is the reason the round needs them.
- **DIC** shears the field by a fraction of the resolution limit and recombines
  with a bias. Zero bias gives no relief (a gradient and its opposite look
  identical) and the relief follows the shear axis exactly, which is why you rotate
  the specimen and not the prism.

### M12 — game shell (done)

- **Parts budget** enforced, counting everything except probes, stops and the lamp:
  a white card is a measuring instrument, and charging a slot for it would penalise
  the player for looking. A test asserts every round's own reference build fits its
  budget, so no round is unwinnable.
- **Three-star rating.** One star for solving it; the second for no outstanding
  warnings; the third for staying inside budget. The second star is the interesting
  one — it marks the difference between "it worked" and "a demonstrator would still
  raise an eyebrow", which is what a WARN has meant since M5.
- **Structural diff** after a failure, offered rather than forced. It says what
  differs — missing, extra, moved, changed — and deliberately not what to do about
  it, so working out why the difference matters is still the round.
- **Progress** saves atomically (write to a sibling, then rename) into the user
  config dir, never beside the executable: a one-file binary on a shared lab machine
  may live somewhere read-only, and two people should not overwrite each other. Best
  result is kept rather than latest, a corrupt save starts a fresh game instead of
  refusing to open, and a save from a newer format is refused rather than misread.
- **Nine specimens**, including the birefringent one. Rounds unlock on a
  prerequisite, with the contrast techniques gated on Köhler rather than on the
  round before them.

### M13 — verification and polish (partly done)

Done:

- **`--verify-catalog`** prints a per-field checklist of what needs confirming
  against which product, so the datasheet pass is mechanical rather than research.
- **The loader refuses a catalog** in which any entry is marked `verified = true`
  without a `source` and a `checked_on` date. Without that guard the flag is just an
  assertion, and the provenance scheme rests on it meaning something a reader can
  follow up.
- **The workspace says so.** A permanent status-bar banner states that catalog
  specifications are unverified and that aberration budgets are pedagogical by
  design — where the audience will actually see it, not buried in a file header.
- **Onboarding** (F1) explains the ray colours, the two ribbon rows, the white card
  and the Run loop.
- **Ribbon labelling**: rows are captioned and every tick is named, with its
  position on hover.

**Not done, and not doable from here:**

- **The datasheet check itself.** Every one of the 13 objective entries is still
  `verified = false`: the specifications were written from secondary knowledge and
  have not been confirmed against Nikon's current datasheets. The tooling makes that
  pass straightforward; it does not perform it, and nothing in this repository
  should be read as claiming these numbers are datasheet-accurate.
- **The ribbon playtest.** `docs/PLAYTEST.md` is the script: who to test with, what
  to watch for, and a decision rule that includes cutting the ribbon if the white
  card turns out to do its job better. It needs a person who has not seen the
  workspace, which is the one thing that cannot be automated.

### Risk register

| Risk | State |
|---|---|
| Partial-coherence synthesis defensible and fast enough | **Closed.** Abbe source integration, not Hopkins TCC; ~0.3 s at 384². The incoherent-PSF fallback is now unavailable anyway, since decision 4 needs complex amplitude |
| Conjugate ribbon makes Köhler legible *to a person* | **Open, and unanswerable from code.** Rows are now captioned and ticks named, which is the most that can be done without a subject. `docs/PLAYTEST.md` is the script, including a decision rule that permits cutting the ribbon if the white card does its job better |
| Aberration budgets a student will check against their own bench | **Partly open.** Provenance tracking, the UI disclaimer, a verification checklist and a loader guard that refuses an uncited `verified = true` are all built. All 13 catalog entries remain `verified = false`: the specifications were written from secondary knowledge and have not been confirmed against a datasheet. The Zernike budgets stay `pedagogical` permanently and by design |
| Frozen-build portability | **Closed by building locally**: PyInstaller ≥ 6.22 for numpy 2.4, xcb libraries present at build time, oldest-supported Linux runner |
| Unsigned macOS and Windows binaries | **Open by choice.** First launch needs Right-click → Open or "Run anyway"; stated in the release body and README. Signing needs an Apple Developer ID and a Windows certificate — a cost decision, not a technical one |
| 3D view | **Deferred by design** (decision 3). The bench is 3D-native and the renderer sits behind a protocol, so this stays a second renderer rather than a rewrite |

---

## 7. Open questions — RESOLVED

All four answered; see §0 for the decisions and their consequences. New questions that
arise during M1–M2 get appended here rather than blocking work.
