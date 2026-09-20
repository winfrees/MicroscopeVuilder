"""The optical invariants themselves.

Each check takes a bench (plus whatever context it needs) and returns one
:class:`RuleResult`. They are deliberately independent so a round can compose the
subset it cares about, and so a failure names one idea rather than a build.
"""

from __future__ import annotations

from ..bench.bench import Bench
from ..optics.paraxial import Ray
from ..optics.psf import abbe_resolution_um, nyquist_pixel_size_um, rayleigh_resolution_um
from .base import RuleResult, Status, tolerance_result


def check_image_lands_on_detector(
    bench: Bench,
    s_object: float,
    detector_name: str,
    tolerance_mm: float = 0.05,
    note: str = "",
) -> RuleResult:
    """Is the image actually in focus where the player put the sensor or eyepoint?"""
    system = bench.to_paraxial()
    detector = bench.get(detector_name)
    s_img = system.image_plane(s_object, search_to=detector.s + 1.0)

    if s_img is None:
        return RuleResult(
            name="Focus",
            status=Status.FAIL,
            summary="the light leaving this system is collimated, so no image forms",
            equation="B = 0 has no solution: the outgoing bundle is parallel",
            culprit=detector_name,
            remedy=(
                "an infinity space needs a tube lens to bring it to an image; "
                "add one, or move the specimen off the objective's front focal plane"
            ),
        )

    return tolerance_result(
        name="Focus",
        measured=s_img,
        target=detector.s,
        tolerance=tolerance_mm,
        units="mm",
        equation=f"image forms at s = {s_img:.3f} mm; {detector_name} sits at {detector.s:.3f} mm",
        culprit=detector_name,
        remedy=(
            f"move {detector_name} to s = {s_img:.3f} mm, or refocus by moving the "
            f"specimen by roughly {(detector.s - s_img) / 100:.4f} mm"
        ),
        relative=False,
        note=note,
    )


def check_magnification(
    bench: Bench,
    s_object: float,
    detector_name: str,
    target: float,
    tolerance: float = 0.05,
) -> RuleResult:
    """Transverse magnification at the detector, signed convention ignored."""
    system = bench.to_paraxial()
    detector = bench.get(detector_name)
    m = abs(system.magnification(s_object, detector.s))
    return tolerance_result(
        name="Magnification",
        measured=m,
        target=target,
        tolerance=tolerance,
        units="x",
        equation=f"M = A of the object-to-image matrix = {m:.4g}",
        culprit=detector_name,
        remedy="adjust the lens focal length or the conjugate distances",
    )


def check_no_unintended_clipping(
    bench: Bench, s_object: float, field_height_mm: float = 0.0
) -> RuleResult:
    """Does any element other than the intended stop cut into the chief-ray bundle?

    Vignetting that the player did not ask for is the classic invisible failure:
    the image is dim at the edges and nobody can say why.
    """
    system = bench.to_paraxial()
    stop = system.aperture_stop(s_object)
    marginal = system.marginal_ray(s_object)
    if stop is None or marginal is None:
        return RuleResult(
            name="Clear aperture",
            status=Status.NOT_APPLICABLE,
            summary="no aperture-limited bundle to check",
        )

    chief = system.chief_ray(s_object, field_height_mm)
    if chief is None:
        return RuleResult(
            name="Clear aperture",
            status=Status.NOT_APPLICABLE,
            summary="no chief ray: the object plane is imaged onto the aperture stop",
        )
    worst_name, worst_fill = None, 0.0
    for element, ray in system.trace_profile(marginal, s_object):
        if element.name == stop.name:
            continue
        _, chief_ray = next(
            (e, r) for e, r in system.trace_profile(chief, s_object) if e.name == element.name
        )
        reach = abs(ray.y) + abs(chief_ray.y)
        fill = reach / element.semi_diameter
        if fill > worst_fill:
            worst_name, worst_fill = element.name, fill

    if worst_name is None:
        return RuleResult(
            name="Clear aperture",
            status=Status.PASS,
            summary="only the aperture stop limits the bundle",
        )

    status = Status.PASS if worst_fill <= 1.0 else Status.FAIL
    return RuleResult(
        name="Clear aperture",
        status=status,
        summary=(
            f"the bundle fills {worst_fill * 100:.0f}% of {worst_name}"
            + ("" if status is Status.PASS else ", so it is vignetting the field")
        ),
        equation=f"|y_marginal| + |y_chief| = {worst_fill:.2f} x semi-diameter at {worst_name}",
        measured=worst_fill,
        target=1.0,
        culprit=None if status is Status.PASS else worst_name,
        remedy=f"use a larger clear aperture at {worst_name}, or reduce the field height",
    )


def check_resolution(
    na_objective: float,
    na_condenser: float,
    wavelength_um: float,
    required_um: float,
) -> RuleResult:
    """Abbe limit against what the round asks the player to resolve."""
    d = abbe_resolution_um(wavelength_um, na_objective, na_condenser)
    status = Status.PASS if d <= required_um else Status.FAIL
    return RuleResult(
        name="Resolution",
        status=status,
        summary=f"resolves {d:.3f} um; the target needs {required_um:.3f} um",
        equation=(
            f"d = lambda / (NA_obj + NA_cond) = {wavelength_um:.4f} / "
            f"({na_objective:.2f} + {na_condenser:.2f}) = {d:.3f} um"
        ),
        measured=d,
        target=required_um,
        units="um",
        remedy=(
            "raise the objective NA, or open the condenser: closing it down to "
            "NA_cond = 0 costs a factor of two in resolution"
        ),
    )


def check_condenser_na_match(
    na_objective: float, na_condenser: float, low: float = 0.7, high: float = 1.0
) -> RuleResult:
    """Is the condenser matched to the objective?

    Conventionally NA_cond is set to 70-100% of NA_obj: matched gives full
    resolution, slightly under trades a little of it for contrast, and far under
    throws away half the resolution the objective was paid for.
    """
    if na_objective <= 0:
        return RuleResult("Condenser match", Status.NOT_APPLICABLE, "no objective")
    ratio = na_condenser / na_objective
    if low <= ratio <= high:
        status = Status.PASS
    elif ratio > high or ratio >= low * 0.75:
        status = Status.WARN
    else:
        status = Status.FAIL
    note = {
        Status.PASS: "matched",
        Status.WARN: (
            "over-open: the extra illumination NA adds glare without resolution"
            if ratio > high
            else "a little closed: some resolution traded for contrast"
        ),
        Status.FAIL: "closed far down: you are throwing away resolution for contrast",
    }[status]
    return RuleResult(
        name="Condenser match",
        status=status,
        summary=f"NA_cond / NA_obj = {ratio:.2f} -- {note}",
        equation=f"{na_condenser:.2f} / {na_objective:.2f} = {ratio:.2f}, want {low:.2f}-{high:.2f}",
        measured=ratio,
        target=(low + high) / 2,
        culprit="condenser",
        remedy="open the condenser aperture diaphragm until it matches the objective",
    )


def check_sampling(
    wavelength_um: float, na: float, magnification: float, pixel_um: float
) -> RuleResult:
    """Nyquist at the sensor: did the camera keep what the optics resolved?"""
    limit = nyquist_pixel_size_um(wavelength_um, na, magnification)
    status = Status.PASS if pixel_um <= limit else Status.FAIL
    return RuleResult(
        name="Sampling",
        status=status,
        summary=(
            f"{pixel_um:.2f} um pixels against a {limit:.2f} um Nyquist limit"
            + ("" if status is Status.PASS else " -- undersampled, detail is being discarded")
        ),
        equation=(
            f"pixel <= 0.61 lambda M / (2 NA) = 0.61 x {wavelength_um:.4f} x "
            f"{magnification:.0f} / (2 x {na:.2f}) = {limit:.2f} um"
        ),
        measured=pixel_um,
        target=limit,
        units="um",
        culprit="camera",
        remedy=(
            "use smaller pixels or add magnification before the sensor; note that "
            "adding magnification costs field of view and light"
        ),
    )


def check_optical_tube_length(
    bench: Bench,
    objective_name: str,
    image_name: str,
    target_mm: float = 160.0,
    tolerance: float = 0.02,
) -> RuleResult:
    """Finite-tube convention: the objective is corrected for one tube length.

    The quantity that sets the objective's magnification is the **optical** tube
    length: back focal plane to intermediate image, ``M = L / f``. It is not the
    objective-to-image distance (that is ``L + f``), and it is not the mechanical
    tube length either -- DIN's 160 mm is a shoulder-to-eyepiece-seat dimension, and
    the optical tube length inside it is shorter. Getting this wrong is worth a
    full magnification step at 10x, which is why the rule measures it explicitly.
    """
    objective = bench.get(objective_name)
    image = bench.get(image_name)
    f = objective.focal_length_mm or 0.0
    length = image.s - (objective.s + f)
    return tolerance_result(
        name="Optical tube length",
        measured=length,
        target=target_mm,
        tolerance=tolerance,
        units="mm",
        equation=(
            f"L = {image_name} - (back focal plane of {objective_name}) = "
            f"{image.s:.2f} - ({objective.s:.2f} + {f:.2f}) = {length:.2f} mm; "
            f"M = L / f = {length / f if f else float('nan'):.2f}x"
        ),
        culprit=objective_name,
        remedy=(
            "a finite-tube objective is only corrected at its design tube length; "
            "move the intermediate image, or switch to an infinity system"
        ),
    )


def check_infinity_space(bench: Bench, s_object: float, objective_name: str) -> RuleResult:
    """Is the space behind the objective genuinely collimated?

    This is what lets filters and dichroics be inserted without shifting focus,
    and it is the defining property the player has to establish in round 11.
    """
    system = bench.to_paraxial()
    objective = bench.get(objective_name)
    out = system.trace(Ray(0.0, 1e-3), s_object, objective.s + 1e-6)
    residual = abs(out.u) / 1e-3

    status = Status.PASS if residual < 0.02 else Status.FAIL
    return RuleResult(
        name="Infinity space",
        status=status,
        summary=(
            "the space behind the objective is collimated"
            if status is Status.PASS
            else f"the bundle behind the objective still converges ({residual:.1%} of input angle)"
        ),
        equation=(
            f"u_out / u_in = {residual:.4f}; collimated means 0 "
            f"(specimen exactly at the front focal plane)"
        ),
        measured=residual,
        target=0.0,
        culprit=objective_name,
        remedy=(
            f"place the specimen at the objective's front focal plane, "
            f"s = {objective.s - (objective.focal_length_mm or 0):.3f} mm"
        ),
    )


def check_relaxed_eye(bench: Bench, image_name: str, eyepiece_name: str) -> RuleResult:
    """Is the intermediate image at the eyepiece's front focal plane?

    If it is, the eyepiece sends a collimated bundle and the eye focuses at
    infinity -- what "relaxed" means, and why a correctly built scope is not
    tiring. If it is not, the user accommodates to make up the difference, which
    works (so nothing looks obviously wrong) and causes eyestrain over an hour.
    A rule is the only way this becomes visible in a game.
    """
    image = bench.get(image_name)
    eyepiece = bench.get(eyepiece_name)
    f_eye = eyepiece.focal_length_mm or 0.0
    separation = eyepiece.s - image.s

    return tolerance_result(
        name="Relaxed eye",
        measured=separation,
        target=f_eye,
        tolerance=0.02,
        units="mm",
        equation=(
            f"{eyepiece_name} - {image_name} = {separation:.2f} mm; the eyepiece's "
            f"front focal length is {f_eye:.2f} mm, so collimated output needs them equal"
        ),
        culprit=eyepiece_name,
        remedy=(
            f"move {eyepiece_name} to s = {image.s + f_eye:.2f} mm; further out and "
            "the eye must accommodate, which is invisible at first and tiring by lunchtime"
        ),
    )


def _conjugate_within_horizon(system, s_from: float, s_to: float, horizon: float) -> float | None:
    """The conjugate of ``s_from`` before ``s_to``, or None if effectively at infinity.

    A bundle that is *almost* collimated solves to a conjugate hundreds of metres
    away. That is arithmetically true and physically meaningless on a bench, and
    printing it ("images 176486 mm away") reads as a bug. Anything past the horizon
    is reported as collimated, which is what it is.
    """
    s_img = system.image_plane(s_from, search_to=s_to + 1e-6)
    if s_img is None or abs(s_img) > horizon:
        return None
    return s_img


def check_conjugate(
    bench: Bench,
    s_from: float,
    to_name: str,
    label: str,
    tolerance_mm: float = 1.0,
    from_label: str = "source",
) -> RuleResult:
    """Does the plane at ``s_from`` image onto ``to_name``?

    The workhorse of Koehler alignment. Everything about the four-plane conjugacy
    reduces to asking this question four times, so the rule reports the miss
    distance in millimetres rather than a bare verdict -- a player 3 mm out needs
    to know which way and how far.
    """
    target = bench.get(to_name)
    system = bench.to_paraxial()
    s_img = system.image_plane(s_from, search_to=target.s + 1e-6)

    if s_img is None:
        return RuleResult(
            name=label,
            status=Status.FAIL,
            summary=f"{from_label} is not imaged anywhere: the bundle leaves collimated",
            equation=f"no finite conjugate of s = {s_from:.2f} mm before {to_name}",
            culprit=to_name,
            remedy="add or move a lens so this plane is actually imaged",
        )

    return tolerance_result(
        name=label,
        measured=s_img,
        target=target.s,
        tolerance=tolerance_mm,
        units="mm",
        equation=(
            f"the conjugate of {from_label} (s = {s_from:.2f} mm) lands at "
            f"s = {s_img:.2f} mm; {to_name} is at s = {target.s:.2f} mm"
        ),
        culprit=to_name,
        remedy=f"move {to_name} to s = {s_img:.2f} mm, or refocus the lens before it",
        relative=False,
    )


def check_conjugate_to_plane(
    bench: Bench,
    s_from: float,
    s_target: float,
    label: str,
    target_label: str,
    tolerance_mm: float = 1.0,
    from_label: str = "source",
) -> RuleResult:
    """Conjugacy against a bare plane rather than a named element.

    The objective's back focal plane is not a component you can place -- it is a
    position derived from the objective -- so the Koehler aperture-set check needs
    to compare against a coordinate. Passing a loose tolerance against the
    objective's own position instead would let a build off by a full focal length
    pass, which is the whole quantity in question.
    """
    system = bench.to_paraxial()
    s_img = system.image_plane(s_from, search_to=s_target + 1e-6)

    if s_img is None:
        return RuleResult(
            name=label,
            status=Status.FAIL,
            summary=f"{from_label} is not imaged onto {target_label}: the bundle leaves collimated",
            equation=f"no finite conjugate of s = {s_from:.2f} mm before {target_label}",
            remedy="refocus the lens ahead of this plane",
        )

    return tolerance_result(
        name=label,
        measured=s_img,
        target=s_target,
        tolerance=tolerance_mm,
        units="mm",
        equation=(
            f"the conjugate of {from_label} (s = {s_from:.2f} mm) lands at "
            f"s = {s_img:.2f} mm; {target_label} is at s = {s_target:.2f} mm"
        ),
        remedy=f"adjust so the conjugate falls on {target_label} at s = {s_target:.2f} mm",
        relative=False,
    )


def check_not_conjugate(
    bench: Bench,
    s_from: float,
    to_name: str,
    label: str,
    minimum_mm: float = 5.0,
    from_label: str = "source",
) -> RuleResult:
    """Does the plane at ``s_from`` stay *off* ``to_name``?

    The half of Koehler that is easy to forget: the lamp filament must NOT be
    imaged onto the specimen. Critical illumination puts it there and you see the
    filament across your field; Koehler deliberately pushes it to the aperture
    plane instead.
    """
    target = bench.get(to_name)
    system = bench.to_paraxial()
    horizon = max(bench.extent() * 10.0, 1000.0)
    s_img = _conjugate_within_horizon(system, s_from, target.s, horizon)

    if s_img is None:
        return RuleResult(
            name=label,
            status=Status.PASS,
            summary=(
                f"{from_label} is collimated at {to_name}: every point of it fills the "
                "whole field, which is exactly what Köhler is for"
            ),
            equation=(
                f"the conjugate of s = {s_from:.2f} mm lies beyond the bench "
                f"(> {horizon:.0f} mm), i.e. the bundle is collimated"
            ),
        )

    miss = abs(s_img - target.s)
    status = Status.PASS if miss >= minimum_mm else Status.FAIL
    return RuleResult(
        name=label,
        status=status,
        summary=(
            f"{from_label} images {miss:.1f} mm away from {to_name}"
            if status is Status.PASS
            else f"{from_label} is imaged onto {to_name} (only {miss:.2f} mm off) -- "
            "its structure will be visible in the field"
        ),
        equation=(
            f"conjugate of {from_label} at s = {s_img:.2f} mm vs {to_name} at "
            f"s = {target.s:.2f} mm; need at least {minimum_mm:.0f} mm of separation"
        ),
        measured=miss,
        target=minimum_mm,
        units="mm",
        culprit="collector",
        remedy=(
            "refocus the collector so the lamp images onto the aperture diaphragm "
            "instead of onto the specimen -- that is the difference between critical "
            "and Koehler illumination"
        ),
    )


def check_illumination_uniformity(
    bench: Bench,
    lamp_name: str,
    specimen_name: str,
    condenser_name: str,
    advisory: bool = False,
) -> RuleResult:
    """Is the field evenly lit?

    In Koehler each lamp point fills the whole field, so filament structure averages
    away. The measurable proxy is how far the lamp's image is from the specimen,
    scaled by the condenser focal length: at the specimen you see the filament, one
    focal length away you see nothing of it.
    """
    lamp = bench.get(lamp_name)
    specimen = bench.get(specimen_name)
    condenser = bench.get(condenser_name)
    system = bench.to_paraxial()
    horizon = max(bench.extent() * 10.0, 1000.0)
    s_img = _conjugate_within_horizon(system, lamp.s, specimen.s, horizon)

    if s_img is None:
        return RuleResult(
            name="Field uniformity",
            status=Status.PASS,
            summary="even field: the lamp is collimated at the specimen, so no filament structure",
            equation=(
                f"the lamp's conjugate lies beyond the bench (> {horizon:.0f} mm): "
                "each filament point illuminates the entire field"
            ),
        )

    f_cond = condenser.focal_length_mm or 1.0
    defocus_in_focal_lengths = abs(s_img - specimen.s) / f_cond
    status = (
        Status.PASS if defocus_in_focal_lengths >= 0.5
        else Status.WARN if defocus_in_focal_lengths >= 0.2
        else Status.FAIL
    )
    if advisory and status is Status.FAIL:
        status = Status.WARN
    return RuleResult(
        name="Field uniformity",
        status=status,
        summary=(
            "even field: no filament structure visible"
            if status is Status.PASS
            else (
                "the filament is imaged onto the specimen and will be visible across "
                "the field -- expected for critical illumination"
                if advisory
                else f"the filament is {'faintly ' if status is Status.WARN else ''}"
                "visible across the field"
            )
        ),
        equation=(
            f"|lamp image - specimen| / f_condenser = "
            f"|{s_img:.2f} - {specimen.s:.2f}| / {f_cond:.2f} = "
            f"{defocus_in_focal_lengths:.2f} focal lengths"
        ),
        measured=defocus_in_focal_lengths,
        target=0.5,
        culprit="collector",
        remedy="move the collector so the lamp images onto the aperture diaphragm",
    )


def condenser_na(bench: Bench, diaphragm_name: str, condenser_name: str, n: float = 1.0) -> float:
    """Illumination NA set by the aperture diaphragm radius and condenser focal length.

    Paraxially ``NA = n * r / f``. The diaphragm sits at the condenser's front focal
    plane in a Koehler build, so its radius maps directly onto illumination angle --
    which is why that one knob controls resolution and contrast together.
    """
    diaphragm = bench.get(diaphragm_name)
    condenser = bench.get(condenser_name)
    f = condenser.focal_length_mm or 1.0
    return n * diaphragm.semi_diameter_mm / f


def check_illumination_throughput(
    bench: Bench, lamp_name: str, collector_name: str, required_na: float = 0.4
) -> RuleResult:
    """How much of the lamp's output the collector actually gathers.

    A lamp radiates into a hemisphere; the collector catches the cone it subtends,
    ``NA = r / d``, and irradiance at the specimen scales as ``NA^2``. Moving the
    collector 50% further away costs more than half the light.

    This is the constraint that makes round 3 a real puzzle. The lamp-to-specimen
    conjugate on its own is nearly insensitive -- the condenser demagnifies the lamp
    image so strongly that any collector from 28 to 55 mm lands it within a
    millimetre of the specimen -- so focus alone cannot decide the round. Throughput
    can: of the two collector positions that focus correctly, only the near one
    gathers enough light.
    """
    lamp = bench.get(lamp_name)
    collector = bench.get(collector_name)
    distance = collector.s - lamp.s
    if distance <= 0:
        return RuleResult(
            name="Throughput",
            status=Status.FAIL,
            summary="the collector is not downstream of the lamp",
            equation=f"collector at s = {collector.s:.2f} mm, lamp at s = {lamp.s:.2f} mm",
            culprit=collector_name,
            remedy="put the collector after the lamp",
        )

    collected_na = collector.semi_diameter_mm / distance
    relative = (collected_na / required_na) ** 2
    status = Status.PASS if collected_na >= required_na else Status.FAIL
    return RuleResult(
        name="Throughput",
        status=status,
        summary=(
            f"the collector gathers NA {collected_na:.3f}"
            + ("" if status is Status.PASS else f" -- only {relative * 100:.0f}% of the light needed")
        ),
        equation=(
            f"NA_collected = r / d = {collector.semi_diameter_mm:.1f} / {distance:.1f} = "
            f"{collected_na:.3f}; irradiance scales as NA^2, so this is "
            f"{relative:.2f}x the requirement"
        ),
        measured=collected_na,
        target=required_na,
        culprit=collector_name,
        remedy=(
            "move the collector closer to the lamp: it subtends a larger cone there, "
            "and irradiance goes as the square of the collected NA"
        ),
    )


def check_epi_separation(bench: Bench, beamsplitter_name: str, objective_name: str) -> RuleResult:
    """Does the episcopic illumination actually reach the specimen through the objective?

    In an epi stand the objective is its own condenser: illumination enters at the
    beamsplitter, travels *back* down through the objective to the specimen, and
    the returning light comes up the same path. The beamsplitter must therefore sit
    between the objective and the image, not anywhere else -- put it below the
    objective and there is nothing to illuminate through.
    """
    splitter = bench.get(beamsplitter_name)
    objective = bench.get(objective_name)
    status = Status.PASS if splitter.s > objective.s else Status.FAIL
    return RuleResult(
        name="Epi geometry",
        status=status,
        summary=(
            "the beamsplitter sits above the objective, so illumination reaches the "
            "specimen through it"
            if status is Status.PASS
            else "the beamsplitter is below the objective: illumination never reaches the specimen"
        ),
        equation=(
            f"{beamsplitter_name} at s = {splitter.s:.2f} mm, {objective_name} at "
            f"s = {objective.s:.2f} mm"
        ),
        culprit=None if status is Status.PASS else beamsplitter_name,
        remedy="move the beamsplitter between the objective and the intermediate image",
    )


def check_filter_set(cube, fluorophore, minimum_signal: float = 1.0,
                     maximum_bleedthrough: float = 0.25) -> RuleResult:
    """Does this filter cube actually work with this fluorophore?

    Two independent ways to fail, and a build has to clear both. A mismatched cube
    delivers no excitation where the dye absorbs, so there is nothing to see. A cube
    whose excitation and emission bands overlap leaks excitation straight to the
    detector -- and since excitation is orders of magnitude brighter than emission,
    even a small leak turns a black background grey.
    """
    signal = cube.signal(fluorophore)
    bleed = cube.bleedthrough()

    if signal < minimum_signal:
        return RuleResult(
            name="Filter set",
            status=Status.FAIL,
            summary=(
                f"the {cube.name} cube barely excites {fluorophore.name}: relative "
                f"signal {signal:.3g}"
            ),
            equation=(
                f"signal = (excitation x dichroic reflectance x dye absorption) x "
                f"(dye emission x dichroic transmittance x emission filter) = {signal:.3g}"
            ),
            measured=signal,
            target=minimum_signal,
            culprit="filter cube",
            remedy=(
                f"{fluorophore.name} absorbs near {fluorophore.excitation_peak_nm:.0f} nm "
                f"and emits near {fluorophore.emission_peak_nm:.0f} nm; pick a cube whose "
                "bands sit there"
            ),
        )

    status = Status.PASS if bleed <= maximum_bleedthrough else Status.FAIL
    return RuleResult(
        name="Filter set",
        status=status,
        summary=(
            f"{cube.name} on {fluorophore.name}: signal {signal:.3g}, bleedthrough {bleed:.3g}"
            + ("" if status is Status.PASS else " -- excitation is reaching the detector")
        ),
        equation=(
            f"bleedthrough = excitation x dichroic reflect x dichroic transmit x emission "
            f"= {bleed:.3g}; excitation band {cube.excitation.label()}, "
            f"emission band {cube.emission.label()}, dichroic edge "
            f"{cube.dichroic.edge_nm:.0f} nm"
        ),
        measured=bleed,
        target=maximum_bleedthrough,
        culprit="filter cube",
        remedy=(
            "separate the excitation and emission bands across the dichroic edge; "
            "they must not overlap, because excitation is far brighter than emission"
        ),
    )


def check_stokes_shift(fluorophore, cube) -> RuleResult:
    """Is the dichroic edge actually between excitation and emission?

    The Stokes shift is the entire reason fluorescence can be separated from its own
    illumination. If the dichroic edge does not fall inside that gap, no combination
    of filters will help.
    """
    edge = cube.dichroic.edge_nm
    low, high = fluorophore.excitation_peak_nm, fluorophore.emission_peak_nm
    inside = low < edge < high
    return RuleResult(
        name="Dichroic placement",
        status=Status.PASS if inside else Status.FAIL,
        summary=(
            f"the dichroic edge at {edge:.0f} nm sits inside {fluorophore.name}'s "
            f"{fluorophore.stokes_shift_nm:.0f} nm Stokes shift"
            if inside
            else f"the dichroic edge at {edge:.0f} nm is outside the "
            f"{low:.0f}-{high:.0f} nm gap it has to split"
        ),
        equation=(
            f"excitation peak {low:.0f} nm < dichroic {edge:.0f} nm < emission peak {high:.0f} nm"
        ),
        measured=edge,
        target=(low + high) / 2,
        units="nm",
        culprit="dichroic",
        remedy=(
            "choose a dichroic whose edge falls between the dye's excitation and "
            "emission peaks -- that gap is the whole basis of the technique"
        ),
    )


def check_filter_is_focus_neutral(
    bench: Bench,
    s_object: float,
    detector_name: str,
    filter_name: str,
    tolerance_mm: float = 0.005,
) -> RuleResult:
    """Does inserting the filter move the image?

    The practical payoff of an infinity space, and the only reason anyone went to
    the trouble of redesigning objectives around it. A plane-parallel plate in
    *converging* light displaces focus by ``t(1 - 1/n)``; in *collimated* light a
    ray ``(y, 0)`` is unchanged by any transfer, so the plate does nothing. That is
    why you can drop a filter, a dichroic or a DIC prism into an infinity space and
    not refocus -- and why on a finite-tube stand you must.
    """
    detector = bench.get(detector_name)
    plate = bench.get(filter_name)

    without = Bench(
        [e for e in bench.elements if e.name != filter_name],
        bench.folds, tuple(bench.origin), tuple(bench.initial_direction),
        list(bench.arms.values()),
    )
    s_with = bench.to_paraxial().image_plane(s_object, search_to=detector.s + 1e-6)
    s_without = without.to_paraxial().image_plane(s_object, search_to=detector.s + 1e-6)

    if s_with is None or s_without is None:
        return RuleResult(
            name="Filter neutrality",
            status=Status.FAIL,
            summary="no image forms with or without the filter, so nothing can be compared",
            equation="B = 0 has no solution",
            culprit=filter_name,
        )

    shift = abs(s_with - s_without)
    thickness = float(plate.metadata.get("thickness_mm", 0.0))
    index = float(plate.metadata.get("index", 1.52))
    predicted = thickness * (1.0 - 1.0 / index) if index else 0.0

    status = Status.PASS if shift <= tolerance_mm else Status.FAIL
    return RuleResult(
        name="Filter neutrality",
        status=status,
        summary=(
            f"inserting {filter_name} moves the image by {shift:.4f} mm"
            + (" -- it sits in collimated light, so it is focus-neutral"
               if status is Status.PASS
               else f" -- it is in converging light, where a plate shifts focus by t(1-1/n)")
        ),
        equation=(
            f"t(1 - 1/n) = {thickness:.2f} x (1 - 1/{index:.3f}) = {predicted:.4f} mm "
            f"in converging light; measured shift {shift:.4f} mm"
        ),
        measured=shift,
        target=0.0,
        units="mm",
        culprit=None if status is Status.PASS else filter_name,
        remedy=(
            "move the filter into the infinity space between the objective and the "
            "tube lens, where the bundle is collimated and a plate has no effect"
        ),
    )


def check_parfocality(bench: Bench, objective_names: list[str], parfocal_mm: float = 60.0) -> RuleResult:
    """Do all the turret objectives put their front focal plane in the same place?

    Parfocality is a mechanical promise dressed as an optical one: every objective
    in a matched set has the same shoulder-to-specimen distance, so rotating the
    turret keeps focus. For CFI60 that distance is 60 mm. Different focal lengths
    sit at different heights above the specimen, and the mount takes up the
    difference.
    """
    if not objective_names:
        return RuleResult("Parfocality", Status.NOT_APPLICABLE, "no turret fitted")

    shoulders = {}
    for name in objective_names:
        objective = bench.get(name)
        f = objective.focal_length_mm or 0.0
        # The specimen sits at the front focal plane, so the lens is f above it and
        # the shoulder is parfocal_mm above the specimen.
        shoulders[name] = objective.s - f + parfocal_mm

    spread = max(shoulders.values()) - min(shoulders.values())
    status = Status.PASS if spread <= 0.01 else Status.FAIL
    detail = ", ".join(f"{n} at {v:.2f}" for n, v in shoulders.items())
    return RuleResult(
        name="Parfocality",
        status=status,
        summary=(
            f"all {len(objective_names)} objectives share a shoulder height to "
            f"within {spread:.4f} mm"
            if status is Status.PASS
            else f"shoulder heights differ by {spread:.2f} mm, so the turret loses focus"
        ),
        equation=f"shoulder = objective_s - f + {parfocal_mm:.0f} mm: {detail}",
        measured=spread,
        target=0.0,
        units="mm",
        remedy=(
            f"a matched {parfocal_mm:.0f} mm parfocal set puts every objective's "
            "front focal plane at the same height; mixing series breaks that"
        ),
    )


def check_ring_matches_annulus(bench: Bench, ring_name: str, annulus_name: str) -> RuleResult:
    """Is the phase ring the same size as the condenser annulus?

    They are conjugate planes, so the undiffracted light from the annulus lands as
    a ring in the objective back focal plane, and the phase ring must cover exactly
    that. Too small and some surround light escapes unshifted, diluting contrast;
    too large and it starts catching diffracted light, which is the signal.
    """
    ring = bench.get(ring_name)
    annulus = bench.get(annulus_name)
    ring_in = float(ring.metadata.get("inner", 0.0))
    ring_out = float(ring.metadata.get("outer", 0.0))
    ann_in = float(annulus.metadata.get("inner", 0.0))
    ann_out = float(annulus.metadata.get("outer", 0.0))

    mismatch = max(abs(ring_in - ann_in), abs(ring_out - ann_out))
    status = (
        Status.PASS if mismatch <= 0.03
        else Status.WARN if mismatch <= 0.08
        else Status.FAIL
    )
    return RuleResult(
        name="Ring and annulus match",
        status=status,
        summary=(
            f"ring {ring_in:.2f}-{ring_out:.2f} against annulus {ann_in:.2f}-{ann_out:.2f}"
            + ("" if status is Status.PASS else f" -- out by {mismatch:.2f} of the pupil radius")
        ),
        equation=(
            f"|ring - annulus| = {mismatch:.3f} in normalized pupil radius; "
            "they are conjugate planes and must coincide"
        ),
        measured=mismatch,
        target=0.0,
        culprit=ring_name,
        remedy=(
            "match the ring to the annulus: too small lets surround light past "
            "unshifted, too large starts phase-shifting the diffracted signal"
        ),
    )


def check_ring_at_back_focal_plane(
    bench: Bench, ring_name: str, objective_name: str, tolerance_mm: float = 1.0
) -> RuleResult:
    """Is the phase ring actually in the objective's back focal plane?

    Nowhere else will do. The back focal plane is where illumination angle maps to
    position, so it is the only plane at which undiffracted light occupies a
    compact ring separable from the diffracted light spread across the pupil.
    """
    objective = bench.get(objective_name)
    ring = bench.get(ring_name)
    bfp_s = objective.s + (objective.focal_length_mm or 0.0)
    return tolerance_result(
        name="Ring at the back focal plane",
        measured=ring.s,
        target=bfp_s,
        tolerance=tolerance_mm,
        units="mm",
        equation=(
            f"{ring_name} at s = {ring.s:.2f} mm; back focal plane at "
            f"{objective.s:.2f} + {objective.focal_length_mm or 0.0:.2f} = {bfp_s:.2f} mm"
        ),
        culprit=ring_name,
        remedy=f"move the phase ring to s = {bfp_s:.2f} mm",
        relative=False,
    )


def check_crossed_polars(bench: Bench, polarizer_name: str, analyzer_name: str) -> RuleResult:
    """Are the polarizer and analyzer crossed?

    Everything in polarization microscopy is read against a dark background, so the
    two have to be at ninety degrees. A few degrees off and the background lifts
    enough to bury a weakly birefringent specimen.
    """
    import math as _math

    polarizer = float(bench.get(polarizer_name).metadata.get("angle_deg", 0.0))
    analyzer = float(bench.get(analyzer_name).metadata.get("angle_deg", 90.0))
    separation = abs((analyzer - polarizer) % 180.0)
    error = abs(separation - 90.0)

    # Malus's law: the background rises as sin^2 of the error from crossed.
    leak = _math.sin(_math.radians(error)) ** 2
    status = Status.PASS if error <= 1.0 else Status.WARN if error <= 5.0 else Status.FAIL
    return RuleResult(
        name="Crossed polars",
        status=status,
        summary=(
            f"polarizer and analyzer are {separation:.1f} degrees apart"
            + ("" if status is Status.PASS else f" -- {leak * 100:.1f}% of the light leaks through")
        ),
        equation=(
            f"|analyzer - polarizer| = {separation:.1f} deg; Malus gives a background "
            f"of sin^2({error:.1f} deg) = {leak:.4f}"
        ),
        measured=separation,
        target=90.0,
        units="deg",
        culprit=analyzer_name,
        remedy="rotate the analyzer to ninety degrees from the polarizer",
    )


def check_dic_prism(
    bench: Bench, prism_name: str, na: float, wavelength_um: float
) -> RuleResult:
    """Is the Wollaston shear right for this objective, and is there any bias?

    Two settings, two ways to get it wrong. A shear much larger than the resolution
    limit produces a visible double image rather than a relief; much smaller and
    there is no difference to detect. Zero bias gives a symmetric, dark-field-like
    image in which a gradient and its opposite look identical, so the relief has no
    direction.
    """
    prism = bench.get(prism_name)
    fraction = float(prism.metadata.get("shear_fraction", 0.0))
    bias = float(prism.metadata.get("bias_waves", 0.0))

    problems = []
    if not 0.2 <= fraction <= 1.0:
        problems.append(
            f"shear is {fraction:.2f} of the resolution limit"
            + (" -- too large, you will see a double image" if fraction > 1.0
               else " -- too small to generate a difference")
        )
    if not 0.05 <= bias <= 0.35:
        problems.append(
            f"bias is {bias:.2f} waves"
            + (" -- at zero bias the relief has no direction" if bias < 0.05
               else " -- past a quarter wave the background washes out")
        )

    status = Status.PASS if not problems else Status.FAIL
    return RuleResult(
        name="DIC prism",
        status=status,
        summary="shear and bias are both in range" if not problems else "; ".join(problems),
        equation=(
            f"shear = {fraction:.2f} x (0.61 lambda / NA) = "
            f"{fraction * 0.61 * wavelength_um / max(na, 1e-3):.3f} um; "
            f"bias = {bias:.2f} waves"
        ),
        measured=fraction,
        target=0.6,
        culprit=prism_name,
        remedy=(
            "shear a little under the resolution limit, and bias around an eighth "
            "of a wave to put the working point on the steep part of the sine"
        ),
    )
